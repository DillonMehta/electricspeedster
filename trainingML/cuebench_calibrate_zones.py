#!/usr/bin/env python3
"""
cuebench_calibrate_zones.py — set the 6 quality-zone boundaries from real data
==============================================================================
Runs the PRODUCTION engine (local 4D model + EB shrinkage) over the whole
labelled corpus (data/*.jsonl), then reports:

  1. the n_effective (support) distribution  -> picks the thin-session gate,
  2. the population prior (mean raw composite) -> the EB shrinkage target,
  3. the EB-shrunk composite distribution over SUBSTANTIVE sessions
     (those that pass the gate) -> where the 6 zones should fall.

We deliberately do NOT use academic letter-grade cutoffs. We look at where the
substantive sessions actually cluster and propose 6 bands such that the urgent
tier (zones 5 & 6) is a small minority. The numbers this prints become the
named constants in cuebench_signals.py (treat them as v1 estimates to retune).

Usage:
  python cuebench_calibrate_zones.py                 # full corpus
  python cuebench_calibrate_zones.py --limit 1000    # quick sample
  python cuebench_calibrate_zones.py --gate 6 --k 6  # try other gate/K values
"""
from __future__ import annotations
import argparse, glob, json, math, re, sys

HEADER_RE = re.compile(r"PROMPTS=(\d+)\s+TOOLS=(\d+)", re.I)


def support_from_text(text: str) -> int | None:
    """n_effective = n_prompts + n_tools, parsed from a digest's header line.
    Mirrors cuebench_signals.n_effective so calibration == production."""
    m = HEADER_RE.search(text or "")
    if not m:
        return None
    return int(m.group(1)) + int(m.group(2))


def percentiles(xs, ps):
    xs = sorted(xs)
    out = {}
    n = len(xs)
    for p in ps:
        if n == 0:
            out[p] = None
            continue
        k = (n - 1) * (p / 100.0)
        lo, hi = math.floor(k), math.ceil(k)
        if lo == hi:
            out[p] = xs[int(k)]
        else:
            out[p] = xs[lo] * (hi - k) + xs[hi] * (k - lo)
    return out


def histogram(xs, lo=0, hi=100, width=5):
    bins = {}
    for x in xs:
        b = int(min(hi - 1e-9, max(lo, x)) // width * width)
        bins[b] = bins.get(b, 0) + 1
    n = max(1, len(xs))
    lines = []
    for b in range(lo, hi, width):
        c = bins.get(b, 0)
        bar = "#" * round(40 * c / n)
        lines.append(f"  [{b:3d},{b+width:3d})  {c:5d} {100*c/n:5.1f}%  {bar}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cuebench_model")
    ap.add_argument("--data", default="data/*.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="cap sessions (0 = all)")
    ap.add_argument("--gate", type=int, default=None,
                    help="thin-session min support to test (default: auto from p10)")
    ap.add_argument("--k", type=float, default=None,
                    help="EB prior strength K (default: = gate)")
    ap.add_argument("--out", default=None, help="write per-session rows to this CSV")
    a = ap.parse_args()

    paths = sorted(glob.glob(a.data))
    if not paths:
        sys.exit(f"no data matched {a.data}")

    from model_infer import ModelScorer
    import cuebench_signals as sig
    import cuebench_calibrate as calib
    scorer = ModelScorer(a.model)
    print(f"[calibrate] model loaded from {scorer.loaded_from}; files={paths}", file=sys.stderr)

    rows = []  # (sid, support, raw_composite)
    n_seen = 0
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = rec.get("text", "")
            sup = support_from_text(text)
            if sup is None:
                continue
            vectors = scorer.score(text)
            vectors = calib.decorrelate(vectors, sup)   # mirror production: length-correct,
            vectors = calib.recenter(vectors)           # global level recenter,
            raw = sig.composite(vectors)                # then WEIGHTED composite (not even mean)
            rows.append((rec.get("sid", ""), sup, raw))
            n_seen += 1
            if n_seen % 500 == 0:
                print(f"  scored {n_seen}…", file=sys.stderr)
            if a.limit and n_seen >= a.limit:
                break
        if a.limit and n_seen >= a.limit:
            break

    sups = [s for _, s, _ in rows]
    raws = [r for _, _, r in rows]
    print(f"\n==== corpus: {len(rows)} sessions ====")

    # ---- 1. support (n_effective) distribution -> the thin-session gate ----
    sp = percentiles(sups, [1, 5, 10, 25, 50, 75, 90])
    print("\n-- support (n_effective = prompts + tools) --")
    print("  percentiles:", {k: round(v, 1) for k, v in sp.items()})
    print("  support histogram (0..60, width 3):")
    print(histogram([min(60, s) for s in sups], 0, 60, 3))

    gate = a.gate if a.gate is not None else int(round(sp[10]))
    gate = max(3, gate)
    K = a.k if a.k is not None else float(gate)
    n_thin = sum(1 for s in sups if s < gate)
    print(f"\n  => THIN gate (min support) = {gate}  "
          f"({n_thin} / {len(rows)} = {100*n_thin/len(rows):.1f}% gated as Insufficient)")

    # ---- 2. population prior = mean raw composite over SUBSTANTIVE sessions ----
    sub = [(sid, s, r) for sid, s, r in rows if s >= gate]
    sub_raw = [r for _, _, r in sub]
    prior = sum(sub_raw) / max(1, len(sub_raw))
    print(f"\n-- EB shrinkage --")
    print(f"  PRIOR_MEAN (mean raw composite, substantive) = {prior:.2f}")
    print(f"  PRIOR_STRENGTH K = {K}")

    def eb(raw, sup):
        return (sup * raw + K * prior) / (sup + K)

    eb_sub = [eb(r, s) for _, s, r in sub]

    # ---- 3. EB-shrunk distribution over substantive sessions -> 6 bands ----
    ep = percentiles(eb_sub, [2, 5, 8, 12, 25, 50, 60, 75, 85, 90, 95])
    print(f"\n-- EB-shrunk composite over {len(sub)} SUBSTANTIVE sessions --")
    print("  percentiles:", {k: round(v, 1) for k, v in ep.items()})
    print("  EB-shrunk histogram (0..100, width 5):")
    print(histogram(eb_sub, 0, 100, 5))

    # Target shares per zone (definition-driven), urgent tier kept a small minority:
    #   1 Dialed in  ~ top 15%        4 Inconsistent    ~ next ~17%
    #   2 Solid      ~ next ~30%      5 Needs attention ~ next ~7%   (urgent)
    #   3 Developing ~ next ~23%      6 Critical        ~ bottom ~3% (urgent)
    # We read the cutoffs straight off the EB-shrunk percentiles, then the agent
    # can hand-snap to round numbers near any natural valley in the histogram.
    cut_p = percentiles(eb_sub, [3, 10, 33, 56, 85])  # lower bounds of zones 6,5,4,3,2->1
    print("\n-- proposed zone cutoffs (lower bound of each zone, on EB-shrunk score) --")
    labels = ["z6 Critical (bottom)", "z5 Needs attention", "z4 Inconsistent",
              "z3 Developing", "z2 Solid", "z1 Dialed in (top)"]
    cuts = [round(cut_p[3]), round(cut_p[10]), round(cut_p[33]),
            round(cut_p[56]), round(cut_p[85])]
    print(f"  ZONE_CUTOFFS (z2,z3,z4,z5,z6 ascending lower bounds) ~ "
          f"{sorted(set(cuts))}")
    print("  meaning (ascending):")
    bounds = [0] + sorted(cuts) + [101]
    names = ["6 Critical", "5 Needs attention", "4 Inconsistent",
             "3 Developing", "2 Solid", "1 Dialed in"]
    for i, nm in enumerate(names):
        lo, hi = bounds[i], bounds[i + 1] - 1
        share = sum(1 for v in eb_sub if lo <= v <= hi) / max(1, len(eb_sub))
        urgent = " (URGENT)" if i < 2 else ""
        print(f"    {nm:18s}{urgent:9s} [{lo:3d}..{hi:3d}]  {100*share:5.1f}% of substantive")
    urg = sum(1 for v in eb_sub if v < sorted(cuts)[1]) / max(1, len(eb_sub))
    print(f"\n  urgent tier (zones 5+6) = {100*urg:.1f}% of substantive "
          f"(should be a small minority)")

    if a.out:
        import csv
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sid", "support", "raw_composite", "eb_score", "gated"])
            for sid, s, r in rows:
                w.writerow([sid, s, round(r, 2),
                            round(eb(r, s), 2) if s >= gate else "",
                            int(s < gate)])
        print(f"\n[wrote] {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
