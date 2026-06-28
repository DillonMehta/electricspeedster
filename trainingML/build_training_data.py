#!/usr/bin/env python3
"""
CueBench — build_training_data (distill the judge into a local model)
====================================================================
Turns swechat_rich.json (digests) + judgments.json (gpt-5.5 4D scores) into
train/val/test JSONL of {text, labels:[del,desc,disc,dil], mask:[...]}.

The text is the SAME digest the judge scored (clean distillation: learn the
mapping digest -> judge score). Token-budgeted so it fits a small encoder.
'mask' marks which axes are present (judge returned N/A for some) so training
can skip absent axes per-session instead of inventing a 0.

  python build_training_data.py --rich swechat_rich.json --judgments judgments.json --out data/
"""
from __future__ import annotations
import argparse, json, os, random

AXES = ["delegation", "description", "discernment", "diligence"]

def jscores(entry):
    """Return ([4 scores], [4 mask]) averaging the two judge runs where present."""
    if not isinstance(entry, dict): return None
    if "judgment" in entry and isinstance(entry["judgment"], dict):
        runs = [entry.get("judgment") or {}, entry.get("judgment_2") or {}]
    else:
        runs = [entry]
    out, mask = [], []
    for ax in AXES:
        vals = []
        for j in runs:
            v = j.get(ax)
            if isinstance(v, dict) and isinstance(v.get("score"), (int, float)):
                vals.append(float(v["score"]))
        if vals:
            out.append(sum(vals)/len(vals)/100.0)  # scale to 0-1 for regression
            mask.append(1.0)
        else:
            out.append(0.0); mask.append(0.0)
    if sum(mask) == 0: return None
    return out, mask

def digest_text(rec, max_prompt_chars=4000):
    """Token-budgeted text: signals line + prompts (the driving content)."""
    s = rec["inputs"]
    head = (f"PROMPTS={s['n_prompts']} TOOLS={s['n_tools']} EDITS={s['n_edits']} "
            f"VERIFY={s['verify_runs']} LOOPS={s['loops']} COMMITS={s['n_commits']} "
            f"CHURN={s['churn']} SURVIVAL={s.get('survival_proxy')} REVERTS={s['reverts']}")
    vc = ""  # verify commands aren't in rich inputs; signals line carries the count
    prompts = []
    budget = max_prompt_chars
    for i, p in enumerate(s.get("prompts", []), 1):
        if budget <= 0: break
        chunk = f" [P{i}] {p[:600]}"
        prompts.append(chunk); budget -= len(chunk)
    return head + "\nPROMPTS:" + "".join(prompts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rich", required=True)
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rich = json.load(open(a.rich))
    J = json.load(open(a.judgments))

    rows = []
    for sid, rec in rich.items():
        js = jscores(J.get(sid, {}))
        if js is None: continue
        labels, mask = js
        text = digest_text(rec)
        if not text.strip() or rec["inputs"].get("n_prompts", 0) == 0:
            continue
        rows.append({"sid": sid, "text": text, "labels": labels, "mask": mask})

    random.Random(a.seed).shuffle(rows)
    n = len(rows)
    n_test = max(int(n*0.10), 50); n_val = max(int(n*0.10), 50)
    test, val, train = rows[:n_test], rows[n_test:n_test+n_val], rows[n_test+n_val:]
    for name, part in [("train", train), ("val", val), ("test", test)]:
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            for r in part: f.write(json.dumps(r)+"\n")
    print(f"total usable: {n}  ->  train {len(train)}  val {len(val)}  test {len(test)}")

    # axis coverage (how many sessions have a real score per axis)
    import numpy as np
    M = np.array([r["mask"] for r in rows])
    for i, ax in enumerate(AXES):
        print(f"  {ax:12} labeled in {int(M[:,i].sum())}/{n} sessions")
    # label distribution sanity
    L = np.array([r["labels"] for r in rows])
    for i, ax in enumerate(AXES):
        col = L[M[:,i]==1, i]
        if len(col): print(f"  {ax:12} score range {col.min()*100:.0f}-{col.max()*100:.0f} mean {col.mean()*100:.0f}")
    print(f"\nwrote train/val/test to {a.out}/")

if __name__ == "__main__":
    main()
