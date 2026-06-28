#!/usr/bin/env python3
"""
cuebench_dryrun.py — prove the new payload + caching/dedup on session 6e076b3b
==============================================================================
Runs the REAL scoring model and the REAL payload-assembly / cache / dedup code on the
this-session golden (6e076b3b). Two scenarios:

  (1) NO BYOK key  -> title == last-use date/time (no generated name), no insights/trace.
  (2) WITH a key   -> content-aware title (from the first prompt, via the user's OWN BYOK
                      endpoint) + numbers-only insights; raw prompt itself never in payload.

In BOTH scenarios the same session is run TWICE to prove:
  * the 2nd run makes ZERO BYOK calls (cached generation is reused), and
  * the 2nd run does NOT re-POST (already-sent dedup on sessionId).

No real network/keys are used: scenario (2) uses a fake generator (counts calls, returns
canned NEUTRAL text), and POST acceptance is simulated. A temp SQLite db is used so real
local state is untouched.

  python3 cuebench_dryrun.py [path-to-6e076b3b.jsonl]
"""
from __future__ import annotations
import json, os, sys, tempfile

import cuebench_agent as agent
from cuebench_gen import Generator
from cuebench_store import Store
from model_infer import ModelScorer

GOLDEN = sys.argv[1] if len(sys.argv) > 1 else "6e076b3b-f864-464a-a952-5de43fb31ae4.jsonl"
KEEP_KEYS = {"employeeId", "sessionId", "title", "taskType", "taskLabel", "model", "cli", "date",
             "duration", "score", "quality", "vectors", "breakdown", "cost",
             "insights", "trace", "specificity"}


class FakeGenerator(Generator):
    """A Generator that never hits the network — returns canned NEUTRAL output and counts
    calls, so we can prove cache behaviour offline without spending BYOK quota."""
    def _complete(self, system: str, user: str, max_tokens: int = 600):
        if not self.enabled:
            return None
        self.calls += 1
        if "extremely short title" in system:        # content_title (from the first prompt)
            return "Thorough, well-verified build session"
        if "strengths" in system:
            return json.dumps({
                "strengths": [
                    {"axis": "diligence", "point": "Verification was run consistently before committing work."},
                    {"axis": "delegation", "point": "Work was decomposed into well-sized, mostly clean tool calls."},
                ],
                "improvements": [
                    {"axis": "discernment", "point": "A surfaced error was not caught before the session ended."},
                    {"axis": "description", "point": "Requests could have stated explicit acceptance criteria up front."},
                ],
            })
        if "timeline" in system:
            return json.dumps([
                {"t": "00:00", "sig": "info", "label": "Session started", "detail": "Work began across several prompts."},
                {"t": "30:00", "sig": "good", "label": "Verified and committed", "detail": "Tests/lint were run and changes committed."},
            ])
        return None
    # specificity is no longer a BYOK method (computed locally now) — nothing to fake here.


def _raw_strings(parsed: dict) -> list[str]:
    """Distinctive raw fragments that must NEVER appear in the payload."""
    out = []
    if parsed.get("cwd"):
        out.append(parsed["cwd"])                       # a file path
    for p in parsed.get("prompts", []):
        if p and len(p) >= 25:
            out.append(p[:25])                          # raw prompt fragment
    for c in parsed.get("tool_cmds", []):
        if c and len(c) >= 25:
            out.append(c[:25])                          # raw command fragment
    return out


def _check_payload_shape(payload: dict) -> list[str]:
    problems = []
    for forbidden in ("checklist", "task", "_meta", "grade"):
        if forbidden in payload:
            problems.append(f"payload still contains forbidden field {forbidden!r}")
    extra = set(payload) - KEEP_KEYS
    if extra:
        problems.append(f"payload has unexpected keys: {sorted(extra)}")
    return problems


def _neutrality_scan(payload: dict, parsed: dict,
                     content_insights: bool = False) -> tuple[bool, list[str]]:
    # `title` and `trace` are intentional CONTENT-AWARE opt-ins (generated via the user's own
    # BYOK from the first prompt / the session timeline — they are MEANT to name files,
    # commands, and operator words). When prompt-informed insights is ON, `insights` joins them
    # (coaching may reference what was asked). Scan only the fields that MUST stay numbers-only;
    # a raw fragment in any of those would be a real leak.
    exempt = {"title", "trace"} | ({"insights"} if content_insights else set())
    numbers_only = {k: v for k, v in payload.items() if k not in exempt}
    blob = json.dumps(numbers_only, ensure_ascii=False)
    leaks = [frag for frag in _raw_strings(parsed) if frag in blob]
    return (not leaks), leaks


def run_scenario(label: str, gen: Generator, scorer: ModelScorer, parsed: dict):
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    sid = agent.make_sid(parsed["session_uuid"])
    db = os.path.join(tempfile.mkdtemp(prefix="cuebench_dry_"), "state.db")
    store = Store(db)

    # ---------- RUN 1 ----------
    print(f"\n--- RUN 1 (gen status: {gen.status()}) ---")
    before = gen.calls
    payload, status = agent.build_payload(parsed, scorer, store, gen)
    run1_calls = gen.calls - before
    print(f"gen_status={status}   BYOK calls this run={run1_calls}")
    print("EXACT payload that WOULD be sent:")
    print(json.dumps(payload, indent=2))

    shape = _check_payload_shape(payload)
    print("\nPayload shape:", "OK (no checklist/task/_meta; only the allowed fields)"
          if not shape else "PROBLEMS: " + "; ".join(shape))
    ok, leaks = _neutrality_scan(payload, parsed, content_insights=gen.insights_prompts_enabled)
    _aware = "title/trace" + ("/insights" if gen.insights_prompts_enabled else "")
    print("Neutrality scan:", "PASS — no raw prompt/path/command in the numbers-only fields"
          f" ({_aware} are content-aware by design)"
          if ok else f"FAIL — leaked fragments: {leaks}")
    if "insights" in payload:
        print("insights present:", sum(len(payload['insights'].get(b, [])) for b in ('strengths', 'improvements')),
              "items, each tied to an axis")
    else:
        print("insights: ABSENT (omitted)")
    print("trace:", "present" if "trace" in payload else "ABSENT (omitted)")
    print("specificity:", payload["specificity"] if "specificity" in payload else "ABSENT (omitted)")
    print(f"title -> {payload['title']!r}   (local keyphrase extraction — short, not verbatim, no BYOK)")
    print(f"taskType -> {payload['taskType']!r}  taskLabel -> {payload['taskLabel']!r}"
          "   (local embedding classifier)")

    # Simulate the dashboard accepting the POST.
    store.mark_posted(sid, parsed["n_lines"])
    print("POST accepted (simulated) -> recorded as sent for", sid)

    # ---------- RUN 2 (same session again) ----------
    print(f"\n--- RUN 2 (same session re-processed) ---")
    before = gen.calls
    prev_n = store.get_n_lines(sid)
    appended = prev_n is not None and parsed["n_lines"] > prev_n
    already = store.is_posted(sid)
    payload2, status2 = agent.build_payload(parsed, scorer, store, gen)
    run2_calls = gen.calls - before
    post_action = "SKIP re-POST (already sent, unchanged)" if (already and not appended) else "would re-POST as UPDATE"
    print(f"generation: {('REUSED cache' if status2 == 'cache' else status2)}   BYOK calls this run={run2_calls}")
    print(f"POST dedup: is_posted={already}  appended={appended}  -> {post_action}")

    # Assertions that back the printed claims.
    assert run2_calls == 0, f"expected ZERO BYOK calls on run 2, got {run2_calls}"
    assert payload2 == payload, "cached generation must be byte-stable across runs"
    if gen.enabled:
        assert status2 == "cache", f"run 2 should reuse cache, got {status2}"
    assert already and not appended, "run 2 must be deduped (already sent, unchanged)"
    print("\nSUMMARY:", f"run1 BYOK calls={run1_calls}, run2 BYOK calls={run2_calls} (cache reused), "
          f"run2 POST={post_action}.")
    store.close()


def main():
    print(f"[load] scoring model (CPU) ...", file=sys.stderr)
    scorer = ModelScorer(agent.MODEL_DIR)
    parsed = agent.parse_transcript(GOLDEN)
    print(f"[parsed] {os.path.basename(GOLDEN)}: sid={agent.make_sid(parsed['session_uuid'])} "
          f"prompts={parsed['n_prompts'] if 'n_prompts' in parsed else len(parsed['prompts'])} "
          f"n_lines={parsed['n_lines']} repo={parsed['cwd']}", file=sys.stderr)

    # Scenario 1: NO key -> generation OFF.
    run_scenario("SCENARIO 1 — NO BYOK KEY (generation OFF)",
                 Generator(key="", provider="anthropic"), scorer, parsed)

    # Scenario 2: WITH a key (OpenAI provider, specificity opt-in ON) -> content-aware title
    # (from the first prompt) + numbers-only insights + the specificity number. Fake completion
    # + fake specificity -> no real spend, no raw prompts actually sent anywhere.
    run_scenario("SCENARIO 2 — WITH BYOK KEY (numbers-only insights; fake). Title/taskType/"
                 "specificity are local regardless of the key.",
                 FakeGenerator(key="sk-FAKEKEY-not-real", provider="openai",
                               model="gpt-4o-mini"), scorer, parsed)


if __name__ == "__main__":
    main()
