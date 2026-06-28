#!/usr/bin/env python3
"""Regression tests for CueBench signal extraction (cuebench_agent.parse_transcript
+ cuebench_signals). These lock the signals to independently-verified ground truth
for two real sessions so the extraction bugs we fixed can't silently come back:

  - churn must be the written-content volume (was 0 — git-diff path returned nothing)
  - loops must count EXACT-duplicate re-runs (was 47 from a consecutive-pair heuristic)
  - n_tools must count ALL tool calls (was 189 — bash only)
  - prompts must apply the human-origin filter (was 47 incl. skill/interrupt noise)
  - commits must be isolated to the session's time window (was 7 leaked from history)
  - duration must be ACTIVE time, not idle-inflated wall-clock (was 19h)

Run:  python3 -m pytest test_extraction.py -v     (or just: python3 test_extraction.py)
"""
from __future__ import annotations
import os
import pytest

from cuebench_agent import parse_transcript
import cuebench_signals as sig

HERE = os.path.dirname(os.path.abspath(__file__))

# 6e076b3b — a ~6h, 44-prompt build session in /Users/dillonmehta/cuebench (10 commits).
SESS_BUILD = "6e076b3b-f864-464a-a952-5de43fb31ae4.jsonl"
# 63c1b9c9 — a 1-prompt, immediately-interrupted exploration in .../cuebenchv2 that
#            committed nothing. The repo it sat in has 7 commits from LATER sessions.
SESS_INTERRUPTED = "63c1b9c9-e4ab-4a27-b625-a759fe96802c.jsonl"


def _find(name: str) -> str:
    for cand in (os.path.join(HERE, name),
                 os.path.expanduser(f"~/{name}"),
                 os.path.expanduser(f"~/.claude/projects/-Users-dillonmehta-cuebench/{name}")):
        if os.path.isfile(cand):
            return cand
    pytest.skip(f"fixture transcript not found: {name}")


def _inputs(name: str) -> tuple[dict, dict]:
    """parse + git + build_inputs, exactly as the agent's build_payload does."""
    parsed = parse_transcript(_find(name))
    repo = parsed["cwd"]
    git = (sig.git_signals(repo, parsed["first_ts"], parsed["last_ts"]) if repo
           else {"n_commits": 0, "reverts": 0, "survival_proxy": None})
    inputs = sig.build_inputs(parsed["prompts"], parsed["tool_cmds"], parsed["n_tools"],
                              parsed["n_edits"], parsed["loops"], parsed["churn"], git)
    return parsed, inputs


# ---------------------------------------------------------------------------
# 6e076b3b — the reference build session
# ---------------------------------------------------------------------------

def test_build_session_core_signals():
    parsed, i = _inputs(SESS_BUILD)
    # human-origin prompt filter: 44 real turns, NOT the 47 the old filter let through
    assert i["n_prompts"] == 44, "human-origin prompt filter (skill/interrupt noise dropped)"
    # ALL tool calls, not bash-only (was 189)
    assert i["n_tools"] == 415, "n_tools must count every tool call, not just Bash"
    assert i["n_edits"] == 190
    assert i["verify_runs"] == 30
    # exact-duplicate loops ~7 (1.7% of tools), NOT 47 from the consecutive-pair bug
    assert i["loops"] == 7, "loops = exact-duplicate re-runs (full-input signature)"
    assert i["loops"] / i["n_tools"] < 0.05
    # churn is written-content volume, not 0 (the git-diff path returned nothing)
    assert 8000 <= i["churn"] <= 10000, f"churn should be ~9000 lines, got {i['churn']}"
    assert i["churn"] != 0


def test_build_session_commits_windowed():
    parsed, i = _inputs(SESS_BUILD)
    if not (parsed["cwd"] and os.path.isdir(os.path.join(parsed["cwd"], ".git"))):
        pytest.skip("session repo not present to verify commit attribution")
    assert i["n_commits"] == 10
    assert i["reverts"] == 0
    assert i["survival_proxy"] == 1.0


def test_build_session_active_duration_not_wall_clock():
    parsed, _ = _inputs(SESS_BUILD)
    wall = (sig._git_when(parsed["last_ts"]), sig._git_when(parsed["first_ts"]))
    assert all(wall)
    dur = parsed["duration_s"]
    # active time ~5.6h (≈20k s), and DRAMATICALLY less than the 19h (~68k s) wall span
    assert 10_000 <= dur <= 30_000, f"active duration out of band: {dur}s"
    assert dur < 40_000, "duration must exclude the multi-hour idle gap (was 19h wall-clock)"


# ---------------------------------------------------------------------------
# 63c1b9c9 — the interrupted session that must NOT inherit the repo's history
# ---------------------------------------------------------------------------

def test_interrupted_session_no_commit_leak():
    parsed, i = _inputs(SESS_INTERRUPTED)
    # The session committed nothing; the repo's 7 commits are from later sessions.
    assert i["n_commits"] == 0, "commits must be windowed to the session (no history leak)"
    assert i["n_edits"] == 0
    assert i["churn"] == 0
    assert i["loops"] == 0


def test_interrupted_session_prompt_filter_drops_interrupt():
    parsed, i = _inputs(SESS_INTERRUPTED)
    # 1 real human prompt; the "[Request interrupted by user]" sentinel is NOT a turn.
    assert i["n_prompts"] == 1, "interrupt sentinel must not count as a human prompt"


# ---------------------------------------------------------------------------
# Quality scale — confidence gate + 6 zones + EB shrinkage (cuebench_signals.quality)
# These are model-free: quality() takes a raw composite + the signal inputs, so we can
# lock the gate/zoning/shrinkage behaviour without loading the scorer.
# ---------------------------------------------------------------------------

def _mk(n_prompts, n_tools):
    """Minimal inputs dict carrying just what n_effective reads."""
    return {"n_prompts": n_prompts, "n_tools": n_tools}


def test_confidence_gate_runs_first_and_is_not_a_zone():
    # Below the support threshold -> Insufficient signal, NOT placed on the scale,
    # regardless of how high the raw composite is.
    q = sig.quality(95.0, _mk(1, sig.THIN_SESSION_MIN_SUPPORT - 9))
    assert q["state"] == "insufficient_signal"
    assert q["label"] == sig.INSUFFICIENT_SIGNAL
    assert "zone" not in q, "Insufficient signal must not be assigned one of the 6 zones"
    assert q["support"] < sig.THIN_SESSION_MIN_SUPPORT
    assert q["urgent"] is False


def test_at_or_above_gate_is_assessed_into_a_zone():
    # Exactly at the threshold -> assessed (gate is a floor, inclusive).
    q = sig.quality(70.0, _mk(2, sig.THIN_SESSION_MIN_SUPPORT - 2))
    assert q["support"] == sig.THIN_SESSION_MIN_SUPPORT
    assert q["state"] == "assessed"
    assert 1 <= q["zone"] <= 6
    assert q["key"] in {z["key"] for z in sig.ZONES}


def test_n_effective_is_prompts_plus_tools():
    assert sig.n_effective(_mk(3, 14)) == 17
    # fixtures: the build session is deeply substantive; the interrupted one is borderline
    _, b = _inputs(SESS_BUILD)
    assert sig.n_effective(b) == b["n_prompts"] + b["n_tools"] >= 400


def test_eb_shrinkage_pulls_thin_sessions_toward_prior():
    # A high raw on thin support is regularised DOWN toward the prior; the same raw on
    # heavy support keeps (≈) its value. This is the whole point of the shrinkage.
    thin = sig.quality(92.0, _mk(2, sig.THIN_SESSION_MIN_SUPPORT))     # just substantive
    busy = sig.quality(92.0, _mk(40, 400))                            # heavy support
    assert thin["eb_score"] < thin["raw_score"], "thin high score must shrink toward prior"
    assert abs(busy["eb_score"] - busy["raw_score"]) <= 1, "busy session keeps its raw score"
    assert thin["eb_score"] < busy["eb_score"]
    # And a very low raw on thin support is pulled UP toward the prior (symmetric).
    low_thin = sig.quality(15.0, _mk(2, sig.THIN_SESSION_MIN_SUPPORT))
    assert low_thin["eb_score"] > low_thin["raw_score"]


def test_zone_boundaries_match_named_constants():
    # On a heavy-support session shrinkage ~0, so eb_score ≈ raw and we can check the
    # cutoffs directly. Walk each boundary; the lower const is the inclusive floor.
    big = _mk(50, 500)
    cases = [
        (sig.ZONE_CRITICAL_MAX - 1,        "critical",       True),
        (sig.ZONE_CRITICAL_MAX,            "needs_attention", True),
        (sig.ZONE_NEEDS_ATTENTION_MAX - 1, "needs_attention", True),
        (sig.ZONE_NEEDS_ATTENTION_MAX,     "inconsistent",   False),
        (sig.ZONE_INCONSISTENT_MAX - 1,    "inconsistent",   False),
        (sig.ZONE_INCONSISTENT_MAX,        "developing",     False),
        (sig.ZONE_DEVELOPING_MAX - 1,      "developing",     False),
        (sig.ZONE_DEVELOPING_MAX,          "solid",          False),
        (sig.ZONE_SOLID_MAX - 1,           "solid",          False),
        (sig.ZONE_SOLID_MAX,               "dialed_in",      False),
    ]
    for raw, key, urgent in cases:
        q = sig.quality(float(raw), big)
        assert q["eb_score"] == raw, f"heavy support should keep raw {raw}, got {q['eb_score']}"
        assert q["key"] == key, f"eb {raw} -> {q['key']}, expected {key}"
        assert q["urgent"] is urgent, f"urgent flag wrong for zone {key}"


def test_urgent_tier_is_only_zones_5_and_6():
    urgent_keys = {z["key"] for z in sig.ZONES if z["urgent"]}
    assert urgent_keys == {"needs_attention", "critical"}
    assert all(z["zone"] in (5, 6) for z in sig.ZONES if z["urgent"])


def test_zone_and_reported_score_are_coherent():
    # The reported eb_score must be the exact value the zone was chosen from (no
    # off-by-rounding drift between the headline zone and the number shown).
    for raw in range(0, 101, 7):
        q = sig.quality(float(raw), _mk(30, 300))
        assert sig.zone_for_score(q["eb_score"])["key"] == q["key"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
