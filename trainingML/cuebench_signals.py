#!/usr/bin/env python3
"""
cuebench_signals.py — consolidated deterministic signals for production scoring
==============================================================================
Everything the live agent needs to turn ONE Claude Code session into the inputs
the trained model expects, plus the deterministic Ds and checklist derivation.

This is the clean production version of logic that was scattered across the
research scripts (cuebench_extract2.py, cuebench_mine.py, cuebench_specificity.py,
build_training_data.py). Import from here; do not reverse-engineer the research files.

CRITICAL: digest_text() below MUST stay byte-for-byte compatible with the format
the model was trained on (build_training_data.py). If you change it, the model's
inputs drift and scores become meaningless. Don't touch it without retraining.
"""
from __future__ import annotations
import re, math, subprocess, json
from collections import Counter

# ============================================================================
# 1. VERIFICATION CLASSIFIER  (Diligence signal)  — from cuebench_mine.py
#    Detects test/typecheck/lint/build commands in a bash string.
# ============================================================================
_PAT = {
 "test": re.compile(r"\b(pytest|py\.test|unittest|nose2|jest|vitest|mocha|jasmine|ava|go\s+test|cargo\s+test|rspec|phpunit|dotnet\s+test|bun\s+test|mvn\s+test|gradle\s+test|ctest)\b|\bnpm\s+(run\s+)?test\b|\byarn\s+test\b|\bpnpm\s+(run\s+)?test\b", re.I),
 "typecheck": re.compile(r"\b(tsc|mypy|pyright|pyre|flow\s+check)\b", re.I),
 "lint": re.compile(r"\b(ruff|eslint|flake8|pylint|golangci-lint|clippy|cargo\s+clippy|shellcheck|vet|go\s+vet)\b", re.I),
 "build": re.compile(r"\b(make|cmake|cargo\s+build|go\s+build|mvn\s+(package|install|compile)|gradle\s+build|webpack|vite\s+build|tsc\s+-b|dotnet\s+build)\b|\bnpm\s+run\s+build\b|\byarn\s+build\b|\bpnpm\s+(run\s+)?build\b", re.I)}
_VC = {"test","typecheck","lint","build"}
_SP = re.compile(r"&&|\|\||;|\n|\|")
_HD = re.compile(r"<<-?\s*(['\"]?)(?P<tag>\w+)\1.*?\n[ \t]*(?P=tag)[ \t]*(?:\n|$)", re.DOTALL)
_Q  = re.compile(r"'[^']*'|\"[^\"]*\"", re.DOTALL)

def is_verification(cmd: str) -> bool:
    """True if a bash command runs tests/typecheck/lint/build."""
    if not isinstance(cmd, str) or not cmd: return False
    c = _Q.sub(" ", _HD.sub(" ", cmd)); cats = set()
    for part in _SP.split(c):
        for k, p in _PAT.items():
            if p.search(part): cats.add(k)
    return bool(cats & _VC)

# ============================================================================
# 2. GIT SIGNALS  (Delegation/Diligence/outcome-risk) — production replacement
#    for the parquet-based extraction. Run against the repo the session edited.
#    commits / reverts / survival anchored to the SESSION'S TIME WINDOW.
#
#    NOTE: churn is NOT computed here. It is derived from the written edit content
#    in the transcript (see cuebench_agent.parse_transcript), which is robust and
#    always available; git-diff churn was fragile (it returned 0 whenever the
#    repo had <N commits or the start ref didn't resolve) and is decoupled now.
# ============================================================================
_REVERT = re.compile(r"\brevert(s|ed|ing)?\b|\brollback\b|\bundo\b", re.I)
_COMMIT_GRACE_S = 300   # commits up to 5min after the last logged event still count
                        # (a session often commits as its final act, just after its
                        #  last tool result), but not a *later* session's commits.

def _git_when(ts) -> str | None:
    """A JSONL timestamp (ISO-8601 string or epoch float) -> a tz-AWARE ISO string
    git's --since/--until parse unambiguously. Passing a naive string makes git
    interpret it in LOCAL time, which silently shifts the window by the tz offset
    (the bug that made the session-window filter miss every commit)."""
    from datetime import datetime, timezone, timedelta
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return None

def _git_until(last_ts) -> str | None:
    from datetime import datetime, timezone, timedelta
    w = _git_when(last_ts)
    if w is None:
        return None
    try:
        return (datetime.fromisoformat(w) + timedelta(seconds=_COMMIT_GRACE_S)).isoformat()
    except Exception:
        return w

def git_signals(repo_dir: str, first_ts=None, last_ts=None) -> dict:
    """
    Commits / reverts / survival for ONE session, isolated to its time window.

    Commits are attributed to the session ONLY if their commit date falls within
    [first_ts, last_ts + grace] (tz-aware). This fixes the leak where a 1-prompt
    session in an untracked subfolder of a busy repo inherited the parent repo's
    entire history: there is no "last N commits" fallback any more — a session
    that committed nothing reads as zero commits, not as all of git history.

    Falls back to zeros if not a git repo / no window. NEVER raises.
    """
    def run(args):
        try:
            return subprocess.run(["git","-C",repo_dir]+args, capture_output=True,
                                  text=True, timeout=20).stdout
        except Exception:
            return ""
    out = {"n_commits": 0, "reverts": 0, "survival_proxy": None}
    if not repo_dir:
        return out
    since, until = _git_when(first_ts), _git_until(last_ts)
    if not since or not until:
        # Without a resolvable window we cannot isolate the session's commits, and
        # guessing (last N) is exactly what leaked. Report zero rather than lie.
        return out
    log = run(["log", "--no-merges", "--pretty=%s",
               f"--since={since}", f"--until={until}", "HEAD"])
    msgs = [l for l in log.splitlines() if l.strip()]
    out["n_commits"] = len(msgs)
    out["reverts"] = sum(1 for m in msgs if _REVERT.search(m))
    if out["n_commits"]:
        out["survival_proxy"] = max(0.0, 1.0 - out["reverts"] / out["n_commits"])
    return out

# ============================================================================
# 3. SESSION INPUTS  — assemble the dict digest_text() needs.
#    The agent's JSONL parser fills prompts/tool/edit/verify/loops; git fills the rest.
# ============================================================================
def build_inputs(prompts: list[str], tool_cmds: list[str], n_tools: int,
                 n_edits: int, loops: int, churn: int, git: dict) -> dict:
    """Assemble the digest inputs. Matches the training-time semantics
    (cuebench_extract2.py): `n_tools` is the count of ALL tool calls (not just
    bash), `churn` is total written-line volume, `loops` is exact-duplicate tool
    re-runs. `tool_cmds` is the bash-only list, used solely for the verification
    classifier — it is NOT the tool count."""
    verify_runs = sum(1 for c in tool_cmds if is_verification(c))
    return {
        "prompts": prompts,
        "n_prompts": len(prompts),
        "n_tools": int(n_tools),
        "n_edits": int(n_edits),
        "verify_runs": int(verify_runs),
        "loops": int(loops),
        "n_commits": git.get("n_commits", 0),
        "churn": int(churn),
        "survival_proxy": git.get("survival_proxy"),
        "reverts": git.get("reverts", 0),
    }

# ============================================================================
# 4. DIGEST  — EXACT format the model was trained on. DO NOT MODIFY.
#    (copied verbatim from build_training_data.py digest_text)
# ============================================================================
def digest_text(inputs: dict, max_prompt_chars: int = 4000) -> str:
    s = inputs
    head = (f"PROMPTS={s['n_prompts']} TOOLS={s['n_tools']} EDITS={s['n_edits']} "
            f"VERIFY={s['verify_runs']} LOOPS={s['loops']} COMMITS={s['n_commits']} "
            f"CHURN={s['churn']} SURVIVAL={s.get('survival_proxy')} REVERTS={s['reverts']}")
    prompts = []
    budget = max_prompt_chars
    for i, p in enumerate(s.get("prompts", []), 1):
        if budget <= 0: break
        chunk = f" [P{i}] {p[:600]}"
        prompts.append(chunk); budget -= len(chunk)
    return head + "\nPROMPTS:" + "".join(prompts)

# ============================================================================
# 5. SPECIFICITY  (Description deterministic signal) — BYOK embeddings.
#    Optional: the model already scores Description; this is the deterministic
#    cross-check / no-LLM-tier value. Uses the org's BYOK key.
# ============================================================================
_VAGUE = ["make it nice","fix it","make it better","clean it up","improve this",
          "do the thing","make it work","handle it","just make it good"]
_SPECIFIC = ["add a 300ms debounce to the search handler in search.js",
    "fix the off-by-one in pagination offset when page=0",
    "return 404 instead of 500 when the record id is missing",
    "add a unit test asserting the parser rejects inputs over 1MB"]

def specificity(prompts: list[str], openai_client) -> float | None:
    """0-100 specificity via embedding distance to specific/vague poles. BYOK."""
    texts = [p for p in prompts if p and len(p.split())>=2]
    if not texts or openai_client is None: return None
    import numpy as np
    def emb(ts):
        r = openai_client.embeddings.create(model="text-embedding-3-large",
                                            input=[t[:2000] for t in ts])
        v = np.array([d.embedding for d in r.data], float)
        n = np.linalg.norm(v,axis=1,keepdims=True); n[n==0]=1; return v/n
    sv, vv, pv = emb(_SPECIFIC), emb(_VAGUE), emb(texts)
    margin = (pv@sv.T).max(1) - (pv@vv.T).max(1)
    return float(np.clip((margin.mean()+0.05)/0.25, 0, 1) * 100)

# ============================================================================
# 6. CHECKLIST DERIVATION  — from REAL deterministic signals only.
#    Each item's pass/fail is grounded in a measured signal. No fabrication.
#    Returns the CueBench checklist shape. Questions only included when the
#    signal backing them actually exists.
# ============================================================================
def derive_checklist(inputs: dict, vectors: dict) -> dict:
    i = inputs
    cl = {"delegation": [], "description": [], "discernment": [], "diligence": []}
    # Diligence — verification + commit discipline (real signals)
    cl["diligence"].append({"q":"Ran tests / typecheck / lint / build",
                            "pass": i["verify_runs"] > 0})
    if i["n_commits"]:
        cl["diligence"].append({"q":"Work was committed", "pass": True})
        cl["diligence"].append({"q":"No reverts/rollbacks in session",
                                "pass": i["reverts"] == 0})
    # Delegation — scoping signals (churn vs commits, survival)
    if i.get("survival_proxy") is not None:
        cl["delegation"].append({"q":"Committed work survived (not reverted)",
                                 "pass": i["survival_proxy"] >= 0.8})
    cl["delegation"].append({"q":"Bounded session (not excessive churn)",
                             "pass": i["churn"] < 2000})
    # Discernment — loops / edit behaviour. A handful of exact-duplicate re-runs
    # in a long session is normal; only a high loop RATE is a real weakness.
    # (Testing loops==0 manufactured a failure on healthy sessions — e.g. 7
    # duplicate calls out of 415 is a 1.7% rate, not a problem.)
    loop_rate = i["loops"] / i["n_tools"] if i.get("n_tools") else 0.0
    cl["discernment"].append({"q":"No unproductive loops detected",
                              "pass": loop_rate < 0.05})
    cl["discernment"].append({"q":"Reviewed before committing (edits then commit)",
                              "pass": i["n_edits"] > 0 and i["n_commits"] > 0})
    # Description — prompt presence + specificity (if computed, stored in vectors meta)
    cl["description"].append({"q":"Provided explicit prompts",
                              "pass": i["n_prompts"] > 0})
    return cl

# ============================================================================
# 7. QUALITY SCALE  — confidence gate + 6 quality zones (replaces letter grades)
# ============================================================================
# A session's headline is a QUALITY ZONE, not a letter grade. Two stages, in order:
#
#   (a) CONFIDENCE GATE — runs FIRST. A session must carry enough behavioural
#       evidence to be placed on the scale at all. `support` (= n_effective =
#       operator prompts + tool calls) is the count of judgeable actions the four
#       axes read. Below THIN_SESSION_MIN_SUPPORT there isn't enough to assess
#       reliably, so the session gets the SEPARATE state "Insufficient signal" and
#       is NOT zoned. Insufficient signal is NOT one of the 6 zones — it is gated
#       BEFORE zoning, so a 3-action stub never lands in (or distorts) the scale.
#
#   (b) ZONING — substantive sessions are placed in one of 6 zones by their
#       EB-SHRUNK composite. Empirical-Bayes shrinkage pulls a session's raw
#       composite toward the population mean in proportion to how thin it is: a
#       busy session (large support) keeps its raw score; a borderline-thin one is
#       regularised toward typical so a handful of lucky/unlucky actions can't fling
#       it into "Dialed in" or "Critical" on noise.
#
# The boundaries below are NOT academic letter cutoffs. They were read off the
# EB-shrunk score distribution of the REAL labelled corpus (5,781 sessions; see
# cuebench_calibrate_zones.py) so that each zone's DEFINITION fits the sessions
# landing in it and the urgent tier (zones 5 & 6) stays a small minority (~9% of
# substantive sessions in v1). Treat every constant here as a v1 ESTIMATE to
# retune as more production sessions accrue — rerun the calibrator to refresh them.

# Confidence gate
THIN_SESSION_MIN_SUPPORT = 11      # support (prompts+tools) below this -> Insufficient signal
                                   # (v1: ~p10 of the corpus; gates ~9% of sessions)

# Empirical-Bayes shrinkage of the 0-100 composite toward the population prior
EB_PRIOR_MEAN     = 58.8           # mean raw composite over substantive sessions (shrink target)
EB_PRIOR_STRENGTH = 11.0           # K: pseudo-observations of prior weight. support>>K -> ~raw;
                                   # support==K -> halfway to prior. Tied to the gate by design.

# Lower bounds (on the EB-shrunk, rounded composite) separating the 6 zones.
# Ascending; a score in [MAX_below, MAX_at) lands in the zone named by the upper const.
ZONE_CRITICAL_MAX         = 38     # eb_score <  38            -> 6 Critical        (URGENT)
ZONE_NEEDS_ATTENTION_MAX  = 46     # 38 <= eb_score < 46       -> 5 Needs attention (URGENT)
ZONE_INCONSISTENT_MAX     = 56     # 46 <= eb_score < 56       -> 4 Inconsistent
ZONE_DEVELOPING_MAX       = 65     # 56 <= eb_score < 65       -> 3 Developing
ZONE_SOLID_MAX            = 80     # 65 <= eb_score < 80       -> 2 Solid
#                                    eb_score >= 80            -> 1 Dialed in

INSUFFICIENT_SIGNAL = "Insufficient signal"   # the gated state's label (NOT a zone)

# The 6 zones, best (1) to worst (6). `urgent` marks the act-now tier (zones 5 & 6).
ZONES = [
    {"zone": 1, "key": "dialed_in",       "label": "Dialed in",       "urgent": False,
     "definition": "Strong, deliberate driving across the axes."},
    {"zone": 2, "key": "solid",           "label": "Solid",           "urgent": False,
     "definition": "Effective driving — a few areas to sharpen."},
    {"zone": 3, "key": "developing",      "label": "Developing",      "urgent": False,
     "definition": "Fundamentals present, with clear headroom."},
    {"zone": 4, "key": "inconsistent",    "label": "Inconsistent",    "urgent": False,
     "definition": "Quality swings — uneven work, targeted practice needed."},
    {"zone": 5, "key": "needs_attention", "label": "Needs attention", "urgent": True,
     "definition": "Driving is limiting outcomes — act now."},
    {"zone": 6, "key": "critical",        "label": "Critical",        "urgent": True,
     "definition": "Core practices breaking down — address before the next session."},
]
_ZONE_BY_KEY = {z["key"]: z for z in ZONES}


def n_effective(inputs: dict) -> int:
    """Support: how many judgeable actions the session gives the four axes to read.
    = operator prompts + total tool calls. Edits/verify/commits are already a subset
    of tool calls, so they are counted once here, not double-counted."""
    return int(inputs.get("n_prompts", 0) or 0) + int(inputs.get("n_tools", 0) or 0)


def eb_shrink(raw_composite: float, support: int,
              prior: float = EB_PRIOR_MEAN, k: float = EB_PRIOR_STRENGTH) -> float:
    """Empirical-Bayes shrink of a 0-100 composite toward `prior`, weighted by support.
    support >> k -> ~raw_composite; support == k -> halfway; support == 0 -> prior."""
    s = max(0, int(support))
    denom = s + k
    return (s * float(raw_composite) + k * float(prior)) / denom if denom else float(raw_composite)


def zone_for_score(eb_score) -> dict:
    """Map an EB-shrunk composite to its zone dict. Assumes the caller already passed
    the confidence gate — this never returns Insufficient signal."""
    if eb_score < ZONE_CRITICAL_MAX:        return _ZONE_BY_KEY["critical"]
    if eb_score < ZONE_NEEDS_ATTENTION_MAX: return _ZONE_BY_KEY["needs_attention"]
    if eb_score < ZONE_INCONSISTENT_MAX:    return _ZONE_BY_KEY["inconsistent"]
    if eb_score < ZONE_DEVELOPING_MAX:      return _ZONE_BY_KEY["developing"]
    if eb_score < ZONE_SOLID_MAX:           return _ZONE_BY_KEY["solid"]
    return _ZONE_BY_KEY["dialed_in"]


def quality(raw_composite: float, inputs: dict) -> dict:
    """Headline quality verdict for a session. Confidence gate FIRST, then zoning.

    Returns ONE of two shapes (always with `state` and `support`):

      Insufficient signal (gated — NOT placed on the scale):
        {"state": "insufficient_signal", "label": "Insufficient signal",
         "message": "...", "support": int, "min_support": int, "urgent": False}

      Assessed (placed in one of the 6 zones):
        {"state": "assessed", "zone": 1..6, "key": str, "label": str,
         "definition": str, "urgent": bool, "support": int,
         "eb_score": int, "raw_score": int}

    The zone is computed from the EB-shrunk composite and `eb_score` is that exact
    (rounded) value, so the headline zone and the score it reports are coherent."""
    support = n_effective(inputs)
    raw = int(round(float(raw_composite)))
    if support < THIN_SESSION_MIN_SUPPORT:
        return {
            "state": "insufficient_signal",
            "label": INSUFFICIENT_SIGNAL,
            "message": "Session too short/thin to assess reliably.",
            "support": support,
            "min_support": THIN_SESSION_MIN_SUPPORT,
            "urgent": False,
        }
    # Round the shrunk composite BEFORE zoning so the reported eb_score and the
    # chosen zone are derived from the identical value (no off-by-rounding drift).
    eb = int(round(eb_shrink(raw_composite, support)))
    z = zone_for_score(eb)
    return {
        "state": "assessed",
        "zone": z["zone"],
        "key": z["key"],
        "label": z["label"],
        "definition": z["definition"],
        "urgent": z["urgent"],
        "support": support,
        "eb_score": eb,
        "raw_score": raw,
    }
