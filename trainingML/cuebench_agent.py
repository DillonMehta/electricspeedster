#!/usr/bin/env python3
"""
cuebench_agent.py — watch Claude Code sessions, score them, POST to the dashboard
=================================================================================
Pipeline per finished session:
  1. Parse the Claude Code JSONL transcript (prompts / tools / edits / tokens / cwd).
  2. Compute git signals on the repo the session edited.
  3. Assemble inputs via cuebench_signals.build_inputs -> digest_text (the EXACT format
     the model trained on) and score with the local 4D model (model_infer).
  4. Generate a NEUTRAL title + per-axis insights (and an optional trace) via cuebench_gen
     from DERIVED NUMBERS ONLY — never raw prompts/code/paths/commands. OFF when no BYOK
     key; cached per-session in cuebench_store so it is generated (and paid for) once.
  5. POST the privacy-safe payload to the dashboard (deduped per sessionId; never re-POSTed;
     an appended session is re-sent as an UPDATE, not a duplicate insert).

Privacy (non-negotiable): only derived numbers and AI-generated NEUTRAL text are sent.
Raw prompts, code, file paths, and verbatim commands never leave the device.

Run:
  python cuebench_agent.py                       # watcher loop over ~/.claude/projects
  python cuebench_agent.py --once <path.jsonl>   # score one transcript (testing)
  python cuebench_agent.py --once <path.jsonl> --dry-run   # print payload, don't POST

Secrets are read from env ONLY (never hardcoded). Missing keys degrade gracefully.
See AGENT_README.md for env vars and JSONL-format assumptions.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, glob
from datetime import datetime, timezone
from urllib import request as urlrequest, error as urlerror

import cuebench_signals as sig
from model_infer import ModelScorer
from train_cuebench import AXES  # ["delegation","description","discernment","diligence"]
from cuebench_gen import Generator        # one provider-switched generation module
from cuebench_store import Store          # local generation cache + POST-sent dedup

# ----------------------------------------------------------------------------
# Config (env only)
# ----------------------------------------------------------------------------
API_URL        = os.environ.get("CUEBENCH_API_URL", "https://app.cuebench.dev/api/session")
DATA_URL       = os.environ.get("CUEBENCH_DATA_URL", "https://app.cuebench.dev/api/data")
API_KEY        = os.environ.get("CUEBENCH_API_KEY")
EMPLOYEE_ID    = os.environ.get("CUEBENCH_EMPLOYEE_ID", "e1")
PROJECTS_DIR   = os.path.expanduser(os.environ.get("CUEBENCH_PROJECTS_DIR", "~/.claude/projects"))
MODEL_DIR      = os.environ.get("CUEBENCH_MODEL_DIR", "cuebench_model")
STATE_DB       = os.environ.get("CUEBENCH_STATE_DB", "cuebench_state.db")
SCORED_FILE    = os.environ.get("CUEBENCH_SCORED_FILE", "scored_sessions.json")  # legacy posted-set, seeded once
# BYOK generation (provider / key / model / trace) is owned entirely by cuebench_gen.Generator.
# There is ONE credential path; do not read BYOK_* here.

POLL_INTERVAL  = int(os.environ.get("CUEBENCH_POLL_INTERVAL", "20"))   # seconds between dir scans
STABLE_SECONDS = int(os.environ.get("CUEBENCH_STABLE_SECONDS", "60"))  # no growth for this long => finished

EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit", "str_replace_based_edit_tool"}

# ----------------------------------------------------------------------------
# 1. JSONL transcript parser  (schema confirmed by inspecting a real ~/.claude file;
#    see AGENT_README.md "JSONL assumptions". Defensive: tolerate missing fields.)
# ----------------------------------------------------------------------------
_SYSREM = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_CMDOUT = re.compile(r"<(local-command-stdout|command-name|command-message|command-args)>.*?</\1>", re.DOTALL)

def _text_from_content(content) -> str:
    """User/assistant message content -> plain text (string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "\n".join(parts)
    return ""

def _clean_prompt(t: str) -> str:
    t = _SYSREM.sub(" ", t or "")
    t = _CMDOUT.sub(" ", t)
    return t.strip()

def _is_human_prompt(rec: dict) -> bool:
    """True only for genuine human turns. A `type=="user"` record is overloaded:
    it also carries tool results, skill/subagent injections, interruption
    sentinels, and compaction summaries. A real prompt has a human origin signal
    (origin.kind=="human" or a promptSource) and is none of the synthetic kinds.
    Without this filter the prompt count is inflated by injected/pasted noise
    (e.g. a skill's 'Base directory for this skill: …' block, a '[Request
    interrupted by user]' sentinel)."""
    msg = rec.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
        return False
    if rec.get("isMeta") or rec.get("sourceToolUseID") or rec.get("interruptedMessageId"):
        return False
    if msg.get("role") != "user":
        return False
    origin = rec.get("origin") or {}
    return origin.get("kind") == "human" or bool(rec.get("promptSource"))

def _written_lines(name: str, inp: dict) -> int:
    """Lines of content WRITTEN by one edit-type tool call — the churn signal.
    Edit COUNT alone is blind to size (190 edits could be 190 one-liners or a
    full rewrite); churn keys size-aware logic. Sums the written side of each
    tool: Write.content, Edit.new_string/new_str, every MultiEdit sub-edit's
    new_string, NotebookEdit.new_source."""
    def _lines(s: str) -> int:
        return (s.count("\n") + 1) if s else 0
    if name == "Write":
        return _lines(inp.get("content", "") or "")
    if name in ("Edit", "str_replace_based_edit_tool"):
        return _lines(inp.get("new_string", inp.get("new_str", "")) or "")
    if name == "MultiEdit":
        return sum(_lines(e.get("new_string", "") or "") for e in (inp.get("edits") or []))
    if name == "NotebookEdit":
        return _lines(inp.get("new_source", "") or "")
    return 0

def parse_transcript(path: str) -> dict:
    """Extract the signal inputs + metadata from one session JSONL. Never raises on
    bad lines (skips them) so a malformed transcript can't crash the watcher."""
    prompts, tool_cmds = [], []
    n_edits = n_tools = churn = 0
    tool_sigs: dict[str, int] = {}     # tool|full-input -> count (for loop detection)
    in_tok = out_tok = 0
    models, timestamps = [], []
    cwd = git_branch = session_uuid = None
    n_lines = 0

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n_lines += 1
            t = rec.get("type")
            if rec.get("timestamp"):
                timestamps.append(rec["timestamp"])
            if cwd is None and rec.get("cwd"):
                cwd = rec.get("cwd")
            if git_branch is None and rec.get("gitBranch"):
                git_branch = rec.get("gitBranch")
            if session_uuid is None and rec.get("sessionId"):
                session_uuid = rec.get("sessionId")

            if t == "user":
                # Real operator prompts only — see _is_human_prompt (drops tool
                # results, skill/subagent injections, interrupt sentinels).
                if not _is_human_prompt(rec):
                    continue
                msg = rec.get("message") or {}
                txt = _clean_prompt(_text_from_content(msg.get("content")))
                if txt:
                    prompts.append(txt)

            elif t == "assistant":
                msg = rec.get("message") or {}
                if msg.get("model"):
                    models.append(msg["model"])
                u = msg.get("usage") or {}
                in_tok  += int(u.get("input_tokens", 0) or 0)
                in_tok  += int(u.get("cache_read_input_tokens", 0) or 0)
                in_tok  += int(u.get("cache_creation_input_tokens", 0) or 0)
                out_tok += int(u.get("output_tokens", 0) or 0)
                for b in (msg.get("content") or []):
                    if not isinstance(b, dict) or b.get("type") != "tool_use":
                        continue
                    name = b.get("name", "")
                    inp = b.get("input") or {}
                    n_tools += 1     # ALL tool calls (matches training n_tools), not bash-only
                    # loop signature: tool + FULL input. Counting exact-duplicate
                    # re-runs (re-ran the same command / re-applied the same edit),
                    # NOT consecutive same-target calls — editing one file 40× in a
                    # row or repeating `git status` is normal work, not a loop.
                    sig = name + "|" + json.dumps(inp, sort_keys=True)
                    tool_sigs[sig] = tool_sigs.get(sig, 0) + 1
                    if name == "Bash":
                        cmd = inp.get("command", "")
                        if cmd:
                            tool_cmds.append(cmd)   # bash-only list, for the verify classifier
                    elif name in EDIT_TOOLS:
                        n_edits += 1
                        churn += _written_lines(name, inp)   # size, not just count

    if session_uuid is None:
        session_uuid = os.path.splitext(os.path.basename(path))[0]

    return {
        "session_uuid": session_uuid,
        "cli": "claude code",
        "path": path,
        "prompts": prompts,
        "tool_cmds": tool_cmds,
        "n_tools": n_tools,
        "n_edits": n_edits,
        "churn": churn,
        "loops": detect_loops(tool_sigs),
        "model": _most_recent(models),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        # first/last by VALUE, not file order: a resumed session can carry an
        # out-of-order leading record, and the active-duration math needs the
        # true min/max.
        "first_ts": min(timestamps) if timestamps else None,
        "last_ts": max(timestamps) if timestamps else None,
        "duration_s": active_duration(timestamps),
        "cwd": cwd,
        "git_branch": git_branch,
        "n_lines": n_lines,
    }

def _most_recent(models: list) -> str | None:
    return models[-1] if models else None

def detect_loops(tool_sigs: dict) -> int:
    """Unproductive-loop count = EXACT-DUPLICATE tool re-runs.

    A signature is (tool name + full JSON input); loops = Σ(occurrences − 1) over
    signatures seen more than once — i.e. how many calls merely repeated an
    earlier identical call (re-ran the same command, re-applied the same edit).
    This is the training-time definition (cuebench_extract2.py) but keyed on the
    FULL input rather than its first 200 chars, so distinct-but-similar commands
    (all starting `cd repo && …`) don't collide into phantom loops.

    The previous heuristic counted *consecutive* (tool, primary-target) repeats,
    which fired on normal work: editing one file 40× in a row, or running
    `git status` twice, each read as a loop. On the reference session that
    inflated loops to 47 (≈ the prompt count, by coincidence); the real figure
    is 7. Over-counting here manufactures a Discernment weakness that isn't there.
    """
    return sum(c - 1 for c in tool_sigs.values() if c > 1)

# ----------------------------------------------------------------------------
# 2. Time helpers
# ----------------------------------------------------------------------------
def _parse_ts(ts: str | None):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def duration_seconds(first_ts, last_ts) -> int:
    a, b = _parse_ts(first_ts), _parse_ts(last_ts)
    if a and b:
        return max(0, int((b - a).total_seconds()))
    return 0

_IDLE_CAP_S = 300   # a gap longer than this is "away", not "working"

def active_duration(timestamps: list, idle_cap: int = _IDLE_CAP_S) -> int:
    """ACTIVE session seconds: Σ over consecutive events of min(gap, idle_cap).

    Wall-clock min→max is wrong for any session left open: the reference session
    spans 19h end-to-end but the human only worked in two bursts around an ~12h
    overnight gap. Capping each inter-event gap at `idle_cap` excludes idle time
    so the figure reflects time actually spent (no epoch/units bug was involved —
    the 19h was a real, mostly-idle span)."""
    ts = sorted(t for t in (_parse_ts(x) for x in (timestamps or [])) if t is not None)
    if len(ts) < 2:
        return 0
    total = 0.0
    for i in range(1, len(ts)):
        gap = (ts[i] - ts[i - 1]).total_seconds()
        if gap > 0:
            total += min(gap, idle_cap)
    return int(total)

def fmt_duration(secs: int) -> str:
    m, s = divmod(secs, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h {m}m"
    return f"{m}m {s}s" if m else f"{s}s"

def fmt_mmss(secs: int) -> str:
    """MM:SS, rolling into H:MM:SS past an hour so a multi-hour session doesn't
    render as e.g. '338:40' (and the old idle-inflated bug as '1140:14')."""
    m, s = divmod(max(0, secs), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def commafmt(n) -> str:
    try:
        return f"{int(round(float(n))):,}"
    except Exception:
        return "0"

def fmt_date(ts: str | None) -> str:
    """Human-readable date for the payload, e.g. 'Jun 28, 2026, 14:32' (v1 schema `date`).
    Uses the session's last timestamp (converted to local time) when available, else now."""
    dt = _parse_ts(ts) if ts else None
    if dt is None:
        dt = datetime.now()
    else:
        try:
            dt = dt.astimezone()        # UTC ISO -> local wall clock
        except Exception:
            pass
    return f"{dt.strftime('%b')} {dt.day}, {dt.year}, {dt.strftime('%H:%M')}"

# ----------------------------------------------------------------------------
# 3b. Session timeline (CONTENT-bearing — built only when the trace is enabled).
#     This is the one extraction that carries raw file names / commands / prompt
#     text; it is fed ONLY to the user's own BYOK endpoint by cuebench_gen.trace
#     (never POSTed raw). Skipped entirely on the default path (trace off).
# ----------------------------------------------------------------------------
# tool name -> the input key whose value is the salient, human-meaningful argument
_TOOL_ARG_KEY = {
    "Bash": "command", "Read": "file_path", "Edit": "file_path", "Write": "file_path",
    "MultiEdit": "file_path", "NotebookEdit": "notebook_path",
    "str_replace_based_edit_tool": "path", "Grep": "pattern", "Glob": "pattern",
}

def _tool_descriptor(name: str, inp: dict) -> str:
    """One-line 'Tool: salient-arg' descriptor for a tool_use block (e.g.
    'Bash: pytest tests/test_auth.py', 'Edit: session.py'). Truncated; newlines flattened."""
    val = str((inp or {}).get(_TOOL_ARG_KEY.get(name, ""), "") or "")
    if not val:                              # unknown tool -> first stringy arg, if any
        for v in (inp or {}).values():
            if isinstance(v, str) and v.strip():
                val = v
                break
    val = " ".join(val.split())
    return f"{name}: {val[:140]}" if val else (name or "tool")

def build_timeline(path: str, first_ts, *, max_tool_events: int = 120,
                   max_prompts: int = 40) -> dict:
    """Ordered list of session events with elapsed MM:SS timestamps and turn numbers,
    for the content-grounded trace. Each event: {turn, t, kind, text}. Keeps every operator
    prompt (high value, few) and the tool/edit sequence; on a long session keeps the start
    and end of the tool sequence and records how many middle events were omitted. Bounded
    so a huge transcript can't blow up memory or BYOK cost."""
    base = _parse_ts(first_ts)
    prompts, tools, seq, turn = [], [], 0, 0
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                ts = _parse_ts(rec.get("timestamp"))
                t = fmt_mmss(int((ts - base).total_seconds())) if (ts and base) else "00:00"
                typ = rec.get("type")
                if typ == "user" and _is_human_prompt(rec):
                    turn += 1
                    txt = _clean_prompt(_text_from_content((rec.get("message") or {}).get("content")))
                    if txt:
                        prompts.append({"i": seq, "turn": turn, "t": t,
                                        "kind": "prompt", "text": txt[:240]})
                        seq += 1
                elif typ == "assistant":
                    for b in ((rec.get("message") or {}).get("content") or []):
                        if not isinstance(b, dict) or b.get("type") != "tool_use":
                            continue
                        name, inp = b.get("name", ""), (b.get("input") or {})
                        desc = _tool_descriptor(name, inp)
                        if name in EDIT_TOOLS:
                            n = _written_lines(name, inp)
                            ev = {"i": seq, "turn": turn, "t": t, "kind": "edit",
                                  "text": f"{desc} (+{n} lines)" if n else desc}
                        else:
                            ev = {"i": seq, "turn": turn, "t": t, "kind": "tool", "text": desc}
                        tools.append(ev)
                        seq += 1
    except Exception:
        return {"events": [], "tool_events_omitted": 0}

    prompts = prompts[:max_prompts]
    omitted = 0
    if len(tools) > max_tool_events:                 # keep the start AND the end of a long run
        head = int(max_tool_events * 0.7)
        kept = tools[:head] + tools[-(max_tool_events - head):]
        omitted = len(tools) - max_tool_events
    else:
        kept = tools
    events = sorted(prompts + kept, key=lambda e: e["i"])
    for e in events:
        e.pop("i", None)
    return {"events": events, "tool_events_omitted": omitted}

# ----------------------------------------------------------------------------
# 4. (BYOK generation lives in cuebench_gen.Generator — one provider switch, off-if-no-key.
#    insights are fed DERIVED NUMBERS ONLY; the TITLE (from the first prompt) and the TRACE
#    (from the ordered session timeline — file names / commands / operator quotes) are
#    content-aware opt-ins: their raw inputs go only to the user's own endpoint, and only the
#    short generated text reaches the payload. trace is additionally double-gated by
#    CUEBENCH_TRACE. The old helpers that echoed verbatim prompts into the payload were removed.)
# ----------------------------------------------------------------------------
# (the three legacy raw-transcript helpers — generate_trace / _fallback_trace /
#  make_task_title — were deleted. Their replacements in cuebench_gen are fed derived
#  numbers only, so they cannot echo raw prompt text into the payload.)

# ----------------------------------------------------------------------------
# 5. Payload assembly + POST
# ----------------------------------------------------------------------------
def make_sid(session_uuid: str) -> str:
    """Canonical dashboard sessionId. Single source of truth so the watcher dedups on
    the EXACT value that gets POSTed (the bug was dedup-by-filename letting many files
    that resolve to one sessionId each POST the same sid)."""
    return "S-" + (session_uuid or "").replace("-", "")[:8].upper()

def build_payload(parsed: dict, scorer: ModelScorer, store: Store, gen: Generator,
                  *, regenerate: bool = False) -> tuple[dict, str]:
    """Assemble the privacy-safe POST body. Returns (payload, gen_status) where gen_status is
    one of 'cache' | 'generated' | 'gen-failed' | 'off' — for logging/dry-run visibility.

    The payload contains derived numbers + AI-generated text. Two fields are CONTENT-AWARE
    when BYOK is on (both generated via the user's OWN endpoint; the raw inputs are never
    posted): `title` (from the first prompt — see cuebench_gen.content_title; falls back to
    the session's last-use date/time when generation is off) and, when CUEBENCH_TRACE is
    enabled, `trace` (a coaching timeline grounded in real events — file names, commands,
    operator quotes). insights and specificity (a 0-100 number) stay numbers-only. The
    payload still carries no checklist, no verbatim transcript dump, and no repo path.

    `quality` is the headline verdict (replaces the old letter grade): a confidence gate
    first (thin sessions -> "Insufficient signal", NOT placed on the scale), else one of 6
    quality zones from the EB-shrunk composite. See cuebench_signals.quality.
    """
    repo = parsed["cwd"]
    git = sig.git_signals(repo, parsed["first_ts"], parsed["last_ts"]) if repo else \
          {"n_commits": 0, "reverts": 0, "survival_proxy": None}

    inputs = sig.build_inputs(parsed["prompts"], parsed["tool_cmds"], parsed["n_tools"],
                              parsed["n_edits"], parsed["loops"], parsed["churn"], git)
    digest = sig.digest_text(inputs)
    vectors = scorer.score(digest)               # {axis: 0-100}
    raw_composite = sum(vectors.values()) / 4.0
    score = round(raw_composite)
    # Headline verdict: confidence gate first (thin -> "Insufficient signal"), else a zone.
    quality = sig.quality(raw_composite, inputs)

    dur = parsed["duration_s"]
    sid = make_sid(parsed["session_uuid"])

    # Numbers-only metrics for generation. NO prompts/commands/paths — only derived signals,
    # so the generator structurally cannot echo raw content into the payload.
    metrics = {k: inputs[k] for k in ("n_prompts", "n_tools", "n_edits", "verify_runs",
                                      "loops", "n_commits", "churn", "survival_proxy", "reverts")}
    metrics["duration_seconds"] = dur

    # Generation: reuse the cached, paid-for-once artifacts unless regenerate is forced.
    cached = None if regenerate else store.get_generation(sid)
    if cached:
        title, insights, trace = cached["title"], cached["insights"], cached["trace"]
        specificity = cached.get("specificity")
        gen_status = "cache"
    elif gen.enabled:
        # Content-aware title from the session's FIRST prompt (opt-in: see
        # cuebench_gen.content_title). The raw prompt goes only to the user's own BYOK
        # endpoint; only the short title is kept. insights/trace below stay numbers-only.
        first_prompt = parsed["prompts"][0] if parsed["prompts"] else None
        title = gen.content_title(first_prompt)
        # insights are numbers-only UNLESS prompt-informed mode is on (CUEBENCH_INSIGHTS_PROMPTS):
        # then gen.insights feeds these prompts to its own BYOK endpoint so coaching can
        # reference what was asked. The flag is checked inside insights(); passing prompts
        # unconditionally is safe — they're ignored when the opt-in is off.
        insights = gen.insights(metrics, vectors, quality, prompts=parsed["prompts"])
        # Content-grounded coaching timeline, built from the ACTUAL transcript events and
        # generated via the user's own BYOK endpoint (opt-in: double-gated by CUEBENCH_TRACE).
        trace = None
        if gen.trace_enabled:
            timeline = build_timeline(parsed["path"], parsed["first_ts"])
            header = {
                "model": parsed["model"] or "unknown", "duration": fmt_mmss(dur),
                "score": score, "n_prompts": metrics["n_prompts"],
                "n_tools": metrics["n_tools"], "n_edits": metrics["n_edits"],
                "loops": metrics["loops"], "n_commits": metrics["n_commits"],
                "cost_tokens": parsed["input_tokens"] + parsed["output_tokens"],
            }
            trace = gen.trace(timeline, header, quality)
        # specificity is the ONE input that sees raw prompt TEXT (opt-in, OpenAI-only). It is
        # sent only to the user's own embeddings endpoint; just the 0-100 NUMBER is kept here.
        specificity = gen.specificity(parsed["prompts"])
        if title is not None:                    # provider reachable -> persist (paid once)
            store.save_generation(sid, title, insights, trace, specificity, gen.provider,
                                  gen.model, parsed["n_lines"])
            gen_status = "generated"
        else:                                    # transient failure -> don't cache; retry next run
            gen_status = "gen-failed"
    else:
        title, insights, trace, specificity = None, None, None, None
        gen_status = "off"

    total_tok = parsed["input_tokens"] + parsed["output_tokens"]
    tool_tok = round(total_tok * 0.15)           # can't isolate tool tokens -> ~15% (plan)

    payload = {
        "employeeId": EMPLOYEE_ID,
        "sessionId": sid,
        # Content-aware title when BYOK is on; else a provider-supplied name (Cursor composers
        # carry one — free and accurate); else the session's last-use date/time (no key -> no
        # generated name, so label by WHEN it ran, not the opaque sessionId).
        "title": title or parsed.get("title_hint") or fmt_date(parsed["last_ts"]),
        "model": parsed["model"] or "unknown",
        # Which CLI/tool produced the session. Each provider parser stamps it; "ext" is the
        # fallback for any source that doesn't (an unknown/external transcript).
        "cli": parsed.get("cli", "ext"),          # "claude code" | "cursor" | "codex" | "ext"
        "date": fmt_date(parsed["last_ts"]),      # v1 schema: human-readable date
        "duration": fmt_duration(dur),
        "score": score,
        # v1 quality object — exactly the four documented fields (eb_score drives the rolling
        # average). `quality` from sig.quality() is the richer internal verdict; we send only
        # the v1 subset, falling back to the raw score / state when a field is absent.
        "quality": {
            "eb_score": quality.get("eb_score", score),
            "label": quality.get("label"),
            "urgent": bool(quality.get("urgent", False)),
            "state": quality.get("state"),
        },
        "vectors": vectors,
        "breakdown": [{"key": ax, "v": vectors[ax]} for ax in AXES],
        "cost": {
            "input": commafmt(parsed["input_tokens"]),
            "output": commafmt(parsed["output_tokens"]),
            "tool": commafmt(tool_tok),
            "total": commafmt(total_tok),
            "time": fmt_duration(dur),
            # dollars / baseline / timeBaseline are optional in v1 and NOT measured here —
            # omitted rather than fabricated (consistent with the rest of the payload).
        },
    }
    if insights:                                 # omit entirely when absent (null-safe)
        payload["insights"] = insights
    if trace:                                    # optional, off by default; omit when absent
        payload["trace"] = trace
    if specificity is not None:                  # 0-100 number; omit when not computed
        payload["specificity"] = specificity
    return payload, gen_status

def post_payload(payload: dict) -> tuple[bool, str]:
    if not API_KEY:
        return False, "CUEBENCH_API_KEY not set"
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(API_URL, data=body, method="POST",
                             headers={"Content-Type": "application/json", "x-api-key": API_KEY,
                                      # A real product User-Agent: the default urllib UA
                                      # ("Python-urllib/x") trips Cloudflare's bot signature ban
                                      # (HTTP 403, "error code: 1010") in front of the dashboard.
                                      "User-Agent": "CueBench-Agent/1.0",
                                      # Stable per-session idempotency key. Lets the endpoint
                                      # collapse a crash-window re-POST (same sessionId) into the
                                      # same record even before our local sent-flag is committed.
                                      "Idempotency-Key": payload.get("sessionId", "")})
    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            return True, f"{resp.status} {resp.read(200).decode('utf-8', 'replace')}"
    except urlerror.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read(300).decode('utf-8', 'replace')}"
    except Exception as e:
        return False, repr(e)

# ----------------------------------------------------------------------------
# 6. dedup / generation cache (cuebench_store) + watcher
# ----------------------------------------------------------------------------
def peek_session_uuid(path: str) -> str:
    """Cheaply resolve a transcript's session uuid (first `sessionId` field, else the
    filename stem) WITHOUT parsing the whole file — so dedup can run before scoring.
    Matches parse_transcript's session_uuid logic so the peeked sid == the POSTed sid."""
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("sessionId"):
                    return rec["sessionId"]
    except Exception:
        pass
    return os.path.splitext(os.path.basename(path))[0]

def parse_session(path: str) -> dict:
    """Provider dispatch: a Codex CLI rollout is parsed by cuebench_codex into the SAME dict
    shape this module's parse_transcript returns (so Codex sessions score identically); any
    other JSONL is treated as a Claude Code transcript. Falls back to the Claude parser if the
    Codex adapter is unavailable for any reason."""
    try:
        import cuebench_codex
        if cuebench_codex.is_codex_rollout(path):
            return cuebench_codex.parse_codex_transcript(path)
    except Exception:
        pass
    return parse_transcript(path)

def process_one(path: str, scorer: ModelScorer, store: Store, gen: Generator,
                *, dry_run: bool = False, regenerate: bool = False) -> tuple[str, str | None]:
    """Parse a transcript FILE (Claude or Codex) and score/dedup/POST it. Thin wrapper over
    process_parsed so non-file providers (e.g. Cursor's SQLite composers) share the exact same
    dedup + score + POST core."""
    return process_parsed(parse_session(path), scorer, store, gen, dry_run=dry_run,
                          regenerate=regenerate, source=os.path.basename(path))

def process_parsed(parsed: dict, scorer: ModelScorer, store: Store, gen: Generator,
                   *, dry_run: bool = False, regenerate: bool = False,
                   source: str = "") -> tuple[str, str | None]:
    """Returns (status, sid). status in {'posted','dup','skip','fail'}:
      posted = scored & accepted (or dry-run printed); dup = already sent and unchanged;
      skip = nothing to score (don't retry); fail = transient POST failure (retry later).

    Dedup + append handling key on the stable sessionId: a session with MORE turns than when
    last sent is re-sent as an UPDATE (the dashboard upserts on sessionId); an unchanged
    already-sent session is skipped. A failed POST is NOT marked sent (so it retries) and does
    NOT regenerate (the cached generation is reused). Provider-agnostic: `parsed` is the dict
    shape parse_transcript / parse_codex_transcript / parse_cursor_composer all return."""
    if not parsed["prompts"]:
        print(f"  [skip] {source or parsed.get('session_uuid', '?')}: no operator prompts found")
        return "skip", None
    sid = make_sid(parsed["session_uuid"])

    prev_n = store.get_n_lines(sid)
    appended = prev_n is not None and parsed["n_lines"] > prev_n
    if store.is_posted(sid) and not appended and not dry_run:
        print(f"  [dup] {sid} already POSTed and unchanged; skipping")
        return "dup", sid

    payload, gen_status = build_payload(parsed, scorer, store, gen, regenerate=regenerate)
    if dry_run:
        print(json.dumps(payload, indent=2))
        ok, info = True, "(dry-run, not POSTed)"
    else:
        ok, info = post_payload(payload)
        if ok:
            # CRASH ORDERING (load-bearing): we record "sent" ONLY after the server returns OK,
            # and mark_posted() commits it to disk IMMEDIATELY (conn.commit() with
            # synchronous=FULL) before this function returns and the watcher moves to the next
            # file. The only crash window is between the server accepting and this commit
            # landing — a crash there re-POSTs on restart (at-least-once), made harmless by the
            # server's UPSERT on sessionId. We never mark "sent" before the server confirms, so
            # a crash mid-POST also re-POSTs rather than silently dropping the session.
            store.mark_posted(sid, parsed["n_lines"])   # UPSERT on sid: record sent + line count
    tag = " UPDATE" if appended else ""
    q = payload["quality"]
    verdict = (q.get("label") or q.get("state") or "?") + (
        f" (eb{q['eb_score']})" if q.get("eb_score") is not None else "")
    print(f"  sid={sid} title={payload['title']!r} score={payload['score']} "
          f"quality={verdict!r} gen={gen_status} POST={'OK' if ok else 'FAIL'}{tag} {info}")
    return ("posted" if ok else "fail"), sid

def watch(scorer: ModelScorer, store: Store, gen: Generator):
    """On-demand watcher entry point (unchanged behaviour): require CUEBENCH_API_KEY and
    always POST. The poll/stable-size loop itself lives in run_watch_loop() so the background
    daemon (cuebench_daemon.py) reuses the EXACT same loop instead of reimplementing it. With
    these arguments the result is byte-for-byte the original behaviour."""
    if not API_KEY:
        print("[fatal] CUEBENCH_API_KEY not set — the watcher would score sessions but never POST.\n"
              "        Set it (export CUEBENCH_API_KEY=...) and rerun. For local testing without a\n"
              "        key, use:  python cuebench_agent.py --once <transcript.jsonl> --dry-run",
              file=sys.stderr)
        sys.exit(2)
    print(f"[watch] {PROJECTS_DIR}  poll={POLL_INTERVAL}s stable={STABLE_SECONDS}s "
          f"employee={EMPLOYEE_ID} api=set gen={gen.status()}")
    run_watch_loop(scorer, store, gen, dry_run=False)


def run_watch_loop(scorer: ModelScorer, store: Store, gen: Generator,
                   *, dry_run: bool = False, on_cycle=None, roots=None):
    """The poll/stable-size watcher loop — extracted verbatim from watch() so both the
    on-demand entry point and the background daemon share ONE implementation of the
    stable-size detection, poll/stable thresholds, dedup, append-detection and
    retry-later logic.

    Additive, behaviour-preserving knobs (defaults reproduce the original loop exactly):
      dry_run  -- threaded into process_one: print the payload, never POST, never mark sent.
      on_cycle -- optional zero-arg callback invoked once per completed scan pass (just
                  before the sleep) so a wrapper can record a liveness heartbeat without
                  duplicating the loop. Exceptions it raises propagate (used for clean
                  shutdown at a cycle boundary).
      roots    -- list of directories to scan for *.jsonl (default [PROJECTS_DIR]). The daemon
                  passes both ~/.claude/projects AND ~/.codex/sessions so Claude and Codex
                  sessions are watched together; parse_session dispatches per file."""
    roots = roots or [PROJECTS_DIR]
    sizes: dict[str, tuple[int, float]] = {}     # path -> (size, first_seen_at_this_size)
    handled: set[str] = set()                    # paths terminal at their current size (this run)
    while True:
        now = time.time()
        scan = [p for root in roots
                for p in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)]
        for path in scan:
            name = os.path.basename(path)
            # Subagent transcripts (agent-*.jsonl) are NOT standalone sessions — they share the
            # parent's sessionId. Only score top-level UUID-named session files.
            if name.startswith("agent-"):
                continue
            try:
                sz = os.path.getsize(path)
            except OSError:
                continue
            prev = sizes.get(path)
            if prev is None or prev[0] != sz:
                sizes[path] = (sz, now)                 # size changed -> reset stability clock
                handled.discard(path)                   # growth re-opens it (append -> UPDATE)
                continue
            if now - prev[1] < STABLE_SECONDS:
                continue                                # not stable long enough yet
            if path in handled:
                continue                                # already terminal at this size
            print(f"[finished] {name}")
            # Dedup, append-detection, generation cache, and POST all key on the stable
            # sessionId inside process_one (via cuebench_store) — not on the filename.
            try:
                status, _sid = process_one(path, scorer, store, gen, dry_run=dry_run)
            except Exception as e:
                print(f"  [error] processing {path}: {e!r}", file=sys.stderr)
                status = "skip"                         # unparseable transcript: don't loop on it
            if status in ("posted", "dup", "skip"):
                handled.add(path)                       # terminal at this size; growth re-opens it
            else:
                # transient POST failure (network/5xx): leave unmarked + unsent, and back off this
                # file's stability clock so we retry on a later poll rather than every poll. The
                # cached generation is reused on retry — no second BYOK call.
                sizes[path] = (sz, now)
                print(f"  [retry-later] {name} will be retried", file=sys.stderr)
        if on_cycle is not None:
            on_cycle()                                  # heartbeat / clean-shutdown hook
        time.sleep(POLL_INTERVAL)

# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CueBench live scoring agent")
    ap.add_argument("--once", metavar="JSONL", help="score a single transcript and exit")
    ap.add_argument("--dry-run", action="store_true", help="print payload instead of POSTing")
    ap.add_argument("--model", default=MODEL_DIR, help="path to cuebench_model/")
    ap.add_argument("--state-db", default=STATE_DB, help="local generation-cache / dedup SQLite db")
    ap.add_argument("--trace", action="store_true",
                    help="also generate the optional neutral trace (off by default)")
    ap.add_argument("--specificity", action="store_true",
                    help="compute the prompt-specificity number via OpenAI embeddings "
                         "(OpenAI BYOK only; sends raw prompt TEXT to your embeddings endpoint — "
                         "off by default; only the 0-100 number is sent to the dashboard)")
    ap.add_argument("--regenerate", action="store_true",
                    help="ignore cached generation and regenerate (spends BYOK quota)")
    a = ap.parse_args()

    store = Store(a.state_db)
    seeded = store.seed_posted_from_json(SCORED_FILE)   # one-time migrate legacy posted-set
    if seeded:
        print(f"[migrate] seeded {seeded} already-POSTed sessionId(s) from {SCORED_FILE}",
              file=sys.stderr)
    # A flag forces the feature on; absent -> None -> the env default (CUEBENCH_TRACE / _SPECIFICITY).
    gen = Generator(trace_enabled=(True if a.trace else None),
                    specificity_enabled=(True if a.specificity else None))
    print(f"[gen] {gen.status()}", file=sys.stderr)

    print(f"[load] model from {a.model} ...", file=sys.stderr)
    scorer = ModelScorer(a.model)
    print(f"[load] ok (weights={scorer.loaded_from})", file=sys.stderr)

    if a.once:
        process_one(a.once, scorer, store, gen, dry_run=a.dry_run, regenerate=a.regenerate)
        return
    watch(scorer, store, gen)

if __name__ == "__main__":
    main()
