#!/usr/bin/env python3
"""
cuebench_cursor.py — Cursor (Composer/agent) session → the Claude-shaped session dict
=====================================================================================
A provider adapter, like cuebench_codex. Cursor stores agent conversations in a SQLite
key-value store (NOT per-session files):

  ~/Library/Application Support/Cursor/User/globalStorage/state.vscdb
    table cursorDiskKV:
      composerData:<composerId>          -> JSON: the conversation (name, status, model,
                                            totalLinesAdded, trackedGitRepos, createdAt,
                                            and fullConversationHeadersOnly = ordered bubbles)
      bubbleId:<composerId>:<bubbleId>   -> JSON: one message {type 1=user|2=assistant,
                                            text, toolFormerData (tool call), tokenCount, …}

This module reads a composer and returns the SAME dict cuebench_agent.parse_transcript
returns, so a Cursor session scores through the identical pipeline and looks like a Claude
session downstream (same S-XXXX id scheme, same fields).

Mapping:
  prompts      ← bubbles type==1 (.text)
  tool_cmds    ← toolFormerData of terminal tools (.command)        (verify classifier)
  n_tools      ← every bubble with toolFormerData
  n_edits      ← toolFormerData of edit tools (edit_file/search_replace/…),
                 else filesChangedCount when lines were added
  churn        ← composerData.totalLinesAdded                       (Cursor records it directly)
  loops        ← exact-duplicate (name+args) tool calls
  model        ← composerData.modelConfig.modelName
  tokens       ← Σ bubble.tokenCount (best-effort; cosmetic — not used in the digest/score)
  cwd / branch ← composerData.trackedGitRepos[0]
  session_uuid ← composerId (→ make_sid → S-XXXX)
  first/last_ts← createdAt / lastUpdatedAt (ms epoch → ISO)
"""
from __future__ import annotations
import json
import os
import sqlite3
from datetime import datetime, timezone

DEFAULT_CURSOR_DB = os.path.expanduser(
    "~/Library/Application Support/Cursor/User/globalStorage/state.vscdb")

# Cursor tool names (toolFormerData.name) → our classes. Names are versioned (edit_file_v2,
# run_terminal_command_v2, …) so match by prefix.
_SHELL_PREFIXES = ("run_terminal", "run_command", "terminal", "exec_command", "exec")
_EDIT_PREFIXES = ("edit_file", "search_replace", "apply_patch", "create_file", "str_replace",
                  "multi_edit", "write_file", "delete_file", "reapply")


def _is_shell(name: str) -> bool:
    n = (name or "").lower()
    return any(n.startswith(p) for p in _SHELL_PREFIXES)


def _is_edit(name: str) -> bool:
    n = (name or "").lower()
    return any(n.startswith(p) for p in _EDIT_PREFIXES)


def _db_path(db: str | None = None) -> str:
    return db or os.environ.get("CUEBENCH_CURSOR_DB") or DEFAULT_CURSOR_DB


def _connect(db: str):
    # immutable=1: read the main db file without taking a lock (Cursor may be running).
    return sqlite3.connect(f"file:{db}?immutable=1", uri=True, timeout=3)


def _ms_to_iso(ms) -> str | None:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).isoformat()
    except Exception:
        return None


def _tf_args(tf: dict) -> dict:
    """Tool args live in `rawArgs` and/or `params` — EITHER can be a JSON string or a dict
    (e.g. run_terminal_command_v2 leaves rawArgs empty and puts the command in params.command)."""
    out: dict = {}
    for field in ("rawArgs", "params"):
        v = tf.get(field)
        if isinstance(v, str) and v.strip():
            try:
                out.update(json.loads(v))
            except Exception:
                out.setdefault("_raw", v)
        elif isinstance(v, dict):
            out.update(v)
    return out


def _tf_cmd(args: dict) -> str:
    for k in ("command", "cmd", "terminalCommand", "text", "script"):
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list):
            return " ".join(str(x) for x in v)
    return ""


def list_cursor_composers(db: str | None = None) -> list[dict]:
    """Lightweight list of composers (no bubble fetches): id/name/status/last_updated/count."""
    p = _db_path(db)
    if not os.path.exists(p):
        return []
    out = []
    try:
        c = _connect(p)
        rows = c.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'").fetchall()
        c.close()
    except Exception:
        return []
    for k, v in rows:
        if v is None:
            continue
        try:
            cd = json.loads(v)
        except Exception:
            continue
        out.append({
            "composer_id": cd.get("composerId") or k.split(":", 1)[-1],
            "name": cd.get("name"),
            "status": cd.get("status"),
            "last_updated": cd.get("lastUpdatedAt"),
            "n_bubbles": len(cd.get("fullConversationHeadersOnly") or []),
        })
    return out


def parse_cursor_composer(composer_id: str, db: str | None = None) -> dict:
    """One Cursor composer → the same dict shape as cuebench_agent.parse_transcript."""
    from cuebench_agent import active_duration   # lazy: avoid import cycle
    p = _db_path(db)
    c = _connect(p)

    def get(key):
        r = c.execute("SELECT value FROM cursorDiskKV WHERE key=?", (key,)).fetchone()
        return r[0] if r and r[0] is not None else None

    raw = get(f"composerData:{composer_id}")
    if not raw:
        c.close()
        return _empty(composer_id)
    try:
        cd = json.loads(raw)
    except Exception:
        c.close()
        return _empty(composer_id)

    cid = cd.get("composerId") or composer_id
    headers = cd.get("fullConversationHeadersOnly") or []
    prompts, tool_cmds = [], []
    n_tools = n_edits = 0
    tool_sigs: dict[str, int] = {}
    in_tok = out_tok = 0
    timestamps = []

    for h in headers:
        bid = h.get("bubbleId")
        v = get(f"bubbleId:{cid}:{bid}")
        if not v:
            continue
        try:
            b = json.loads(v)
        except Exception:
            continue
        if b.get("type") == 1:                      # user turn
            txt = (b.get("text") or "").strip()
            if txt:
                prompts.append(txt)
        tf = b.get("toolFormerData")
        if isinstance(tf, dict):
            n_tools += 1
            name = tf.get("name") or ""          # NB: tf["tool"] is an int type-id, not a name
            args = _tf_args(tf)
            cmd = _tf_cmd(args) if _is_shell(name) else ""
            if cmd:
                tool_cmds.append(cmd)            # shell command → verification classifier
            if _is_edit(name):
                n_edits += 1
            # Loop = an unproductive EXACT re-run. Only repeated identical shell COMMANDS count:
            # Cursor stores only the file PATH in an edit/read tool's args (not the content), so
            # keying those on args would collapse distinct edits to one file into phantom loops
            # and manufacture a Discernment penalty. Non-shell calls get a unique key instead.
            sig = ("sh|" + cmd) if cmd else (name + "|" + (tf.get("toolCallId") or str(n_tools)))
            tool_sigs[sig] = tool_sigs.get(sig, 0) + 1
        tc = b.get("tokenCount")
        if isinstance(tc, dict):
            in_tok += int(tc.get("inputTokens", 0) or 0)
            out_tok += int(tc.get("outputTokens", 0) or 0)
        if isinstance(b.get("createdAt"), str):
            timestamps.append(b["createdAt"])

    churn = int(cd.get("totalLinesAdded") or 0)
    if n_edits == 0 and churn > 0:                  # edits via tools not labelled — fall back to file count
        n_edits = int(cd.get("filesChangedCount") or len(cd.get("newlyCreatedFiles") or []) or 0)

    repos = cd.get("trackedGitRepos") or []
    cwd = repos[0].get("repoPath") if repos and isinstance(repos[0], dict) else None
    branch = None
    if repos and repos[0].get("branches"):
        branch = (repos[0]["branches"][0] or {}).get("branchName")

    created, updated = cd.get("createdAt"), cd.get("lastUpdatedAt")
    if timestamps:
        duration = active_duration(timestamps)      # per-event cap if bubbles carry timestamps
    elif created and updated and updated > created:
        duration = int((updated - created) / 1000)  # else wall-clock start→end
    else:
        duration = 0

    model = None
    mc = cd.get("modelConfig")
    if isinstance(mc, dict):
        model = mc.get("modelName")

    c.close()
    return {
        "session_uuid": cid,
        "cli": "cursor",
        "title_hint": (cd.get("name") or "").strip() or None,   # Cursor's own composer name
        "prompts": prompts,
        "tool_cmds": tool_cmds,
        "n_tools": n_tools,
        "n_edits": n_edits,
        "churn": churn,
        "loops": sum(x - 1 for x in tool_sigs.values() if x > 1),
        "model": model,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "first_ts": _ms_to_iso(created),
        "last_ts": _ms_to_iso(updated),
        "duration_s": duration,
        "cwd": cwd,
        "git_branch": branch,
        "n_lines": len(headers),
    }


def _empty(cid: str) -> dict:
    return {"session_uuid": cid, "cli": "cursor", "title_hint": None,
            "prompts": [], "tool_cmds": [], "n_tools": 0, "n_edits": 0,
            "churn": 0, "loops": 0, "model": None, "input_tokens": 0, "output_tokens": 0,
            "first_ts": None, "last_ts": None, "duration_s": 0, "cwd": None,
            "git_branch": None, "n_lines": 0}
