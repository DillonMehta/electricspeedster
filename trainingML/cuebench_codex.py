#!/usr/bin/env python3
"""
cuebench_codex.py — Codex CLI rollout → the SAME session dict Claude Code produces
==================================================================================
A provider adapter. The CueBench engine scores a `parse_transcript()` result dict
(prompts / tool_cmds / n_tools / n_edits / churn / loops / model / tokens / ts / cwd …).
This module produces that EXACT dict from an OpenAI Codex CLI rollout so a Codex session
flows through the identical pipeline (build_inputs → digest_text → model → POST) and looks
just like a Claude Code session to everything downstream.

Codex rollout format (`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`): one JSON object per
line, `{timestamp, type, payload}`:
  type=session_meta   payload.{session_id, cwd, model_provider, ...}      (1, first line)
  type=turn_context   payload.{model, cwd, ...}                           (per turn)
  type=event_msg      payload.type ∈ {user_message, agent_message, token_count, task_*}
  type=response_item  payload.type ∈ {message, function_call, function_call_output}

Mapping to the Claude dict:
  prompts      ← event_msg/user_message.message                 (the human turns)
  tool_cmds    ← function_call shell tools (exec_command…).cmd   (for the verify classifier)
  n_tools      ← every function_call                            (all tool calls)
  n_edits      ← function_calls that are edits (apply_patch…)
  churn        ← added (+) lines in those patches
  loops        ← exact-duplicate (name+args) re-runs            (same defn as the engine)
  model        ← last turn_context.model
  in/out tokens← last token_count.total_token_usage (cumulative)
  first/last_ts← min/max record timestamp; duration via the engine's active_duration
  cwd          ← session_meta.cwd (→ git signals)
  session_uuid ← session_meta.session_id (→ make_sid → S-XXXX, same scheme as Claude)
"""
from __future__ import annotations
import json
import os
import re

# shell-style tools whose command text should feed the verification classifier
_SHELL_NAMES = {"exec_command", "shell", "local_shell", "bash", "container.exec", "exec"}
# tools that edit files (dedicated patch/write tools)
_EDIT_NAMES = {"apply_patch", "edit", "write_file", "create_file", "str_replace",
               "update_file", "str_replace_editor"}
# injected (non-human) user turns to ignore, mirroring Claude's human-prompt filter
_INJECTED_PREFIXES = ("<environment_context", "<permissions", "<user_instructions",
                      "<system", "# permissions")


def _payload(rec: dict) -> dict:
    p = rec.get("payload")
    return p if isinstance(p, dict) else {}


def _args(p: dict) -> dict:
    """function_call.arguments is a JSON string in the rollout (OpenAI convention); be
    tolerant of a dict too."""
    a = p.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return {"_raw": a}
    return a if isinstance(a, dict) else {}


def _cmd_text(args: dict) -> str:
    c = args.get("cmd") or args.get("command") or args.get("script") or args.get("_raw") or ""
    if isinstance(c, list):
        c = " ".join(str(x) for x in c)
    return c if isinstance(c, str) else ""


def _patch_added_lines(text: str) -> int:
    """Count added lines in an apply_patch body — `+` lines, excluding the `+++` header
    (mirrors the engine's written-line churn signal)."""
    if not text:
        return 0
    return sum(1 for ln in text.splitlines()
               if ln.startswith("+") and not ln.startswith("+++"))


def is_codex_rollout(path: str) -> bool:
    """True if this looks like a Codex rollout (by filename, else by the first record)."""
    if os.path.basename(path).startswith("rollout-"):
        return True
    try:
        with open(path, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    return False
                p = _payload(rec)
                return rec.get("type") == "session_meta" or p.get("session_id") is not None
    except Exception:
        return False
    return False


def codex_session_uuid(path: str) -> str:
    """Resolve the session uuid WITHOUT a full parse (so dedup matches the POSTed sid)."""
    try:
        with open(path, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                p = _payload(rec)
                sid = p.get("session_id") or p.get("id")
                if sid:
                    return sid
    except Exception:
        pass
    return os.path.splitext(os.path.basename(path))[0]


def parse_codex_transcript(path: str) -> dict:
    """Codex rollout → the same dict shape as cuebench_agent.parse_transcript. Never raises
    on bad lines (skips them) so a malformed rollout can't crash the watcher."""
    from cuebench_agent import active_duration   # lazy: avoid an import cycle

    prompts, tool_cmds = [], []
    n_tools = n_edits = churn = 0
    tool_sigs: dict[str, int] = {}
    models, timestamps = [], []
    cwd = git_branch = session_uuid = None
    last_token_usage = None
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
            if rec.get("timestamp"):
                timestamps.append(rec["timestamp"])
            t = rec.get("type")
            p = _payload(rec)
            pt = p.get("type")

            if t == "session_meta":
                session_uuid = session_uuid or p.get("session_id") or p.get("id")
                cwd = cwd or p.get("cwd")
                git = p.get("git") if isinstance(p.get("git"), dict) else {}
                git_branch = git_branch or git.get("branch")

            elif t == "turn_context":
                if p.get("model"):
                    models.append(p["model"])
                cwd = cwd or p.get("cwd")

            elif t == "event_msg" and pt == "user_message":
                msg = p.get("message")
                if isinstance(msg, str):
                    txt = msg.strip()
                    low = txt.lower()
                    if txt and not any(low.startswith(pre) for pre in _INJECTED_PREFIXES):
                        prompts.append(txt)

            elif t == "event_msg" and pt == "token_count":
                info = p.get("info") if isinstance(p.get("info"), dict) else p
                tu = info.get("total_token_usage")
                if isinstance(tu, dict):
                    last_token_usage = tu     # cumulative; keep the latest

            elif t == "response_item" and pt == "function_call":
                name = p.get("name", "")
                args = _args(p)
                n_tools += 1
                sig = name + "|" + json.dumps(args, sort_keys=True, default=str)
                tool_sigs[sig] = tool_sigs.get(sig, 0) + 1
                cmd = _cmd_text(args)
                is_shell = name in _SHELL_NAMES
                # an edit = a dedicated patch tool, OR a shell call that runs apply_patch
                patch_text = (args.get("patch") or args.get("input")
                              or args.get("file_text") or args.get("content") or "")
                if not isinstance(patch_text, str):
                    patch_text = ""
                is_edit = (name in _EDIT_NAMES
                           or "apply_patch" in cmd or "*** Begin Patch" in cmd)
                if is_shell and cmd:
                    tool_cmds.append(cmd)     # bash-equivalent → verify classifier
                if is_edit:
                    n_edits += 1
                    churn += _patch_added_lines(patch_text or cmd)

    if session_uuid is None:
        session_uuid = codex_session_uuid(path)

    in_tok = out_tok = 0
    if last_token_usage:
        in_tok = int(last_token_usage.get("input_tokens", 0) or 0)
        out_tok = int(last_token_usage.get("output_tokens", 0) or 0)
        total = int(last_token_usage.get("total_tokens", 0) or 0)
        if total and total > in_tok + out_tok:    # fold reasoning tokens into output
            out_tok = total - in_tok

    return {
        "session_uuid": session_uuid,
        "prompts": prompts,
        "tool_cmds": tool_cmds,
        "n_tools": n_tools,
        "n_edits": n_edits,
        "churn": churn,
        "loops": sum(c - 1 for c in tool_sigs.values() if c > 1),
        "model": models[-1] if models else None,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "first_ts": min(timestamps) if timestamps else None,
        "last_ts": max(timestamps) if timestamps else None,
        "duration_s": active_duration(timestamps),
        "cwd": cwd,
        "git_branch": git_branch,
        "n_lines": n_lines,
    }
