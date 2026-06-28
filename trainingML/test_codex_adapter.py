#!/usr/bin/env python3
"""
test_codex_adapter.py — Codex rollout → the Claude-shaped session dict.

Builds a synthetic Codex rollout (incl. an apply_patch edit, a duplicate tool call, and an
injected non-human user turn) and asserts cuebench_codex maps it to the SAME dict shape the
engine scores — so a Codex session flows through the identical pipeline. No model, no network.

Run:  python3 -m pytest test_codex_adapter.py -q
"""
from __future__ import annotations
import json
import cuebench_codex as cx
import cuebench_agent as a


def _rollout(tmp_path):
    """A realistic Codex rollout: meta, turn_context, two user turns (one injected), two
    identical shell calls (a loop) + an apply_patch edit, and a cumulative token count."""
    recs = [
        {"timestamp": "2026-06-28T11:00:00.000Z", "type": "session_meta",
         "payload": {"session_id": "abcd1234-0000-1111-2222-333344445555",
                     "cwd": "/Users/x/proj", "model_provider": "openai"}},
        {"timestamp": "2026-06-28T11:00:01.000Z", "type": "turn_context",
         "payload": {"model": "gpt-5.5", "cwd": "/Users/x/proj"}},
        # injected (non-human) user turn — must be ignored
        {"timestamp": "2026-06-28T11:00:02.000Z", "type": "event_msg",
         "payload": {"type": "user_message", "message": "<environment_context><cwd>/Users/x/proj</cwd>"}},
        # the real prompt
        {"timestamp": "2026-06-28T11:00:03.000Z", "type": "event_msg",
         "payload": {"type": "user_message", "message": "add a debounce to search"}},
        # two identical shell calls -> one loop; both feed the verify classifier
        {"timestamp": "2026-06-28T11:00:10.000Z", "type": "response_item",
         "payload": {"type": "function_call", "name": "exec_command",
                     "arguments": json.dumps({"cmd": "npm test"})}},
        {"timestamp": "2026-06-28T11:00:20.000Z", "type": "response_item",
         "payload": {"type": "function_call", "name": "exec_command",
                     "arguments": json.dumps({"cmd": "npm test"})}},
        # a dedicated apply_patch edit -> n_edits=1, churn = 2 added lines
        {"timestamp": "2026-06-28T11:00:30.000Z", "type": "response_item",
         "payload": {"type": "function_call", "name": "apply_patch",
                     "arguments": json.dumps({"patch": "*** Begin Patch\n*** Update File: s.js\n"
                                              "+const a = 1\n+const b = 2\n*** End Patch"})}},
        # an edit performed VIA the shell (apply_patch heredoc) -> also counts as an edit
        {"timestamp": "2026-06-28T11:00:40.000Z", "type": "response_item",
         "payload": {"type": "function_call", "name": "exec_command",
                     "arguments": json.dumps({"cmd": "apply_patch <<'EOF'\n*** Begin Patch\n"
                                              "+x\n*** End Patch\nEOF"})}},
        {"timestamp": "2026-06-28T11:01:00.000Z", "type": "event_msg",
         "payload": {"type": "token_count",
                     "total_token_usage": {"input_tokens": 1000, "output_tokens": 200,
                                           "total_tokens": 1200}}},
    ]
    p = tmp_path / "rollout-2026-06-28T11-00-00-abcd1234.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    return str(p)


def test_is_codex_rollout(tmp_path):
    p = _rollout(tmp_path)
    assert cx.is_codex_rollout(p) is True
    claude = tmp_path / "x.jsonl"
    claude.write_text(json.dumps({"type": "user", "sessionId": "s",
                                  "message": {"role": "user", "content": "hi"}}))
    assert cx.is_codex_rollout(str(claude)) is False


def test_parse_codex_maps_to_claude_shape(tmp_path):
    d = cx.parse_codex_transcript(_rollout(tmp_path))
    assert d["session_uuid"] == "abcd1234-0000-1111-2222-333344445555"
    assert d["cli"] == "codex"                                    # provider tag for the POST
    assert a.make_sid(d["session_uuid"]) == "S-ABCD1234"          # same sid scheme as Claude
    assert d["prompts"] == ["add a debounce to search"]           # injected turn dropped
    assert d["n_tools"] == 4                                      # 3 exec + 1 apply_patch
    assert d["tool_cmds"].count("npm test") == 2                 # shell cmds → verify classifier
    assert d["n_edits"] == 2                                      # apply_patch fn + shell apply_patch
    assert d["churn"] == 3                                        # 2 + 1 added lines
    assert d["loops"] == 1                                        # the duplicate `npm test`
    assert d["model"] == "gpt-5.5"
    assert d["input_tokens"] == 1000 and d["output_tokens"] == 200
    assert d["cwd"] == "/Users/x/proj"
    # exact same keys the Claude parser returns — so everything downstream is identical
    assert set(d.keys()) == {
        "session_uuid", "cli", "prompts", "tool_cmds", "n_tools", "n_edits", "churn", "loops",
        "model", "input_tokens", "output_tokens", "first_ts", "last_ts", "duration_s",
        "cwd", "git_branch", "n_lines"}


def test_dispatcher_routes_codex(tmp_path):
    p = _rollout(tmp_path)
    viaagent = a.parse_session(p)            # engine dispatcher
    direct = cx.parse_codex_transcript(p)
    assert viaagent["session_uuid"] == direct["session_uuid"]
    assert viaagent["n_edits"] == direct["n_edits"] == 2


def test_token_reasoning_folded_into_output(tmp_path):
    # when total > input+output (reasoning tokens), fold the remainder into output
    p = tmp_path / "rollout-r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"timestamp": "2026-06-28T11:00:00.000Z", "type": "session_meta",
         "payload": {"session_id": "r", "cwd": "/x"}},
        {"timestamp": "2026-06-28T11:00:01.000Z", "type": "event_msg",
         "payload": {"type": "token_count",
                     "total_token_usage": {"input_tokens": 100, "output_tokens": 10,
                                           "reasoning_output_tokens": 40, "total_tokens": 150}}},
    ]))
    d = cx.parse_codex_transcript(str(p))
    assert d["input_tokens"] == 100 and d["output_tokens"] == 50   # 150 - 100
