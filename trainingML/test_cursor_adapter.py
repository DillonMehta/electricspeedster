#!/usr/bin/env python3
"""
test_cursor_adapter.py — Cursor composer (SQLite) → the Claude-shaped session dict, and the
daemon's stable-scan logic for the (non-file) Cursor source.

Builds a synthetic Cursor state.vscdb and asserts cuebench_cursor maps a composer to the same
dict shape the engine scores — including the tricky bits learned from the real DB: args live
in a JSON-string `params`, the edit tool is `edit_file_v2`, and repeated edits to the SAME file
must NOT count as loops (only repeated identical shell commands do). No model, no network.

Run:  python3 -m pytest test_cursor_adapter.py -q
"""
from __future__ import annotations
import json
import sqlite3
import cuebench_cursor as cur
import cuebench_agent as a
import cuebench_daemon as daemon

CID = "cur12345-aaaa-bbbb-cccc-dddddddddddd"


def _make_db(tmp_path):
    p = tmp_path / "state.vscdb"
    c = sqlite3.connect(str(p))
    c.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)")
    headers = [{"bubbleId": b, "type": t} for b, t in
               [("b1", 1), ("b2", 2), ("b3", 2), ("b4", 2), ("b5", 2)]]
    cd = {"composerId": CID, "name": "Build app", "status": "completed",
          "createdAt": 1_000_000, "lastUpdatedAt": 1_300_000, "totalLinesAdded": 42,
          "modelConfig": {"modelName": "composer-2.5"},
          "trackedGitRepos": [{"repoPath": "/Users/x/proj",
                               "branches": [{"branchName": "main"}]}],
          "fullConversationHeadersOnly": headers}
    rows = {f"composerData:{CID}": cd,
            f"bubbleId:{CID}:b1": {"type": 1, "text": "do X please"},
            # shell tool — command lives in a JSON-string `params` (rawArgs empty, like real)
            f"bubbleId:{CID}:b2": {"type": 2, "toolFormerData": {
                "name": "run_terminal_command_v2", "toolCallId": "t2", "rawArgs": "",
                "params": json.dumps({"command": "npm test"})}},
            # two edits to the SAME file — args only carry the path; must NOT be a loop
            f"bubbleId:{CID}:b3": {"type": 2, "toolFormerData": {
                "name": "edit_file_v2", "toolCallId": "t3",
                "params": json.dumps({"relativeWorkspacePath": "a.js"})}},
            f"bubbleId:{CID}:b4": {"type": 2, "toolFormerData": {
                "name": "edit_file_v2", "toolCallId": "t4",
                "params": json.dumps({"relativeWorkspacePath": "a.js"})}},
            # duplicate shell command -> ONE real loop
            f"bubbleId:{CID}:b5": {"type": 2, "toolFormerData": {
                "name": "run_terminal_command_v2", "toolCallId": "t5", "rawArgs": "",
                "params": json.dumps({"command": "npm test"})}}}
    for k, v in rows.items():
        c.execute("INSERT INTO cursorDiskKV VALUES (?,?)", (k, json.dumps(v)))
    c.commit()
    c.close()
    return str(p)


def test_list_cursor_composers(tmp_path):
    db = _make_db(tmp_path)
    comps = cur.list_cursor_composers(db)
    assert len(comps) == 1
    assert comps[0]["name"] == "Build app"
    assert comps[0]["status"] == "completed"
    assert comps[0]["n_bubbles"] == 5


def test_parse_cursor_maps_to_claude_shape(tmp_path):
    d = cur.parse_cursor_composer(CID, _make_db(tmp_path))
    assert d["session_uuid"] == CID
    assert d["cli"] == "cursor"                    # provider tag for the POST
    assert a.make_sid(CID) == "S-CUR12345"
    assert d["title_hint"] == "Build app"          # composer name → free, accurate title
    assert d["prompts"] == ["do X please"]
    assert d["n_tools"] == 4                       # 2 terminal + 2 edit
    assert d["tool_cmds"] == ["npm test", "npm test"]   # command pulled from params JSON string
    assert d["n_edits"] == 2                       # both edit_file_v2 calls
    assert d["churn"] == 42                        # totalLinesAdded
    assert d["loops"] == 1                         # dup `npm test` only — edits to a.js do NOT loop
    assert d["model"] == "composer-2.5"
    assert d["cwd"] == "/Users/x/proj" and d["git_branch"] == "main"
    assert set(d.keys()) == {
        "session_uuid", "cli", "title_hint", "prompts", "tool_cmds", "n_tools", "n_edits",
        "churn", "loops", "model", "input_tokens", "output_tokens", "first_ts", "last_ts",
        "duration_s", "cwd", "git_branch", "n_lines"}


def _fake_agent():
    calls = []

    class FakeAgent:
        def process_parsed(self, parsed, scorer, store, gen, *, dry_run, source):
            calls.append(source)
            return ("posted", "S-X")
    return FakeAgent(), calls


def test_scan_cursor_stability_and_dedup(monkeypatch, tmp_path):
    """Cursor is read only after the DB mtime settles for stable_seconds, then scored once.
    (mtime-gating means the daemon reads Cursor's TCC-gated folder ~once per finished session,
    not once per poll — so macOS doesn't re-prompt every cycle.)"""
    monkeypatch.setattr(cur, "list_cursor_composers",
                        lambda db=None: [{"composer_id": "c1", "name": "n",
                                          "last_updated": 5, "n_bubbles": 3}])
    monkeypatch.setattr(cur, "parse_cursor_composer",
                        lambda cid, db=None: {"session_uuid": cid, "prompts": ["x"]})
    db = tmp_path / "state.vscdb"
    db.write_bytes(b"x" * 32)
    db = str(db)
    st, (fa, calls) = {}, _fake_agent()
    daemon.scan_cursor_once(fa, None, None, None, dry_run=True, state=st, stable_seconds=0, db=db)
    assert calls == []                              # first sight of this mtime → start the clock
    daemon.scan_cursor_once(fa, None, None, None, dry_run=True, state=st, stable_seconds=0, db=db)
    assert calls == ["cursor:c1"]                   # mtime settled → read + scored once
    daemon.scan_cursor_once(fa, None, None, None, dry_run=True, state=st, stable_seconds=0, db=db)
    assert calls == ["cursor:c1"]                   # same stable mtime → not re-scored


def test_scan_cursor_disables_on_denied_access(monkeypatch, tmp_path):
    """A macOS PermissionError disables Cursor scanning for the run — so the daemon stops
    re-prompting every poll instead of nagging the user indefinitely."""
    def boom(*a, **k):
        raise PermissionError("Operation not permitted")
    monkeypatch.setattr(cur, "list_cursor_composers", boom)
    db = tmp_path / "state.vscdb"
    db.write_bytes(b"x" * 32)
    db = str(db)
    st, (fa, calls) = {}, _fake_agent()
    daemon.scan_cursor_once(fa, None, None, None, dry_run=True, state=st, stable_seconds=0, db=db)
    daemon.scan_cursor_once(fa, None, None, None, dry_run=True, state=st, stable_seconds=0, db=db)
    assert st["disabled"] is True                   # denial → scanning turned off
    assert calls == []
    daemon.scan_cursor_once(fa, None, None, None, dry_run=True, state=st, stable_seconds=0, db=db)
    assert st["disabled"] is True                   # stays off; no further read attempts
