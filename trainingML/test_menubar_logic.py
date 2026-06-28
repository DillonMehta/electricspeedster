#!/usr/bin/env python3
"""
test_menubar_logic.py — GUI-free logic of the menu bar app.

Covers the parts that have no UI: parsing scored sessions out of the daemon log, and the
Settings read/merge/write round-trip (so saving the form preserves unrelated keys and clears
emptied ones). No AppKit, no daemon, no model.

Run:  python3 -m pytest test_menubar_logic.py -q
"""
from __future__ import annotations
import cuebench_menubar as m
import cuebench_daemon as d


SAMPLE_LOG = """\
[daemon] starting  mode=DRY-RUN (no POST)  pid=999  time=2026-06-27 21:22:59
[daemon] model loaded (weights=model.safetensors); entering watch loop
[finished] aaa.jsonl
  sid=S-AAAAAAAA title='S-AAAAAAAA' score=36 quality='Insufficient signal' gen=off POST=OK (dry-run, not POSTed)
[finished] bbb.jsonl
  sid=S-BBBBBBBB title='Careful, well-verified work' score=81 quality='Dialed in (z1)' gen=anthropic POST=OK 200 {"ok":true}
[finished] ccc.jsonl
  [skip] ccc.jsonl: no operator prompts found
[finished] ddd.jsonl
  sid=S-DDDDDDDD title='S-DDDDDDDD' score=49 quality='Inconsistent (z4)' gen=off POST=FAIL HTTP 500: oops
"""


def test_parse_recent_sessions_order_and_fields():
    rows = m.parse_recent_sessions(SAMPLE_LOG, limit=6)
    assert [r["sid"] for r in rows] == ["S-DDDDDDDD", "S-BBBBBBBB", "S-AAAAAAAA"]  # most recent first
    latest = rows[0]
    assert latest["score"] == 49 and latest["quality"] == "Inconsistent (z4)"
    assert latest["posted"] is False               # POST=FAIL -> queued, not posted

    posted = rows[1]
    assert posted["sid"] == "S-BBBBBBBB"
    assert posted["title"] == "Careful, well-verified work"   # quotes stripped
    assert posted["posted"] is True and posted["dry_run"] is False

    dry = rows[2]
    assert dry["dry_run"] is True and dry["posted"] is False  # dry-run sentinel detected
    assert posted["file"] == "bbb.jsonl"                      # transcript captured for AI rename


def test_shorten_title():
    assert m._shorten_title('"Add debounce to search"') == "Add debounce to search"   # quotes stripped
    assert m._shorten_title("Refactor\nignored second line") == "Refactor"            # first line only
    long = m._shorten_title("Add comprehensive retry and backoff to the upload pipeline", max_chars=32)
    assert len(long) <= 32 and not long.endswith(" ") and " " in long                # word-boundary trim
    assert m._shorten_title("Fix bug.") == "Fix bug"                                  # trailing punct trimmed


def test_first_human_prompt(tmp_path):
    import json
    p = tmp_path / "s.jsonl"
    lines = [
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": []}}),
        # a tool-result user turn (must be skipped)
        json.dumps({"type": "user", "message": {"role": "user",
                    "content": [{"type": "tool_result", "content": "x"}]}}),
        # the real first prompt (with a system-reminder to strip)
        json.dumps({"type": "user", "origin": {"kind": "human"}, "message": {"role": "user",
                    "content": "Add a debounce <system-reminder>noise</system-reminder> to search"}}),
        json.dumps({"type": "user", "origin": {"kind": "human"}, "message": {"role": "user",
                    "content": "second prompt"}}),
    ]
    p.write_text("\n".join(lines))
    res = m.first_human_prompt(str(p))
    assert res.startswith("Add a debounce") and res.endswith("to search")  # first human prompt
    assert "system-reminder" not in res and "noise" not in res             # reminder stripped
    assert "second" not in res                                             # only the FIRST prompt
    assert m.first_human_prompt(str(tmp_path / "missing.jsonl")) is None


def test_title_override_roundtrip_and_applied(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "TITLE_OVERRIDES_PATH", str(tmp_path / "ov.json"))
    assert m.read_title_overrides() == {}
    m.set_title_override("S-DDDDDDDD", "My short name")
    assert m.read_title_overrides()["S-DDDDDDDD"] == "My short name"

    # recent_sessions should apply the override to the matching row's title
    log = tmp_path / "out.log"
    log.write_text(SAMPLE_LOG)
    monkeypatch.setattr(d, "OUT_LOG", str(log))
    rows = m.recent_sessions(6)
    ddd = next(r for r in rows if r["sid"] == "S-DDDDDDDD")
    assert ddd["title"] == "My short name"


def test_parse_dedupes_rescored_session_keeping_latest():
    # The daemon re-scores a growing session, logging the same sid repeatedly (gen=cache,
    # POST UPDATE). The menu should show it ONCE, with the most recent score/title.
    log = """\
[finished] zzz.jsonl
  sid=S-REPEAT01 title='Early draft' score=60 quality='Developing (z3)' gen=generated POST=OK (dry-run, not POSTed)
[finished] other.jsonl
  sid=S-OTHER999 title='Another session' score=70 quality='Solid (z2)' gen=generated POST=OK (dry-run, not POSTed)
[finished] zzz.jsonl
  sid=S-REPEAT01 title='Productive work' score=77 quality='Solid (z2)' gen=cache POST=OK UPDATE (dry-run, not POSTed)
[finished] zzz.jsonl
  sid=S-REPEAT01 title='Productive work' score=77 quality='Solid (z2)' gen=cache POST=OK UPDATE (dry-run, not POSTed)
"""
    rows = m.parse_recent_sessions(log, limit=6)
    assert [r["sid"] for r in rows] == ["S-REPEAT01", "S-OTHER999"]   # one row per sid
    latest = next(r for r in rows if r["sid"] == "S-REPEAT01")
    assert latest["score"] == 77 and latest["title"] == "Productive work"   # most recent kept


def test_parse_limit_and_empty():
    assert m.parse_recent_sessions(SAMPLE_LOG, limit=1)[0]["sid"] == "S-DDDDDDDD"
    assert m.parse_recent_sessions("", limit=5) == []
    assert m.parse_recent_sessions("nothing scored here", limit=5) == []


def test_settings_roundtrip_merge_and_clear(tmp_path, monkeypatch):
    envf = str(tmp_path / "daemon.env")
    # pre-existing file with a managed key AND an unrelated key that must be preserved
    with open(envf, "w") as f:
        f.write("CUEBENCH_API_KEY=oldkey\nCUEBENCH_UNRELATED=keepme\n")

    # set + clear: employee gets set, the old API key is cleared (emptied), unrelated preserved
    d.save_env_file({"CUEBENCH_EMPLOYEE_ID": "e9",
                     "CUEBENCH_API_KEY": "",
                     "CUEBENCH_BYOK_KEY": "sk-test"}, path=envf)
    parsed = d.parse_env_file(envf)
    assert parsed["CUEBENCH_EMPLOYEE_ID"] == "e9"
    assert parsed["CUEBENCH_BYOK_KEY"] == "sk-test"
    assert "CUEBENCH_API_KEY" not in parsed            # emptied -> removed (generation/POST off)
    assert parsed["CUEBENCH_UNRELATED"] == "keepme"    # non-managed key preserved

    # file is private
    import os, stat
    assert stat.S_IMODE(os.stat(envf).st_mode) == 0o600


def test_read_settings_reads_managed_only(tmp_path, monkeypatch):
    envf = str(tmp_path / "daemon.env")
    with open(envf, "w") as f:
        f.write("CUEBENCH_EMPLOYEE_ID=e3\nCUEBENCH_UNRELATED=x\n")
    monkeypatch.setattr(d, "ENVFILE", envf)
    s = m.read_settings()
    assert s["CUEBENCH_EMPLOYEE_ID"] == "e3"
    assert set(s.keys()) == set(d.MANAGED_ENV_KEYS)    # only managed keys surface in the pane


def test_extract_payload_json():
    # mimics `cuebench_agent.py --once --dry-run` stdout: banner, payload, then a summary line
    stdout = (
        "[load] ok\n"
        '{\n  "employeeId": "e1",\n  "sessionId": "S-ABCD",\n  "score": 81,\n'
        '  "quality": {"label": "Dialed in", "zone": 1}\n}\n'
        "  sid=S-ABCD title='x' score=81 quality='Dialed in (z1)' gen=off POST=OK (dry-run, not POSTed)\n"
    )
    payload = m._extract_payload_json(stdout)
    assert payload["sessionId"] == "S-ABCD"
    assert payload["score"] == 81
    assert payload["quality"]["zone"] == 1            # nested braces handled
    assert m._extract_payload_json("no json here") is None


def test_latest_session_path_skips_agents(tmp_path):
    import os
    d = tmp_path / "proj"
    d.mkdir()
    old = d / "old.jsonl"; old.write_text("{}")
    new = d / "new.jsonl"; new.write_text("{}")
    agent = d / "agent-sub.jsonl"; agent.write_text("{}")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    os.utime(agent, (3000, 3000))                     # newest, but must be skipped
    assert m.latest_session_path(str(tmp_path)) == str(new)
    assert m.latest_session_path(str(tmp_path / "empty")) is None


def test_env_truthy():
    assert m._env_truthy("1") and m._env_truthy("true") and m._env_truthy("ON")
    assert not m._env_truthy("") and not m._env_truthy("0") and not m._env_truthy(None)


def test_trace_is_a_managed_toggle():
    assert "CUEBENCH_TRACE" in d.MANAGED_ENV_KEYS                     # editable in Settings
    assert ("CUEBENCH_TRACE", "Session trace", "toggle") in m.SETTINGS_FIELDS


def test_mask_key():
    assert m._mask_key("") == "(none set)"
    assert m._mask_key("sk-ant-abcdefghijklmnop") == "sk-ant…mnop"   # masked middle
    assert "…" in m._mask_key("short")


def test_diagnose_post_status_mapping():
    assert m._diagnose_post("u", "k", 200, '{"ok":true}', None).startswith("✓")
    assert "401" in m._diagnose_post("u", "k", 401, "Invalid x-api-key", None)
    assert "no API key" in m._diagnose_post("u", "", 401, "x-api-key header required", None)
    assert "employeeId" in m._diagnose_post("u", "k", 404, "Employee not found", None)
    assert "400" in m._diagnose_post("u", "k", 400, "employeeId required", None)
    assert "Cloudflare" in m._diagnose_post("u", "k", 403, "error code: 1010", None)  # UA ban
    assert "permission" in m._diagnose_post("u", "k", 403, "Forbidden", None).lower()  # generic 403
    assert m._diagnose_post("u", "k", None, None, "Connection refused").startswith("✗")


def test_debug_test_post_no_url(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "ENVFILE", str(tmp_path / "none.env"))   # no daemon.env -> no URL
    monkeypatch.delenv("CUEBENCH_API_URL", raising=False)
    report = m.debug_test_post()
    assert "CUEBENCH_API_URL is not set" in report                  # never throws; reports clearly
    assert "request body" in report and "employeeId" in report      # shows what it would send


def test_build_app_bundle(tmp_path):
    app = m.build_app_bundle(str(tmp_path))
    import os
    assert app.endswith("CueBench.app")
    assert os.access(os.path.join(app, "Contents", "MacOS", "CueBench"), os.X_OK)
    import plistlib
    info = plistlib.load(open(os.path.join(app, "Contents", "Info.plist"), "rb"))
    assert info["LSUIElement"] is True                  # menu-bar only (no Dock icon)
    assert info["CFBundleExecutable"] == "CueBench"
