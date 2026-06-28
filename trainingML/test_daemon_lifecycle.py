#!/usr/bin/env python3
"""
test_daemon_lifecycle.py — the background daemon's lifecycle guarantees.

These tests exercise the SHARED watch loop (cuebench_agent.run_watch_loop) the way the
daemon drives it, plus the daemon's own helpers — WITHOUT loading the real RoBERTa model
(FakeScorer) and WITHOUT the live dashboard (a local mock HTTP server records POSTs).

What is proven here (maps to the user's 7 guards):
  * start → new session appears → scored + POSTed exactly ONCE                  (core flow)
  * restart → the already-sent session is NOT re-POSTed (dedup honored)         (guard #4)
  * dry-run mode scores but NEVER POSTs                                         (default safety)
  * no key / offline → never crashes; session is queued and POSTs once a key
    is available (no loss, no duplicate)                                        (guard #5)
  * the on-demand watch() entry point is UNCHANGED (still exits 2 with no key)  (guard #1)
  * the daemon's env-file loader and single-instance pid lock work             (guard #2)

Run:  python3 -m pytest test_daemon_lifecycle.py -q
"""
from __future__ import annotations
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import cuebench_agent as agent
import cuebench_daemon as daemon
from cuebench_store import Store
from cuebench_gen import Generator

SESSION_UUID = "daemonsess-aaaa-bbbb-cccc-ddddeeeeffff"
SID = agent.make_sid(SESSION_UUID)


# ---- doubles ---------------------------------------------------------------
class FakeScorer:
    """Fixed vectors — no RoBERTa load (keeps the test fast and RAM-light)."""
    loaded_from = "fake"
    def score(self, digest):
        return {"delegation": 70, "description": 68, "discernment": 72, "diligence": 75}


class _Stop(Exception):
    """Raised from on_cycle to break the otherwise-infinite watch loop in a test."""


def stop_after(n: int):
    state = {"i": 0}
    def cb():
        state["i"] += 1
        if state["i"] >= n:
            raise _Stop()
    return cb


def run_loop(scorer, store, gen, *, dry_run, cycles=4):
    """Drive the real run_watch_loop for a few cycles then stop (POLL/STABLE forced to 0)."""
    try:
        agent.run_watch_loop(scorer, store, gen, dry_run=dry_run, on_cycle=stop_after(cycles))
    except _Stop:
        pass


# ---- a real (local) mock dashboard ----------------------------------------
class MockDashboard:
    """Counts POSTs two ways: total sends, and unique sessionIds (UPSERT view)."""
    def __init__(self):
        self.sends = []          # every accepted POST (a re-POST shows up as a 2nd entry)
        self.by_sid = {}         # sid -> last payload (UPSERT: len == stored rows)

    def __enter__(self):
        dash = self
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n) or b"{}")
                dash.sends.append(payload["sessionId"])
                dash.by_sid[payload["sessionId"]] = payload
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/api/session"
        self.t = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()


# ---- fixtures --------------------------------------------------------------
@pytest.fixture
def projects(tmp_path):
    """A temp ~/.claude/projects layout with one valid finished session."""
    d = tmp_path / "projects" / "proj-hash"
    d.mkdir(parents=True)
    p = d / f"{SESSION_UUID}.jsonl"
    tools = [{"type": "tool_use", "name": "Bash", "input": {"command": f"echo step{i}"}}
             for i in range(12)]
    lines = [
        json.dumps({"type": "user", "sessionId": SESSION_UUID, "origin": {"kind": "human"},
                    "timestamp": "2026-01-01T00:00:00.000Z",
                    "message": {"role": "user",
                                "content": "please implement the described feature with enough detail here"}}),
        json.dumps({"type": "assistant", "sessionId": SESSION_UUID,
                    "timestamp": "2026-01-01T00:05:00.000Z",
                    "message": {"role": "assistant", "model": "claude-opus-4-8",
                                "usage": {"output_tokens": 60}, "content": tools}}),
    ]
    p.write_text("\n".join(lines))
    return str(tmp_path / "projects")


@pytest.fixture
def fast_loop(monkeypatch, projects):
    """Point the loop at the temp projects dir and make poll/stable instant."""
    monkeypatch.setattr(agent, "PROJECTS_DIR", projects)
    monkeypatch.setattr(agent, "POLL_INTERVAL", 0)
    monkeypatch.setattr(agent, "STABLE_SECONDS", 0)


# ---- 1: start → score+POST once → restart → no re-POST  (guard #4) ----------
def test_post_once_then_no_repost_on_restart(tmp_path, monkeypatch, fast_loop):
    db = str(tmp_path / "state.db")
    gen = Generator(key="")                              # generation off (isolate POST dedup)
    with MockDashboard() as dash:
        monkeypatch.setattr(agent, "API_URL", dash.url)
        monkeypatch.setattr(agent, "API_KEY", "fake-key")

        # START — first run: the new session is found, scored, POSTed once.
        store1 = Store(db)
        run_loop(FakeScorer(), store1, gen, dry_run=False)
        store1.close()
        assert dash.sends == [SID]                       # exactly one POST
        assert len(dash.by_sid) == 1

        # RESTART — fresh in-memory state, same on-disk dedup DB: must NOT re-POST.
        store2 = Store(db)
        assert store2.is_posted(SID) is True
        run_loop(FakeScorer(), store2, gen, dry_run=False)
        store2.close()
        assert dash.sends == [SID]                       # STILL one — no duplicate
        assert len(dash.by_sid) == 1


# ---- 2: dry-run scores but never POSTs  (default safety) --------------------
def test_dry_run_never_posts(tmp_path, monkeypatch, fast_loop):
    db = str(tmp_path / "state.db")
    gen = Generator(key="")
    with MockDashboard() as dash:
        monkeypatch.setattr(agent, "API_URL", dash.url)
        monkeypatch.setattr(agent, "API_KEY", "fake-key")
        # also make a real POST explode if ever attempted in dry-run
        def boom(_p):
            raise AssertionError("post_payload must NOT be called in dry-run")
        monkeypatch.setattr(agent, "post_payload", boom)

        store = Store(db)
        run_loop(FakeScorer(), store, gen, dry_run=True)
        assert dash.sends == []                          # nothing POSTed
        assert store.is_posted(SID) is False             # nothing marked sent
        store.close()


# ---- 3: no key / offline → queue + retry, never crash, no loss  (guard #5) --
def test_no_key_queues_then_posts_when_key_arrives(tmp_path, monkeypatch, fast_loop):
    db = str(tmp_path / "state.db")
    gen = Generator(key="")
    with MockDashboard() as dash:
        monkeypatch.setattr(agent, "API_URL", dash.url)

        # No key yet: live run scores but the POST fails -> queued (retry-later), no crash.
        monkeypatch.setattr(agent, "API_KEY", None)
        store = Store(db)
        run_loop(FakeScorer(), store, gen, dry_run=False)
        assert dash.sends == []                          # nothing sent
        assert store.is_posted(SID) is False             # not marked -> will retry
        store.close()

        # Key arrives: next run POSTs exactly once. The session was never lost.
        monkeypatch.setattr(agent, "API_KEY", "fake-key")
        store = Store(db)
        run_loop(FakeScorer(), store, gen, dry_run=False)
        assert dash.sends == [SID]
        assert store.is_posted(SID) is True
        store.close()


# ---- 4: the on-demand watch() entry point is UNCHANGED  (guard #1) ----------
def test_watch_entrypoint_still_requires_key(monkeypatch):
    monkeypatch.setattr(agent, "API_KEY", None)
    with pytest.raises(SystemExit) as ei:
        agent.watch(FakeScorer(), object(), Generator(key=""))
    assert ei.value.code == 2                            # same hard-exit as before the refactor


# ---- 5: daemon env-file loader (bridges launchd's missing shell env) --------
def test_env_file_loader_setdefault(tmp_path, monkeypatch):
    envf = tmp_path / "daemon.env"
    envf.write_text("# comment\nexport CUEBENCH_EMPLOYEE_ID=e7\nCUEBENCH_FOO=\"bar baz\"\n")
    monkeypatch.delenv("CUEBENCH_EMPLOYEE_ID", raising=False)
    monkeypatch.delenv("CUEBENCH_FOO", raising=False)
    monkeypatch.setenv("CUEBENCH_ALREADY", "keepme")
    n = daemon.load_env_file(str(envf))
    assert n == 2
    assert __import__("os").environ["CUEBENCH_EMPLOYEE_ID"] == "e7"
    assert __import__("os").environ["CUEBENCH_FOO"] == "bar baz"
    # existing env is never overridden (plist / launchctl setenv wins)
    monkeypatch.setenv("CUEBENCH_EMPLOYEE_ID", "preset")
    daemon.load_env_file(str(envf))
    assert __import__("os").environ["CUEBENCH_EMPLOYEE_ID"] == "preset"


# ---- 6: single-instance pid lock helpers -----------------------------------
def test_pid_lock_helpers():
    import os
    assert daemon._pid_alive(os.getpid()) is True
    assert daemon._pid_alive(2_000_000_000) is False     # implausible pid
    assert daemon._pid_alive(None) is False


# ---- 7: clear sent cache (resend) — resets posted, keeps generation --------
def test_clear_sent_cache(tmp_path, monkeypatch):
    import os
    db = str(tmp_path / "state.db")
    scored = str(tmp_path / "scored.json")
    store = Store(db)
    store.save_generation("S-KEEP", "Nice title", None, None, None, "anthropic", "m", 100)
    store.mark_posted("S-KEEP", 100)
    assert store.is_posted("S-KEEP") is True
    store.close()
    with open(scored, "w") as f:
        f.write('["S-KEEP"]')                            # legacy seed that would restore posted

    monkeypatch.setenv("CUEBENCH_STATE_DB", db)
    monkeypatch.setenv("CUEBENCH_SCORED_FILE", scored)
    res = daemon.clear_sent_cache()

    assert res["cleared"] == 1
    store2 = Store(db)
    assert store2.is_posted("S-KEEP") is False            # will re-POST now
    assert store2.get_generation("S-KEEP")["title"] == "Nice title"  # generation kept (no re-bill)
    store2.close()
    assert not os.path.exists(scored)                     # legacy seed sidelined...
    assert os.path.exists(scored + ".bak")               # ...to a backup (not deleted)


def test_clear_sent_cache_wipe_generation(tmp_path, monkeypatch):
    db = str(tmp_path / "state.db")
    store = Store(db)
    store.save_generation("S-G", "t", None, None, None, "anthropic", "m", 10)
    store.mark_posted("S-G", 10)
    store.close()
    monkeypatch.setenv("CUEBENCH_STATE_DB", db)
    monkeypatch.setenv("CUEBENCH_SCORED_FILE", str(tmp_path / "none.json"))
    res = daemon.clear_sent_cache(wipe_generation=True)
    assert res["generation_wiped"] is True
    store2 = Store(db)
    assert store2.is_posted("S-G") is False
    assert store2.get_generation("S-G") is None           # wiped -> re-generates (with trace)
    store2.close()
