#!/usr/bin/env python3
"""
test_crash_safety.py — crash-safety of the POST / dedup / generation-cache flow.

Simulates process death / power-off at the dangerous points and asserts:
  * a crash BETWEEN server-OK and the local sent-flag commit re-POSTs on restart
    (at-least-once), and the sessionId-keyed UPSERT collapses that into ONE record
    (no duplicate) — while a naive insert WOULD duplicate;
  * a COMMITTED sent-flag prevents any re-POST (at-most-once, no server reliance);
  * the sent-flag is durable across close/reopen (proxy for power-off, since
    synchronous=FULL fsyncs on commit);
  * generation cached before a crashed POST is reused on restart with ZERO new BYOK calls.

No model and no network: a FakeScorer returns fixed vectors and post_payload is
monkeypatched to a fake server. Run:  python3 -m pytest test_crash_safety.py -q
"""
from __future__ import annotations
import json
import pytest

import cuebench_agent as agent
from cuebench_store import Store
from cuebench_gen import Generator

SESSION_UUID = "crashsess-1111-2222-3333-444455556666"
SID = agent.make_sid(SESSION_UUID)


# ---- doubles ---------------------------------------------------------------
class FakeScorer:
    loaded_from = "fake"
    def score(self, digest):
        return {"delegation": 72, "description": 72, "discernment": 74, "diligence": 80}


class FakeServer:
    """Records every accepted POST two ways so a test can compare outcomes:
      inserts -> naive append (NO dedup): its length = number of stored rows if the
                 endpoint INSERTs. >1 for one sid means a DUPLICATE.
      by_sid  -> UPSERT keyed on sessionId: its length = stored rows if the endpoint
                 UPSERTs. Always 1 per sid = SAFE."""
    def __init__(self):
        self.inserts = []
        self.by_sid = {}
    def record(self, payload):
        sid = payload["sessionId"]
        self.inserts.append(sid)
        self.by_sid[sid] = payload


def make_fake_post(server):
    def _post(payload):
        server.record(payload)
        return True, "200 OK (fake server)"
    return _post


class CountingGen(Generator):
    """Counts BYOK calls; returns canned NEUTRAL output (no network)."""
    def _complete(self, system, user, max_tokens=600):
        if not self.enabled:
            return None
        self.calls += 1
        if "extremely short title" in system:        # content_title (from the first prompt)
            return "Steady, well-verified work"
        if "strengths" in system:
            return json.dumps({"strengths": [{"axis": "diligence", "point": "Ran checks before committing."}],
                               "improvements": [{"axis": "description", "point": "State acceptance criteria up front."}]})
        return None


# ---- fixtures --------------------------------------------------------------
@pytest.fixture
def transcript(tmp_path):
    """A minimal but valid session: 1 human prompt + 12 tool calls (support 13 >= gate)."""
    p = tmp_path / "session.jsonl"
    tool_blocks = [{"type": "tool_use", "name": "Bash", "input": {"command": f"echo step{i}"}}
                   for i in range(12)]
    lines = [
        json.dumps({"type": "user", "sessionId": SESSION_UUID, "origin": {"kind": "human"},
                    "timestamp": "2026-01-01T00:00:00.000Z",
                    "message": {"role": "user",
                                "content": "please implement the described feature with enough words here"}}),
        json.dumps({"type": "assistant", "sessionId": SESSION_UUID,
                    "timestamp": "2026-01-01T00:05:00.000Z",
                    "message": {"role": "assistant", "model": "claude-opus-4-8",
                                "usage": {"output_tokens": 50}, "content": tool_blocks}}),
    ]
    p.write_text("\n".join(lines))
    return str(p)


# ---- 1 & 2: crash between server-OK and flag-commit ------------------------
def test_crash_between_post_success_and_flag_commit_no_duplicate_with_upsert(
        tmp_path, monkeypatch, transcript):
    db = str(tmp_path / "state.db")
    server = FakeServer()
    monkeypatch.setattr(agent, "post_payload", make_fake_post(server))
    scorer, gen = FakeScorer(), Generator(key="")          # generation OFF -> isolate POST dedup

    # RUN 1 — server accepts, then the process dies BEFORE the flag commits.
    store1 = Store(db)
    def boom(*a, **k):
        raise RuntimeError("power off between server-OK and commit")
    monkeypatch.setattr(store1, "mark_posted", boom)
    with pytest.raises(RuntimeError):
        agent.process_one(transcript, scorer, store1, gen)
    store1.close()
    assert server.inserts == [SID]                          # server DID accept the first POST

    # RESTART — fresh handle on the same db file; the flag was never committed.
    store2 = Store(db)
    assert store2.is_posted(SID) is False                   # => restart will re-POST (at-least-once)
    status, sid = agent.process_one(transcript, scorer, store2, gen)
    assert (status, sid) == ("posted", SID)
    assert store2.is_posted(SID) is True                    # now durably committed
    store2.close()

    # OUTCOME: the client re-POSTed (2 sends). WITHOUT upsert that is a duplicate; WITH
    # upsert on sessionId the dashboard holds exactly one record => harmless.
    assert server.inserts == [SID, SID]                     # 2 sends (would duplicate if endpoint INSERTs)
    assert len(server.by_sid) == 1                          # UPSERT on sessionId => no duplicate (SAFE)


def test_committed_flag_prevents_resend(tmp_path, monkeypatch, transcript):
    db = str(tmp_path / "state.db")
    server = FakeServer()
    monkeypatch.setattr(agent, "post_payload", make_fake_post(server))
    scorer, gen = FakeScorer(), Generator(key="")

    store1 = Store(db)
    assert agent.process_one(transcript, scorer, store1, gen)[0] == "posted"
    store1.close()
    assert server.inserts == [SID]

    # RESTART — flag is committed, so the session is skipped (no second POST at all).
    store2 = Store(db)
    assert store2.is_posted(SID) is True
    status, _ = agent.process_one(transcript, scorer, store2, gen)
    assert status == "dup"
    store2.close()
    assert server.inserts == [SID]                          # at-most-once: never re-sent


# ---- 4: durability across reopen (power-off proxy) -------------------------
def test_sent_flag_durable_across_reopen(tmp_path):
    db = str(tmp_path / "state.db")
    store1 = Store(db)
    store1.mark_posted(SID, 100)                            # conn.commit() with synchronous=FULL -> fsynced
    store1.close()                                          # simulate process exit
    store2 = Store(db)                                      # simulate restart
    assert store2.is_posted(SID) is True
    store2.close()


# ---- 5: generation cached before a crashed POST is reused, no re-BYOK ------
def test_generation_cached_before_crash_is_reused_no_rebyok(tmp_path, monkeypatch, transcript):
    db = str(tmp_path / "state.db")
    scorer = FakeScorer()

    # RUN 1 — generation succeeds (and commits), THEN the POST dies.
    def crash_post(payload):
        raise RuntimeError("crash during POST, after generation was saved")
    monkeypatch.setattr(agent, "post_payload", crash_post)
    gen1 = CountingGen(key="sk-ant-FAKE", provider="anthropic", model="claude-haiku-4-5")
    store1 = Store(db)
    with pytest.raises(RuntimeError):
        agent.process_one(transcript, scorer, store1, gen1)
    assert gen1.calls == 1                                  # insights is the ONLY BYOK call now
                                                            # (the title is generated locally, free)
    assert store1.get_generation(SID) is not None          # ...and committed BEFORE the POST
    assert store1.is_posted(SID) is False                  # POST never succeeded
    store1.close()

    # RESTART — fresh gen; cache must be reused with ZERO new BYOK calls, then POST.
    server = FakeServer()
    monkeypatch.setattr(agent, "post_payload", make_fake_post(server))
    gen2 = CountingGen(key="sk-ant-FAKE", provider="anthropic", model="claude-haiku-4-5")
    store2 = Store(db)
    status, _ = agent.process_one(transcript, scorer, store2, gen2)
    assert status == "posted"
    assert gen2.calls == 0                                  # cached insights survived: NO re-pay
    # Title is now LOCAL: a short keyphrase-extracted title (verb + salient phrase), NOT a BYOK
    # call and NOT the verbatim prompt. Present even on the cache-reuse path. The prompt
    # "please implement the described feature ..." -> "Implement described feature".
    assert server.by_sid[SID]["title"] == "Implement described feature"
    assert len(server.by_sid[SID]["title"].split()) <= 5     # always short
    assert server.by_sid[SID]["taskType"] == "feature"
    store2.close()
