#!/usr/bin/env python3
"""
cuebench_store.py — local generation cache + POST-sent dedup (SQLite)
=====================================================================
Two SEPARATE correctness guarantees, both keyed on the stable sessionId (never the
filename or run count):

1. Generation cache — once title/insights/trace are generated for a session, persist them.
   Re-runs reuse the cached text and make ZERO BYOK calls. Rationale: regenerating burns the
   user's BYOK quota (their money) and yields different text each run; a session's generated
   artifacts should be stable and paid-for-once.

2. POST-sent record — once a session has been successfully POSTed and accepted, never POST it
   again. Rationale: re-POSTing a re-scored session corrupts the dashboard's rolling average
   (a duplicate-POST bug previously dragged an employee 85 -> 75).

The two are SEPARATE flags: a session can be generated-but-not-sent (POST failed) and must
then be sent later WITHOUT regenerating. A failed POST must not mark the session sent (so it
retries) and must not trigger regeneration (the cached generation is reused).

Append handling (invalidation): identity is the sessionId. If a session file grows (more
turns appended in a later run) we treat it as the SAME session and re-score + re-POST it as
an UPDATE — never a second insert. The dashboard endpoint must therefore UPSERT on sessionId.
We persist n_lines so the agent can tell "grew" (-> update) from "unchanged" (-> dedup skip).
Cached generation is reused across appends (paid-for-once); use regenerate=True to override.
"""
from __future__ import annotations
import json, os, sqlite3, time

DEFAULT_DB = os.environ.get("CUEBENCH_STATE_DB", "cuebench_state.db")


class Store:
    def __init__(self, path: str = DEFAULT_DB):
        self.path = path
        self.conn = sqlite3.connect(path)
        # Durability: a COMMIT must survive a power-off. Rollback-journal mode with
        # synchronous=FULL fsyncs on every commit, so once mark_posted()/save_generation()
        # call conn.commit() and it returns, the row is on disk. We deliberately do NOT use
        # WAL: WAL+synchronous=NORMAL can drop the last commit(s) on power loss (only durable
        # at checkpoint), which is exactly the lost-sent-flag we must avoid. Set explicitly so
        # we don't depend on the compiled-in default.
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id    TEXT PRIMARY KEY,
                title         TEXT,
                insights      TEXT,            -- JSON or NULL
                trace         TEXT,            -- JSON or NULL
                specificity   INTEGER,         -- 0-100 prompt-specificity (OpenAI embeddings) or NULL
                gen_provider  TEXT,
                gen_model     TEXT,
                generated_at  REAL,            -- NULL until generation succeeds
                n_lines       INTEGER,         -- transcript line count at last process
                posted        INTEGER DEFAULT 0,
                posted_at     REAL
            )""")
        self.conn.commit()
        # additive migration for dbs created before `specificity` existed
        try:
            self.conn.execute("ALTER TABLE sessions ADD COLUMN specificity INTEGER")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # column already present

    # --- generation cache (guarantee 1) --------------------------------------------
    def get_generation(self, sid: str) -> dict | None:
        """Cached generation for a session, or None if never generated. A row that exists
        only because the session was POSTed-without-generation has generated_at IS NULL and
        returns None here (so enabling a key later still generates)."""
        row = self.conn.execute(
            "SELECT title, insights, trace, specificity, gen_provider, gen_model, n_lines "
            "FROM sessions WHERE session_id=? AND generated_at IS NOT NULL", (sid,)).fetchone()
        if not row:
            return None
        title, insights, trace, specificity, prov, model, n_lines = row
        return {
            "title": title,
            "insights": json.loads(insights) if insights else None,
            "trace": json.loads(trace) if trace else None,
            "specificity": specificity,
            "provider": prov, "model": model, "n_lines": n_lines,
        }

    def save_generation(self, sid: str, title, insights, trace, specificity, provider, model, n_lines):
        """Persist generated artifacts (paid-for-once). Preserves any existing posted flag."""
        self.conn.execute("""
            INSERT INTO sessions (session_id, title, insights, trace, specificity,
                                  gen_provider, gen_model, generated_at, n_lines)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(session_id) DO UPDATE SET
                title=excluded.title, insights=excluded.insights, trace=excluded.trace,
                specificity=excluded.specificity,
                gen_provider=excluded.gen_provider, gen_model=excluded.gen_model,
                generated_at=excluded.generated_at, n_lines=excluded.n_lines
            """, (sid, title,
                  json.dumps(insights) if insights is not None else None,
                  json.dumps(trace) if trace is not None else None,
                  specificity, provider, model, time.time(), n_lines))
        self.conn.commit()

    # --- POST-sent record (guarantee 2) --------------------------------------------
    def is_posted(self, sid: str) -> bool:
        row = self.conn.execute("SELECT posted FROM sessions WHERE session_id=?", (sid,)).fetchone()
        return bool(row and row[0])

    def mark_posted(self, sid: str, n_lines: int):
        """Record a successful, accepted POST. Updates n_lines so a later append is detected
        as growth (-> update) rather than a duplicate."""
        self.conn.execute("""
            INSERT INTO sessions (session_id, posted, posted_at, n_lines)
            VALUES (?,1,?,?)
            ON CONFLICT(session_id) DO UPDATE SET
                posted=1, posted_at=excluded.posted_at, n_lines=excluded.n_lines
            """, (sid, time.time(), n_lines))
        self.conn.commit()

    def get_n_lines(self, sid: str) -> int | None:
        row = self.conn.execute("SELECT n_lines FROM sessions WHERE session_id=?", (sid,)).fetchone()
        return row[0] if row and row[0] is not None else None

    # --- one-time migration of the legacy posted-set --------------------------------
    def seed_posted_from_json(self, json_path: str) -> int:
        """Seed the posted record from a legacy scored_sessions.json (a list of sessionIds).
        Only canonical sids (starting 'S-') are migrated; legacy path entries are ignored.
        Idempotent. Returns how many were seeded."""
        try:
            data = json.load(open(json_path))
        except Exception:
            return 0
        n = 0
        for entry in data if isinstance(data, list) else []:
            if isinstance(entry, str) and entry.startswith("S-") and not self.is_posted(entry):
                self.conn.execute(
                    "INSERT INTO sessions (session_id, posted, posted_at) VALUES (?,1,?) "
                    "ON CONFLICT(session_id) DO UPDATE SET posted=1",
                    (entry, time.time()))
                n += 1
        self.conn.commit()
        return n

    def close(self):
        self.conn.close()
