#!/usr/bin/env python3
"""
demo_daemon_e2e.py — live end-to-end proof of the background daemon.

Runs the REAL daemon process (cuebench_daemon.py run) with the REAL trained model against a
throwaway ~/.claude/projects dir and a localhost mock dashboard. Demonstrates, on real code:

  1. start the daemon → it idles (near-zero CPU)
  2. a NEW session file appears → it is scored + POSTed exactly ONCE
  3. measure idle CPU% and resident memory while it sits there
  4. stop the daemon, then RESTART it against the SAME dedup DB
  5. confirm it does NOT re-POST the already-sent session

Everything is local: POSTs go to 127.0.0.1, nothing touches the real dashboard, and a throwaway
state DB is used so your real cuebench_state.db is untouched.

  python3 demo_daemon_e2e.py
"""
from __future__ import annotations
import json, os, shutil, signal, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
SESSION_UUID = "demosess-1234-5678-9abc-def012345678"
SID = "S-" + SESSION_UUID.replace("-", "")[:8].upper()


# ---- localhost mock dashboard (counts POSTs) -------------------------------
class Mock:
    def __init__(self):
        self.sends, self.lock = [], threading.Lock()

    def start(self):
        outer = self
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                with outer.lock:
                    outer.sends.append(body.get("sessionId"))
                self.send_response(200); self.end_headers(); self.wfile.write(b'{"ok":true}')
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    def count(self):
        with self.lock:
            return list(self.sends)


# ---- helpers ----------------------------------------------------------------
def write_session(projects_dir):
    d = os.path.join(projects_dir, "proj-demo")
    os.makedirs(d, exist_ok=True)
    tools = [{"type": "tool_use", "name": "Bash", "input": {"command": f"echo step{i}"}}
             for i in range(14)]
    edits = [{"type": "tool_use", "name": "Edit",
              "input": {"file_path": "x.py", "old_string": "a", "new_string": "b\nc\nd"}}]
    lines = [
        json.dumps({"type": "user", "sessionId": SESSION_UUID, "origin": {"kind": "human"},
                    "timestamp": "2026-01-01T00:00:00.000Z",
                    "message": {"role": "user",
                                "content": "implement the described feature with clear acceptance criteria and tests"}}),
        json.dumps({"type": "assistant", "sessionId": SESSION_UUID,
                    "timestamp": "2026-01-01T00:08:00.000Z",
                    "message": {"role": "assistant", "model": "claude-opus-4-8",
                                "usage": {"input_tokens": 5000, "output_tokens": 800},
                                "content": tools + edits}}),
    ]
    path = os.path.join(d, f"{SESSION_UUID}.jsonl")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def cpu_seconds(pid):
    """Total CPU seconds the process has used (utime+stime), via ps cputime."""
    out = subprocess.run(["ps", "-p", str(pid), "-o", "cputime="],
                         capture_output=True, text=True).stdout.strip()
    if not out:
        return None
    parts = out.replace("-", ":").split(":")
    parts = [float(p) for p in parts]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def rss_mb(pid):
    out = subprocess.run(["ps", "-p", str(pid), "-o", "rss="],
                         capture_output=True, text=True).stdout.strip()
    return round(int(out) / 1024, 1) if out else None


def wait_until(fn, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fn():
            return True
        time.sleep(0.5)
    print(f"  !! timed out waiting for: {what}")
    return False


def tail(path, n=400):
    try:
        return "".join(open(path).readlines()[-n:])
    except FileNotFoundError:
        return "(no output)"


# ---- the demo ---------------------------------------------------------------
def main():
    tmp = tempfile.mkdtemp(prefix="cuebench-demo-")
    projects = os.path.join(tmp, "projects")
    os.makedirs(projects)
    state_db = os.path.join(tmp, "demo_state.db")
    out_log = os.path.join(tmp, "daemon.out.log")
    err_log = os.path.join(tmp, "daemon.err.log")

    mock = Mock().start()
    env = {
        **os.environ,
        "CUEBENCH_API_URL": f"http://127.0.0.1:{mock.port}/api/session",
        "CUEBENCH_API_KEY": "demo-localhost-key",
        "CUEBENCH_PROJECTS_DIR": projects,
        "CUEBENCH_STATE_DB": state_db,
        "CUEBENCH_MODEL_DIR": os.path.join(APP_DIR, "cuebench_model"),
        "CUEBENCH_DAEMON_LIVE": "1",          # POST for real — but only to localhost
        "CUEBENCH_POLL_INTERVAL": "2",
        "CUEBENCH_STABLE_SECONDS": "2",
        "CUEBENCH_BYOK_KEY": "",              # generation off (no spend in the demo)
    }

    def launch():
        of, ef = open(out_log, "a"), open(err_log, "a")
        return subprocess.Popen([PYTHON, os.path.join(APP_DIR, "cuebench_daemon.py"), "run"],
                                env=env, stdout=of, stderr=ef, cwd=APP_DIR)

    print("=" * 78)
    print("CueBench daemon — live end-to-end demo (real process, real model, localhost POST)")
    print("=" * 78)
    print(f"  projects dir : {projects}")
    print(f"  state DB     : {state_db}   (throwaway)")
    print(f"  mock POST URL: http://127.0.0.1:{mock.port}/api/session")
    print()

    # ---- 1) start ----
    print("[1] Starting daemon (LIVE → localhost) …")
    proc = launch()
    ok = wait_until(lambda: "entering watch loop" in tail(out_log), 120,
                    "model load + watch loop")
    if not ok:
        print(tail(err_log)); proc.terminate(); return 1
    print(f"    daemon up (pid {proc.pid}); model loaded; watching.")
    print(f"    POSTs so far: {mock.count()}  (none expected — no session yet)")
    print()

    # ---- 2) a new session appears ----
    print("[2] Dropping a NEW session transcript into the watched dir …")
    write_session(projects)
    ok = wait_until(lambda: mock.count() == [SID], 60, "exactly one POST")
    print(f"    POSTs after session appeared: {mock.count()}")
    assert mock.count() == [SID], f"expected one POST of {SID}, got {mock.count()}"
    print(f"    ✓ scored and POSTed exactly once  ({SID})")
    print()

    # ---- 3) idle footprint ----
    print("[3] Measuring idle footprint (daemon sitting in its poll loop) …")
    rss = rss_mb(proc.pid)
    c0 = cpu_seconds(proc.pid); w0 = time.time()
    time.sleep(15)
    c1 = cpu_seconds(proc.pid); w1 = time.time()
    idle_pct = 100.0 * (c1 - c0) / (w1 - w0) if (c1 is not None and c0 is not None) else None
    print(f"    resident memory (RSS): {rss} MB")
    print(f"    idle CPU over {w1 - w0:.0f}s window: {idle_pct:.2f}% "
          f"({c1 - c0:.2f} CPU-s used while idle)")
    print()

    # ---- 4) restart against the SAME dedup DB ----
    print("[4] Stopping the daemon (SIGTERM) and RESTARTING against the same state DB …")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
    print(f"    stopped (exit {proc.returncode}).")
    proc = launch()
    ok = wait_until(lambda: "entering watch loop" in tail(out_log).split("shutting down")[-1]
                    or tail(out_log).count("entering watch loop") >= 2, 120, "restart")
    # give it a few poll cycles to (not) re-POST the existing session
    time.sleep(8)
    print(f"    POSTs after restart: {mock.count()}")
    print()

    # ---- 5) verdict ----
    print("=" * 78)
    if mock.count() == [SID]:
        print("RESULT ✓  Exactly ONE POST across start → score → restart. No duplicate. No re-score-resend.")
        rc = 0
    else:
        print(f"RESULT ✗  Expected [{SID}], got {mock.count()}")
        rc = 1
    print("=" * 78)

    # cleanup
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    mock.srv.shutdown()

    print("\n----- daemon stdout (captured) -----")
    print(tail(out_log))
    err = tail(err_log).strip()
    if err:
        print("----- daemon stderr (captured) -----")
        print(err)

    shutil.rmtree(tmp, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
