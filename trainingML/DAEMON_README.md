# CueBench Daemon — background lifecycle wrapper

This packages the existing CueBench agent as a background service that watches
`~/.claude/projects/` and **scores + (optionally) POSTs** sessions automatically as they
finish — no manual runs. It is a **wrapper around the existing engine**; the
scoring / extraction / dedup / generation logic is unchanged.

> **It is DRY-RUN by default** — it scores and logs the payload but does **not** POST.
> Flip to live only when you're ready: `install --live`.

---

## TL;DR — the exact commands

```bash
cd "/Users/dillonmehta/training ML"
PY=/opt/anaconda3/bin/python        # the interpreter that has torch/transformers

# Install + start (auto-starts on login, restarts on crash). DRY-RUN by default.
$PY cuebench_daemon.py install

# Is it running?  (clear RUNNING ✓ / STOPPED / NOT INSTALLED indicator)
$PY cuebench_daemon.py status

# See the logs (add -f to follow)
$PY cuebench_daemon.py logs
$PY cuebench_daemon.py logs -f

# Stop now (instant kill switch; no respawn)        / Start again
$PY cuebench_daemon.py stop
$PY cuebench_daemon.py start

# Fully remove the launchd job + plist (one clean command)
$PY cuebench_daemon.py uninstall
```

When you're ready to POST for real (see "Going live" below):

```bash
$PY cuebench_daemon.py uninstall          # if currently installed in dry-run
$PY cuebench_daemon.py install --live
```

---

## What it is (and what it deliberately is NOT)

| | |
|---|---|
| **New files** | `cuebench_daemon.py` (the wrapper), `test_daemon_lifecycle.py`, `demo_daemon_e2e.py`, this README |
| **One surgical edit** | `cuebench_agent.py`: the watcher loop body was extracted into `run_watch_loop(...)`; `watch()` now calls it. **Behaviour-preserving** — see "Existing flows unchanged" below. |
| **Reused as-is** | `cuebench_store.py` (crash-safe SQLite dedup), `cuebench_gen.py` (BYOK generation), `model_infer.py`, `cuebench_signals.py` — **not touched**. |
| **NOT done** | No new scoring/extraction/dedup logic. No reimplemented watcher loop. No new entry-point behaviour. |

The daemon runs the **exact same** `run_watch_loop` the on-demand `python cuebench_agent.py`
watcher runs (same poll interval, same stable-size detection, same dedup, same retry-later),
only with a `dry_run` flag and a liveness-heartbeat hook added.

---

## The 7 guards — how each is handled (and verified)

### 1. Doesn't destabilize the existing on-demand flow ✅
The only change to `cuebench_agent.py` is mechanical: the `while True:` loop inside `watch()`
was moved verbatim into a new `run_watch_loop(scorer, store, gen, *, dry_run=False, on_cycle=None)`.
`watch()` keeps its exact API-key check, banner, and always-POST behaviour and simply calls
`run_watch_loop(..., dry_run=False)`. With the default args the loop is byte-for-byte the
original.
- `--once`, `--once --dry-run`, and `python cuebench_agent.py` (`--watch`) are unchanged.
- **Verified:** `test_watch_entrypoint_still_requires_key` asserts `watch()` still `sys.exit(2)`
  without a key; the 4 original `test_crash_safety.py` tests still pass; `--once --dry-run` on a
  real transcript prints the same payload + status line (`POST=OK (dry-run, not POSTed)`).

### 2. Doesn't run away with resources ✅
It uses the existing poll loop: scan → `time.sleep(POLL_INTERVAL=20s)`. No busy-loop. The plist
sets `ProcessType=Background` (low-priority scheduling) and `ThrottleInterval=30` (caps any
crash-loop). A single-instance PID lock prevents a manual `run` from loading a **second** model
next to the launchd one (a real risk on a RAM-constrained Mac).
- **Measured (live daemon, real model, idle in its poll loop):**
  - **Idle CPU: ~0.13%** over a 15 s window (**0.02 CPU-seconds** used while idle).
  - **Resident memory (RSS): ~780 MB** — dominated by the resident PyTorch/RoBERTa model
    (loaded once, scores many; this is the engine's load-once design, not the wrapper).
  - *Note on RAM:* idle CPU is effectively zero. The ~780 MB is the cost of keeping the model
    hot for sub-second scoring; under memory pressure macOS compresses/pages its idle pages.
    If you ever want zero idle RSS you'd trade latency for a load-per-session model — that would
    require changing the engine, so it's intentionally **not** done here.

### 3. A crash doesn't kill it silently ✅
launchd `KeepAlive = {SuccessfulExit: false}` restarts it on any abnormal exit, **and** every
start logs a timestamped `[daemon] starting …` line, so a restart leaves a trail. `status` shows
`last exit` (non-zero is flagged "← check err log") and a **heartbeat** age — so a *wedged* (not
crashed) process is visible too (`RUNNING ⚠ heartbeat STALE`).
- **Verified:** `kill -9` of the running daemon → launchd relaunched it with a new PID, `status`
  showed `RUNNING ✓` + `last exit 9 ← non-zero: check err log`, and the log had two `starting`
  lines.

### 4. Doesn't double-POST or corrupt scores on restart ✅
It reuses `cuebench_store.py` exactly, pointed at the **same** `cuebench_state.db` the on-demand
flow uses. Dedup keys on the stable `sessionId`; a successfully-POSTed session is committed with
`synchronous=FULL` and skipped on restart; the POST also carries an `Idempotency-Key`.
- **Verified (real daemon process):** start → new session → POSTed once → SIGTERM → restart
  against the same DB → log shows `[dup] S-… already POSTed and unchanged; skipping` → **still
  exactly one POST**. Also covered by `test_post_once_then_no_repost_on_restart`.

### 5. Doesn't require a network or a key to function ✅
- **No BYOK key →** generation is off (`gen off (no CUEBENCH_BYOK_KEY)`); scores still computed.
- **No dashboard key / no network (live mode) →** the POST fails, the session is **not** marked
  sent (so it's queued), and the existing retry-later path re-attempts it on a later poll. It
  **never crashes and never drops a session.** (The daemon calls `run_watch_loop` directly, so it
  does **not** hard-exit on a missing key the way the standalone `watch()` does.)
- **Dry-run (default) needs no keys at all.**
- **Verified:** `test_no_key_queues_then_posts_when_key_arrives` — no key → 0 POSTs, not marked
  sent; key arrives → POSTs exactly once, never lost.

### 6. Easy to turn off ✅
`stop` does `launchctl bootout` — an **instant** kill switch with no respawn (KeepAlive only
applies while loaded). `uninstall` boots it out **and** deletes the plist in one command. Both
verified to leave **no process** and no launchd job.

### 7. No destructive writes to your data ✅
The engine only **reads** session JSONLs (`open(path, "r")`) and runs **read-only** `git log`
(`cuebench_signals.git_signals`). The daemon writes only to its own files: the state DB, the logs
in `~/Library/Logs/CueBench/`, and control files in `~/.cuebench/daemon/`. The daemon banner even
prints `watching … (READ-ONLY)`. `uninstall` leaves your state DB and logs in place.

---

## Existing flows unchanged — confirmation

```
$ python -m pytest test_crash_safety.py -q          → 4 passed   (the original engine tests)
$ python cuebench_agent.py --once <real>.jsonl --dry-run
    [gen] off (no CUEBENCH_BYOK_KEY)
    [load] model from cuebench_model ...
    { … full payload … }
      sid=S-051D051D … POST=OK (dry-run, not POSTed)   ← identical to before
$ python cuebench_agent.py            (no key)       → [fatal] CUEBENCH_API_KEY not set … exit 2
```

The diff to `cuebench_agent.py` is **only** the loop extraction; no observable behaviour changed
for `--once` / `--dry-run` / `--watch`.

---

## The lifecycle/dedup test

`test_daemon_lifecycle.py` (fast — uses a fake scorer + a real localhost mock dashboard, no model
load):

```
$ /opt/anaconda3/bin/python -m pytest test_crash_safety.py test_daemon_lifecycle.py -q
10 passed
```

It proves: POST-once → no re-POST on restart (guard 4); dry-run never POSTs; no-key → queue then
post-once (guard 5); `watch()` still exits 2 without a key (guard 1); env-file loader + PID lock.

### Live end-to-end demo (real process, real model, localhost POST)

```
$ /opt/anaconda3/bin/python demo_daemon_e2e.py
[1] Starting daemon (LIVE → localhost) …
    daemon up (pid …); model loaded; watching.
    POSTs so far: []  (none expected — no session yet)
[2] Dropping a NEW session transcript into the watched dir …
    POSTs after session appeared: ['S-DEMOSESS']
    ✓ scored and POSTed exactly once  (S-DEMOSESS)
[3] Measuring idle footprint …
    resident memory (RSS): 779.6 MB
    idle CPU over 15s window: 0.13% (0.02 CPU-s used while idle)
[4] Stopping the daemon (SIGTERM) and RESTARTING against the same state DB …
    POSTs after restart: ['S-DEMOSESS']
RESULT ✓  Exactly ONE POST across start → score → restart. No duplicate.
```

---

## Going live (when you're ready to hit the real dashboard)

launchd agents do **not** inherit your shell environment, so put secrets/config in a private
env file the daemon reads at startup (never hardcoded, never written into the plist):

```bash
mkdir -p ~/.cuebench/daemon
cat > ~/.cuebench/daemon/daemon.env <<'EOF'
CUEBENCH_API_KEY=...           # required for live POST
CUEBENCH_EMPLOYEE_ID=e1
# CUEBENCH_BYOK_KEY=...         # optional: enables neutral title/insights generation
EOF
chmod 600 ~/.cuebench/daemon/daemon.env

# Reinstall in live mode
/opt/anaconda3/bin/python cuebench_daemon.py uninstall
/opt/anaconda3/bin/python cuebench_daemon.py install --live
```

`status` will then show `mode  LIVE (POSTing)`. If you ever want to bail, `stop` is the instant
kill switch and `uninstall` removes it entirely.

---

## Menu bar app (`CueBench.app`)

A native macOS **menu bar** app (top-right of the screen, next to Wi-Fi/battery) — built with
PyObjC/AppKit, **no extra dependencies**. It is a lightweight control surface: it does **not**
load the model or score anything (the headless daemon does that and keeps running even if you
quit the app). Click the gauge icon to get:

- **Status** — Running / Stopped / Not installed, dry-run vs live, heartbeat age.
- **Most recent scored session(s)** — sid · score · quality (parsed from the daemon log).
- **Start / Stop / Switch mode / Install / Uninstall / Open logs / Launch-at-login / Quit.**
- **⚙︎ Settings…** — edit Employee ID, the dashboard (forward) URL + API key, and the BYOK
  key/model. Saved to `~/.cuebench/daemon/daemon.env` (chmod 600) and applied on the next
  start (saving restarts the daemon if it's running). Secrets are never put in any plist.

```bash
# Build the double-clickable app (uses the interpreter you build it with):
/opt/anaconda3/bin/python cuebench_menubar.py --build-app
open ./CueBench.app                 # launch it — gauge icon appears in the menu bar
# (or just run it without bundling:)
/opt/anaconda3/bin/python cuebench_menubar.py
```

To start the **icon** at login, use the app's "Launch CueBench at login" menu item (a separate
GUI LaunchAgent `dev.cuebench.menubar` — distinct from the scoring daemon's own job). The
scoring daemon's login auto-start is governed by `install` as described above. To stop the app,
use its **Quit** menu item; to remove it from /Applications just delete `CueBench.app`.

---

## Files & locations

| Path | What |
|---|---|
| `cuebench_daemon.py` | the wrapper + CLI (run/install/uninstall/start/stop/restart/status/logs) |
| `cuebench_menubar.py` / `CueBench.app` | the menu bar app + double-clickable bundle |
| `~/Library/LaunchAgents/dev.cuebench.agent.plist` | the scoring daemon's launchd job (written by `install`) |
| `~/Library/LaunchAgents/dev.cuebench.menubar.plist` | the menu bar app's login item (optional) |
| `cuebench_state.db` (project dir) | crash-safe dedup DB — **shared** with the on-demand flow |
| `~/Library/Logs/CueBench/cuebench-daemon.{out,err}.log` | stdout / stderr logs |
| `~/.cuebench/daemon/{daemon.pid,status.json,daemon.env}` | PID lock, heartbeat, optional env |

## Other platforms

Only macOS/launchd is implemented. The `ServiceManager` abstraction in `cuebench_daemon.py`
(`get_service_manager()`) documents exactly where a Linux `systemd --user` unit or a Windows
Task Scheduler / NSSM backend slots in — the four lifecycle verbs map cleanly; only the OS calls
differ.
```
