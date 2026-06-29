#!/usr/bin/env python3
"""
cuebench_daemon.py — background lifecycle WRAPPER around the CueBench scoring agent
==================================================================================
This file adds nothing to the scoring/extraction/dedup/generation ENGINE. It is a thin
lifecycle layer that:

  * runs the EXISTING watcher loop (cuebench_agent.run_watch_loop) continuously in the
    background, watching ~/.claude/projects for finished sessions;
  * auto-starts on login and is restarted on crash via macOS launchd (KeepAlive);
  * exposes start / stop / restart / status / logs / install / uninstall commands and a
    clear "is it running" indicator;
  * reuses the EXISTING crash-safe SQLite dedup (cuebench_store) by pointing at the SAME
    state DB the on-demand flow uses — so it never re-scores, re-pays BYOK, or re-POSTs.

It is DRY-RUN BY DEFAULT (scores + logs the payload, never POSTs) so you can watch it work
safely before it touches the real dashboard. Flip to live with `install --live`.

The on-demand flow is untouched: `python cuebench_agent.py --once/--dry-run` and the
`python cuebench_agent.py` watcher behave exactly as before. This wrapper only adds a new
layer that calls the same code.

Commands
--------
  python cuebench_daemon.py install [--live]   # write launchd job + start it (login + crash restart)
  python cuebench_daemon.py status             # is it running? mode, pid, heartbeat, last exit
  python cuebench_daemon.py logs [-n N] [-f]    # tail stdout+stderr logs
  python cuebench_daemon.py start | stop | restart
  python cuebench_daemon.py uninstall          # stop + remove the launchd job entirely
  python cuebench_daemon.py run [--live]       # foreground loop (what launchd executes; also for manual testing)

Secrets: never hardcoded, never written into the plist. Optional per-machine env (the API
key, BYOK key, employee id, …) is read from ~/.cuebench/daemon/daemon.env at startup — this
is required because launchd agents do NOT inherit your shell environment. Dry-run needs no key.

Cross-platform: only macOS/launchd is implemented today. Linux (systemd --user) and Windows
(Task Scheduler / NSSM) slot in as new ServiceManager subclasses — see get_service_manager().
"""
from __future__ import annotations
import argparse
import os
import plistlib
import re
import signal
import subprocess
import sys
import time

# ----------------------------------------------------------------------------
# Paths & identity (all absolute — launchd runs from "/" with a minimal env)
# ----------------------------------------------------------------------------
LABEL        = "dev.cuebench.agent"
APP_DIR      = os.path.dirname(os.path.abspath(__file__))      # the project dir (engine lives here)
HOME         = os.path.expanduser("~")
PYTHON       = os.path.abspath(sys.executable)                 # install with the interpreter you ran
DAEMON_PY    = os.path.abspath(__file__)

CONTROL_DIR  = os.path.join(HOME, ".cuebench", "daemon")       # pid / status / env (our own subdir)
LOG_DIR      = os.path.join(HOME, "Library", "Logs", "CueBench")
PLIST_PATH   = os.path.join(HOME, "Library", "LaunchAgents", f"{LABEL}.plist")

PIDFILE      = os.path.join(CONTROL_DIR, "daemon.pid")
STATUSFILE   = os.path.join(CONTROL_DIR, "status.json")        # heartbeat + live state (JSON)
ENVFILE      = os.path.join(CONTROL_DIR, "daemon.env")         # optional KEY=VALUE secrets/config
OUT_LOG      = os.path.join(LOG_DIR, "cuebench-daemon.out.log")
ERR_LOG      = os.path.join(LOG_DIR, "cuebench-daemon.err.log")

# Engine paths — default to the project dir so the daemon shares the SAME state DB / model
# as the on-demand flow (critical for dedup: a session scored either way must not re-POST).
DEFAULT_STATE_DB    = os.path.join(APP_DIR, "cuebench_state.db")
DEFAULT_MODEL_DIR   = os.path.join(APP_DIR, "cuebench_model")
DEFAULT_SCORED_FILE = os.path.join(APP_DIR, "scored_sessions.json")
DEFAULT_PROJECTS    = os.path.join(HOME, ".claude", "projects")     # Claude Code transcripts
DEFAULT_CODEX       = os.path.join(HOME, ".codex", "sessions")       # Codex CLI rollouts

# launchd agents start with a near-empty PATH; git_signals shells out to `git`, so put git's
# dir (and the interpreter's) on PATH explicitly.
SERVICE_PATH = ":".join([
    os.path.dirname(PYTHON), "/usr/bin", "/bin", "/usr/sbin", "/sbin",
    "/usr/local/bin", "/opt/homebrew/bin",
])


def _truthy(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _ensure_dirs():
    for d in (CONTROL_DIR, LOG_DIR, os.path.dirname(PLIST_PATH)):
        os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------------
# Optional env file (KEY=VALUE) — bridges launchd's missing shell env. NEVER hardcodes a
# key; only loads what the user put in a 0600 file. Existing env wins (setdefault).
# These engine-facing keys are what the menu bar Settings pane edits.
# ----------------------------------------------------------------------------
MANAGED_ENV_KEYS = (
    "CUEBENCH_EMPLOYEE_ID",   # which operator this machine is
    "CUEBENCH_API_URL",       # dashboard "forward" endpoint
    "CUEBENCH_API_KEY",       # dashboard write key (forward) — required for live POST
    "CUEBENCH_BYOK_KEY",      # BYOK key — enables neutral title/insights generation
    "CUEBENCH_BYOK_MODEL",    # BYOK model id (provider inferred)
    "CUEBENCH_TRACE",         # "1" -> also generate the session timeline trace (needs BYOK)
    "CUEBENCH_INSIGHTS_PROMPTS",  # "1" -> prompt-informed insights: feed prompts to coaching (needs BYOK)
)


def parse_env_file(path: str = ENVFILE) -> dict:
    """Parse a KEY=VALUE env file into an ordered dict (tolerates `export `, #comments,
    blank lines, and surrounding quotes). Returns {} if the file is absent/unreadable."""
    out: dict[str, str] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k:
                    out[k] = v
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[daemon] warning: could not read {path}: {e!r}", file=sys.stderr)
    return out


def load_env_file(path: str = ENVFILE) -> int:
    """Populate os.environ from the env file WITHOUT overriding anything already set
    (setdefault semantics). Returns how many keys were newly set."""
    n = 0
    for k, v in parse_env_file(path).items():
        if k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


def save_env_file(updates: dict, path: str = ENVFILE,
                  managed_keys=MANAGED_ENV_KEYS) -> None:
    """Merge `updates` into the env file and write it back at 0600.

    For each managed key: a non-empty value is written; an empty/missing value is REMOVED
    (so clearing the BYOK key in Settings turns generation off). Any non-managed keys already
    in the file are preserved untouched."""
    _ensure_dirs()
    existing = parse_env_file(path)
    for k in managed_keys:
        v = (updates.get(k) or "").strip()
        if v:
            existing[k] = v
        else:
            existing.pop(k, None)
    lines = ["# CueBench daemon settings — written by the menu bar app. Secrets; keep private.",
             "# launchd agents don't inherit your shell env, so the daemon reads keys from here.", ""]
    for k, v in existing.items():
        lines.append(f"{k}={v}")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def clear_sent_cache(*, wipe_generation: bool = False) -> dict:
    """Reset the crash-safe POST-sent dedup so finished sessions are re-POSTed on the next
    scan ("resend"). By default the generation cache (paid-for titles/insights) is KEPT, so
    nothing is re-billed. Also sidelines the legacy scored_sessions.json (renaming it .bak) so
    its one-time seed can't re-mark those sessions as sent on the next start.

    Returns a summary dict. Does NOT delete the DB or touch ~/.claude. Reversible-ish: restore
    the .bak to undo the seed change (the posted flags reset is intentional and not undone)."""
    import sqlite3
    db = os.environ.get("CUEBENCH_STATE_DB") or DEFAULT_STATE_DB
    scored = os.environ.get("CUEBENCH_SCORED_FILE") or DEFAULT_SCORED_FILE
    out = {"db": db, "cleared": 0, "generation_wiped": False, "scored_backup": None}
    try:
        conn = sqlite3.connect(db)
        try:
            out["cleared"] = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE posted=1").fetchone()[0]
            conn.execute("UPDATE sessions SET posted=0, posted_at=NULL")
            if wipe_generation:                      # forces re-generation -> re-pays BYOK
                conn.execute("UPDATE sessions SET generated_at=NULL")
                out["generation_wiped"] = True
            conn.commit()
        finally:
            conn.close()
    except sqlite3.OperationalError:
        pass                                          # no DB / no table yet => nothing to clear
    if os.path.exists(scored):
        bak = scored + ".bak"
        try:
            os.replace(scored, bak)                   # keep a backup; don't lose the record
            out["scored_backup"] = bak
        except OSError:
            pass
    return out


# ----------------------------------------------------------------------------
# PID file (single-instance lock so a manual `run` can't double-load the model alongside
# the launchd one — a real concern on a RAM-constrained machine).
# ----------------------------------------------------------------------------
def _read_pid(path: str = PIDFILE):
    try:
        return int(open(path).read().strip())
    except Exception:
        return None


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, owned by someone else
    except Exception:
        return False


def _write_pid():
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))


def _clear_pid():
    # only remove if it's still ours (don't clobber a newer instance's pidfile)
    if _read_pid() == os.getpid():
        try:
            os.remove(PIDFILE)
        except OSError:
            pass


# ----------------------------------------------------------------------------
# Heartbeat / status (so `status` can tell RUNNING-AND-HEALTHY from RUNNING-BUT-WEDGED)
# ----------------------------------------------------------------------------
def write_status(mode: str, cycles: int, poll: int, projects: str, processed: int):
    import json
    tmp = STATUSFILE + ".tmp"
    data = {
        "pid": os.getpid(), "ts": time.time(), "mode": mode, "cycles": cycles,
        "poll_interval": poll, "projects_dir": projects, "processed_this_run": processed,
    }
    try:
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, STATUSFILE)   # atomic
    except Exception:
        pass


def read_status() -> dict | None:
    import json
    try:
        with open(STATUSFILE) as f:
            return json.load(f)
    except Exception:
        return None


def scan_cursor_once(agent, scorer, store, gen, *, dry_run, state, stable_seconds, db=None):
    """Scan Cursor composers once and score/POST the finished, changed ones. Cursor isn't a file
    source, so finished-ness is detected on the DB's mtime: once Cursor stops writing for
    `stable_seconds`, the (now-settled) composers are read and scored once. Dedup/append/POST are
    handled by agent.process_parsed (keyed on the stable sessionId), like Claude/Codex files.

    state = {"disabled":bool, "mtime":float, "mtime_at":float, "done_mtime":float,
             "handled": {cid: last_updated}}.  `db` overrides the DB path (tests).
    """
    if state.get("disabled"):
        return
    try:
        import cuebench_cursor as cur
    except Exception:
        return
    path = db or cur._db_path()
    now = time.time()

    # TCC/privacy gate: macOS gates the (launchd-spawned) daemon's reads of Cursor's app-data
    # folder behind a prompt that doesn't reliably persist. To avoid nagging every poll, we only
    # OPEN the DB once Cursor has finished writing — its mtime has CHANGED and then held steady
    # for stable_seconds — so a finished session reads ~once, not once per 20s poll. (Grant Full
    # Disk Access to the daemon's Python for silent, persistent access.)
    try:
        mtime = os.stat(path).st_mtime
    except FileNotFoundError:
        return
    except (PermissionError, OSError) as e:
        _disable_cursor(state, e)
        return
    if mtime != state.get("mtime"):               # Cursor just wrote -> (re)start stability clock
        state["mtime"], state["mtime_at"] = mtime, now
        return
    if now - state.get("mtime_at", now) < stable_seconds:   # still settling
        return
    if state.get("done_mtime") == mtime:          # already scored this stable point
        return

    try:
        comps = cur.list_cursor_composers(path)
        if not comps:                             # list swallows errors -> probe to tell denied vs empty
            with open(path, "rb") as fh:
                fh.read(16)
    except FileNotFoundError:
        return
    except (PermissionError, OSError) as e:       # macOS denied access -> stop asking, guide the user
        _disable_cursor(state, e)
        return
    state["done_mtime"] = mtime
    handled = state.setdefault("handled", {})      # cid -> last_updated we scored (append re-opens)
    for comp in comps:
        cid = comp.get("composer_id")
        lu = comp.get("last_updated") or 0
        if not cid or (comp.get("n_bubbles") or 0) <= 0 or handled.get(cid) == lu:
            continue
        print(f"[finished] cursor:{comp.get('name') or cid}", flush=True)
        try:
            parsed = cur.parse_cursor_composer(cid, path)
            status, _ = agent.process_parsed(parsed, scorer, store, gen, dry_run=dry_run,
                                             source=f"cursor:{cid[:8]}")
        except (PermissionError, OSError) as e:
            _disable_cursor(state, e)
            return
        except Exception as e:
            print(f"  [error] cursor {cid}: {e!r}", file=sys.stderr, flush=True)
            continue
        if status in ("posted", "dup", "skip"):
            handled[cid] = lu                     # remember the version scored; growth re-scores


def _disable_cursor(state, err=None):
    """macOS denied the daemon access to Cursor's data: disable Cursor scanning for the rest of
    this run (don't re-prompt every poll) and tell the user exactly how to enable it for good."""
    state["disabled"] = True
    print("[cursor] macOS denied access to Cursor's data; Cursor scanning is OFF for this run"
          + (f" ({err})" if err else "") + ".\n"
          "         To score Cursor sessions, grant Full Disk Access to the daemon's Python:\n"
          "           System Settings > Privacy & Security > Full Disk Access > '+'\n"
          "           (Cmd-Shift-G) /opt/anaconda3/bin/python  -> enable -> "
          "python cuebench_daemon.py restart\n"
          "         Or disable Cursor entirely: launchctl setenv CUEBENCH_CURSOR 0 (then restart).",
          file=sys.stderr, flush=True)


# ============================================================================
# The foreground loop launchd executes (also runnable by hand for testing).
# ============================================================================
def cmd_run(args) -> int:
    _ensure_dirs()
    load_env_file()

    # Resolve engine config to absolute paths BEFORE importing the engine (its module-level
    # config is read at import time). setdefault: anything already in env (plist/daemon.env)
    # wins; otherwise fall back to the project-dir defaults that match the on-demand flow.
    os.environ.setdefault("CUEBENCH_STATE_DB", DEFAULT_STATE_DB)
    os.environ.setdefault("CUEBENCH_MODEL_DIR", DEFAULT_MODEL_DIR)
    os.environ.setdefault("CUEBENCH_SCORED_FILE", DEFAULT_SCORED_FILE)
    os.environ.setdefault("CUEBENCH_PROJECTS_DIR", DEFAULT_PROJECTS)
    os.environ.setdefault("CUEBENCH_CODEX_DIR", DEFAULT_CODEX)

    live = bool(args.live) or _truthy(os.environ.get("CUEBENCH_DAEMON_LIVE"))
    dry_run = not live
    mode = "LIVE (POSTing)" if live else "DRY-RUN (no POST)"

    # Single-instance guard: refuse to start a second copy that would load a second model.
    existing = _read_pid()
    if _pid_alive(existing) and existing != os.getpid():
        print(f"[daemon] already running (pid {existing}); refusing to start a second instance.",
              file=sys.stderr)
        return 0
    _write_pid()
    import atexit
    atexit.register(_clear_pid)

    # Early heartbeat: marks us alive BEFORE the (possibly slow, RAM-pressured) model load so
    # `status` reads healthy during startup instead of a false "wedged" alarm.
    _poll0 = int(os.environ.get("CUEBENCH_POLL_INTERVAL", "20") or "20")
    write_status(mode, 0, _poll0, os.environ["CUEBENCH_PROJECTS_DIR"], processed=0)

    # Clean shutdown: launchd sends SIGTERM on stop/bootout. Log why, then exit 0 so
    # KeepAlive(SuccessfulExit=false) does NOT treat a deliberate stop as a crash.
    def _on_signal(signum, _frame):
        name = signal.Signals(signum).name
        print(f"[daemon] received {name}; shutting down cleanly", file=sys.stderr, flush=True)
        _clear_pid()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    sys.path.insert(0, APP_DIR)   # ensure the engine modules import regardless of cwd
    # Lazy import: keeps the lifecycle commands (status/logs/...) fast and dependency-free.
    import cuebench_agent as agent

    projects = os.environ["CUEBENCH_PROJECTS_DIR"]
    codex = os.environ.get("CUEBENCH_CODEX_DIR", "")
    roots = [projects] + ([codex] if codex and os.path.isdir(codex) else [])
    # Cursor lives in a SQLite KV store (not files); it's scanned separately each cycle.
    cursor_db = os.environ.get("CUEBENCH_CURSOR_DB", "")
    if not cursor_db:
        try:
            import cuebench_cursor as _cc
            cursor_db = _cc.DEFAULT_CURSOR_DB
        except Exception:
            cursor_db = ""
    # Kill-switch: CUEBENCH_CURSOR=0 disables Cursor scanning entirely (e.g. to silence the
    # macOS Full-Disk-Access prompt if the user would rather not grant it).
    cursor_enabled = os.environ.get("CUEBENCH_CURSOR", "1").strip().lower() not in (
        "0", "false", "no", "off")
    cursor_on = bool(cursor_enabled and cursor_db and os.path.exists(cursor_db))
    watching = "; ".join(roots) + ("  + Cursor" if cursor_on else "")
    print(f"[daemon] starting  mode={mode}  pid={os.getpid()}  "
          f"time={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[daemon] watching  {watching}  (READ-ONLY)", flush=True)
    if cursor_on:
        print(f"[daemon] cursor    {cursor_db}", flush=True)
    if codex and not os.path.isdir(codex):
        print(f"[daemon] (codex dir {codex} not present yet — will pick it up if created "
              "after restart)", flush=True)
    print(f"[daemon] state_db  {os.environ['CUEBENCH_STATE_DB']}", flush=True)
    print(f"[daemon] model_dir {os.environ['CUEBENCH_MODEL_DIR']}", flush=True)
    print(f"[daemon] poll={agent.POLL_INTERVAL}s stable={agent.STABLE_SECONDS}s "
          f"employee={agent.EMPLOYEE_ID}", flush=True)

    if live and not agent.API_KEY:
        # Guard #5: never crash on a missing key. Score + queue; the loop's retry-later path
        # re-attempts the POST each poll once a key/network is available.
        print("[daemon] WARNING: live mode but CUEBENCH_API_KEY is not set. Sessions will be "
              "SCORED and QUEUED; the POST is retried each poll until a key is provided.",
              file=sys.stderr, flush=True)

    # Generation off when no BYOK key (guard #5). Store/seed exactly as the on-demand main().
    gen = agent.Generator()
    print(f"[daemon] gen {gen.status()}", flush=True)
    store = agent.Store(os.environ["CUEBENCH_STATE_DB"])
    seeded = store.seed_posted_from_json(os.environ["CUEBENCH_SCORED_FILE"])
    if seeded:
        print(f"[daemon] seeded {seeded} already-POSTed sessionId(s) from legacy file", flush=True)

    model_dir = os.environ["CUEBENCH_MODEL_DIR"]
    if not os.path.isdir(model_dir):
        print(f"[daemon] FATAL: model dir not found: {model_dir}", file=sys.stderr, flush=True)
        return 3
    print(f"[daemon] loading model from {model_dir} …", flush=True)
    try:
        scorer = agent.ModelScorer(model_dir)
    except Exception as e:
        print(f"[daemon] FATAL: model load failed: {e!r}", file=sys.stderr, flush=True)
        return 3
    print(f"[daemon] model loaded (weights={scorer.loaded_from}); entering watch loop", flush=True)

    # Heartbeat closure: bump a cycle counter, write status, and (since Cursor isn't a file
    # source) scan Cursor composers each scan pass.
    state = {"cycles": 0}
    cursor_state = {}

    def _heartbeat():
        state["cycles"] += 1
        write_status(mode, state["cycles"], agent.POLL_INTERVAL, watching, processed=0)
        if cursor_on:
            scan_cursor_once(agent, scorer, store, gen, dry_run=dry_run, state=cursor_state,
                             stable_seconds=agent.STABLE_SECONDS, db=cursor_db)

    write_status(mode, 0, agent.POLL_INTERVAL, watching, processed=0)
    # Reuse the EXACT existing loop (stable-size detection, dedup, retry-later) — only the
    # dry_run flag, heartbeat hook, and multi-root (Claude + Codex) scanning are added.
    agent.run_watch_loop(scorer, store, gen, dry_run=dry_run, on_cycle=_heartbeat, roots=roots)
    return 0   # run_watch_loop loops forever; reached only via SystemExit from the signal handler


# ============================================================================
# Service manager abstraction (macOS now; Linux/Windows later)
# ============================================================================
class ServiceManager:
    """Interface every platform backend implements. Only LaunchdServiceManager exists today."""
    def install(self, *, live: bool): raise NotImplementedError
    def uninstall(self): raise NotImplementedError
    def start(self): raise NotImplementedError
    def stop(self): raise NotImplementedError
    def restart(self): raise NotImplementedError
    def status(self) -> dict: raise NotImplementedError


class LaunchdServiceManager(ServiceManager):
    """macOS launchd (per-user gui domain) backend."""

    def __init__(self):
        self.uid = os.getuid()
        self.domain = f"gui/{self.uid}"
        self.target = f"{self.domain}/{LABEL}"

    # -- low-level helpers -------------------------------------------------
    @staticmethod
    def _launchctl(*args) -> tuple[int, str, str]:
        p = subprocess.run(["launchctl", *args], capture_output=True, text=True)
        return p.returncode, p.stdout.strip(), p.stderr.strip()

    def _plist_dict(self, live: bool) -> dict:
        env = {
            "PATH": SERVICE_PATH,
            "PYTHONUNBUFFERED": "1",                       # flush logs promptly
            "HOME": HOME,
            "CUEBENCH_STATE_DB": DEFAULT_STATE_DB,         # SHARED with on-demand flow (dedup)
            "CUEBENCH_MODEL_DIR": DEFAULT_MODEL_DIR,
            "CUEBENCH_SCORED_FILE": DEFAULT_SCORED_FILE,
            "CUEBENCH_PROJECTS_DIR": DEFAULT_PROJECTS,
            "CUEBENCH_DAEMON_LIVE": "1" if live else "0",  # dry-run by default; visible in the plist
        }
        return {
            "Label": LABEL,
            "ProgramArguments": [PYTHON, DAEMON_PY, "run"],
            "RunAtLoad": True,                              # start on login / load
            # Restart on crash (non-zero/abnormal exit) but NOT on a clean stop (exit 0). A
            # deliberate stop should stay stopped; a crash should come back and leave a log.
            "KeepAlive": {"SuccessfulExit": False},
            "ThrottleInterval": 30,                        # cap crash-loop relaunch rate (guard #2/#3)
            "ProcessType": "Background",                   # low-priority scheduling (idle-friendly)
            "WorkingDirectory": APP_DIR,
            "EnvironmentVariables": env,
            "StandardOutPath": OUT_LOG,
            "StandardErrorPath": ERR_LOG,
        }

    def write_plist(self, live: bool):
        _ensure_dirs()
        with open(PLIST_PATH, "wb") as f:
            plistlib.dump(self._plist_dict(live), f)
        os.chmod(PLIST_PATH, 0o644)

    def _is_loaded(self) -> bool:
        rc, _, _ = self._launchctl("list", LABEL)
        return rc == 0

    # -- lifecycle ---------------------------------------------------------
    def install(self, *, live: bool):
        _ensure_dirs()
        if self._is_loaded():
            self._launchctl("bootout", self.target)        # replace any stale job
        self.write_plist(live)
        rc, out, err = self._launchctl("bootstrap", self.domain, PLIST_PATH)
        if rc != 0 and "already" not in (err + out).lower():
            # Fallback to legacy load for older launchctl
            rc2, _, err2 = self._launchctl("load", "-w", PLIST_PATH)
            if rc2 != 0:
                raise SystemExit(f"launchctl bootstrap failed: {err or out}\n(load fallback: {err2})")
        self._launchctl("enable", self.target)
        print(f"[install] wrote {PLIST_PATH}")
        print(f"[install] mode: {'LIVE (POSTing)' if live else 'DRY-RUN (no POST)'}")
        print(f"[install] launchd job '{LABEL}' bootstrapped (RunAtLoad + KeepAlive-on-crash).")

    def uninstall(self):
        had_plist = os.path.exists(PLIST_PATH)
        if self._is_loaded():
            self._launchctl("bootout", self.target)
        # legacy unload too, harmless if already out
        if had_plist:
            self._launchctl("unload", "-w", PLIST_PATH)
            try:
                os.remove(PLIST_PATH)
            except OSError:
                pass
        # stop any stray foreground instance
        pid = _read_pid()
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        print(f"[uninstall] launchd job removed{' + plist deleted' if had_plist else ''}.")
        print("[uninstall] state DB and logs were left in place (your data; delete manually if desired).")

    def start(self):
        if not os.path.exists(PLIST_PATH):
            raise SystemExit("Not installed. Run:  python cuebench_daemon.py install")
        if self._is_loaded():
            self._launchctl("kickstart", "-k", self.target)
            print(f"[start] kickstarted '{LABEL}'.")
        else:
            rc, out, err = self._launchctl("bootstrap", self.domain, PLIST_PATH)
            if rc != 0 and "already" not in (err + out).lower():
                self._launchctl("load", "-w", PLIST_PATH)
            print(f"[start] bootstrapped '{LABEL}'.")

    def stop(self):
        # bootout = instant, no respawn (KeepAlive only applies while loaded). The kill switch.
        if self._is_loaded():
            self._launchctl("bootout", self.target)
            print(f"[stop] booted out '{LABEL}' (stopped; will not respawn until start/login).")
        else:
            print(f"[stop] '{LABEL}' was not loaded.")
        pid = _read_pid()
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

    def restart(self):
        if self._is_loaded():
            self._launchctl("kickstart", "-k", self.target)
            print(f"[restart] kickstarted '{LABEL}'.")
        else:
            self.start()

    def status(self) -> dict:
        out = {"installed": os.path.exists(PLIST_PATH), "loaded": False,
               "pid": None, "last_exit": None}
        rc, txt, _ = self._launchctl("list", LABEL)
        if rc == 0:
            out["loaded"] = True
            m = re.search(r'"PID"\s*=\s*(\d+)', txt)
            if m:
                out["pid"] = int(m.group(1))
            m = re.search(r'"LastExitStatus"\s*=\s*(-?\d+)', txt)
            if m:
                out["last_exit"] = int(m.group(1))
        return out


def get_service_manager() -> ServiceManager:
    """Pick the platform backend. macOS → launchd (implemented). Others raise with a pointer.

    To add a platform: subclass ServiceManager.
      * Linux:   SystemdUserServiceManager — write ~/.config/systemd/user/cuebench.service
                 ([Service] ExecStart=PYTHON DAEMON_PY run, Restart=on-failure;
                 [Install] WantedBy=default.target), then `systemctl --user enable --now`.
                 `loginctl enable-linger` makes it survive logout/reboot.
      * Windows: TaskSchedulerServiceManager (schtasks /create … /sc ONLOGON) or an NSSM
                 service wrapping `PYTHON DAEMON_PY run`.
    All four lifecycle verbs map cleanly onto each backend; only the OS calls differ.
    """
    if sys.platform == "darwin":
        return LaunchdServiceManager()
    raise SystemExit(
        f"Unsupported platform '{sys.platform}'. Only macOS/launchd is implemented today; "
        "Linux (systemd --user) and Windows (Task Scheduler/NSSM) can be added as new "
        "ServiceManager subclasses (see get_service_manager()).")


# ============================================================================
# Status / logs presentation
# ============================================================================
def _fmt_age(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s ago"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m ago"


def status_dict() -> dict:
    """Structured status for both the CLI and the menu bar app (one source of truth for the
    RUNNING/healthy/stale/stopped interpretation)."""
    svc = get_service_manager()
    st = svc.status()
    hb = read_status()
    now = time.time()
    poll = (hb or {}).get("poll_interval", 20)
    hb_age = (now - hb["ts"]) if hb and hb.get("ts") else None
    fresh = hb_age is not None and hb_age < max(60, poll * 3)
    running = bool(st["loaded"] and st["pid"] and _pid_alive(st["pid"]))
    if not st["installed"]:
        indicator, dot = "Not installed", "○"
    elif running and fresh:
        indicator, dot = "Running", "●"
    elif running and not fresh:
        indicator, dot = "Running (heartbeat stale)", "◐"
    elif st["loaded"]:
        indicator, dot = "Loaded, not running", "◌"
    else:
        indicator, dot = "Stopped", "○"
    return {**st, "running": running, "fresh": fresh, "hb_age": hb_age,
            "mode": (hb or {}).get("mode"), "cycles": (hb or {}).get("cycles"),
            "projects_dir": (hb or {}).get("projects_dir"),
            "indicator": indicator, "dot": dot}


def cmd_status(_args) -> int:
    svc = get_service_manager()
    st = svc.status()
    hb = read_status()
    now = time.time()

    if not st["installed"]:
        print("● CueBench daemon: NOT INSTALLED")
        print("  install with:  python cuebench_daemon.py install")
        return 0

    running = bool(st["loaded"] and st["pid"] and _pid_alive(st["pid"]))
    # Heartbeat freshness: healthy if the loop wrote status within ~3 poll intervals.
    poll = (hb or {}).get("poll_interval", 20)
    hb_age = (now - hb["ts"]) if hb and hb.get("ts") else None
    fresh = hb_age is not None and hb_age < max(60, poll * 3)

    if running and fresh:
        indicator = "RUNNING ✓ (healthy)"
    elif running and not fresh:
        indicator = "RUNNING ⚠  (heartbeat STALE — possibly wedged)"
    elif st["loaded"]:
        indicator = "LOADED but NOT running (stopped/crashed)"
    else:
        indicator = "INSTALLED but STOPPED"

    print(f"● CueBench daemon: {indicator}")
    print(f"  label         {LABEL}")
    print(f"  loaded        {st['loaded']}")
    print(f"  pid           {st['pid'] if st['pid'] else '—'}")
    if st["last_exit"] is not None:
        note = " (clean)" if st["last_exit"] == 0 else "  ← non-zero: check err log"
        print(f"  last exit     {st['last_exit']}{note}")
    if hb:
        print(f"  mode          {hb.get('mode', '?')}")
        print(f"  heartbeat     {_fmt_age(hb_age)}  (cycles={hb.get('cycles', '?')})" if hb_age is not None else "  heartbeat     —")
        print(f"  watching      {hb.get('projects_dir', '?')}")
    print(f"  plist         {PLIST_PATH}")
    print(f"  logs          {OUT_LOG}")
    print(f"                {ERR_LOG}")
    print(f"  tail logs:    python cuebench_daemon.py logs -f")
    return 0


def cmd_logs(args) -> int:
    logs = [p for p in (OUT_LOG, ERR_LOG) if os.path.exists(p)]
    if not logs:
        print(f"(no logs yet at {LOG_DIR})")
        return 0
    cmd = ["tail", "-n", str(args.n)]
    if args.follow:
        cmd.append("-F")
    cmd += logs
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        return 0


def cmd_install(args) -> int:
    get_service_manager().install(live=bool(args.live))
    print()
    cmd_status(args)
    return 0


def cmd_uninstall(_args) -> int:
    get_service_manager().uninstall()
    return 0


def cmd_start(_args) -> int:
    get_service_manager().start()
    return 0


def cmd_stop(_args) -> int:
    get_service_manager().stop()
    return 0


def cmd_restart(_args) -> int:
    get_service_manager().restart()
    return 0


# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="cuebench_daemon.py",
        description="Background lifecycle wrapper for the CueBench scoring agent (dry-run by default).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="foreground loop (what launchd executes / manual testing)")
    p_run.add_argument("--live", action="store_true", help="POST for real (default: dry-run, no POST)")
    p_run.set_defaults(func=cmd_run)

    p_inst = sub.add_parser("install", help="write the launchd job and start it")
    p_inst.add_argument("--live", action="store_true", help="install in live POST mode (default: dry-run)")
    p_inst.set_defaults(func=cmd_install)

    sub.add_parser("uninstall", help="stop and remove the launchd job").set_defaults(func=cmd_uninstall)
    sub.add_parser("start", help="start the installed job").set_defaults(func=cmd_start)
    sub.add_parser("stop", help="stop the job now (no respawn)").set_defaults(func=cmd_stop)
    sub.add_parser("restart", help="restart the job").set_defaults(func=cmd_restart)
    sub.add_parser("status", help="is it running? mode, pid, heartbeat, last exit").set_defaults(func=cmd_status)

    p_logs = sub.add_parser("logs", help="tail stdout+stderr logs")
    p_logs.add_argument("-n", type=int, default=40, help="lines to show (default 40)")
    p_logs.add_argument("-f", "--follow", action="store_true", help="follow (Ctrl-C to stop)")
    p_logs.set_defaults(func=cmd_logs)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
