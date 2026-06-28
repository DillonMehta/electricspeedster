#!/usr/bin/env python3
"""
cuebench_menubar.py — macOS menu bar app for the CueBench scoring daemon
=======================================================================
A native menu bar item (top-right of the screen, next to Wi-Fi/battery). Click it to see:
  * the daemon's status (Running / Stopped / Not installed, dry-run vs live)
  * your most recent scored session(s)
  * Start / Stop / Mode toggle / Install / Uninstall / Open logs / Launch-at-login
  * ⚙︎ Settings… — edit Employee ID, the dashboard (forward) URL + API key, and the BYOK
    key/model. Saved to ~/.cuebench/daemon/daemon.env (0600) and applied on the next start.

This is a LIGHTWEIGHT control surface. It does NOT load the model or score anything — the
headless launchd daemon (cuebench_daemon.py) does the scoring and keeps running even if you
quit this app. The app just drives that daemon and reads its status + logs.

Built with PyObjC/AppKit (already installed) — no extra dependencies.

Run:
  python cuebench_menubar.py            # launch the menu bar app
  python cuebench_menubar.py --build-app [DEST]   # build a double-clickable CueBench.app
  python cuebench_menubar.py --status   # print parsed status (debug; no GUI)
"""
from __future__ import annotations
import glob
import json
import os
import plistlib
import re
import subprocess
import sys

import cuebench_daemon as d

# GUI auto-start (separate from the scoring daemon's own LaunchAgent)
MENUBAR_LABEL = "dev.cuebench.menubar"
MENUBAR_PLIST = os.path.join(d.HOME, "Library", "LaunchAgents", f"{MENUBAR_LABEL}.plist")
MENUBAR_PY = os.path.abspath(__file__)
_HERE = os.path.dirname(MENUBAR_PY)
ICON_PATH = os.path.join(_HERE, "cuebench_icon.png")   # menu bar template (CueBench mark)
ICNS_PATH = os.path.join(_HERE, "CueBench.icns")        # .app Finder/Dock icon

# Settings fields shown in the gear pane: (env key, label, kind)
#   kind: "text" | "secret" | "choice"
SETTINGS_FIELDS = [
    ("CUEBENCH_EMPLOYEE_ID", "Employee ID", "text"),
    ("CUEBENCH_API_URL",     "Dashboard URL (forward)", "text"),
    ("CUEBENCH_API_KEY",     "Dashboard API key", "secret"),
    ("CUEBENCH_BYOK_KEY",    "BYOK key (generation)", "secret"),
    ("CUEBENCH_BYOK_MODEL",  "BYOK model", "choice"),
]

# BYOK model dropdown — (label, value). Empty value = let the engine pick the cheapest
# default for the provider (anthropic→claude-haiku-4-5, openai→gpt-4o-mini). Generation is
# a cheap "title + insights from numbers" call, so the cheapest model per provider is the
# sensible default; the provider is inferred from the model id (claude* → Anthropic, else
# OpenAI), so it must match whatever BYOK key you set above.
MODEL_CHOICES = [
    ("Default — cheapest for the provider", ""),
    ("Anthropic · Haiku 4.5  (cheapest, recommended)", "claude-haiku-4-5"),
    ("Anthropic · Sonnet 4.6", "claude-sonnet-4-6"),
    ("Anthropic · Opus 4.8", "claude-opus-4-8"),
    ("Anthropic · Fable 5  (most capable)", "claude-fable-5"),
    ("OpenAI · GPT-4o mini  (cheapest)", "gpt-4o-mini"),
    ("OpenAI · GPT-4o", "gpt-4o"),
]

# Placeholder text shown in empty Settings fields (guidance without cluttering labels).
SETTINGS_PLACEHOLDERS = {
    "CUEBENCH_EMPLOYEE_ID": "e.g. dillonmehta",
    "CUEBENCH_API_URL": "https://…/api/ingest   (blank = don't forward)",
    "CUEBENCH_API_KEY": "paste dashboard API key",
    "CUEBENCH_BYOK_KEY": "sk-…   (blank = no generated titles)",
}

# A score line printed by the engine's process_one(), e.g.:
#   sid=S-051D051D title='Steady, verified work' score=81 quality='Dialed in (z1)' gen=off POST=OK ...
_SCORE_RE = re.compile(
    r"sid=(?P<sid>S-\S+)\s+title=(?P<title>.+?)\s+score=(?P<score>\d+)\s+"
    r"quality=(?P<quality>.+?)\s+gen=(?P<gen>\S+)\s+POST=(?P<post>OK|FAIL)(?P<tail>.*)")


# ============================================================================
# Pure logic (GUI-free, unit-tested)
# ============================================================================
def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def parse_recent_sessions(log_text: str, limit: int = 6) -> list[dict]:
    """Most-recent-first list of scored sessions parsed from the daemon's stdout log. Each
    row also carries `file` — the transcript basename from the preceding `[finished]` line,
    used to find the session's first prompt for AI rename."""
    rows = []
    last_file = None
    for line in (log_text or "").splitlines():
        s = line.strip()
        if s.startswith("[finished] "):
            last_file = s[len("[finished] "):].strip()
            continue
        m = _SCORE_RE.search(line)
        if m:
            rows.append({
                "sid": m.group("sid"),
                "title": _unquote(m.group("title")),
                "score": int(m.group("score")),
                "quality": _unquote(m.group("quality")),
                "gen": m.group("gen"),
                "posted": m.group("post") == "OK" and "dry-run" not in m.group("tail"),
                "dry_run": "dry-run" in m.group("tail"),
                "file": last_file,
            })
            last_file = None
    # The daemon re-scores a session every time its transcript grows, logging a fresh line
    # each time — so one session shows up repeatedly. Collapse to the most recent line per
    # sid (the store is already deduped on session_id; this keeps the menu consistent with it).
    rows.reverse()                       # newest occurrence first
    seen, deduped = set(), []
    for r in rows:
        if r["sid"] in seen:
            continue
        seen.add(r["sid"])
        deduped.append(r)
    return deduped[:limit]


def recent_sessions(limit: int = 6) -> list[dict]:
    try:
        with open(d.OUT_LOG, errors="replace") as f:
            text = f.read()
    except FileNotFoundError:
        return []
    rows = parse_recent_sessions(text, limit)
    overrides = read_title_overrides()              # user-renamed labels win
    for r in rows:
        if r["sid"] in overrides:
            r["title"] = overrides[r["sid"]]
    return rows


def read_settings() -> dict:
    """Current values of the managed settings (for prefilling the Settings pane)."""
    env = d.parse_env_file(d.ENVFILE)   # resolve path at call time (don't bind the default)
    return {k: env.get(k, "") for k in d.MANAGED_ENV_KEYS}


# ---- AI rename of a session's menu label ----------------------------------
# NOTE: this is the ONE place that sends a raw prompt off-device. The daemon's automatic
# titles are numbers-only; this is an explicit, user-initiated, per-session action that
# sends just that session's FIRST prompt to YOUR OWN BYOK provider to make a short title.
# The result only relabels the menu (it is not re-sent to the dashboard).
TITLE_OVERRIDES_PATH = os.path.join(d.CONTROL_DIR, "title_overrides.json")
_SYSREM_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_CMDOUT_RE = re.compile(r"<(local-command-stdout|command-name|command-message|command-args)>.*?</\1>",
                        re.DOTALL)


def byok_key_present() -> bool:
    env = d.parse_env_file(d.ENVFILE)
    return bool(env.get("CUEBENCH_BYOK_KEY") or os.environ.get("CUEBENCH_BYOK_KEY"))


def read_title_overrides() -> dict:
    try:
        with open(TITLE_OVERRIDES_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def set_title_override(sid: str, title: str) -> None:
    d._ensure_dirs()
    data = read_title_overrides()
    data[sid] = title
    tmp = TITLE_OVERRIDES_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, TITLE_OVERRIDES_PATH)


def resolve_session_path(basename: str | None) -> str | None:
    if not basename:
        return None
    projects = d.parse_env_file(d.ENVFILE).get("CUEBENCH_PROJECTS_DIR") or d.DEFAULT_PROJECTS
    for p in glob.glob(os.path.join(projects, "**", basename), recursive=True):
        return p
    return None


def first_human_prompt(path: str) -> str | None:
    """The first genuine operator prompt in a transcript (mirrors the engine's human-prompt
    filter; read-only, no torch import)."""
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
                if rec.get("type") != "user":
                    continue
                msg = rec.get("message") or {}
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if isinstance(content, list) and any(
                        isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                    continue
                if rec.get("isMeta") or rec.get("sourceToolUseID") or rec.get("interruptedMessageId"):
                    continue
                origin = rec.get("origin") or {}
                if not (origin.get("kind") == "human" or rec.get("promptSource")):
                    continue
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "\n".join(b.get("text", "") for b in content
                                     if isinstance(b, dict) and b.get("type") == "text")
                else:
                    text = ""
                text = _CMDOUT_RE.sub(" ", _SYSREM_RE.sub(" ", text)).strip()
                if text:
                    return text
    except Exception:
        return None
    return None


def _shorten_title(raw: str, max_chars: int = 32) -> str:
    """Trim a model title to something that fits a menu without cutting a word mid-way."""
    title = (raw or "").strip().strip('"').strip("'").strip()
    title = title.splitlines()[0].strip() if title else ""
    if len(title) > max_chars:
        title = title[:max_chars].rsplit(" ", 1)[0].strip() or title[:max_chars].strip()
    return title.rstrip(" .")


def make_short_title(first_prompt: str, max_chars: int = 32) -> tuple[bool, str]:
    """Generate a very short title from a session's first prompt via the BYOK provider.
    Returns (ok, title_or_error). Sends the prompt to the user's own BYOK endpoint only."""
    env = d.parse_env_file(d.ENVFILE)
    key = env.get("CUEBENCH_BYOK_KEY") or os.environ.get("CUEBENCH_BYOK_KEY")
    if not key:
        return False, "No BYOK key set — add one in Settings to enable AI rename."
    try:
        from cuebench_gen import Generator   # lightweight: no torch
    except Exception as e:
        return False, f"Could not load the generator: {e!r}"
    gen = Generator(key=key,
                    provider=env.get("CUEBENCH_BYOK_PROVIDER") or None,
                    model=env.get("CUEBENCH_BYOK_MODEL") or None)
    system = ("You write extremely short titles for a coding session. Given the operator's "
              f"first prompt, reply with ONLY a title of at most 5 words and at most {max_chars} "
              "characters. No quotes, no trailing punctuation, no emoji. Name the task, not filler.")
    raw = gen._complete(system, (first_prompt or "")[:2000], max_tokens=24)
    if not raw:
        return False, "The model returned nothing (check your BYOK key / provider / model)."
    title = _shorten_title(raw, max_chars)
    return (True, title) if title else (False, "Generated an empty title.")


# ---- Developer: preview the exact privacy-safe payload that gets shipped ----
PREVIEW_PATH = os.path.join(d.CONTROL_DIR, "last_payload_preview.json")


def latest_session_path(projects_dir: str | None = None) -> str | None:
    """Most-recently-modified real session transcript (skips agent-* subagent files)."""
    projects_dir = (projects_dir
                    or d.parse_env_file(d.ENVFILE).get("CUEBENCH_PROJECTS_DIR")
                    or d.DEFAULT_PROJECTS)
    newest, newest_mtime = None, -1.0
    for p in glob.glob(os.path.join(projects_dir, "**", "*.jsonl"), recursive=True):
        if os.path.basename(p).startswith("agent-"):
            continue
        try:
            mt = os.path.getmtime(p)
        except OSError:
            continue
        if mt > newest_mtime:
            newest, newest_mtime = p, mt
    return newest


def _extract_payload_json(stdout: str) -> dict | None:
    """Pull the single payload object out of `cuebench_agent.py --once --dry-run` stdout
    (which prints json.dumps(payload, indent=2) then a one-line summary with no braces)."""
    i, j = stdout.find("{"), stdout.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    try:
        return json.loads(stdout[i:j + 1])
    except Exception:
        return None


def generate_payload_preview(session_path: str, out_path: str = PREVIEW_PATH) -> tuple[bool, str]:
    """Run the REAL engine in dry-run on one session and write the exact POST payload to
    out_path. Returns (ok, out_path). Nothing is sent — this is the privacy-safe body the
    daemon would POST (derived numbers + neutral generated text only)."""
    d._ensure_dirs()
    env = {**os.environ, **d.parse_env_file(d.ENVFILE)}
    env.setdefault("CUEBENCH_MODEL_DIR", d.DEFAULT_MODEL_DIR)
    env.setdefault("CUEBENCH_STATE_DB", d.DEFAULT_STATE_DB)
    env.setdefault("CUEBENCH_SCORED_FILE", d.DEFAULT_SCORED_FILE)
    env.setdefault("CUEBENCH_PROJECTS_DIR", d.DEFAULT_PROJECTS)
    agent_py = os.path.join(d.APP_DIR, "cuebench_agent.py")
    try:
        proc = subprocess.run([d.PYTHON, agent_py, "--once", session_path, "--dry-run"],
                              capture_output=True, text=True, cwd=d.APP_DIR, env=env, timeout=300)
        payload = _extract_payload_json(proc.stdout)
        if payload is None:
            with open(out_path, "w") as f:
                f.write("// Could not parse a payload from the engine output.\n"
                        f"// session: {session_path}\n// --- stdout ---\n{proc.stdout}\n"
                        f"// --- stderr ---\n{proc.stderr}\n")
            return False, out_path
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)   # EXACT shipped body (dry-run; not sent)
        return True, out_path
    except Exception as e:
        with open(out_path, "w") as f:
            f.write(f"// payload preview failed: {e!r}\n")
        return False, out_path


# ---- GUI auto-start (login item) ------------------------------------------
MENUBAR_OUT_LOG = os.path.join(d.LOG_DIR, "cuebench-menubar.out.log")
MENUBAR_ERR_LOG = os.path.join(d.LOG_DIR, "cuebench-menubar.err.log")


def gui_login_enabled() -> bool:
    return os.path.exists(MENUBAR_PLIST)


def enable_gui_login():
    d._ensure_dirs()
    plist = {
        "Label": MENUBAR_LABEL,
        "ProgramArguments": [d.PYTHON, MENUBAR_PY],
        "RunAtLoad": True,            # show the menu bar icon at login
        # Restart on a crash (non-zero exit) but stay quit after a clean Quit (exit 0).
        # Without this a crash left the icon silently gone — exactly the symptom we hit.
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 10,
        "ProcessType": "Interactive",
        "EnvironmentVariables": {"PATH": d.SERVICE_PATH, "PYTHONUNBUFFERED": "1"},
        "StandardOutPath": MENUBAR_OUT_LOG,   # so a crash is visible, not silent
        "StandardErrorPath": MENUBAR_ERR_LOG,
    }
    with open(MENUBAR_PLIST, "wb") as f:
        plistlib.dump(plist, f)
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/{MENUBAR_LABEL}"],
                   capture_output=True)
    subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", MENUBAR_PLIST],
                   capture_output=True)


def disable_gui_login():
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/{MENUBAR_LABEL}"], capture_output=True)
    try:
        os.remove(MENUBAR_PLIST)
    except OSError:
        pass


# ============================================================================
# .app bundle builder (a real double-clickable app)
# ============================================================================
def build_app_bundle(dest_dir: str | None = None) -> str:
    """Create CueBench.app — a menu-bar (LSUIElement) app whose launcher execs this script."""
    dest_dir = dest_dir or d.APP_DIR
    app = os.path.join(dest_dir, "CueBench.app")
    macos = os.path.join(app, "Contents", "MacOS")
    res = os.path.join(app, "Contents", "Resources")
    os.makedirs(macos, exist_ok=True)
    os.makedirs(res, exist_ok=True)

    info = {
        "CFBundleName": "CueBench",
        "CFBundleDisplayName": "CueBench",
        "CFBundleIdentifier": "dev.cuebench.menubar.app",
        "CFBundleExecutable": "CueBench",
        "CFBundlePackageType": "APPL",
        "CFBundleVersion": "1.0",
        "CFBundleShortVersionString": "1.0",
        "LSUIElement": True,                 # menu-bar only: no Dock icon, no main window
        "LSMinimumSystemVersion": "11.0",
        "NSHighResolutionCapable": True,
    }
    # Finder/Dock icon (the CueBench logo) if it's been generated (see make_icon.py)
    if os.path.exists(ICNS_PATH):
        import shutil
        shutil.copy2(ICNS_PATH, os.path.join(res, "CueBench.icns"))
        info["CFBundleIconFile"] = "CueBench"
    with open(os.path.join(app, "Contents", "Info.plist"), "wb") as f:
        plistlib.dump(info, f)

    launcher = os.path.join(macos, "CueBench")
    with open(launcher, "w") as f:
        f.write("#!/bin/bash\n"
                "# Launch the CueBench menu bar app with the interpreter it was built against.\n"
                f'exec {sh_quote(d.PYTHON)} {sh_quote(MENUBAR_PY)}\n')
    os.chmod(launcher, 0o755)
    return app


def sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


# ============================================================================
# Settings window (pure construction, so it can be rendered/verified headlessly)
# ============================================================================
def build_settings_window(cur, target=None,
                          save_action="doSaveSettings:", cancel_action="doCancelSettings:"):
    """Build the Settings window + its fields and return
    (window, fields, field_kinds, model_value_by_title).

    Layout is computed so nothing overlaps: a title + wrapping hint at the top, one
    comfortably-spaced row per field, then a divider and a footer (note + buttons) that
    sit in their own band at the bottom with a clear gap above. `target` (the app
    delegate) receives the Save/Cancel actions; pass None to just render it."""
    from AppKit import (
        NSWindow, NSTextField, NSSecureTextField, NSButton, NSPopUpButton, NSBox,
        NSColor, NSFont, NSMakeRect,
        NSWindowStyleMaskTitled, NSWindowStyleMaskClosable, NSBackingStoreBuffered,
    )

    PAD = 24
    LABEL_W, LABEL_GAP = 196, 12
    FIELD_X = PAD + LABEL_W + LABEL_GAP            # 232
    W = 600
    FIELD_W = W - FIELD_X - PAD                    # 344
    ROW_H, FIELD_H = 44, 24

    # vertical bands: header (title+hint) on top, footer (divider+note+buttons) on bottom
    TOP_PAD = 88                                   # title + hint live here
    FOOT = 116                                     # divider + buttons + note live here
    rows = len(SETTINGS_FIELDS)
    H = TOP_PAD + rows * ROW_H + FOOT

    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, W, H),
        NSWindowStyleMaskTitled | NSWindowStyleMaskClosable,
        NSBackingStoreBuffered, False)
    win.setTitle_("CueBench Settings")
    content = win.contentView()

    title = NSTextField.labelWithString_("CueBench settings")
    title.setFont_(NSFont.boldSystemFontOfSize_(16))
    title.setFrame_(NSMakeRect(PAD, H - 40, W - 2 * PAD, 22))
    content.addSubview_(title)

    hint = NSTextField.wrappingLabelWithString_(
        "Forwarding POSTs your scores to a dashboard. BYOK generates neutral titles/"
        "insights — pick a model that matches your BYOK key's provider.")
    hint.setFont_(NSFont.systemFontOfSize_(11))
    hint.setTextColor_(NSColor.secondaryLabelColor())
    hint.setFrame_(NSMakeRect(PAD, H - 78, W - 2 * PAD, 32))
    content.addSubview_(hint)

    fields, field_kinds, model_value_by_title = {}, {}, {}
    for i, (key, label, kind) in enumerate(SETTINGS_FIELDS):
        y = H - TOP_PAD - (i + 1) * ROW_H + 10
        lab = NSTextField.labelWithString_(label + ":")
        lab.setAlignment_(2)                       # right
        lab.setFrame_(NSMakeRect(PAD, y + 1, LABEL_W, 20))
        content.addSubview_(lab)
        field_kinds[key] = kind
        if kind == "choice":
            pop = NSPopUpButton.alloc().initWithFrame_pullsDown_(
                NSMakeRect(FIELD_X - 2, y - 3, FIELD_W + 2, 28), False)
            current = cur.get(key, "") or ""
            choices = list(MODEL_CHOICES)
            if current and current not in [v for _, v in MODEL_CHOICES]:
                choices.append((f"Custom · {current}", current))   # keep a hand-set value
            for ctitle, cvalue in choices:
                pop.addItemWithTitle_(ctitle)
                model_value_by_title[ctitle] = cvalue
                if cvalue == current:
                    pop.selectItemWithTitle_(ctitle)
            content.addSubview_(pop)
            fields[key] = pop
        else:
            cls = NSSecureTextField if kind == "secret" else NSTextField
            fld = cls.alloc().initWithFrame_(NSMakeRect(FIELD_X, y, FIELD_W, FIELD_H))
            fld.setStringValue_(cur.get(key, "") or "")
            ph = SETTINGS_PLACEHOLDERS.get(key)
            if ph:
                fld.setPlaceholderString_(ph)      # hint shown when the field is empty
            content.addSubview_(fld)
            fields[key] = fld

    # footer band: divider, then buttons, then the save-location note at the very bottom
    sep = NSBox.alloc().initWithFrame_(NSMakeRect(PAD, 96, W - 2 * PAD, 1))
    sep.setBoxType_(2)                             # NSBoxSeparator
    content.addSubview_(sep)

    save = NSButton.alloc().initWithFrame_(NSMakeRect(W - PAD - 104, 50, 104, 32))
    save.setTitle_("Save")
    save.setBezelStyle_(1)
    save.setKeyEquivalent_("\r")                   # default (blue) button
    if target is not None:
        save.setTarget_(target)
        save.setAction_(save_action)
    content.addSubview_(save)

    cancel = NSButton.alloc().initWithFrame_(NSMakeRect(W - PAD - 104 - 12 - 104, 50, 104, 32))
    cancel.setTitle_("Cancel")
    cancel.setBezelStyle_(1)
    cancel.setKeyEquivalent_("\033")               # Esc
    if target is not None:
        cancel.setTarget_(target)
        cancel.setAction_(cancel_action)
    content.addSubview_(cancel)

    note = NSTextField.labelWithString_(
        "Saved to ~/.cuebench/daemon/daemon.env (0600). Applied on next start/restart.")
    note.setFont_(NSFont.systemFontOfSize_(11))
    note.setTextColor_(NSColor.tertiaryLabelColor())
    note.setFrame_(NSMakeRect(PAD, 22, W - 2 * PAD, 16))
    content.addSubview_(note)

    win.center()
    return win, fields, field_kinds, model_value_by_title


# ============================================================================
# The menu bar app (PyObjC) — built lazily so the logic above stays import-safe
# ============================================================================
def run_menubar():
    from AppKit import (
        NSApplication, NSApplicationActivationPolicyAccessory, NSStatusBar,
        NSVariableStatusItemLength, NSMenu, NSMenuItem, NSImage, NSAlert,
        NSWindow, NSTextField, NSSecureTextField, NSButton, NSPopUpButton,
        NSWindowStyleMaskTitled, NSWindowStyleMaskClosable, NSBackingStoreBuffered,
        NSColor, NSFont, NSMakeRect, NSApp,
        NSAttributedString, NSMutableAttributedString,
        NSFontAttributeName, NSForegroundColorAttributeName, NSKernAttributeName,
        NSFontWeightRegular, NSFontWeightBold,
    )
    from Foundation import NSObject
    from PyObjCTools import AppHelper
    import objc

    QUALITY_ICON = {  # a tiny visual cue per zone
        "Dialed in": "🟢", "Solid": "🟢", "Developing": "🟡",
        "Inconsistent": "🟠", "Needs attention": "🔴", "Critical": "🔴",
        "Insufficient signal": "⚪",
    }

    def q_icon(quality: str) -> str:
        for k, v in QUALITY_ICON.items():
            if quality.startswith(k):
                return v
        return "•"

    class AppDelegate(NSObject):
        @objc.python_method
        def _install_edit_menu(self):
            # An accessory app has no menu bar, so the standard Edit menu (and its
            # ⌘X/⌘C/⌘V/⌘A key equivalents) is absent — without it, paste never reaches
            # the Settings text fields. Installing a main menu with an Edit submenu wires
            # the shortcuts to the focused field editor via the responder chain.
            main = NSMenu.alloc().init()
            app_item = NSMenuItem.alloc().init()
            main.addItem_(app_item)
            app_menu = NSMenu.alloc().init()
            app_item.setSubmenu_(app_menu)
            app_menu.addItemWithTitle_action_keyEquivalent_("Quit CueBench", "terminate:", "q")
            edit_item = NSMenuItem.alloc().init()
            main.addItem_(edit_item)
            edit_menu = NSMenu.alloc().initWithTitle_("Edit")
            edit_item.setSubmenu_(edit_menu)
            for label, action, key in (
                ("Undo", "undo:", "z"), ("Redo", "redo:", "Z"),
                ("Cut", "cut:", "x"), ("Copy", "copy:", "c"),
                ("Paste", "paste:", "v"), ("Select All", "selectAll:", "a"),
            ):
                edit_menu.addItemWithTitle_action_keyEquivalent_(label, action, key)
            NSApp.setMainMenu_(main)

        def applicationDidFinishLaunching_(self, _note):
            NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
            self._install_edit_menu()      # enables ⌘V/⌘C/etc. in the Settings fields
            self.statusItem = NSStatusBar.systemStatusBar().statusItemWithLength_(
                NSVariableStatusItemLength)
            btn = self.statusItem.button()
            img = None
            if os.path.exists(ICON_PATH):                       # the CueBench logo mark
                img = NSImage.alloc().initWithContentsOfFile_(ICON_PATH)
                if img is not None:
                    img.setSize_((18, 18))                      # menu bar height
                    img.setTemplate_(True)                      # adapt to light/dark menu bar
            if img is None:                                     # fallbacks if the png is missing
                img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                    "gauge.with.dots.needle.bottom.50percent", "CueBench")
                if img is not None:
                    img.setTemplate_(True)
            if img is not None:
                btn.setImage_(img)                  # logo only (the C + underline mark)
            else:
                btn.setTitle_("CB")                 # text fallback only if the png is missing
            btn.setToolTip_("CueBench")
            self.menu = NSMenu.alloc().init()
            self.menu.setAutoenablesItems_(False)
            self.menu.setDelegate_(self)
            self.statusItem.setMenu_(self.menu)
            self.settingsWindow = None
            self.fields = {}
            self.rebuild()

        # rebuild the menu each time it opens so status/scores are fresh
        def menuWillOpen_(self, _menu):
            self.rebuild()

        # -- helpers (pure Python; @python_method keeps PyObjC from bridging them as
        #    Objective-C selectors, which requires arg-count==underscore-count) ----------
        @objc.python_method
        def _add(self, title, action=None, key="", enabled=None, indent=0, attr=None):
            it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "" if attr is not None else title, action, key)
            if attr is not None:
                it.setAttributedTitle_(attr)
            if action:
                it.setTarget_(self)
            it.setEnabled_(bool(action) if enabled is None else enabled)
            if indent:
                it.setIndentationLevel_(indent)
            self.menu.addItem_(it)
            return it

        @objc.python_method
        def _attr(self, text, size=13.0, bold=False, mono=False, color=None, kern=None):
            """An NSAttributedString for a menu item — the lever for typographic hierarchy
            (bold headers, monospaced/aligned scores, quiet secondary subtext, tracked labels)."""
            if mono:
                font = NSFont.monospacedSystemFontOfSize_weight_(
                    size, NSFontWeightBold if bold else NSFontWeightRegular)
            else:
                font = NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size)
            attrs = {NSFontAttributeName: font}
            if color is not None:
                attrs[NSForegroundColorAttributeName] = color
            if kern is not None:
                attrs[NSKernAttributeName] = kern
            return NSAttributedString.alloc().initWithString_attributes_(text, attrs)

        @objc.python_method
        def _sep(self):
            self.menu.addItem_(NSMenuItem.separatorItem())

        @objc.python_method
        def _add_session_item(self, r, has_byok):
            # A session row is a SUBMENU (macOS can't right-click a menu item): the visible
            # NAME is the title; hover to reveal details + the AI-rename action.
            tag = "dry-run" if r["dry_run"] else ("posted" if r["posted"] else "queued")
            name = r["title"] or r["sid"]
            parent = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("", None, "")
            parent.setAttributedTitle_(
                self._attr(f"{q_icon(r['quality'])}  {name}", size=13, color=NSColor.labelColor()))
            parent.setEnabled_(True)
            sub = NSMenu.alloc().init()
            sub.setAutoenablesItems_(False)
            detail = NSMutableAttributedString.alloc().init()
            detail.appendAttributedString_(self._attr(f"{r['score']:>3}", size=12.5, mono=True,
                                                      bold=True, color=NSColor.labelColor()))
            detail.appendAttributedString_(self._attr(f"   {r['quality']}  ·  {tag}", size=12.5,
                                                      color=NSColor.secondaryLabelColor()))
            ti = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("", None, "")
            ti.setAttributedTitle_(detail)
            ti.setEnabled_(False)
            sub.addItem_(ti)
            sub.addItem_(NSMenuItem.separatorItem())
            if has_byok:
                ri = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    "✨ Rename with AI", "doRenameSession:", "")
                ri.setTarget_(self)
                ri.setRepresentedObject_(r["sid"])
                ri.setEnabled_(True)
            else:
                ri = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    "Set a BYOK key in Settings to enable AI rename", None, "")
                ri.setEnabled_(False)
            sub.addItem_(ri)
            parent.setSubmenu_(sub)
            self.menu.addItem_(parent)

        @objc.python_method
        def rebuild(self):
            self.menu.removeAllItems()
            st = d.status_dict()
            sec = NSColor.secondaryLabelColor()
            ter = NSColor.tertiaryLabelColor()
            lab = NSColor.labelColor()

            # header: bold name + status, with mode/heartbeat as quiet sub-lines
            self._add("", enabled=False,
                      attr=self._attr(f"{st['dot']}  CueBench", size=14, bold=True, color=lab))
            sub = st.get("indicator", "")
            if st.get("mode"):
                sub += f"  ·  {st['mode'].lower()}"
            self._add("", enabled=False, attr=self._attr(f"        {sub}", size=11.5, color=sec))
            if st.get("running") and st.get("hb_age") is not None:
                self._add("", enabled=False, attr=self._attr(
                    f"        heartbeat {int(st['hb_age'])}s ago  ·  cycle {st.get('cycles', '?')}",
                    size=11, color=ter))

            self._sep()
            self._add("", enabled=False, attr=self._attr(
                "RECENT SESSIONS  ·  HOVER ▸ TO RENAME", size=10, color=ter, kern=0.5))
            rows = recent_sessions(6)
            self._session_files = {}
            if not rows:
                self._add("", enabled=False, indent=1, attr=self._attr(
                    "None yet — scores appear as sessions finish", size=12, color=ter))
            else:
                has_byok = byok_key_present()
                for r in rows:
                    self._session_files[r["sid"]] = r.get("file")
                    self._add_session_item(r, has_byok)

            self._sep()
            if st["running"]:
                self._add("Stop", "doStop:")
            elif st["installed"]:
                self._add("Start", "doStart:")
            else:
                self._add("Install & Start (dry-run)", "doInstall:")

            if st["installed"]:
                live = (st.get("mode") or "").upper().startswith("LIVE")
                self._add("Switch to DRY-RUN" if live else "Switch to LIVE (POST for real)…",
                          "doToggleMode:")
                self._add("Uninstall daemon", "doUninstall:")

            self._sep()
            self._add("Settings…", "doSettings:", key=",")
            login = self._add("Launch CueBench at login", "doToggleLogin:")
            login.setState_(1 if gui_login_enabled() else 0)
            self._add("Open logs", "doOpenLogs:")
            self._add("Refresh", "doRefresh:")

            self._sep()
            self._add("", enabled=False, attr=self._attr("DEVELOPER", size=10, color=ter, kern=0.5))
            self._add("Preview payload JSON (what gets sent)…", "doPreviewPayload:", indent=1)
            self._add("Open config file…", "doOpenConfig:", indent=1)

            self._sep()
            self._add("Quit CueBench", "doQuit:", "q")

        @objc.python_method
        def _alert(self, text, info=""):
            a = NSAlert.alloc().init()
            a.setMessageText_(text)
            if info:
                a.setInformativeText_(info)
            a.runModal()

        @objc.python_method
        def _svc(self):
            try:
                return d.get_service_manager()
            except SystemExit as e:
                self._alert("Unsupported platform", str(e))
                return None

        # -- menu actions ---------------------------------------------------
        def doStart_(self, _s):
            svc = self._svc()
            if svc:
                try: svc.start()
                except Exception as e: self._alert("Start failed", repr(e))
            self.rebuild()

        def doStop_(self, _s):
            svc = self._svc()
            if svc:
                try: svc.stop()
                except Exception as e: self._alert("Stop failed", repr(e))
            self.rebuild()

        def doInstall_(self, _s):
            svc = self._svc()
            if svc:
                try: svc.install(live=False)
                except Exception as e: self._alert("Install failed", repr(e))
            self.rebuild()

        def doUninstall_(self, _s):
            svc = self._svc()
            if svc:
                try: svc.uninstall()
                except Exception as e: self._alert("Uninstall failed", repr(e))
            self.rebuild()

        def doToggleMode_(self, _s):
            svc = self._svc()
            if not svc:
                return
            st = d.status_dict()
            going_live = not (st.get("mode") or "").upper().startswith("LIVE")
            if going_live:
                a = NSAlert.alloc().init()
                a.setMessageText_("Switch to LIVE mode?")
                a.setInformativeText_("Sessions will be POSTed to the real dashboard. Make sure "
                                      "your Dashboard API key is set in Settings.")
                a.addButtonWithTitle_("Go Live")
                a.addButtonWithTitle_("Cancel")
                if a.runModal() != 1000:   # not the first button
                    return
            try:
                svc.install(live=going_live)
            except Exception as e:
                self._alert("Mode switch failed", repr(e))
            self.rebuild()

        def doToggleLogin_(self, _s):
            try:
                if gui_login_enabled():
                    disable_gui_login()
                else:
                    enable_gui_login()
            except Exception as e:
                self._alert("Login-item change failed", repr(e))
            self.rebuild()

        def doOpenLogs_(self, _s):
            target = d.OUT_LOG if os.path.exists(d.OUT_LOG) else d.LOG_DIR
            subprocess.run(["/usr/bin/open", target])

        def doRefresh_(self, _s):
            self.rebuild()

        # -- AI rename of a session label -----------------------------------
        def doRenameSession_(self, sender):
            sid = sender.representedObject()
            basename = (getattr(self, "_session_files", None) or {}).get(sid)
            import threading
            threading.Thread(target=self._do_rename, args=(sid, basename), daemon=True).start()

        @objc.python_method
        def _do_rename(self, sid, basename):
            path = resolve_session_path(basename)
            if not path:
                AppHelper.callAfter(self._alert, "Rename failed",
                                    f"Couldn't locate the transcript for {sid}.")
                return
            prompt = first_human_prompt(path)
            if not prompt:
                AppHelper.callAfter(self._alert, "Rename failed",
                                    "No operator prompt was found in that session.")
                return
            ok, result = make_short_title(prompt)
            if not ok:
                AppHelper.callAfter(self._alert, "Rename failed", result)
                return
            set_title_override(sid, result)
            print(f"[menubar] renamed {sid} -> {result!r}", flush=True)
            AppHelper.callAfter(self.rebuild)
            AppHelper.callAfter(self._alert, "Renamed",
                                f"{sid} is now “{result}”.\n\nGenerated from the session's first "
                                "prompt via your BYOK model. Reopen the menu to see it.")

        # -- developer ------------------------------------------------------
        def doOpenConfig_(self, _s):
            if not os.path.exists(d.ENVFILE):
                d._ensure_dirs()
                with open(d.ENVFILE, "w") as f:
                    f.write("# CueBench daemon settings — edit via the Settings pane.\n")
                os.chmod(d.ENVFILE, 0o600)
            subprocess.run(["/usr/bin/open", "-t", d.ENVFILE])   # -t: force a text editor

        def doPreviewPayload_(self, _s):
            path = latest_session_path()
            if not path:
                self._alert("No sessions found",
                            "No Claude Code session transcripts were found to preview.")
                return
            print(f"[menubar] preview requested for {path}", flush=True)
            # Open a native read-only viewer immediately (no Xcode, no external app); it shows
            # a placeholder, then fills in when the dry-run scoring finishes (~15s).
            self._open_preview_viewer(
                "Generating payload preview…\n\n"
                "Scoring your most recent session in DRY-RUN — nothing is sent.\n"
                "Loading the model takes ~15s; the exact JSON will appear here.")
            import threading
            threading.Thread(target=self._gen_preview, args=(path,), daemon=True).start()

        @objc.python_method
        def _open_preview_viewer(self, text):
            from AppKit import (NSWindow, NSScrollView, NSTextView, NSFont,
                                NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
                                NSWindowStyleMaskResizable, NSBackingStoreBuffered,
                                NSViewWidthSizable, NSViewHeightSizable)
            rect = NSMakeRect(0, 0, 660, 580)
            win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                rect,
                NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable,
                NSBackingStoreBuffered, False)
            win.setTitle_("CueBench — payload that gets sent (dry-run)")
            win.setReleasedWhenClosed_(False)
            scroll = NSScrollView.alloc().initWithFrame_(rect)
            scroll.setHasVerticalScroller_(True)
            scroll.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            tv = NSTextView.alloc().initWithFrame_(rect)
            tv.setEditable_(False)                 # read-only viewer
            tv.setRichText_(False)
            tv.setFont_(NSFont.userFixedPitchFontOfSize_(12))   # monospaced
            tv.setString_(text)
            scroll.setDocumentView_(tv)
            win.setContentView_(scroll)
            win.center()
            NSApp.activateIgnoringOtherApps_(True)
            win.makeKeyAndOrderFront_(None)
            self._preview_win, self._preview_tv = win, tv
            if not hasattr(self, "_viewers"):
                self._viewers = []
            self._viewers.append(win)              # retain so it isn't GC'd

        @objc.python_method
        def _gen_preview(self, path):
            try:
                ok, out_path = generate_payload_preview(path)
                print(f"[menubar] preview {'ok' if ok else 'FAILED'} -> {out_path}", flush=True)
                try:
                    text = open(out_path, errors="replace").read()
                except Exception as e:
                    text = f"// could not read preview file: {e!r}"
                AppHelper.callAfter(self._fill_preview, text)   # back to the main thread
            except Exception as e:
                print(f"[menubar] preview error: {e!r}", flush=True)
                AppHelper.callAfter(self._fill_preview, f"// preview error: {e!r}")

        @objc.python_method
        def _fill_preview(self, text):
            if getattr(self, "_preview_tv", None) is not None:
                self._preview_tv.setString_(text)

        def doQuit_(self, _s):
            NSApp.terminate_(self)

        # -- settings window ------------------------------------------------
        def doSettings_(self, _s):
            win, fields, kinds, model_map = build_settings_window(read_settings(), target=self)
            self.settingsWindow = win
            self.fields = fields
            self.field_kinds = kinds
            self.model_value_by_title = model_map
            NSApp.activateIgnoringOtherApps_(True)
            win.makeKeyAndOrderFront_(None)

        def doCancelSettings_(self, _s):
            if self.settingsWindow:
                self.settingsWindow.close()
                self.settingsWindow = None

        def doSaveSettings_(self, _s):
            updates = {}
            for k in self.fields:
                if self.field_kinds.get(k) == "choice":
                    title = self.fields[k].titleOfSelectedItem()
                    updates[k] = self.model_value_by_title.get(title, "")
                else:
                    updates[k] = self.fields[k].stringValue()
            try:
                d.save_env_file(updates)
            except Exception as e:
                self._alert("Could not save settings", repr(e))
                return
            if self.settingsWindow:
                self.settingsWindow.close()
                self.settingsWindow = None
            # apply: restart the daemon if it's running so it re-reads the env
            st = d.status_dict()
            restarted = False
            if st["installed"]:
                svc = self._svc()
                try:
                    svc.restart()
                    restarted = True
                except Exception:
                    pass
            self._alert("Settings saved",
                        "Daemon restarted with the new settings." if restarted
                        else "They'll apply the next time the daemon starts.")
            self.rebuild()

    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    AppHelper.runEventLoop()


# ============================================================================
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--build-app":
        dest = argv[1] if len(argv) > 1 else None
        app = build_app_bundle(dest)
        print(f"Built {app}")
        print("Double-click it, or drag it to /Applications. To start at login, use the app's "
              "“Launch CueBench at login” menu item.")
        return 0
    if argv and argv[0] == "--status":
        import json
        print(json.dumps(d.status_dict(), indent=2, default=str))
        print("recent:", json.dumps(recent_sessions(6), indent=2))
        return 0
    run_menubar()
    return 0


if __name__ == "__main__":
    sys.exit(main())
