"""
Self-Update (Freebuff-style: type the command, always run the latest)
=====================================================================
Flow when the user launches `mforege`:

  1. main() asks update_check.get_available_update() (24h-cached, silent).
  2. A newer version exists  ->  print a one-line notice and EXIT quickly.
     A small detached helper process then:
       a. `pip install --upgrade mforege` (falls back to `--user`)
       b. restarts `mforege` IN THE SAME TERMINAL — the helper attaches to
          the parent console and the fresh app inherits it, so the new
          instance takes over the terminal the user originally typed in.
  3. No update (or offline / PyPI down)  ->  launch immediately as usual.

Why the exit-then-upgrade dance: a running mforege.exe locks its own file,
so an in-process `pip install --upgrade` fails with WinError 32 ("file in
use"). Exiting first releases the lock; the detached helper then upgrades
and reopens the app — from the user's point of view it's one command that
always ends in the newest version.

Design guarantees (must never trap the user or break CI):
- Loop guard: the helper's relaunch runs with MFOREGE_UPDATING=1; main()
  skips auto-update when that is set, so a failed upgrade can never loop.
- The helper itself is fully detached (DETACHED_PROCESS |
  CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW) and mforege exits before
  pip runs.
- The RELAUNCHED app carries NO creation flags — on Windows it must
  inherit the console the helper just attached, or it would start with no
  console at all (invisible, no input); on POSIX a non-zero creationflags
  raises ValueError.
- Sub-commands travel to the helper as JSON lists: quote-safe through
  argv on every OS, and directly usable by subprocess.run/Popen (a quoted
  *string* command only works on Windows, never POSIX).
- Every failure falls back to the old behavior: run the installed version.
- CI / scripts bypass this entirely: --no-update flag or MFOREGE_NO_UPDATE=1.
"""

import json
import os
import subprocess
import sys

# ── Windows process-creation flags ─────────────────────────────────────
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000
DETACHED_FLAGS = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_NO_WINDOW

ENV_UPDATING = "MFOREGE_UPDATING"    # set for the helper AND the relaunch
ENV_NO_UPDATE = "MFOREGE_NO_UPDATE"  # user/system opt-out


def updates_disabled(env: dict | None = None) -> bool:
    """True when auto-update must not run (opt-out env or inner relaunch)."""
    env = dict(os.environ if env is None else env)
    return bool(env.get(ENV_NO_UPDATE) or env.get(ENV_UPDATING))


def _helper_source() -> str:
    """Source for the tiny helper process (runs via `python -c`).

    Kept as a string so it never executes inside this module. Steps:
      1. pip upgrade (output captured/discarded; --user fallback).
      2. AttachConsole(parent) + relaunch mforege — the fresh instance
         inherits our console AND our environment (incl. the
         MFOREGE_UPDATING=1 loop guard).
    """
    return """
import json, os, subprocess, sys

upgrade_cmd = json.loads(sys.argv[1])   # list[str]: pip upgrade command
relaunch_cmd = json.loads(sys.argv[2])  # list[str]: how to restart mforege

# 1. Upgrade (console-less; pip output is captured and discarded)
rc = 1
try:
    rc = subprocess.run(upgrade_cmd, capture_output=True, timeout=300).returncode
except Exception:
    rc = 1
if rc != 0 and "--user" not in upgrade_cmd:
    try:  # no write access to site-packages? install into the user's home
        subprocess.run(upgrade_cmd + ["--user"], capture_output=True, timeout=300)
    except Exception:
        pass

# 2. Relaunch mforege, attached to the original terminal
if os.name == "nt":
    try:
        import ctypes
        ctypes.windll.kernel32.AttachConsole(-1)  # ATTACH_PARENT_PROCESS
    except Exception:
        pass
try:
    # NO creationflags here: on Windows the child must inherit the console
    # we just attached (DETACHED would start it without any console, i.e.
    # invisible); on POSIX creationflags != 0 raises ValueError.
    subprocess.Popen(relaunch_cmd, close_fds=True, env=dict(os.environ))
except Exception:
    pass
"""


def _helper_command() -> list[str]:
    """The command the detached helper runs: upgrade, then relaunch.

    Both sub-commands are passed as JSON lists — safe to round-trip
    through argv on every OS (no shell-quoting pitfalls) and directly
    usable by subprocess in the helper.
    """
    exe_dir = os.path.dirname(sys.executable)
    pip = os.path.join(exe_dir, "python.exe")  # Windows venv layout
    if not os.path.exists(pip):
        pip = sys.executable
    upgrade = [pip, "-m", "pip", "install", "--upgrade", "mforege"]
    return [
        sys.executable, "-c",
        _helper_source(),
        json.dumps(upgrade),      # argv[1]: upgrade command
        json.dumps(["mforege"]),  # argv[2]: relaunch command
    ]


def _spawn_detached(cmd: list[str], creationflags: int, env: dict) -> None:
    kwargs: dict = {"close_fds": True, "env": env}
    if os.name == "nt":
        kwargs["creationflags"] = creationflags
    else:
        kwargs["start_new_session"] = True  # POSIX detach
    subprocess.Popen(cmd, **kwargs)


def perform_update() -> None:
    """Print a notice, spawn the detached upgrade+relaunch helper, and exit.

    Called from main() when a newer version is available and auto-update is
    allowed. Normally never returns (exits process); falls back to returning
    (old version launches as usual) if the helper could not even start.
    """
    print("[i] Updating mforege to the latest version — restarting…")

    env = dict(os.environ)
    env[ENV_UPDATING] = "1"  # loop guard: relaunched instance must not re-update

    try:
        _spawn_detached(_helper_command(), DETACHED_FLAGS, env)
    except Exception:
        # Helper could not start — fall back to running the old version.
        print("[!] Auto-update failed to start; launching the current version.")
        return

    raise SystemExit(0)  # release the exe lock; the helper takes over from here
