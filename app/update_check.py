"""
Update Check
============
Occasionally asks PyPI whether a newer mforege release exists and, if so,
returns a one-line "upgrade available" banner for the CLI to display.

Design guarantees (this must never annoy or break the chat):
- Network call has a short timeout and every failure is swallowed.
- Result is cached for 24h in ~/.mforege/update_check.json — at most one
  PyPI request per day.
- Compares versions numerically (1.2.10 > 1.2.9), not alphabetically.
- If anything is unclear (no network, PyPI down, weird response), returns
  None and the CLI simply shows nothing.
"""

import json
import os
import urllib.request

CHECK_URL = "https://pypi.org/pypi/mforege/json"
CACHE_PATH = os.path.join(os.path.expanduser("~"), ".mforege", "update_check.json")
CACHE_TTL_SECONDS = 24 * 60 * 60
TIMEOUT = 3.0


def _parse_version(version: str) -> tuple:
    """'1.2.10' -> (1, 2, 10, 1). Non-numeric parts are dropped for safety.

    A non-numeric suffix (e.g. '1.2.3rc1', '1.2.10b2') ends that numeric
    piece — '3rc1' parses as 3, NOT 31 — and marks the version as a
    pre-release with a trailing 0 flag, so a pre-release sorts BELOW its
    own final release: 1.2.3rc1 < 1.2.3, but 1.2.3rc1 == 1.2.3rc1.
    """
    parts = []
    clean = 1
    for piece in (version or "").strip().split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            elif digits:
                clean = 0  # pre-release suffix (rc1, b2, …) on this piece
                break
        if not digits:
            break
        parts.append(int(digits))
    if not parts:
        return (0, 1)
    return tuple(parts) + (clean,)


def is_newer(remote: str, local: str) -> bool:
    """True when `remote` is a strictly higher version than `local`."""
    try:
        return _parse_version(remote) > _parse_version(local)
    except Exception:
        return False


def _read_cache() -> dict:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _write_cache(payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError:
        pass  # cache is best-effort


def check_for_update(local_version: str, force: bool = False) -> "str | None":
    """
    Return an upgrade banner string if PyPI has a newer version, else None.
    Never raises; never blocks longer than TIMEOUT seconds.
    """
    remote = get_available_update(local_version, force=force)
    if not remote:
        return None
    return _banner(remote)


def get_available_update(local_version: str, force: bool = False) -> str:
    """Return the newer remote version string, or '' when up to date.

    Same network guarantees as check_for_update (cached 24h, silent on any
    failure) but returns data instead of a banner so the auto-updater can
    act on it. The cache records `latest`, so a cached "up to date" answer
    is answered locally without touching the network again.
    """
    if not local_version:
        return ""

    if not force:
        cache = _read_cache()
        checked_at = cache.get("checked_at", 0)
        try:
            now = os.path.getmtime(CACHE_PATH)
        except OSError:
            now = 0  # vanished between exists() and getmtime — treat as stale
        import time
        if now and (time.time() - now) < CACHE_TTL_SECONDS and checked_at:
            remote = cache.get("latest")
            if remote and is_newer(str(remote), local_version):
                return str(remote)
            if remote:
                return ""  # cache says we're current
            return ""     # no usable info — fall through to network

    try:
        req = urllib.request.Request(CHECK_URL, headers={"User-Agent": f"mforege/{local_version}"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.load(resp)
        remote = str((data.get("info") or {}).get("version") or "")
    except Exception:
        return ""  # offline, PyPI down, malformed response — stay silent
    if not remote:
        return ""

    _write_cache({"latest": remote, "checked_at": 1})

    if is_newer(remote, local_version):
        return remote
    return ""


def _banner(remote: str) -> str:
    return (
        f"[!] Update available: mforege {remote} — "
        f"run  pip install --upgrade mforege  to get it"
    )
