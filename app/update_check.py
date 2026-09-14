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
    """'1.2.10' -> (1, 2, 10). Non-numeric parts are dropped for safety."""
    parts = []
    for piece in (version or "").strip().split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


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
    if not local_version:
        return None

    # Respect the 24h cache unless explicitly forced
    if not force:
        cache = _read_cache()
        checked_at = cache.get("checked_at", 0)
        now = os.path.getmtime(CACHE_PATH) if os.path.exists(CACHE_PATH) else 0
        import time
        if now and (time.time() - now) < CACHE_TTL_SECONDS and checked_at:
            remote = cache.get("latest")
            if remote and is_newer(str(remote), local_version):
                return _banner(str(remote))
            return None

    try:
        req = urllib.request.Request(CHECK_URL, headers={"User-Agent": f"mforege/{local_version}"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.load(resp)
        remote = str((data.get("info") or {}).get("version") or "")
    except Exception:
        return None  # offline, PyPI down, malformed response — stay silent
    if not remote:
        return None

    _write_cache({"latest": remote, "checked_at": 1})

    if is_newer(remote, local_version):
        return _banner(remote)
    return None


def _banner(remote: str) -> str:
    return (
        f"[!] Update available: mforege {remote} — "
        f"run  pip install --upgrade mforege  to get it"
    )
