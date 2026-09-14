"""
Session Persistence
===================
Saves each conversation as a JSON file under <project root>/data/sessions/
so any chat can be resumed after a restart — like an agent session sidebar.

Design:
- One file per session: data/sessions/<timestamp>_<snippet>.json
- Each file: {"id", "started", "last_active", "title", "messages":[...]}
- Messages stored in OpenAI format (role/content) — exactly what the model
  needs to continue seamlessly.
- Sessions are saved incrementally (after every exchange) so even a hard
  crash loses at most the current message.

Usage:
    store = SessionStore()
    session = store.start()                 # new session
    session.append(user_msg, reply_msg)     # after each exchange
    store.list()                            # for /sessions (sidebar listing)
    store.load(session_id)                  # returns messages for resume
"""

import json
import os
import re
import time
from typing import List, Dict, Optional


def _sessions_dir() -> str:
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(project_root, "data", "sessions")


class StoredSession:
    """One saved conversation."""

    def __init__(self, id: str, title: str, started: str,
                 messages: Optional[List[Dict]] = None,
                 last_active: Optional[float] = None):
        self.id = id
        self.title = title
        self.started = started
        self.last_active = last_active if last_active is not None else time.time()
        self.messages: List[Dict] = messages or []

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "started": self.started,
            "last_active": self.last_active,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "StoredSession":
        return cls(
            id=d.get("id", ""),
            title=d.get("title", "(untitled)"),
            started=d.get("started", ""),
            messages=d.get("messages", []),
            last_active=d.get("last_active"),
        )

    def append(self, user_text: str, reply_text: str) -> None:
        """Record one exchange (OpenAI message format)."""
        self.messages.append({"role": "user", "content": user_text})
        if reply_text:
            self.messages.append({"role": "assistant", "content": reply_text})
        self.last_active = time.time()


class SessionStore:
    """File-backed store of past conversations."""

    MAX_SESSIONS = 50  # oldest files beyond this are pruned on save

    def __init__(self, directory: Optional[str] = None):
        self.directory = directory or _sessions_dir()
        try:
            os.makedirs(self.directory, exist_ok=True)
        except Exception:
            pass

    # ── paths ─────────────────────────────────────────────────────────
    def _path(self, session_id: str) -> str:
        # ids are generated here; sanitize defensively anyway
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)
        return os.path.join(self.directory, f"{safe}.json")

    # ── create / save / load ──────────────────────────────────────────
    _issued_ids: set = set()  # class-level: every id handed out this process

    def start(self) -> StoredSession:
        """Create a fresh session (not yet persisted until first append)."""
        now = time.localtime()
        id = time.strftime("%Y%m%d-%H%M%S", now)
        title = time.strftime("%Y-%m-%d %H:%M", now)
        # Second-resolution timestamps can collide (two sessions started in
        # the same second) — disambiguate with -2, -3, ... against both the
        # ids already issued this process and files on disk.
        base_id = id
        n = 2
        seen = set(SessionStore._issued_ids)
        try:
            seen.update(os.listdir(self.directory))
        except Exception:
            pass
        while f"{id}.json" in seen or id in seen:
            id = f"{base_id}-{n}"
            n += 1
        SessionStore._issued_ids.add(id)
        return StoredSession(
            id=id,
            title=title,
            started=time.strftime("%Y-%m-%d %H:%M:%S", now),
        )

    def save(self, session: StoredSession) -> bool:
        """Persist a session. Auto-titles from the first user message."""
        try:
            if session.title.startswith("20") and session.messages:
                first_user = next(
                    (m for m in session.messages if m.get("role") == "user"), None
                )
                if first_user:
                    text = " ".join(first_user["content"].split())
                    session.title = text[:48] + ("…" if len(text) > 48 else "")
            with open(self._path(session.id), "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            self._prune()
            return True
        except Exception:
            return False

    def load(self, session_id: str) -> Optional[StoredSession]:
        try:
            with open(self._path(session_id), "r", encoding="utf-8") as f:
                return StoredSession.from_dict(json.load(f))
        except Exception:
            return None

    def latest_id(self) -> Optional[str]:
        """Most recently active session, excluding the given one."""
        sessions = self.list()
        return sessions[0]["id"] if sessions else None

    # ── listing (sidebar) ─────────────────────────────────────────────
    def list(self, limit: int = 12) -> List[Dict]:
        """
        Sessions newest-first: [{"id", "title", "last_active", "count"}, ...].
        Corrupt files are skipped silently.
        """
        out: List[Dict] = []
        try:
            names = os.listdir(self.directory)
        except Exception:
            return out
        for name in names:
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.directory, name), "r",
                          encoding="utf-8") as f:
                    d = json.load(f)
                out.append({
                    "id": d.get("id", name[:-5]),
                    "title": d.get("title", "(untitled)"),
                    "last_active": float(d.get("last_active", 0)),
                    "count": len(d.get("messages", [])),
                })
            except Exception:
                continue
        out.sort(key=lambda s: s["last_active"], reverse=True)
        return out[:limit]

    # ── housekeeping ──────────────────────────────────────────────────
    def _prune(self) -> None:
        """Keep only the MAX_SESSIONS newest files."""
        try:
            sessions = self.list(limit=10000)
            for old in sessions[self.MAX_SESSIONS:]:
                try:
                    os.remove(self._path(old["id"]))
                except Exception:
                    pass
        except Exception:
            pass

    def delete(self, session_id: str) -> bool:
        try:
            os.remove(self._path(session_id))
            return True
        except Exception:
            return False
