"""
Long-Term Memory System
=======================
Lets MForege remember facts about the user across CLI sessions.

Design:
- A small JSON file (data/memory.json) holds a list of durable facts,
  e.g. "User's name is Munjur", "User is building a Django e-commerce project".
- Facts are injected into the system prompt every chat() call so the model
  always "knows" them.
- After each reply, a cheap side-call asks the model whether anything durable
  was learned; if yes, the new facts are saved for future sessions.
- Load/save failures never crash the chat — memory is strictly best-effort.
"""

import json
import os
import re
from datetime import datetime
from typing import List, Optional

# ── limits ──────────────────────────────────────────────────────────────────
MAX_FACTS = 100          # cap so the prompt never balloons
MAX_FACT_CHARS = 300     # trim runaway facts
MAX_FACTS_PER_TURN = 3   # at most a few new facts per exchange


class LongTermMemory:
    """
    File-backed store of durable user facts.

    Usage:
        mem = LongTermMemory()                # loads data/memory.json
        mem.inject_into(system_prompt)        # facts appended for the model
        mem.add("User's name is Munjur")      # save a new fact
        mem.all()                             # list all facts
        mem.clear()                           # forget everything
    """

    def __init__(self, path: Optional[str] = None):
        # Default: <MForege project root>/data/memory.json — anchored to the
        # package location so memory stays in ONE place even when the CLI is
        # launched from any other folder (mforege is a global command now).
        if path is None:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            path = os.path.join(project_root, "data", "memory.json")
        self.path = path
        self.facts: List[str] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self) -> None:
        """Load facts from disk (missing/corrupt file -> empty memory)"""
        self.facts = []
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.facts = [
                        str(item) for item in data
                        if isinstance(item, str) and item.strip()
                    ]
        except (OSError, json.JSONDecodeError):
            # Corrupt or unreadable file: start fresh rather than crash
            self.facts = []

    def _save(self) -> bool:
        """Persist facts to disk. Returns True on success (best-effort)."""
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.facts, f, ensure_ascii=False, indent=2)
            return True
        except OSError:
            return False

    # ── mutation ──────────────────────────────────────────────────────────
    def add(self, fact: str) -> bool:
        """
        Add a durable fact. Deduplicates near-identical entries and
        enforces size/count limits. Returns True if the fact was stored.
        """
        fact = (fact or "").strip()
        if not fact:
            return False
        fact = fact[:MAX_FACT_CHARS]

        if self._is_duplicate(fact):
            return False

        self.facts.append(fact)

        # Evict oldest facts (never the profile facts added first) if over cap
        while len(self.facts) > MAX_FACTS:
            self.facts.pop(0)

        return self._save()

    def _is_duplicate(self, fact: str) -> bool:
        """Near-duplicate detection: normalized comparison."""
        norm = self._normalize(fact)
        return any(self._normalize(existing) == norm for existing in self.facts)

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, collapse whitespace/punctuation for fuzzy matching."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clear(self) -> bool:
        """Forget everything and reset the file."""
        self.facts = []
        return self._save()

    def remove(self, index: int) -> bool:
        """Remove a fact by 1-based index (as shown in `all()`)."""
        if 1 <= index <= len(self.facts):
            self.facts.pop(index - 1)
            return self._save()
        return False

    # ── model integration ─────────────────────────────────────────────────

    def guess_name(self) -> Optional[str]:
        """
        Try to find the user's name in stored facts (e.g., a fact like
        "User's name is Munjur"). Returns the name, or None if unknown.
        Used for the startup greeting — no API call needed.
        """
        patterns = [
            r"user'?s? name(?:'s)?(?:\s+is|:)?\s+(.+)",
            r"(?:calls?|called) (?:me|himself|herself|themselves)\s+(.+)",
            r"(?:to be|wants? to be|asked to be|prefers to be) called\s+(.+)",
            r"name is\s+(.+)",
        ]
        for fact in self.facts:
            for pattern in patterns:
                match = re.search(pattern, fact, flags=re.IGNORECASE)
                if match:
                    name = match.group(1).strip().rstrip(".!, ;")
                    words = name.split()
                    if not words or len(name) > 50:
                        continue
                    return " ".join(words[:4])
        return None

    def inject_into(self, system_prompt: str) -> str:
        """
        Return the system prompt with remembered facts appended, so the
        model naturally "knows" them this turn.
        """
        if not self.facts:
            return system_prompt
        lines = "\n".join(f"- {fact}" for fact in self.facts)
        return (
            f"{system_prompt}\n\n"
            "# What you remember about the user (from past sessions)\n"
            f"{lines}\n"
            "Use these memories naturally when relevant — don't recite them."
        )

    def all(self) -> List[str]:
        """All stored facts, oldest first."""
        return list(self.facts)

    @property
    def count(self) -> int:
        """Number of stored facts"""
        return len(self.facts)

    def __len__(self) -> int:
        return len(self.facts)


# ── extraction side-call ────────────────────────────────────────────────────
EXTRACTION_SYSTEM = """You extract long-term memories from a conversation.

Return ONLY a JSON array of short, durable facts about the user worth
remembering for future conversations (preferences, goals, projects, name,
stack choices). Examples: "User's name is Munjur",
"User is building a Django e-commerce project called MForege".

Rules:
- Max 3 facts. Each under 200 characters. Third person ("User ...").
- Skip small talk, greetings, jokes, and anything temporary.
- If nothing is worth remembering, return [].
- Output ONLY the JSON array — no prose, no code fences."""


async def extract_facts(llm_client, model: str, user_message: str,
                        assistant_reply: str, existing: List[str]) -> List[str]:
    """
    Ask the model (cheap side-call) whether anything durable was learned.
    Returns a list of NEW facts (deduped against `existing`). Never raises —
    extraction failures are silently ignored so chatting is never disrupted.
    """
    if not user_message.strip() or not assistant_reply.strip():
        return []

    already = "\n".join(f"- {f}" for f in existing) or "(none yet)"
    user_prompt = (
        f"Already known facts (do not repeat these):\n{already}\n\n"
        f"User message:\n{user_message[:1000]}\n\n"
        f"Assistant reply:\n{assistant_reply[:1000]}\n\n"
        "What new durable facts should be remembered? JSON array only."
    )

    try:
        response = await llm_client.create_chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content or ""
    except Exception:
        # Any LLM problem (network, auth, parse) -> skip remembering this turn
        return []

    return _parse_facts(raw, existing)


def _parse_facts(raw: str, existing: List[str]) -> List[str]:
    """Parse the model's JSON array output, defensively."""
    text = raw.strip()

    # Strip code fences if the model added them anyway
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    known = {LongTermMemory._normalize(f) for f in existing}
    facts: List[str] = []
    for item in data:
        if not isinstance(item, str):
            continue
        fact = item.strip()[:MAX_FACT_CHARS]
        if not fact:
            continue
        norm = LongTermMemory._normalize(fact)
        if norm in known:
            continue
        known.add(norm)
        facts.append(fact)
        if len(facts) >= MAX_FACTS_PER_TURN:
            break
    return facts
