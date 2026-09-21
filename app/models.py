"""
Model Catalog — the ONE place to manage MForege's models
=========================================================

┌─────────────────────────────────────────────────────────────────┐
│  TO ADD A MODEL:  add one Model(...) line in MODELS below.      │
│  TO REMOVE:       delete its line.                              │
│  That's it — /model, /model <name>, and the setup wizard all    │
│  pick it up automatically. No other file needs touching.        │
└─────────────────────────────────────────────────────────────────┘

Each model has:
  id       the provider's EXACT model id (sent to the API)
  name     short friendly name → `/model gpt20b` style switching
  provider key into PROVIDERS (decides base_url / backend / key)
  note     one-line description shown in the menu

All listed models are FREE unless the note says otherwise.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ── Providers (where a model LIVES) ───────────────────────────────

@dataclass(frozen=True)
class Provider:
    key: str
    label: str          # shown in menus
    backend: str        # MForege backend: "custom" | "ollama" | "openai"
    base_url: str       # "" → provider default (OpenAI)
    key_env: Optional[str]   # env var that holds the API key (None = no key)
    key_label: str      # prompt text in the wizard
    key_url: str        # where to get a free key
    key_hint: str


PROVIDERS: Dict[str, Provider] = {p.key: p for p in [
    Provider(
        key="groq", label="Groq", backend="custom",
        base_url="https://api.groq.com/openai/v1",
        key_env="API_KEY", key_label="Groq API key (gsk_...)",
        key_url="https://console.groq.com", key_hint="free — no credit card",
    ),
    Provider(
        key="openrouter", label="OpenRouter", backend="custom",
        base_url="https://openrouter.ai/api/v1",
        key_env="API_KEY", key_label="OpenRouter API key (sk-or-...)",
        key_url="https://openrouter.ai/keys",
        key_hint="free — one key, many ':free' models",
    ),
    Provider(
        key="ollama", label="Ollama (local)", backend="ollama",
        base_url="http://localhost:11434/v1",
        key_env=None, key_label="",
        key_url="https://ollama.com", key_hint="free, 100% local",
    ),
    Provider(
        key="openai", label="OpenAI", backend="openai",
        base_url="",
        key_env="OPENAI_API_KEY", key_label="OPENAI_API_KEY (sk-...)",
        key_url="https://platform.openai.com/api-keys", key_hint="paid",
    ),
]}


# ── Models (add / remove HERE) ────────────────────────────────────

@dataclass(frozen=True)
class Model:
    id: str
    name: str
    provider: str
    note: str


MODELS: List[Model] = [
    # ── Groq — free key from console.groq.com ──────────────────────
    Model(id="openai/gpt-oss-20b",          name="gpt20b",   provider="groq",
          note="fast, good default"),
    Model(id="openai/gpt-oss-120b",         name="gpt120b",  provider="groq",
          note="stronger, still free"),
    Model(id="llama-3.3-70b-versatile",     name="llama70b", provider="groq",
          note="great general chat"),
    Model(id="llama-3.1-8b-instant",        name="llama8b",  provider="groq",
          note="fastest, lightest"),
    Model(id="moonshotai/kimi-k2-instruct", name="kimi",     provider="groq",
          note="long-context agent work"),

    # ── OpenRouter — one free key unlocks every ':free' model ──────
    Model(id="meta-llama/llama-3.3-70b-instruct:free", name="or-llama", provider="openrouter",
          note="Llama 70B, free"),
    Model(id="deepseek/deepseek-chat-v3-0324:free",    name="or-deepseek", provider="openrouter",
          note="DeepSeek V3, free"),
    Model(id="deepseek/deepseek-r1:free",              name="or-r1", provider="openrouter",
          note="reasoning model, free"),
    Model(id="qwen/qwen-2.5-72b-instruct:free",        name="or-qwen", provider="openrouter",
          note="Qwen 72B, free"),
    Model(id="google/gemma-3-27b-it:free",             name="or-gemma", provider="openrouter",
          note="Gemma 27B, free"),

    # ── Ollama — free, 100% local (needs ollama.com) ───────────────
    Model(id="llama3.2",       name="local-llama", provider="ollama",
          note="small, fast local"),
    Model(id="qwen2.5-coder",  name="local-coder", provider="ollama",
          note="code-focused local"),
    Model(id="deepseek-r1",    name="local-r1",    provider="ollama",
          note="reasoning local"),

    # ── OpenAI — paid ───────────────────────────────────────────────
    Model(id="gpt-4o-mini", name="gpt4mini", provider="openai", note="cheap default"),
    Model(id="gpt-4o",      name="gpt4",     provider="openai", note="strongest"),
]


# ── Lookup helpers (used by /model and the setup wizard) ──────────

def models_for(provider: str) -> List[Model]:
    """All catalog models of one provider, in listing order."""
    return [m for m in MODELS if m.provider == provider]


def provider_for_model(model_id: str) -> Optional[str]:
    """Provider key owning this exact model id, else None."""
    for m in MODELS:
        if m.id == model_id:
            return m.provider
    return None


def default_for(provider: str) -> str:
    """First listed model id for a provider (fallback: gpt-4o-mini)."""
    models = models_for(provider)
    return models[0].id if models else "gpt-4o-mini"


def provider_for_current(model_id: str, fallback: str = "groq") -> str:
    """Provider for the active model; custom/unlisted ids keep `fallback`."""
    return provider_for_model(model_id) or fallback


def resolve(token: str, provider: str) -> Tuple[Optional[Model], List[Model]]:
    """Turn what the user typed into a Model.

    Accepts (in priority order):
      1. menu number        → /model 2
      2. exact short name   → /model gpt20b
      3. exact model id     → /model openai/gpt-oss-20b
      4. unique substring   → /model kimi  (matches moonshotai/kimi-k2-...)
      5. any other string   → treated as a custom unlisted model id

    Returns (model, ambiguous_candidates). Exactly one of:
      model is not None            → clean match (or custom id)
      model is None, candidates    → ambiguous, show candidates
      model is None, candidates=[] → not a number, and empty token
    """
    token = (token or "").strip()
    if not token:
        return None, []

    models = models_for(provider)

    # 1) menu number
    if token.isdigit():
        idx = int(token) - 1
        if 0 <= idx < len(models):
            return models[idx], []
        return None, []

    lowered = token.lower()

    # 2) exact short name
    for m in models:
        if m.name.lower() == lowered:
            return m, []

    # 3) exact id
    for m in models:
        if m.id.lower() == lowered:
            return m, []

    # 4) unique substring over name or id
    hits = [m for m in models
            if lowered in m.name.lower() or lowered in m.id.lower()]
    if len(hits) == 1:
        return hits[0], []
    if hits:
        return None, hits

    # 5) custom id not in the catalog — allowed on purpose (BYO endpoint)
    return Model(id=token, name=token, provider=provider, note="custom"), []


# ── Menu rendering ────────────────────────────────────────────────

def render_menu(provider: str, current: str = "") -> List[str]:
    """Numbered menu lines for one provider's models."""
    lines = []
    for i, m in enumerate(models_for(provider), 1):
        marker = "  ← current" if m.id == current else ""
        lines.append(f"   [{i}] {m.name:<12} {m.id}  ({m.note}){marker}")
    return lines
