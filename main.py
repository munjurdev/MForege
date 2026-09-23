"""
MForege — AI Agent CLI
======================
Agent-panel chat UI (prompt_toolkit):
  - colored scrolling transcript on top (live markdown replies, thinking
    blocks, tool activity, colored diffs)
  - status/context line ABOVE the input (model | workspace | ctx | ⏳ timer)
  - input box pinned at the very bottom — the last row on screen

Run with: python main.py [--workspace PATH]

Supports multiple backends:
- OpenAI (requires OPENAI_API_KEY)
- Ollama (free, local)
- Custom API (e.g., Groq)
"""

import asyncio
import argparse
import getpass
import os
import sys

from decouple import config as _env_fallback, Config, RepositoryEnv

# Windows consoles default to cp1252, which cannot encode emojis the model
# may output. Force UTF-8 (with safe replacement) so printing never crashes.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from app.agent import Agent, AgentConfig
from app.agent.sessions import SessionStore
from app import __version__
from app.agent.tools import CalculatorTool, TimeTool
from app.tools import ExaSearchTool
from app.tools.system_tools import create_system_tools, PlanState, NotifyHook
from app.ui import ChatUI, boxed
from app import models as model_catalog
from app.update_check import check_for_update, get_available_update
from app import self_update
from app.llm.client import (
    LLMClient,
    LLMAuthError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
)


def cprint_err(text: str) -> None:
    """Plain terminal error print (used only before the UI starts)."""
    print(f"\033[91m{text}\033[0m", file=sys.stderr)


# ── .env resolution ────────────────────────────────────────────────
# `mforege` can be launched from ANY directory. Settings resolve in order:
#   0. real environment variables (shell/session)
#   1. ./.env            (current folder — per-project override)
#   2. ~/.mforege/.env   (global config written by the setup wizard)
#   3. <MForege repo>/.env (developer install)
# Resolution is per-key: a current-folder .env that lacks a key (e.g. a Django
# project's .env with DEBUG/SECRET_KEY only) must NOT shadow the global config.
_MFOREGE_ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    _root_config = Config(RepositoryEnv(os.path.join(_MFOREGE_ROOT, ".env")))
except Exception:
    _root_config = None

_HOME_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".mforege", ".env")
_home_config = None


def _load_home_config() -> None:
    global _home_config
    if os.path.exists(_HOME_CONFIG_PATH):
        try:
            _home_config = Config(RepositoryEnv(_HOME_CONFIG_PATH))
        except Exception:
            _home_config = None


_load_home_config()


def env_config(key: str, default: str = "") -> str:
    """Read settings: current .env → ~/.mforege/.env → MForege's own .env.

    Falls through **per key**: the first file that actually defines `key`
    wins. A local .env that doesn't define the key (e.g. a Django project's
    .env with only DEBUG/SECRET_KEY) must not shadow ~/.mforege/.env.
    """
    if os.path.exists(".env"):
        # Fresh read per call — picks up edits made after launch (same policy
        # as _load_home_config()). NOTE: decouple's bare `config` object
        # resolves its search path via caller-frame magic, so it must NOT be
        # used here; we build the repository from the real CWD instead.
        try:
            local = Config(RepositoryEnv(os.path.join(os.getcwd(), ".env")))
            if key in local.repository:  # honors os.environ too
                return local(key, default=default)
        except Exception:
            pass  # unreadable local .env → keep falling through
    if _home_config is not None and key in _home_config.repository:
        return _home_config(key, default=default)
    if _root_config is not None and key in _root_config.repository:
        return _root_config(key, default=default)
    # Nothing defined it anywhere — one last chance: the real environment.
    return _env_fallback(key, default=default)


GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ── /model command (shared by plain + UI modes) ────────────────────

def _run_model_command(arg: str, agent) -> list:
    """Handle '/model [token]' → list of (text, kind) output lines.

    kinds: title | dim | ok | error. Performs the switch + persistence
    itself so both CLI modes stay behaviorally identical.
    """
    provider = model_catalog.provider_for_current(agent.config.model)
    out: list = []
    if not arg or arg == "list":
        out.append((f"[Models] {model_catalog.PROVIDERS[provider].label} — /model <#> or /model <name>:", "title"))
        for ln in model_catalog.render_menu(provider, current=agent.config.model):
            out.append((ln, "dim"))
        out.append(("   → /model 2   /model kimi   /model openai/gpt-oss-120b", "dim"))
        return out

    model, ambiguous = model_catalog.resolve(arg, provider)
    if model is None and ambiguous:
        names = ", ".join(m.name for m in ambiguous)
        out.append((f"[Models] '{arg}' matches several — be specific: {names}", "error"))
        return out
    if model is None:
        out.append((f"[Models] No model #{arg} — run /model to see the list.", "error"))
        return out
    if model.id == agent.config.model:
        out.append((f"[Models] Already on {model.id}.", "dim"))
        return out
    try:
        agent.switch_model(model.id)
        _persist_model(model.id)
    except Exception as e:
        out.append((f"[Models] Switch failed: {e}", "error"))
        return out
    out.append((f"[Models] Switched to {model.id} ✓ (next message uses it)", "ok"))
    return out


def _persist_model(model: str) -> None:
    """Save the chosen model to ~/.mforege/.env so it survives restarts.

    Rewrites just the LLM_MODEL line (or appends it); never touches keys.
    Best-effort: a read-only config file must not break a model switch.
    """
    try:
        lines: list[str] = []
        if os.path.exists(_HOME_CONFIG_PATH):
            with open(_HOME_CONFIG_PATH, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        found = False
        for i, ln in enumerate(lines):
            if ln.strip().startswith("LLM_MODEL="):
                lines[i] = f"LLM_MODEL={model}"
                found = True
                break
        if not found:
            lines.append(f"LLM_MODEL={model}")
        os.makedirs(os.path.dirname(_HOME_CONFIG_PATH), exist_ok=True)
        with open(_HOME_CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        _load_home_config()  # pick up the new value immediately
    except Exception:
        pass


def _mask_key(key: str) -> str:
    """'gsk_abcdefghijklmnopqrstuvwxyz12' -> 'gsk_ab••••••••••••zy12'

    Shows enough to confirm WHAT was pasted, hides the secret itself.
    """
    key = (key or "").strip()
    if len(key) <= 10:
        return "•" * len(key)
    return f"{key[:6]}{'•' * min(12, len(key) - 10)}{key[-4:]}"


async def _validate_or_reask(key: str, base_url: str, model: str, backend: str,
                             label: str, max_tries: int = 3) -> str:
    """Ping the backend with the pasted key; re-ask on failure. Returns the key."""
    print(f"  Received: {_mask_key(key)} ({len(key)} chars — hidden for safety)")
    tries = 0
    while True:
        print(f"  Verifying {label} with a tiny test call...", flush=True)
        try:
            client = LLMClient(
                backend=backend, api_key=key, base_url=base_url or None, model=model,
            )
            try:
                await client.ping()
            finally:
                # close the HTTP pool inside the SAME event loop, or httpx
                # complains 'Event loop is closed' during GC
                try:
                    await client.client.close()
                except Exception:
                    pass
            print("  ✓ Key works!\n")
            return key
        except (LLMAuthError, LLMConnectionError, LLMRateLimitError, LLMResponseError) as e:
            tries += 1
            print(f"  ✗ That key didn't work: {e}")
            if tries >= max_tries:
                print("  [!] Three failed attempts — saving nothing. Run `mforege --setup` to try again.")
                sys.exit(1)
            key = getpass.getpass(f"  Paste your {label} again (hidden): ").strip()
            if not key:
                print("  [!] No key entered — setup cancelled.")
                sys.exit(1)
            print(f"  Received: {_mask_key(key)} ({len(key)} chars — hidden for safety)")


def _choose_model(provider: str) -> str:
    """Show the provider's catalog; user picks a number, a short name,
    any model id, or presses Enter for the default."""
    default = model_catalog.default_for(provider)
    if model_catalog.models_for(provider):
        print("\n  Pick a model:")
        for ln in model_catalog.render_menu(provider, current=""):
            print(ln)
    try:
        raw = input(f"  Model number/name/id [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        raw = ""
    model, _amb = model_catalog.resolve(raw, provider)
    if model is None:
        print(f"  [!] No model '{raw}' — using {default}.")
        return default
    return model.id


async def run_setup_wizard() -> None:
    """One-time guided setup (freebuff-style): pick a brain, paste a key, done."""
    print()
    print("=" * 58)
    print("  Welcome to MForege!  One-time setup (about 30 seconds)")
    print("=" * 58)
    print()
    print("  Where should MForege get its brain?")
    print("   [1] Groq        free cloud API — fastest (recommended)")
    print("   [2] OpenRouter  free — ONE key unlocks 20+ free models")
    print("   [3] Ollama      free, 100% local (needs https://ollama.com)")
    print("   [4] OpenAI      paid API (uses OPENAI_API_KEY)")
    print("   [5] Custom      any OpenAI-compatible endpoint")
    print()
    try:
        choice = (input("  Choose 1-5 [1]: ").strip() or "1")
    except (EOFError, KeyboardInterrupt):
        print("\n[!] No interactive terminal — run `mforege --setup` in a real terminal to configure.")
        sys.exit(1)

    if choice == "3":
        model = _choose_model("ollama")
        lines = ["LLM_BACKEND=ollama", f"LLM_MODEL={model}"]
        print("\n  Ollama selected — make sure it's running (`ollama serve`).")
    elif choice == "4":
        try:
            key = getpass.getpass("  Paste your OPENAI_API_KEY (hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            key = ""
        if not key:
            print("  [!] No key entered — setup cancelled.")
            sys.exit(1)
        model = _choose_model("openai")
        key = await _validate_or_reask(key, "", model, "openai", "OPENAI_API_KEY")
        lines = ["LLM_BACKEND=openai", f"OPENAI_API_KEY={key}", f"LLM_MODEL={model}"]
    elif choice == "5":
        try:
            key = getpass.getpass("  Paste your API key (hidden): ").strip()
            base = (input(f"  Base URL [{GROQ_BASE_URL}]: ").strip() or GROQ_BASE_URL)
            model = input("  Model: ").strip()
        except (EOFError, KeyboardInterrupt):
            key = ""
            base, model = "", ""
        if not key or not model:
            print("  [!] Key and model are required — setup cancelled.")
            sys.exit(1)
        key = await _validate_or_reask(key, base, model, "custom", "API key")
        lines = ["LLM_BACKEND=custom", f"API_KEY={key}", f"BASE_URL={base}", f"LLM_MODEL={model}"]
    elif choice == "2":
        provider = model_catalog.PROVIDERS["openrouter"]
        print(f"\n  Get a FREE key at {provider.key_url} — one key unlocks")
        print("  every ':free' model (Llama, DeepSeek, Qwen, Gemma, Mistral...).")
        try:
            key = getpass.getpass(f"  Paste your {provider.key_label} (hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            key = ""
        if not key:
            print("  [!] No key entered — setup cancelled.")
            sys.exit(1)
        model = _choose_model("openrouter")
        key = await _validate_or_reask(key, provider.base_url, model, "custom", "OpenRouter API key")
        lines = [
            "LLM_BACKEND=custom",
            f"API_KEY={key}",
            f"BASE_URL={provider.base_url}",
            f"LLM_MODEL={model}",
        ]
    else:
        provider = model_catalog.PROVIDERS["groq"]
        print(f"\n  Get a FREE key at {provider.key_url} (no credit card needed).")
        try:
            key = getpass.getpass("  Paste your Groq API key (gsk_..., hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            key = ""
        if not key:
            print("  [!] No key entered — setup cancelled.")
            sys.exit(1)
        model = _choose_model("groq")
        key = await _validate_or_reask(key, provider.base_url, model, "custom", "Groq API key")
        lines = [
            "LLM_BACKEND=custom",
            f"API_KEY={key}",
            f"BASE_URL={provider.base_url}",
            f"LLM_MODEL={model}",
        ]

    try:
        os.makedirs(os.path.dirname(_HOME_CONFIG_PATH), exist_ok=True)
        with open(_HOME_CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as e:
        cprint_err(f"[!] Could not save config: {e}")
        sys.exit(1)

    _load_home_config()
    print(f"\n  Done! Config saved to {_HOME_CONFIG_PATH}")
    print("  It works from every folder. A ./.env in the current folder")
    print("  can override it anytime. Re-run `mforege --setup` to change it.")
    print()


COLORS_DIM = "\033[2m"
COLORS_RESET = "\033[0m"


def _pc(text: str, color: str = "", end: str = "\n") -> None:
    """Plain-CLI colored print (never raises)."""
    try:
        print(f"{color}{text}{COLORS_RESET}", end=end, flush=True)
    except Exception:
        pass


async def run_plain_cli(agent, args, workspace, plan_state, notify_hook,
                        search_tool) -> None:
    """Classic line-by-line chat — the --plain fallback.

    Uses only input()/print(): works in every terminal, SSH session, and
    piped script. Same commands as the full-screen UI.
    """
    # confirm callback on stdin — attach to the already-registered tools
    async def confirm_action(action: str) -> bool:
        _pc(f"\n[?] MForege wants to:\n{action}", "\033[93m")
        try:
            ans = input("    Allow? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        return ans in ("y", "yes")

    from app.tools.system_tools import (
        RunCommandTool, CreateFileTool, EditFileTool,
    )
    for tool in agent.tools.list():
        if isinstance(tool, (RunCommandTool, CreateFileTool, EditFileTool)):
            tool._confirm = confirm_action

    def render_activity(event: str, detail: str) -> None:
        if event == "tool_start":
            _pc(f"  · {detail}", COLORS_DIM, end="")
        elif event == "tool_end":
            color = "\033[92m" if "✓" in detail else "\033[91m"
            _pc(f" {detail}", color)
        elif event == "adapting":
            _pc(f"  ↻ {detail}", "\033[95m")
        elif event == "round":
            _pc(f"  ── round {detail} ──", COLORS_DIM)

    agent.on_activity = render_activity

    def render_notify(event: str, detail: str) -> None:
        if event == "diff":
            _pc("  ✓ Applied:", "\033[92m")
            for ln in detail.split("\n"):
                c = "\033[92m" if ln.startswith("+") else (
                    "\033[91m" if ln.startswith("-") else COLORS_DIM)
                _pc("  " + ln, c)
        elif event == "cmd":
            _pc(f"  $ {detail}", COLORS_DIM)

    notify_hook.callback = render_notify

    # header — the same structured box as the full UI (uncolored inside,
    # so the width math is exact; the whole box is printed green)
    _pc(boxed([
        "MForege ✦ your personal AI agent (plain mode)",
        f"backend: {args.backend} │ model: {args.model}",
        f"workspace: {workspace}",
        "",
        "Type / for commands · exit or Ctrl+C to quit.",
    ]), "\033[92m\033[1m")

    if not search_tool.is_configured:
        _pc("[i] Optional: web search is off (free Exa key at dashboard.exa.ai).", COLORS_DIM)
    if os.environ.get("TERM_PROGRAM") == "vscode":
        _pc("[i] VS Code: Shift+Enter needs a one-time keybinding — run /vscode-hint.", COLORS_DIM)
        _pc("    Use Alt+Enter meanwhile.", COLORS_DIM)

    # ── session persistence (resume across restarts) ───────────────
    session_store = SessionStore()
    session = session_store.start()

    # NO auto-continue: every launch starts fresh (like the Codebuff
    # panel). Old chats stay on disk — /sessions lists them, /resume
    # <id> restores one explicitly.

    async def handle_sessions_command(slash, agent, session, pc) -> None:
        """"/sessions" list · "/resume [id]" restore an old conversation."""
        if slash == "/sessions":
            rows = session_store.list()
            if not rows:
                pc("[Sessions] No saved conversations yet.", COLORS_DIM)
                return
            pc(f"[Sessions] Recent conversations (newest first):")
            for r in rows:
                pc(f"   {r['id']}  ({r['count']} msgs)  {r['title'][:44]}", COLORS_DIM)
            pc("   → /resume <id> to continue one", COLORS_DIM)
            return
        # /resume [id]
        rows = session_store.list()
        if not rows:
            pc("[Resume] No saved conversations to resume.", COLORS_DIM)
            return
        wanted = None
        parts = slash.split(maxsplit=1)
        if len(parts) > 1:
            wanted = parts[1].strip()
            target = next((r for r in rows if r["id"].startswith(wanted)), None)
            if target is None:
                pc(f"[Resume] No session matching '{wanted}'. Try /sessions.", "\033[91m")
                return
        else:
            target = rows[0]  # most recent
        stored = session_store.load(target["id"])
        if stored is None or not stored.messages:
            pc("[Resume] That session is empty.", "\033[91m")
            return
        agent.restore_session_messages(stored.messages)
        session.id = stored.id            # keep appending to the same file
        session.title = stored.title
        session.messages = stored.messages
        pc(f"[Resume] Restored '{stored.title}' ({len(stored.messages)} messages).", "\033[92m")
        pc("   Context is back — just continue chatting.", COLORS_DIM)

    async def handle(text: str) -> None:
        nonlocal session
        low = text.lower().strip()
        if text.startswith("/"):
            slash = text.split()[0].lower()
            if slash == "/help":
                _pc("[Commands]")
                for cmd, desc in [
                    ("/plan", "Show the current mission plan"),
                    ("/tools", "List registered tools"),
                    ("/new", "Clear the conversation and start a new chat"),
                    ("/history", "Browse and resume past conversations"),
                    ("/sessions", "List past conversations"),
                    ("/resume", "Resume the most recent chat"),
                    ("/bash", "Run a shell command with the agent's safety guards"),
                    ("/byok", "Show where to configure your API key / model"),
                    ("/model", "Switch model live (/model 2 or /model <id>)"),
                    ("/queue", "Message queueing info"),
                    ("/interview", "Guided Q&A to spec a task before building"),
                    ("/feedback", "Share feedback about MForege"),
                    ("/vscode-hint", "Fix Shift+Enter for VS Code's terminal"),
                    ("/clear", "Reset conversation"),
                    ("/exit", "Quit"),
                ]:
                    _pc(f"  {cmd:<10} {desc}", COLORS_DIM)
                return
            if slash == "/plan":
                rendered = plan_state.render()
                _pc(rendered if rendered else "No active plan.", "\033[95m")
                return
            if slash == "/vscode-hint":
                _pc("[VS Code Shift+Enter fix] (one-time, 30 seconds):", "\033[96m")
                _pc("  1. Ctrl+Shift+P → 'Preferences: Open Keyboard Shortcuts (JSON)'")
                _pc("  2. Add this entry inside the [ ] array:")
                _pc('     { "key": "shift+enter",', COLORS_DIM)
                _pc('       "command": "workbench.action.terminal.sendSequence",', COLORS_DIM)
                _pc('       "args": { "text": "\\u001b\\r" },', COLORS_DIM)
                _pc('       "when": "terminalFocus" }', COLORS_DIM)
                _pc("  3. Save, then restart VS Code completely.")
                _pc("  → This makes VS Code send ESC+CR for Shift+Enter — which MForege")
                _pc("    already treats as a newline. (Same fix Claude Code recommends.)")
                return
            if slash in ("/sessions", "/resume"):
                await handle_sessions_command(slash, agent, session,
                                              _pc)
                return
            text = text[1:]
            low = text.lower()
        if low in ("exit", "quit"):
            _pc("Goodbye! 👋", "\033[92m")
            raise SystemExit(0)
        if low == "clear":
            agent.clear_memory()
            _pc("[*] Conversation cleared", "\033[92m")
            return
        if low in ("memory", "forget", "forget all", "forget everything"):
            _pc("[i] MForege keeps only conversation context now (like an agent):")
            _pc("    auto-continues your last chat; /sessions + /resume manage history.", COLORS_DIM)
            return
        if low == "tools":
            _pc("[Tools]:")
            for t in agent.tools.list():
                _pc(f"   - {t.name}", COLORS_DIM)
            return
        if low == "new":
            agent.clear_memory()
            session = session_store.start()
            _pc("[*] New conversation started — fresh context.", "\033[92m")
            return
        if low == "history":
            rows = session_store.list()
            if not rows:
                _pc("[History] No saved conversations yet.", COLORS_DIM)
                return
            _pc("[History] Recent conversations (newest first):")
            for r in rows:
                _pc(f"   {r['id']}  ({r['count']} msgs)  {r['title'][:44]}", COLORS_DIM)
            _pc("   → /resume <id> to continue one — /new to start fresh.", COLORS_DIM)
            return
        if low == "queue":
            _pc("[Queue] Messages sent while the agent works are queued —")
            _pc("    '✓ Queued' appears, and each runs in order after the current turn.", COLORS_DIM)
            return
        if low == "interview":
            _pc("[Interview] Tell me your goal in one line — I'll ask targeted", "\033[96m")
            _pc("    questions one at a time to pin down scope, constraints and", COLORS_DIM)
            _pc("    trade-offs, then turn it into a step-by-step plan.", COLORS_DIM)
            return
        if low == "bash" or low.startswith("bash "):
            parts = text.split(maxsplit=1)
            cmd = parts[1].strip() if len(parts) > 1 else ""
            if not cmd:
                _pc("[Bash] Usage: /bash <command>", "\033[96m")
                _pc("    Runs with the same safety guards and confirmation as the", COLORS_DIM)
                _pc("    agent's run_command tool.", COLORS_DIM)
                return
            runner = agent.tools.get("run_command")
            if runner is None:
                _pc("[Bash] run_command tool unavailable.", "\033[91m")
                return
            result = await runner.execute(command=cmd)
            _pc(result, "" if not result.startswith("Error") else "\033[91m")
            return
        if low == "byok":
            _pc("[BYOK] Bring your own key — edit your config file:", "\033[96m")
            _pc(f"    {_HOME_CONFIG_PATH}", COLORS_DIM)
            _pc("    Set OPENAI_API_KEY / base_url / model (any OpenAI-compatible", COLORS_DIM)
            _pc("    endpoint works, including a free Groq key). A ./.env in the", COLORS_DIM)
            _pc("    current folder overrides it. Or rerun: mforege --setup", COLORS_DIM)
            return
        if low == "model" or low.startswith("model "):
            parts = text.split(maxsplit=1)
            arg = parts[1].strip() if len(parts) > 1 else ""
            colors = {"title": "\033[96m", "dim": COLORS_DIM,
                      "ok": "\033[92m", "error": "\033[91m"}
            for line, kind in _run_model_command(arg, agent):
                _pc(line, colors.get(kind, ""))
            return
        if low == "feedback":
            _pc("[Feedback] MForege is your project — ideas go straight to the", "\033[96m")
            _pc("    source: https://github.com/munjurdev/MForege/issues 💚", COLORS_DIM)
            return

        _pc("MForege: ", "\033[96m\033[1m")
        try:
            response = await agent.chat(text, stream=True)
            reply_text = ""
            if hasattr(response, "__aiter__"):
                async for chunk in response:
                    reply_text += chunk
                    _pc(chunk, "", end="")
                _pc("")
            else:
                reply_text = response
                _pc(response)
            session.append(text, reply_text)
            session_store.save(session)
        except Exception as e:
            _pc(f"[!] {e}", "\033[91m")
        status = plan_state.progress_line()
        if status:
            _pc(status, "\033[95m")
        _pc("")

    try:
        while True:
            try:
                line = input("You: ")
            except (EOFError, KeyboardInterrupt):
                _pc("\nGoodbye! 👋", "\033[92m")
                break
            text = line.strip()
            if not text:
                continue
            try:
                await handle(text)
            except SystemExit:
                break
    finally:
        try:
            await agent.flush_memory()
        except Exception:
            pass
        try:
            await agent.llm.client.close()
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser. Called AFTER config resolution so defaults like
    LLM_BACKEND/LLM_MODEL reflect what the setup wizard just saved."""
    parser = argparse.ArgumentParser(
        prog="mforege",
        description="MForege — your personal AI agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With Groq (free API key at https://console.groq.com):
    mforege --backend custom --base-url https://api.groq.com/openai/v1

  # With Ollama (free, local — no API key needed):
    mforege --backend ollama --model llama3

  # With OpenAI:
    OPENAI_API_KEY=sk-xxx mforege

  # Work inside a specific project folder:
    mforege --workspace "C:\\path\\to\\project"

Keys go in a .env file in the current folder (API_KEY=..., LLM_BACKEND=custom, BASE_URL=..., LLM_MODEL=...).
        """,
    )
    parser.add_argument("--backend", choices=["openai", "ollama", "custom"],
                        default=env_config("LLM_BACKEND", default="openai"))
    parser.add_argument("--model", default=env_config("LLM_MODEL", default="gpt-4o-mini"))
    parser.add_argument("--base-url", default=env_config("BASE_URL", default=""))
    parser.add_argument("--workspace", default=env_config("WORKSPACE", default="."),
                        help="Project folder MForege may read/write (default: current directory)")
    parser.add_argument("--setup", action="store_true",
                        help="Re-run the one-time setup wizard (choose backend, save API key)")
    parser.add_argument("--version", action="store_true",
                        help="Show the installed mforege version and exit")
    parser.add_argument("--plain", action="store_true",
                        help="Classic line-by-line chat (no full-screen UI) — use if the "
                             "fancy input box doesn't accept your keyboard in this terminal")
    parser.add_argument("--no-update", action="store_true",
                        help="Skip the automatic self-update check (also: MFOREGE_NO_UPDATE=1)")
    return parser


async def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:]) if argv is None else list(argv)

    # --help / --version must win over everything (especially the interactive
    # wizard — CI and scripts call mforege --help with no stdin)
    if "-h" in argv or "--help" in argv:
        _build_parser().parse_args(argv)  # prints help and exits 0
        return
    if "--version" in argv:
        print(f"mforege {__version__}")
        return

    # First pass: detect --setup before config-dependent defaults exist
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--setup", action="store_true")
    pre_args, _ = pre.parse_known_args(argv)

    _no_config = not (
        os.path.exists(".env")
        or os.path.exists(_HOME_CONFIG_PATH)
        or os.path.exists(os.path.join(_MFOREGE_ROOT, ".env"))
    )
    if pre_args.setup or _no_config:
        await run_setup_wizard()
        if pre_args.setup:
            # explicit --setup: relaunch main() so backend/model/base-url are
            # re-read from the freshly saved config, then continue into chat
            print("\n  Starting MForege with your new settings… (Ctrl+C to exit)")
            return await main(argv=[])
        # First-run wizard (no --setup flag): fall through and build the real
        # parser now — env_config() will see the config the wizard just saved.

    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── Freebuff-style auto-update ──────────────────────────────────
    # A newer PyPI release exists (24h-cached, silent when offline) and the
    # user didn't opt out → print a notice, exit to release the exe lock,
    # and let a detached helper upgrade + relaunch into the same terminal.
    if not args.no_update and not self_update.updates_disabled():
        available = get_available_update(__version__)
        if available:
            print(f"[i] New version available: mforege {available} "
                  f"(you have {__version__})")
            self_update.perform_update()   # exits; helper relaunches mforege

    # Resolve and validate the workspace early (human decides the sandbox root)
    workspace = os.path.abspath(args.workspace)
    if not os.path.isdir(workspace):
        cprint_err(f"[!] Workspace does not exist: {workspace}")
        cprint_err("    Create it first, or pass a valid --workspace path.")
        sys.exit(1)

    # Get API key based on backend
    cmd = "mforege"
    api_key: str | None
    if args.backend == "openai":
        api_key = env_config("OPENAI_API_KEY", default="")
        if not api_key:
            cprint_err("[!] OpenAI backend requires OPENAI_API_KEY")
            cprint_err(f"    Put it in a .env file (OPENAI_API_KEY=sk-...) or run: {cmd} --backend ollama  (free, local)")
            sys.exit(1)
    elif args.backend == "ollama":
        api_key = None
    elif args.backend == "custom":
        api_key = env_config("API_KEY", default="")
        if not api_key:
            cprint_err("[!] Custom backend requires API_KEY (e.g., a free Groq key from https://console.groq.com)")
            cprint_err("    Put it in a .env file next to where you run the command:")
            cprint_err("        API_KEY=gsk_...")
            cprint_err("        BASE_URL=https://api.groq.com/openai/v1")
            cprint_err("        LLM_BACKEND=custom")
            cprint_err("        LLM_MODEL=openai/gpt-oss-20b")
            sys.exit(1)
    else:
        api_key = None

    if not args.base_url:
        args.base_url = None

    # Create agent (SYSTEM_PROMPT in .env can override the default identity)
    agent_config = AgentConfig(
        model=args.model,
        temperature=0.7,
        streaming=True,
        system_prompt=env_config("SYSTEM_PROMPT", default=AgentConfig().system_prompt),
    )

    try:
        agent = Agent(
            config=agent_config,
            api_key=api_key,
            backend=args.backend,
            base_url=args.base_url,
        )
    except (LLMAuthError, LLMConnectionError, ValueError) as e:
        cprint_err(f"[!] {e}")
        sys.exit(1)

    # ── Shared state ──────────────────────────────────────────────────
    plan_state = PlanState()
    agent.plan_state = plan_state
    notify_hook = NotifyHook()

    # Pass the key through OUR env chain (./.env → ~/.mforege/.env → repo .env).
    # The tool's bare-decouple fallback can't see ~/.mforege/.env, but the
    # welcome text tells users to put EXA_API_KEY exactly there.
    search_tool = ExaSearchTool(api_key=env_config("EXA_API_KEY", default=""))

    # Register tools BEFORE either UI branch — both modes need them.
    # The plain CLI passes its own async confirm (stdin y/n); the full UI
    # overrides with its question-bar callback.
    agent.register_tools(
        CalculatorTool(),
        TimeTool(),
        search_tool,
        *create_system_tools(workspace=workspace, confirm=None,
                             plan_state=plan_state, notify=notify_hook),
    )

    # ── UI or --plain fallback ───────────────────────────────────────
    if getattr(args, "plain", False):
        await run_plain_cli(agent, args, workspace, plan_state, notify_hook,
                            search_tool)
        return

    base_status = f"{args.model} │ {workspace}"
    ui = ChatUI(status_text=base_status)

    # Esc-to-stop: the UI flips the agent's cooperative flag; the handler
    # task is cancelled by the UI itself (the in-flight HTTP wait ends at
    # its next await point, partial answers stay saved).
    ui.set_stop_hook(lambda: agent.request_stop())
    ui.set_session_start()  # session-bar clock: '◆ model · 12m' next to the model

    # Live context meter (like an agent panel): ~tokens used / window %.
    def refresh_context_meter() -> None:
        try:
            used, window = agent.context_usage()
            pct = int(used * 100 / max(1, window))
            meter = f"~{used / 1000:.1f}K ({pct}%)"
            ui.set_status(f"{base_status} │ ctx {meter}")
        except Exception:
            pass  # display must never break the chat

    refresh_context_meter()

    # ── session persistence (resume across restarts) ───────────────
    session_store = SessionStore()
    session = session_store.start()

    # NO auto-continue: every launch starts fresh (like the Codebuff
    # panel). Old chats stay on disk — /sessions lists them, /resume
    # <id> restores one explicitly.

    # Live TODOS block: PlanState notifies → transcript block redraws in
    # place (Codebuff-style checkmarks). Falls back silently if anything
    # goes wrong — the plan itself is more important than the rendering.
    def render_plan_block() -> None:
        try:
            lines = plan_state.render_block()
            if not lines:
                return
            ui.replace_block("TODOS", [
                ("class:plan", " TODOS\n"),
                *[("class:ok" if ln.startswith("✓") else "class:plan", f" {ln}\n")
                  for ln in lines[1:]],
            ])
        except Exception:
            pass

    plan_state.on_change = render_plan_block

    # /sessions + /resume for the full UI (same semantics as plain mode)
    async def handle_ui_sessions_command(user_input, agent, session,
                                         store, ui) -> None:
        parts = user_input.split(maxsplit=1)
        slash = parts[0].lower()
        if slash == "/sessions":
            rows = store.list()
            if not rows:
                ui.append("[Sessions] No saved conversations yet.", style="class:dim")
                return
            ui.append("[Sessions] Recent conversations (newest first):",
                      style="class:title")
            for r in rows:
                ui.append(f"   {r['id']}  ({r['count']} msgs)  {r['title'][:44]}",
                          style="class:dim")
            ui.append("   → /resume <id> to continue one", style="class:dim")
            return
        # /resume [id]
        rows = store.list()
        if not rows:
            ui.append("[Resume] No saved conversations to resume.", style="class:warn")
            return
        target = None
        if len(parts) > 1:
            wanted = parts[1].strip()
            target = next((r for r in rows if r["id"].startswith(wanted)), None)
            if target is None:
                ui.append(f"[Resume] No session matching '{wanted}'. Try /sessions.",
                          style="class:error")
                return
        else:
            target = rows[0]
        stored = store.load(target["id"])
        if stored is None or not stored.messages:
            ui.append("[Resume] That session is empty.", style="class:error")
            return
        agent.restore_session_messages(stored.messages)
        session.id = stored.id
        session.title = stored.title
        session.messages = list(stored.messages)
        ui.append(f"[Resume] Restored '{stored.title}' ({len(stored.messages)} messages).",
                  style="class:ok")
        ui.append("   Context is back — just continue chatting.", style="class:dim")

    # Upgrade banner (24h-cached PyPI check; silent on any failure).
    # Auto-update already handled it at startup; this banner is the fallback
    # for users with MFOREGE_NO_UPDATE=1 or after a failed upgrade.
    update_banner = None if self_update.updates_disabled() else check_for_update(__version__)
    if update_banner:
        ui.append(update_banner, style="class:warn")

    # The full UI re-registers the system tools with its question-bar
    # confirm (async, deadlock-free: tools await async callbacks).
    async def confirm_action(action: str) -> bool:
        return await ui.confirm(action)

    agent.register_tools(
        *create_system_tools(workspace=workspace, confirm=confirm_action,
                             plan_state=plan_state, notify=notify_hook),
    )

    # Agent activity → colored transcript, live. Also pinned one-liner
    # ("✸ read_file(path=settings.py)") above the input so the user always
    # sees which file/command MForege is touching right now.
    def render_activity(event: str, detail: str) -> None:
        try:
            if event == "tool_start":
                ui.append(f"  · {detail}", end="", style="class:dim")
                ui.set_activity(detail)
            elif event == "tool_end":
                style = "class:ok" if "✓" in detail else "class:error"
                ui.append(f" {detail}", style=style)
                ui.set_activity(None)
            elif event == "round":
                ui.append(f"  ── round {detail} ──", style="class:dim")
            elif event == "adapting":
                ui.append(f"  ↻ {detail}", style="class:warn")
            elif event == "reasoning":
                # Streamed thinking tokens → live italic "Thinking" block,
                # replaced in place until the first real answer token.
                ui.stream_thinking(detail)
            elif event == "stopped":
                # Esc reached the agent — end the phase timer here; the
                # transcript note is printed once by handle_message's
                # finally (single source of truth, no double notes).
                ui.set_phase(None)
        except Exception:
            pass  # display must never break the chat

    agent.on_activity = render_activity

    # Successful mutations → colored diffs in the flow (like my edits)
    def render_notify(event: str, detail: str) -> None:
        try:
            if event == "diff":
                ui.append("  ✓ Applied:", style="class:ok")
                for ln in detail.split("\n"):
                    if ln.startswith("+"):
                        ui.append("  " + ln, style="class:add")
                    elif ln.startswith("-"):
                        ui.append("  " + ln, style="class:del")
                    else:
                        ui.append("  " + ln, style="class:dim")
            elif event == "cmd":
                ui.append(f"  $ {detail}", style="class:dim")
        except Exception:
            pass

    notify_hook.callback = render_notify

    # ── Startup greeting (colored, in-UI) ─────────────────────────────
    # Welcome box FIRST (like my banner first), then the notices — the
    # agent's identity lands before any environment hints.
    ui.append(boxed([
        "MForege ✦ your personal AI agent",
        f"backend: {args.backend} │ model: {args.model}",
        f"workspace: {workspace}",
        "",
        "Nice to meet you! 👋",
        "Type / for the command menu · Enter=send",
        "Shift/Alt+Enter=newline · Esc=stop · Ctrl+T=thinking · Ctrl+C=quit",
    ]), style="class:ok")
    ui.append("")
    if not search_tool.is_configured:
        ui.append("[i] Optional: web search is off. Get a free Exa key at https://dashboard.exa.ai",
                  style="class:warn")
        ui.append("    and add EXA_API_KEY=... to ~/.mforege/.env to enable it. Everything else works.",
                  style="class:warn")
    if os.environ.get("TERM_PROGRAM") == "vscode":
        ui.append("[i] VS Code: Shift+Enter needs a one-time keybinding — /vscode-hint", style="class:dim")
        ui.append("    Use Alt+Enter meanwhile.", style="class:dim")

    # ── Codebuff-parity slash commands ────────────────────────────
    async def handle_buffuff_command(user_input: str) -> None:
        """Freebuff-style commands: /diagnostics /review /copy /export
        /theme:toggle /reasoning /queue /new /history /bash /byok /feedback
        /interview. Keep behavior identical across plain and UI."""
        nonlocal session  # /new replaces the session object
        slash = user_input.split()[0].lower()
        if slash == "/diagnostics":
            used, window = agent.context_usage()
            pct = int(used * 100 / max(1, window))
            ui.append("[Diagnostics]", style="class:title")
            ui.append(f"  version        {__version__}", style="class:dim")
            ui.append(f"  backend        {args.backend}", style="class:dim")
            ui.append(f"  model          {args.model}", style="class:dim")
            ui.append(f"  workspace      {workspace}", style="class:dim")
            ui.append(f"  tools          {len(agent.tools.list())} registered", style="class:dim")
            ui.append(f"  context        ~{used / 1000:.1f}K / {window // 1000}K ({pct}%)", style="class:dim")
            ui.append(f"  history        {len(agent.memory.messages)} messages, summary {'yes' if agent.memory.summary else 'no'}", style="class:dim")
            ui.append(f"  sessions       {len(session_store.list())} saved", style="class:dim")
            return
        if slash == "/review":
            msgs = [m for m in agent.memory.messages if m.role in ("user", "assistant")]
            if not msgs:
                ui.append("[Review] Nothing in this conversation yet.", style="class:warn")
                return
            ui.append("[Review] Changes made via tools this conversation:", style="class:title")
            shown = False
            for ln in ui._segments:
                if ln.get("kind") != "text":
                    continue
                for style, text in ln["lines"]:
                    t = text.strip()
                    if t.startswith("·") or t.startswith("$") or t.startswith("✓ Applied"):
                        ui.append(f"  {t}", style="class:dim")
                        shown = True
            if not shown:
                ui.append("  No tool activity yet — nothing to review.", style="class:dim")
            return
        if slash == "/copy":
            lines = []
            for m in agent.memory.messages:
                who = "You" if m.role == "user" else "MForege"
                lines.append(f"{who}: {m.content}")
            text = "\n".join(lines)
            try:
                import pyperclip # type: ignore
                pyperclip.copy(text)
                ui.append("[Copy] Conversation copied to clipboard.", style="class:ok")
            except Exception:
                import tempfile
                f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
                f.write(text)
                f.close()
                ui.append(f"[Copy] Clipboard unavailable — saved to {f.name}", style="class:warn")
            return
        if slash == "/export":
            import json as _json
            import time as _time
            fname = os.path.join(os.getcwd(), f"mforege-export-{_time.strftime('%Y%m%d-%H%M%S')}.json")
            try:
                with open(fname, "w", encoding="utf-8") as f:
                    _json.dump({
                        "exported": _time.strftime("%Y-%m-%d %H:%M:%S"),
                        "model": args.model,
                        "messages": agent.export_session_messages(),
                    }, f, ensure_ascii=False, indent=2)
                ui.append(f"[Export] Full conversation → {fname}", style="class:ok")
            except Exception as e:
                ui.append(f"[Export] Failed: {e}", style="class:error")
            return
        if slash == "/theme:toggle":
            ui.toggle_theme()
            cur = "dark" if ui._dark_theme else "light"
            ui.append(f"[Theme] Switched to {cur} mode.", style="class:ok")
            return
        if slash == "/reasoning":
            parts = user_input.split(maxsplit=1)
            level = parts[1].strip().lower() if len(parts) > 1 else ""
            if level not in ("low", "high", "max"):
                ui.append("[Reasoning] Usage: /reasoning low | high | max", style="class:warn")
                ui.append(f"   current: {getattr(agent.config, 'reasoning_effort', 'high')}", style="class:dim")
                return
            agent.config.reasoning_effort = level
            ui.append(f"[Reasoning] Set to {level}.", style="class:ok")
            return
        if slash == "/model":
            parts = user_input.split(maxsplit=1)
            arg = parts[1].strip() if len(parts) > 1 else ""
            styles = {"title": "class:title", "dim": "class:dim",
                      "ok": "class:ok", "error": "class:error"}
            for line, kind in _run_model_command(arg, agent):
                ui.append(line, style=styles.get(kind, ""))
            refresh_context_meter()
            return
        if slash == "/queue":
            ui.append("[Queue] Messages sent while the agent works are queued —", style="class:dim")
            ui.append("   '✓ Queued' appears, and each runs in order after the current turn.", style="class:dim")
            return
        if slash == "/new":
            agent.clear_memory()
            session = session_store.start()
            ui.clear()
            ui.append("[*] New conversation started — fresh context.", style="class:ok")
            return
        if slash == "/history":
            rows = session_store.list()
            if not rows:
                ui.append("[History] No saved conversations yet.", style="class:dim")
                return
            ui.append("[History] Recent conversations (newest first):", style="class:title")
            for r in rows:
                ui.append(f"   {r['id']}  ({r['count']} msgs)  {r['title'][:44]}", style="class:dim")
            ui.append("   → /resume <id> to continue one — /new to start fresh.", style="class:dim")
            return
        if slash == "/interview":
            ui.append("[Interview] Tell me your goal in one line — I'll ask targeted", style="class:title")
            ui.append("    questions one at a time to pin down scope, constraints and", style="class:dim")
            ui.append("    trade-offs, then turn it into a step-by-step plan.", style="class:dim")
            return
        if slash == "/bash":
            parts = user_input.split(maxsplit=1)
            cmd = parts[1].strip() if len(parts) > 1 else ""
            if not cmd:
                ui.append("[Bash] Usage: /bash <command>", style="class:warn")
                ui.append("    Runs with the same safety guards and confirmation as the", style="class:dim")
                ui.append("    agent's run_command tool.", style="class:dim")
                return
            runner = agent.tools.get("run_command")
            if runner is None:
                ui.append("[Bash] run_command tool unavailable.", style="class:error")
                return
            result = await runner.execute(command=cmd)
            err = result.startswith("Error")
            for ln in result.split("\n"):
                ui.append(ln, style="class:error" if err else None)
            return
        if slash == "/byok":
            ui.append("[BYOK] Bring your own key — edit your config file:", style="class:title")
            ui.append(f"    {_HOME_CONFIG_PATH}", style="class:dim")
            ui.append("    Set OPENAI_API_KEY / base_url / model (any OpenAI-compatible", style="class:dim")
            ui.append("    endpoint works, including a free Groq key). A ./.env in the", style="class:dim")
            ui.append("    current folder overrides it. Or rerun: mforege --setup", style="class:dim")
            return
        if slash == "/feedback":
            ui.append("[Feedback] MForege is your project — ideas go straight to the", style="class:title")
            ui.append("    source: https://github.com/munjurdev/MForege/issues 💚", style="class:dim")
            return
        ui.append(f"[?] Unknown command {slash} — /help for the list.", style="class:warn")

    # ── Message handler (runs inside the UI event loop) ───────────────
    async def handle_message(user_input: str) -> None:
        nonlocal session
        # The user's message is echoed by the UI at SUBMIT time
        # (echo_submission) — chat-app style, so a message sent while the
        # agent is still working appears immediately and in order.

        low = user_input.lower().strip()

        # Slash commands
        if user_input.startswith("/"):
            slash = user_input.split()[0].lower()
            if slash == "/help":
                ui.append("[Commands]", style="class:title")
                for cmd, desc in [
                    ("/plan", "Show the current mission plan"),
                    ("/tools", "List registered tools"),
                    ("/new", "Clear the conversation and start a new chat"),
                    ("/history", "Browse and resume past conversations"),
                    ("/sessions", "List past conversations"),
                    ("/resume [id]", "Resume a recent chat (default: latest)"),
                    ("/diagnostics", "Version, model, context usage, sessions"),
                    ("/review", "Review changes made this conversation"),
                    ("/copy", "Copy the conversation to clipboard"),
                    ("/export", "Write the conversation to a .json file"),
                    ("/bash", "Run a shell command with the agent's safety guards"),
                    ("/byok", "Show where to configure your API key / model"),
                    ("/queue", "Message queueing info"),
                    ("/interview", "Guided Q&A to spec a task before building"),
                    ("/theme:toggle", "Toggle light/dark mode"),
                    ("/reasoning", "Thinking effort: low / high / max"),
                    ("/feedback", "Share feedback about MForege"),
                    ("/vscode-hint", "Fix Shift+Enter for VS Code's terminal"),
                    ("/clear", "Reset conversation + transcript (like /new)"),
                    ("/exit", "Quit"),
                ]:
                    ui.append(f"  {cmd:<10} {desc}", style="class:dim")
                return
            if slash in ("/diagnostics", "/review", "/copy", "/export",
                         "/theme:toggle", "/reasoning", "/queue", "/new",
                         "/history", "/bash", "/byok", "/feedback", "/interview",
                         "/model"):
                await handle_buffuff_command(user_input)
                return
            if slash == "/plan":
                rendered = plan_state.render()
                ui.append(rendered if rendered else "No active plan.", style="class:plan")
                return
            if slash == "/vscode-hint":
                ui.append("[VS Code Shift+Enter fix] (one-time, 30 seconds):", style="class:title")
                ui.append("  1. Ctrl+Shift+P → 'Preferences: Open Keyboard Shortcuts (JSON)'", style="class:dim")
                ui.append('  2. Add inside the [ ] array:', style="class:dim")
                ui.append('     { "key": "shift+enter",', style="class:dim")
                ui.append('       "command": "workbench.action.terminal.sendSequence",', style="class:dim")
                ui.append('       "args": { "text": "\\u001b\\r" },', style="class:dim")
                ui.append('       "when": "terminalFocus" }', style="class:dim")
                ui.append("  3. Save, then restart VS Code completely.")
                ui.append("  → VS Code then sends ESC+CR for Shift+Enter — which MForege already")
                ui.append("    treats as a newline. (Same fix Claude Code recommends.)")
                return
            if slash in ("/sessions", "/resume"):
                await handle_ui_sessions_command(user_input, agent, session,
                                                 session_store, ui)
                return
            # /tools /new /history /bash /byok /feedback /clear /exit → plain commands
            user_input = user_input[1:]
            low = user_input.lower()

        # Plain commands
        if low in ("exit", "quit"):
            ui.append("Goodbye! 👋", style="class:ok")
            ui.exit()
            return
        if low == "clear":
            agent.clear_memory()
            ui.clear()
            ui.append("[*] Conversation cleared", style="class:ok")
            return
        if low == "new":
            agent.clear_memory()
            session = session_store.start()
            ui.clear()
            ui.append("[*] New conversation started — fresh context.", style="class:ok")
            return
        if low in ("memory", "forget", "forget all", "forget everything"):
            ui.append("[i] MForege keeps only conversation context now (like an agent):",
                      style="class:dim")
            ui.append("    auto-continues your last chat; /sessions + /resume manage history.",
                      style="class:dim")
            return
        if low == "tools":
            ui.append("[Tools] Available tools:", style="class:title")
            for tool in agent.tools.list():
                ui.append(f"   - {tool.name}: {tool.description[:80]}", style="class:dim")
            return

        # Chat (streaming). The reply renders as LIVE MARKDOWN — bold,
        # code blocks, headers and lists appear formatted while streaming
        # (like a coding agent's panel; no raw ** or ``` ever visible).
        # The phase timer starts here and KEEPS running through
        # "responding" — one clock per turn; Esc stops while active.
        ui._stop_requested = False
        ui.set_status("thinking…")
        ui.set_phase("thinking")
        reply_text = ""
        try:
            response = await agent.chat(user_input, stream=True)

            if hasattr(response, "__aiter__"):
                first = True
                async for chunk in response:
                    if first:
                        ui.set_status("responding…")
                        ui.set_phase("responding")  # label swap, clock keeps running
                        ui.finish_thinking()   # collapse the Thinking block
                        ui.begin_reply()       # open the live markdown block
                        first = False
                    reply_text += chunk
                    ui.stream_reply(chunk)
            else:
                ui.finish_thinking()
                ui.begin_reply()
                reply_text = response
                ui.stream_reply(response)
        except (LLMAuthError, LLMConnectionError, LLMRateLimitError,
                LLMResponseError) as e:
            # Backend problems already carry a human-friendly message from
            # the LLM client (key rejected, Ollama down, rate limit, 413…).
            # Show it cleanly instead of a raw traceback-y repr.
            reply_text = ""
            ui.append(f"[!] {e}", style="class:error")
            ui.append("    (Check /byok for config, or /model to switch models.)",
                      style="class:dim")
        finally:
            ui.end_reply()        # close the markdown block (final render stays)
            ui.finish_thinking()  # safety: never leave a dangling block
            ui.set_phase(None)    # stop the live timer, disarm Esc
            if ui._stop_requested:
                # Esc was pressed this turn — confirm it in the transcript.
                # (If the agent stopped mid-stream this is the only note;
                # if it had already finished, it still confirms the key.)
                ui._stop_requested = False
                ui.append("[■] Stopped — tell me what to do next.", style="class:warn")
            refresh_context_meter()  # tokens grew — update the meter
            status = plan_state.progress_line()
            if status:
                cur = ui._status
                sep = " │ " if "ctx" in cur else ""
                ui.set_status(f"{cur}{sep}{status}")

        # save the exchange (incremental — a crash loses at most one message)
        try:
            session.append(user_input, reply_text)
            session_store.save(session)
        except Exception:
            pass

        status = plan_state.progress_line()
        if status:
            ui.append(status, style="class:plan")
        ui.append("")

    # ── Run until the user quits ──────────────────────────────────────
    try:
        try:
            await ui.run(handle_message)
        except KeyboardInterrupt:
            ui.append("\n[*] Interrupted — goodbye! 👋", style="class:warn")
    finally:
        # Remember the last exchange before shutting down
        try:
            await agent.flush_memory()
        except Exception:
            pass
        # Cleanly release the HTTP connection pool
        try:
            await agent.llm.client.close()
        except Exception:
            pass


def cli() -> None:
    """Entry point for the `mforege` console command (installed via pip)."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")


if __name__ == "__main__":
    cli()
