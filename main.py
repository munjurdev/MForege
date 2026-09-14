"""
MForege — AI Agent CLI
======================
IDE-style chat UI (prompt_toolkit):
  - colored scrolling transcript on top
  - input box pinned at the bottom
  - status bar (model | workspace | plan progress)

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

from decouple import config as _cwd_config, Config, RepositoryEnv

# Windows consoles default to cp1252, which cannot encode emojis the model
# may output. Force UTF-8 (with safe replacement) so printing never crashes.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from app.agent import Agent, AgentConfig
from app import __version__
from app.agent.tools import CalculatorTool, TimeTool
from app.tools import ExaSearchTool
from app.tools.system_tools import create_system_tools, PlanState, NotifyHook
from app.ui import ChatUI
from app.update_check import check_for_update
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
#   1. ./.env            (current folder — per-project override)
#   2. ~/.mforege/.env   (global config written by the setup wizard)
#   3. <MForege repo>/.env (developer install)
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
    """Read settings: current .env → ~/.mforege/.env → MForege's own .env."""
    if os.path.exists(".env"):
        return _cwd_config(key, default=default)
    if _home_config is not None:
        return _home_config(key, default=default)
    if _root_config is not None:
        return _root_config(key, default=default)
    return default


GROQ_BASE_URL = "https://api.groq.com/openai/v1"


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


async def run_setup_wizard() -> None:
    """One-time guided setup (freebuff-style): pick a brain, paste a key, done."""
    print()
    print("=" * 58)
    print("  Welcome to MForege!  One-time setup (about 30 seconds)")
    print("=" * 58)
    print()
    print("  Where should MForege get its brain?")
    print("   [1] Groq    free cloud API — fastest option (recommended)")
    print("   [2] Ollama  free, 100% local (needs https://ollama.com installed)")
    print("   [3] OpenAI  paid API (uses OPENAI_API_KEY)")
    print("   [4] Custom  any OpenAI-compatible endpoint")
    print()
    choice = (input("  Choose 1-4 [1]: ").strip() or "1")

    if choice == "2":
        lines = ["LLM_BACKEND=ollama", "LLM_MODEL=llama3"]
        print("\n  Ollama selected — make sure it's running (`ollama serve`).")
    elif choice == "3":
        key = getpass.getpass("  Paste your OPENAI_API_KEY (hidden): ").strip()
        if not key:
            print("  [!] No key entered — setup cancelled.")
            sys.exit(1)
        model = (input("  Model [gpt-4o-mini]: ").strip() or "gpt-4o-mini")
        key = await _validate_or_reask(key, "", model, "openai", "OPENAI_API_KEY")
        lines = ["LLM_BACKEND=openai", f"OPENAI_API_KEY={key}", f"LLM_MODEL={model}"]
    elif choice == "4":
        key = getpass.getpass("  Paste your API key (hidden): ").strip()
        base = (input(f"  Base URL [{GROQ_BASE_URL}]: ").strip() or GROQ_BASE_URL)
        model = input("  Model: ").strip()
        if not key or not model:
            print("  [!] Key and model are required — setup cancelled.")
            sys.exit(1)
        key = await _validate_or_reask(key, base, model, "custom", "API key")
        lines = ["LLM_BACKEND=custom", f"API_KEY={key}", f"BASE_URL={base}", f"LLM_MODEL={model}"]
    else:
        print("\n  Get a FREE key at https://console.groq.com (no credit card needed).")
        key = getpass.getpass("  Paste your Groq API key (gsk_..., hidden): ").strip()
        if not key:
            print("  [!] No key entered — setup cancelled.")
            sys.exit(1)
        model = (input("  Model [openai/gpt-oss-20b]: ").strip() or "openai/gpt-oss-20b")
        key = await _validate_or_reask(key, GROQ_BASE_URL, model, "custom", "Groq API key")
        lines = [
            "LLM_BACKEND=custom",
            f"API_KEY={key}",
            f"BASE_URL={GROQ_BASE_URL}",
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


async def main(argv: list[str] | None = None) -> None:
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
    args = parser.parse_args()

    # Settings resolve: ./.env → ~/.mforege/.env → repo .env. If the user has
    # configured nothing anywhere, launch the guided setup (freebuff-style).
    _no_config = not (
        os.path.exists(".env")
        or os.path.exists(_HOME_CONFIG_PATH)
        or os.path.exists(os.path.join(_MFOREGE_ROOT, ".env"))
    )
    if args.setup or _no_config:
        await run_setup_wizard()
        if args.setup:  # explicit --setup: exit after saving
            return

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

    search_tool = ExaSearchTool()

    # ── UI ────────────────────────────────────────────────────────────
    base_status = f"{args.model} │ {workspace}"
    ui = ChatUI(status_text=base_status)

    # Upgrade banner (24h-cached PyPI check; silent on any failure)
    update_banner = check_for_update(__version__)
    if update_banner:
        ui.append(update_banner, style="class:warn")

    # Confirmation happens in the UI question bar (async, deadlock-free:
    # tools now await async confirm callbacks on the same running loop).
    async def confirm_action(action: str) -> bool:
        return await ui.confirm(action)

    agent.register_tools(
        CalculatorTool(),
        TimeTool(),
        search_tool,
        *create_system_tools(workspace=workspace, confirm=confirm_action,
                             plan_state=plan_state, notify=notify_hook),
    )

    # Agent activity → colored transcript, live
    def render_activity(event: str, detail: str) -> None:
        try:
            if event == "tool_start":
                ui.append(f"  · {detail}", end="", style="class:dim")
            elif event == "tool_end":
                style = "class:ok" if "✓" in detail else "class:error"
                ui.append(f" {detail}", style=style)
            elif event == "round":
                ui.append(f"  ── round {detail} ──", style="class:dim")
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
    name = agent.long_term_memory.guess_name()
    fact_count = agent.long_term_memory.count
    if not search_tool.is_configured:
        ui.append("[!] Web search disabled — set EXA_API_KEY in .env to enable it",
                  style="class:warn")
    ui.append("  MForege — your personal AI agent ✦", style="class:title")
    ui.append(f"  backend: {args.backend} │ model: {args.model}", style="class:dim")
    ui.append(f"  workspace: {workspace}", style="class:dim")
    if name:
        ui.append(f"  Welcome back, {name}! 👋  (I remember {fact_count} "
                  f"thing{'s' if fact_count != 1 else ''})", style="class:ok")
    elif fact_count:
        ui.append(f"  Welcome back! 👋  (I remember {fact_count} "
                  f"thing{'s' if fact_count != 1 else ''})", style="class:ok")
    else:
        ui.append("  Nice to meet you! 👋 I'm MForege.", style="class:ok")
    ui.append("  /help for commands. Enter=send, Shift+Enter=newline, Ctrl+C=quit.",
              style="class:dim")
    ui.append("")

    # ── Message handler (runs inside the UI event loop) ───────────────
    async def handle_message(user_input: str) -> None:
        ui.append(f"You: {user_input}", style="class:you")

        low = user_input.lower().strip()

        # Slash commands
        if user_input.startswith("/"):
            slash = user_input.split()[0].lower()
            if slash == "/help":
                ui.append("[Commands]", style="class:title")
                for cmd, desc in [
                    ("/plan", "Show the current mission plan"),
                    ("/tools", "List registered tools"),
                    ("/memory", "Show what MForege remembers"),
                    ("/forget", "Wipe long-term memory"),
                    ("/clear", "Reset conversation + transcript"),
                    ("/exit", "Quit"),
                ]:
                    ui.append(f"  {cmd:<10} {desc}", style="class:dim")
                return
            if slash == "/plan":
                rendered = plan_state.render()
                ui.append(rendered if rendered else "No active plan.", style="class:plan")
                return
            # /tools /memory /forget /clear /exit → handled as plain commands
            user_input = slash[1:]
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
        if low == "memory":
            facts = agent.long_term_memory.all()
            if facts:
                ui.append(f"[Memory] What MForege remembers about you ({len(facts)}):",
                          style="class:title")
                for i, fact in enumerate(facts, 1):
                    ui.append(f"   {i}. {fact}")
            else:
                ui.append("[Memory] Nothing remembered yet — chat and MForege will learn!",
                          style="class:warn")
            return
        if low in ("forget", "forget all", "forget everything"):
            agent.clear_long_term_memory()
            ui.append("[*] Long-term memory wiped.", style="class:ok")
            return
        if low == "tools":
            ui.append("[Tools] Available tools:", style="class:title")
            for tool in agent.tools.list():
                ui.append(f"   - {tool.name}: {tool.description[:80]}", style="class:dim")
            return

        # Chat (streaming). The pending-extraction happens inside chat();
        # show "thinking" while we wait for the first token.
        ui.set_status("thinking…")
        ui.append("MForege: ", end="", style="class:assistant")
        try:
            response = await agent.chat(user_input, stream=True)

            if hasattr(response, "__aiter__"):
                first = True
                async for chunk in response:
                    if first:
                        ui.set_status("responding…")
                        first = False
                    ui.append(chunk, end="")
                ui.append("")
            else:
                ui.append(response)
        finally:
            status = plan_state.progress_line()
            ui.set_status(f"{base_status} │ {status}" if status else base_status)

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
