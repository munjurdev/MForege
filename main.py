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
import os
import sys

from decouple import config as _cwd_config, Config, RepositoryEnv

# Windows consoles default to cp1252, which cannot encode emojis the model
# may output. Force UTF-8 (with safe replacement) so printing never crashes.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from app.agent import Agent, AgentConfig
from app.agent.tools import CalculatorTool, TimeTool
from app.tools import ExaSearchTool
from app.tools.system_tools import create_system_tools, PlanState, NotifyHook
from app.ui import ChatUI
from app.llm.client import (
    LLMAuthError,
    LLMConnectionError,
)


def cprint_err(text: str) -> None:
    """Plain terminal error print (used only before the UI starts)."""
    print(f"\033[91m{text}\033[0m", file=sys.stderr)


# ── .env resolution ────────────────────────────────────────────────
# `mforege` can be launched from ANY directory (e.g., the user's Django
# project). decouple normally only reads ./.env from the current working
# directory — fall back to MForege's own .env (next to main.py).
_MFOREGE_ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    _root_config = Config(RepositoryEnv(os.path.join(_MFOREGE_ROOT, ".env")))
except Exception:
    _root_config = None


def env_config(key: str, default: str = "") -> str:
    """Read settings: workspace .env first, then MForege's own .env."""
    if os.path.exists(".env"):
        return _cwd_config(key, default=default)
    if _root_config is not None:
        return _root_config(key, default=default)
    return default


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mforege",
        description="MForege — your personal AI agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With Groq (free):
    python main.py

  # With Ollama (local):
    python main.py --backend ollama --model llama3

  # With OpenAI:
    OPENAI_API_KEY=sk-xxx python main.py

  # Work inside a specific project folder:
    python main.py --workspace "C:\\path\\to\\project"
        """,
    )
    parser.add_argument("--backend", choices=["openai", "ollama", "custom"],
                        default=env_config("LLM_BACKEND", default="openai"))
    parser.add_argument("--model", default=env_config("LLM_MODEL", default="gpt-4o-mini"))
    parser.add_argument("--base-url", default=env_config("BASE_URL", default=""))
    parser.add_argument("--workspace", default=env_config("WORKSPACE", default="."),
                        help="Project folder MForege may read/write (default: current directory)")
    args = parser.parse_args()

    # Resolve and validate the workspace early (human decides the sandbox root)
    workspace = os.path.abspath(args.workspace)
    if not os.path.isdir(workspace):
        cprint_err(f"[!] Workspace does not exist: {workspace}")
        cprint_err("    Create it first, or pass a valid --workspace path.")
        sys.exit(1)

    # Get API key based on backend
    api_key: str | None
    if args.backend == "openai":
        api_key = env_config("OPENAI_API_KEY", default="")
        if not api_key:
            cprint_err("[!] OpenAI backend requires OPENAI_API_KEY")
            cprint_err("    Or use the free local backend: python main.py --backend ollama")
            sys.exit(1)
    elif args.backend == "ollama":
        api_key = None
    elif args.backend == "custom":
        api_key = env_config("API_KEY", default="")
        if not api_key:
            cprint_err("[!] Custom backend requires API_KEY")
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
    ui.append("  /help for commands. Enter=send, Alt+Enter=newline, Ctrl+C=quit.",
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
