"""
AI Agent Implementation
========================
A conversational AI agent with support for multiple backends:
- OpenAI API (paid)
- Ollama (free, local)
- Other OpenAI-compatible APIs

Features:
- Full tool-use loop (tool results are sent back to the model)
- Streaming responses (including tool calls)
- Conversation memory management
"""

import asyncio
import json
import time
from typing import Optional, AsyncGenerator, Any, Callable, List, Dict, Tuple, Union
from pydantic import BaseModel

from ..llm import LLMClient
from .memory import ConversationMemory, Message
from .tools import Tool, ToolRegistry

# How many model<->tool round trips a single chat() call may make by default
# (configurable per-agent via AgentConfig.max_tool_rounds or MAX_TOOL_ROUNDS env)
MAX_TOOL_ROUNDS = 10


class AgentConfig(BaseModel):
    """Configuration for the AI Agent"""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_tool_rounds: int = MAX_TOOL_ROUNDS
    # Model context window (tokens). Used for the live context meter and
    # auto-condensation. Conservative default; Groq's gpt-oss-20b serves 131k.
    context_window: int = 131072
    # Reasoning effort for reasoning models (low/high/max). Forwarded to
    # backends that support it; ignored gracefully by those that don't.
    reasoning_effort: str = "high"
    system_prompt: str = """You are MForege, a personal AI assistant with a warm, playful personality.

# Identity
You are NOT ChatGPT, Claude, or any other assistant — never claim another identity, even if asked.
If someone asks who you are, say you are MForege, the user's personal AI assistant.

# Personality
- Friendly and upbeat: chat like a fun, smart friend, not a corporate robot.
- Witty: enjoy a clever one-liner or light joke when it fits — but the answer always comes first.
  Never sacrifice clarity for comedy.
- Emojis: use them like a touch of sparkle ✨ — a few where they fit the mood (usually 1-3 per reply),
  not on every line. Skip them in code blocks, error messages, and serious or sensitive topics.
- Honest and humble: if you don't know something, say so with charm and offer to figure it out
  (you have tools for that!).

# Planning for multi-step tasks
For tasks with 3+ distinct steps (scaffolding, big refactors, "then fix, then test" chains),
create a visible plan first with the todo_plan tool (action "set"), keep it updated as you go
(mark steps done immediately after finishing them), and call "clear" when the mission ends.
Don't use it for simple one-shot questions.

# Response control (very important)
- Match the reply to the question. Casual chat ("give a biscuit?", "sup?") gets a short, playful
  answer — one or two sentences, not an essay.
- Simple factual question → direct answer first; extra detail only if it clearly helps.
- Big requests ("build a project", "teach me X") → give a brief overview (a few bullets) and ask
  which part to dive into. NEVER dump a giant tutorial unprompted.
- Ambiguous question (e.g., "what is apple?" — fruit 🍎 or company 💻?) → ask ONE short clarifying
  question, or answer the most likely meaning and offer the other.
- No padding or filler conclusions ("I hope this helps!"). Answer, then stop.

# Acting on the user's machine (agentic tools)
You have tools to inspect and modify the user's project: list_files, read_file, run_command,
create_file, edit_file, search_code, glob_files.
- Search-first workflow: before touching anything, understand the project. Use glob_files to
  discover files (e.g., '**/*.py'), search_code to find where things are defined or where an
  error string originates, and read_file (with offset/limit for big files) to study the code.
- You operate inside ONE workspace folder, chosen by the user at startup (shown in the header).
  All tool paths are relative to it. If the user names a folder outside this workspace, DO NOT
  say you "can't" or call it a sandbox limitation — instead say: "restart me with
  --workspace <that folder> and I'll work there", then offer to proceed in the current
  workspace meanwhile.
- Relative paths in conversation (e.g., "in the town folder") are fine — tools resolve them
  inside the workspace. Absolute paths outside it are refused by design.
- Before creating anything the user didn't fully specify (project name, location, stack), ask ONE
  short clarifying question or propose sensible defaults and ask "shall I proceed?". NEVER invent
  details and act on them silently.
- Prefer list_files/read_file to ground answers in the actual project before guessing.
- Use create_file for new files (small plan first for big scaffolds); use edit_file for existing
  ones — read the file first, then replace an exact unique snippet.
- Every write/edit asks the user for confirmation — that's by design, don't try to bypass it
  (e.g., don't echo file content through run_command to write files).
- When running commands, explain briefly what you're about to run and why. If the user cancels,
  accept it gracefully and ask what they'd prefer.
- Long build/install commands may take a while — warn the user before starting them.

# Tools
- Prefer tools when they make the answer accurate: web_search for anything current or uncertain,
  calculator for math, get_current_time for dates/times.
- When using a tool, briefly say what you're doing (e.g., "Crunching the numbers... 🧮").
- If a needed tool is unavailable (e.g., search not configured), say so honestly instead of guessing.
- Maintain context across the conversation and offer helpful follow-ups — briefly."""
    streaming: bool = True


class Agent:
    """
    AI Agent with tool use capabilities.

    Supports multiple LLM backends:
    - openai: OpenAI API (requires API key)
    - ollama: Local Ollama (free, no API key needed)
    - custom: Any OpenAI-compatible API

    Features:
    - Full tool loop: when the model requests a tool, the tool runs and its
      result is sent back so the model can incorporate it into the answer
    - Supports streaming responses (tool calls included)
    - Maintains conversation memory
    """

    def __init__(self, config: Optional[AgentConfig] = None, api_key: Optional[str] = None,
                 backend: str = "openai", base_url: Optional[str] = None):
        self.config = config or AgentConfig()
        self.llm = LLMClient(
            backend=backend,
            api_key=api_key,
            base_url=base_url,
            model=self.config.model
        )
        self.memory = ConversationMemory()
        self.tools = ToolRegistry()
        self.on_activity: Optional[Callable[[str, str], None]] = None  # (event, detail) -> UI hook

    def _emit(self, event: str, detail: str = "") -> None:
        """Notify the UI of agent activity (tool calls, rounds). Never raises."""
        if self.on_activity is None:
            return
        try:
            self.on_activity(event, detail)
        except Exception:
            pass

    def add_tool(self, tool: Tool) -> None:
        """Register a tool for the agent to use"""
        self.tools.register(tool)

    def register_tools(self, *tools: Tool) -> None:
        """Register multiple tools at once"""
        for tool in tools:
            self.add_tool(tool)

    async def chat(self, message: str, stream: Optional[bool] = None) -> Union[str, AsyncGenerator[str, None]]:
        """
        Send a message to the agent and get a response.

        Args:
            message: User's message
            stream: Whether to stream the response (default: use config setting)

        Returns:
            If streaming: async generator yielding text chunks
            If not streaming: complete response string
        """
        self.memory.add(Message(role="user", content=message))

        stream_mode = stream if stream is not None else self.config.streaming

        if stream_mode:
            return self._stream_response(user_message=message)

        reply = await self._get_response()
        return reply

    async def flush_memory(self) -> None:
        """Kept for API compatibility (no extraction happens anymore)."""
        return

    # ── session persistence (resume across restarts) ─────────────────

    def export_session_messages(self) -> List[Dict]:
        """OpenAI-format history for saving a session."""
        return self.memory.to_openai_format()

    def restore_session_messages(self, messages: List[Dict]) -> None:
        """
        Replace conversation history with a saved session's messages
        (OpenAI format). Only role/content pairs are accepted; anything
        malformed is skipped. Condensed summary is cleared — the restored
        verbatim history supersedes it.
        """
        clean: List[Message] = []
        for m in messages or []:
            try:
                role = m.get("role")
                content = m.get("content")
                if role in ("user", "assistant") and isinstance(content, str) and content:
                    clean.append(Message(role=role, content=content))
            except Exception:
                continue
        self.memory.messages = clean
        self.memory.summary = ""

    async def _process_pending_memory(self) -> None:
        """No-op placeholder (kept for API compatibility). Cross-session
        persistence now lives entirely in the session store; conversation
        continuity comes from auto-continue + auto-condense."""
        return

    def _build_api_messages(self) -> List[Dict]:
        """Build the message list sent to the API (system prompt + plan + history)"""
        system_prompt = self.config.system_prompt
        plan = getattr(self, "plan_state", None)
        if plan is not None and plan.steps:
            system_prompt += (
                "\n\n# Current mission plan (use todo_plan to update it as you progress)\n"
                + plan.render()
            )
        msgs: List[Dict] = [{"role": "system", "content": system_prompt}]
        # Condensed history (auto-summary of turns folded out of context —
        # like a session summary, so long missions never lose their start)
        if self.memory.summary:
            msgs.append({
                "role": "system",
                "content": "# Earlier conversation (condensed)\n" + self.memory.summary,
            })
        msgs.extend(self.memory.to_openai_format())
        return msgs

    def _tools_schema(self) -> Optional[List[Dict]]:
        return self.tools.to_openai_schema() if self.tools.tools else None

    def context_usage(self) -> tuple[int, int]:
        """(estimated tokens in next request, model context window size)."""
        return self.memory.token_estimate(), self.config.context_window

    # ── Non-streaming path ─────────────────────────────
    async def _get_response(self) -> str:
        """Run the tool loop until the model produces a final text answer"""
        messages = self._build_api_messages()
        # Auto-condense BEFORE sending if approaching the context window
        self.memory.maybe_condense(self.config.context_window)
        tools_schema = self._tools_schema()

        for round_num in range(1, self.config.max_tool_rounds + 1):
            if round_num > 1:
                self._emit("round", f"{round_num}/{self.config.max_tool_rounds}")
            response = await self.llm.create_chat_completion(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                tools=tools_schema,
                tool_choice="auto" if tools_schema else None,
                config=self.config,
            )

            msg = response.choices[0].message

            if not msg.tool_calls:
                content = msg.content or ""
                self.memory.add(Message(role="assistant", content=content))
                return content

            # Record the assistant's tool-call request, run the tools,
            # then feed the results back to the model.
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })
            tool_results = await self._run_tool_calls(msg.tool_calls)
            messages.extend(tool_results)

        # Model kept requesting tools past the limit; force a final answer
        response = await self.llm.create_chat_completion(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            config=self.config,
        )
        content = response.choices[0].message.content or ""
        self.memory.add(Message(role="assistant", content=content))
        return content

    # ── Streaming path ─────────────────────────────────

    async def _stream_response(self, user_message: str = "") -> AsyncGenerator[str, None]:
        """
        Stream a response, handling tool calls if they occur.

        If the model requests tools mid-stream, the tools run and a new
        streamed completion is started; the generator yields the final
        text answer (streamed chunk by chunk).
        """
        messages = self._build_api_messages()
        # Auto-condense BEFORE sending if approaching the context window
        self.memory.maybe_condense(self.config.context_window)
        tools_schema = self._tools_schema()
        full_content = ""
        saved = False

        try:
            for round_num in range(1, self.config.max_tool_rounds + 1):
                if round_num > 1:
                    self._emit("round", f"{round_num}/{self.config.max_tool_rounds}")
                response = await self.llm.create_chat_completion(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=True,
                    tools=tools_schema,
                    tool_choice="auto" if tools_schema else None,
                    config=self.config,
                )

                round_content = ""
                tool_calls: Dict[int, Dict[str, Any]] = {}  # index -> {id, name, arguments}

                async for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    # Reasoning tokens (Groq gpt-oss, DeepSeek-R1, etc.) →
                    # surfaced as events so the UI can render a live
                    # "Thinking" block. Never mixed into the answer text.
                    reasoning_text = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
                    if reasoning_text:
                        self._emit("reasoning", reasoning_text)

                    if delta.content:
                        round_content += delta.content
                        full_content += delta.content
                        yield delta.content

                    # Accumulate streamed tool-call fragments
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            entry = tool_calls.setdefault(
                                tc_delta.index, {"id": "", "name": "", "arguments": ""}
                            )
                            if tc_delta.id:
                                entry["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    entry["name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    entry["arguments"] += tc_delta.function.arguments

                            # No tools requested -> this round is the final answer
                if not tool_calls:
                    self.memory.add(Message(role="assistant", content=full_content))
                    saved = True
                    return

                # Feed the tool-call request back and run the tools
                messages.append({
                    "role": "assistant",
                    "content": round_content,
                    "tool_calls": [
                        {
                            "id": tc["id"] or f"call_{i}",
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }
                        for i, tc in sorted(tool_calls.items())
                    ],
                })
                tool_results = await self._run_tool_calls_from_stream(tool_calls)
                messages.extend(tool_results)
                # Tool progress is surfaced below the stream, not in the text
        finally:
            # Persist whatever was produced, even if the consumer stops early
            if full_content and not saved:
                self.memory.add(Message(role="assistant", content=full_content))

    # ── Tool execution ─────────────────────────────────

    async def _run_tool_calls(self, tool_calls: list) -> List[Dict]:
        """Execute tool calls and return them as OpenAI 'tool' role messages"""
        results = []
        for tc in tool_calls:
            result = await self._execute_one(tc.function.name, tc.function.arguments)
            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        return results

    async def _run_tool_calls_from_stream(self, tool_calls: Dict[int, Dict[str, Any]]) -> List[Dict]:
        """Same as _run_tool_calls but for tool calls accumulated from a stream"""
        results = []
        for i, tc in sorted(tool_calls.items()):
            result = await self._execute_one(tc["name"], tc["arguments"])
            results.append({
                "role": "tool",
                "tool_call_id": tc["id"] or f"call_{i}",
                "content": result,
            })
        return results

    async def _execute_one(self, function_name: str, raw_arguments: str) -> str:
        """Execute a single tool and return its result as a string (never raises)"""
        tool = self.tools.get(function_name)
        if not tool:
            return f"Error: unknown tool '{function_name}'"

        try:
            function_args = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            return f"Error: could not parse arguments for {function_name}: {raw_arguments!r}"

        if not isinstance(function_args, dict):
            return f"Error: arguments for {function_name} must be an object"

        # Show the user what the model is doing right now
        brief = ", ".join(f"{k}={str(v)[:40]}" for k, v in list(function_args.items())[:2])
        self._emit("tool_start", f"{function_name}({brief})")
        start = time.monotonic()
        try:
            result = str(await tool.execute(**function_args))
        except Exception as e:
            result = f"Error in {function_name}: {e}"
        elapsed = time.monotonic() - start
        ok = not result.startswith("Error")
        self._emit("tool_end", f"{function_name} {'✓' if ok else '✗'} ({elapsed:.1f}s)")
        return result

    # ── Utilities ──────────────────────────────────────

    def clear_memory(self) -> None:
        """Clear conversation history (current session)"""
        self.memory.clear()

    def get_history(self) -> List[Message]:
        """Get conversation history"""
        return self.memory.messages

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt (applies to the next chat call)"""
        self.config.system_prompt = prompt
