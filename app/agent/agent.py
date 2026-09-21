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
from ..llm.client import LLMRequestTooLargeError, LLMRateLimitError
from .memory import ConversationMemory, Message
from .tools import Tool, ToolRegistry

# How many model<->tool round trips a single chat() call may make by default
# (configurable per-agent via AgentConfig.max_tool_rounds or MAX_TOOL_ROUNDS env)
MAX_TOOL_ROUNDS = 10

# Adaptive sizing when a backend rejects a request for being too large
# (Groq free tier: ~8000 TPM shared by prompt + completion + tools overhead).
# max_tokens shrinks by this factor per retry, floor at MIN_MAX_TOKENS.
TOO_LARGE_SHRINK = 0.5
MIN_MAX_TOKENS = 512

# Fixed per-request overhead estimate (chars): system prompt + tool schemas
# + JSON scaffolding. The Groq 413 in the field (~8161 requested vs 8000 TPM
# limit while the old meter showed 1%) proved history-only counting lies.
# Measured: system prompt ~8.5K chars + schemas ~4.5K chars + slack.
REQUEST_OVERHEAD_CHARS = 16000


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
- Scaffolding projects: prefer the official scaffolder via run_command when one exists
  (`django-admin startproject`, `npm create vite@latest`, `cargo new`, ...) — one command
  instead of dozens of hand-written files. Hand-write only the files the scaffolder
  doesn't create. This is faster AND keeps requests small (providers rate-limit tokens
  per minute; giant file dumps can hit the limit mid-mission).
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
        # Learned from 413s: max_tokens and condensed-context hint for the
        # rest of the session so later requests pre-shrink instead of failing.
        self._max_tokens_cap: Optional[int] = None
        self._condense_hint = False
        # Cooperative cancellation (Esc): checked before every model call,
        # after each streamed round, and between tool rounds — the
        # in-flight call finishes and state unwinds cleanly instead of
        # being abandoned mid-request.
        self._stop_requested: bool = False
        self._stop_reason_msg: str = ""

    def _emit(self, event: str, detail: str = "") -> None:
        """Notify the UI of agent activity (tool calls, rounds). Never raises."""
        if self.on_activity is None:
            return
        try:
            self.on_activity(event, detail)
        except Exception:
            pass

    # ── cooperative stop (Esc) ─────────────────────────────────────────

    def request_stop(self, reason: str = "") -> None:
        """Ask the agent to stop its current work (Esc key).

        Cooperative: the loop checks the flag before every model call and
        after each streamed round, so the in-flight call finishes and
        state unwinds cleanly — no abandoned requests, no lost partial
        answers.
        """
        self._stop_requested = True
        if reason:
            self._stop_reason_msg = reason

    def _clear_stop(self) -> None:
        """Reset the stop flag (start of a fresh turn)."""
        self._stop_requested = False
        self._stop_reason_msg = ""

    def _stop_reason(self) -> str:
        """Consume the stop: return the message and self-clear the flag,
        so one Esc stops exactly one turn — never the next message."""
        reason = self._stop_reason_msg or "Stopped — tell me what to do next."
        self._stop_requested = False
        self._stop_reason_msg = ""
        return reason

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

    def _overhead_tokens(self) -> int:
        """Estimated tokens of system prompt + tool schemas + message
        scaffolding — the fixed cost every request pays before history."""
        schema = self._tools_schema()
        chars = len(self.config.system_prompt or "")
        chars += len(json.dumps(schema)) if schema else 0
        return chars // 4 + 1000  # +1K slack for per-message JSON framing

    def context_usage(self) -> tuple[int, int]:
        """(estimated tokens in next request, model context window size).

        Counts system prompt + tool schemas too — history-only estimates
        under-reported so badly that a real 413 happened at "ctx 1%".
        """
        return self.memory.token_estimate() + self._overhead_tokens(), self.config.context_window

    def _effective_max_tokens(self) -> int:
        cap = self._max_tokens_cap or self.config.max_tokens
        return max(MIN_MAX_TOKENS, min(cap, self.config.max_tokens))

    # ── Adaptive retry on "request too large" ────────────

    def _pre_shrink(self) -> None:
        """Before sending: if a previous 413 taught us limits, apply them."""
        if self._condense_hint:
            self.memory.force_condense()
        self.memory.maybe_condense(self.config.context_window)

    async def _send_with_recovery(self, call_kwargs: dict, rebuild_messages: Optional[Callable[[], List[Dict]]] = None) -> Any:
        """Send a completion request; on 413, adapt and retry — like an
        agent that hits a rate limit: condense context, shrink max_tokens,
        wait out the TPM window if told to, then try again (max 3 adapts).

        `rebuild_messages` is called after a condense so the retry actually
        ships the smaller history (the caller's list is updated in place).

        Raises the original error if adaptation is exhausted — the user
        sees an honest message instead of a silent failure.
        """
        attempts = 0
        while True:
            try:
                return await self.llm.create_chat_completion(**call_kwargs)
            except LLMRequestTooLargeError as e:
                attempts += 1
                if attempts > 3:
                    raise
                # 1) shrink the completion budget
                old_cap = self._max_tokens_cap or self.config.max_tokens
                new_cap = max(MIN_MAX_TOKENS, int(old_cap * TOO_LARGE_SHRINK))
                improved = new_cap < old_cap
                self._max_tokens_cap = new_cap
                # 2) fold older context into the summary
                condensed = self.memory.force_condense()
                if condensed:
                    self._condense_hint = True
                    if rebuild_messages is not None:
                        call_kwargs["messages"][:] = rebuild_messages()
                self._emit("adapting", (
                    f"request too large for the model's rate limit "
                    f"(limit {e.limit_tokens or '?'}, requested {e.requested_tokens or '?'}) — "
                    + ("condensing context, " if condensed else "")
                    + (f"shrinking output budget to {new_cap}, " if improved else "")
                    + "retrying"
                ))
                if e.retry_after:
                    await asyncio.sleep(min(e.retry_after, 20.0))
                if not improved and not condensed:
                    raise
                call_kwargs["max_tokens"] = self._effective_max_tokens()
            except LLMRateLimitError as e:
                attempts += 1
                if attempts > 3:
                    raise
                wait = getattr(e, "retry_after", None) or 20.0
                self._emit("adapting", f"rate limited — waiting {wait:.0f}s, then retrying")
                await asyncio.sleep(min(wait, 30.0))

    # ── Non-streaming path ─────────────────────────────
    async def _get_response(self) -> str:
        """Run the tool loop until the model produces a final text answer"""
        self._pre_shrink()
        messages = self._build_api_messages()
        base_len = len(messages)  # in-flight suffix (tool rounds) survives a rebuild
        tools_schema = self._tools_schema()

        def _rebuild():
            return self._build_api_messages() + messages[base_len:]

        for round_num in range(1, self.config.max_tool_rounds + 1):
            if round_num > 1:
                self._emit("round", f"{round_num}/{self.config.max_tool_rounds}")
            if self._stop_requested:  # Esc — stop before the next model call
                self._emit("stopped", self._stop_reason())
                return ""
            response = await self._send_with_recovery(
                {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self._effective_max_tokens(),
                    "tools": tools_schema,
                    "tool_choice": "auto" if tools_schema else None,
                    "config": self.config,
                },
                rebuild_messages=_rebuild,
            )

            msg = response.choices[0].message

            if self._stop_requested:  # Esc while the model was working
                self._emit("stopped", self._stop_reason())
                return ""

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
        response = await self._send_with_recovery(
            {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self._effective_max_tokens(),
                "config": self.config,
            },
            rebuild_messages=lambda: self._build_api_messages() + messages[base_len:],
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
        self._pre_shrink()
        messages = self._build_api_messages()
        base_len = len(messages)
        tools_schema = self._tools_schema()

        def _rebuild():
            return self._build_api_messages() + messages[base_len:]

        full_content = ""
        saved = False

        try:
            for round_num in range(1, self.config.max_tool_rounds + 1):
                if round_num > 1:
                    self._emit("round", f"{round_num}/{self.config.max_tool_rounds}")
                if self._stop_requested:  # Esc — stop before the next model call
                    self._emit("stopped", self._stop_reason())
                    return
                response = await self._send_with_recovery(
                    {
                        "model": self.config.model,
                        "messages": messages,
                        "temperature": self.config.temperature,
                        "max_tokens": self._effective_max_tokens(),
                        "stream": True,
                        "tools": tools_schema,
                        "tool_choice": "auto" if tools_schema else None,
                        "config": self.config,
                    },
                    rebuild_messages=_rebuild,
                )

                round_content = ""
                tool_calls: Dict[int, Dict[str, Any]] = {}  # index -> {id, name, arguments}

                async for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    if self._stop_requested:  # Esc — stop before yielding more
                        self._emit("stopped", self._stop_reason())
                        return  # the finally-block persists the partial answer

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
                if self._stop_requested:  # Esc mid-stream — keep the partial answer
                    self._emit("stopped", self._stop_reason())
                    if full_content:
                        self.memory.add(Message(role="assistant", content=full_content))
                        saved = True
                    return
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

    def switch_model(self, model: str) -> None:
        """Switch to a different model at runtime (like a model picker).

        Applies to the NEXT message — conversation history is untouched,
        so mid-mission switches keep full context. Also resets the 413
        recovery cap, since the new model may have different limits.
        """
        model = (model or "").strip()
        if not model:
            raise ValueError("model id is required")
        self.config.model = model
        self.llm.model = model
        self._max_tokens_cap = None

    async def aclose(self) -> None:
        """Release the HTTP connection pool (best-effort)."""
        try:
            await self.llm.client.close()
        except Exception:
            pass
