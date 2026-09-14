"""
Terminal UI for MForege
=======================
IDE-style chat layout using prompt_toolkit (always-running full-screen app):

    ┌─────────────────────────────────────┐
    │  transcript (colored, scrolls)      │
    │  ...                                │
    ├─ ✦ message ────────────────────────┤
    │  > input box                        │
    ├─────────────────────────────────────┤
    │ ◆ status bar                        │
    └─────────────────────────────────────┘

Colors: You=green, Assistant=cyan, activity=dim, diffs=green/red,
plan=magenta, errors=red. The app runs continuously, so tool activity
appears live while the model works (same as an IDE agent panel).
"""

import asyncio
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea


class ChatUI:
    """
    Long-running chat UI.

    Usage:
        ui = ChatUI(status_text="...")
        await ui.run(handle)   # handle(text) coroutine processes each message

    Inside handler (and tool callbacks) use:
        ui.append(text, style="class:you")   # colored transcript
        ok = await ui.confirm("Run X?")      # y/n prompt in the question bar
        ui.set_status("...")                 # status bar text
    """

    def __init__(self, status_text: str = "", output=None):
        self._lines: list[list[tuple[str, str]]] = [[]]  # fragments per line
        self._status = status_text
        self._question: str | None = None
        self._answer: str | None = None
        self._answer_event: asyncio.Event | None = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._injected_output = output

        # Transcript pane (colored fragments, auto-scroll to tail)
        self.transcript = Window(
            content=FormattedTextControl(self._fragments),
            wrap_lines=True,
        )

        # Input box
        self.input = TextArea(height=3, multiline=True, wrap_lines=True)

        self.input_label = Window(
            content=FormattedTextControl(
                lambda: HTML("<label> ✦ message</label>")
            ),
            height=1,
        )
        self.separator = Window(height=1, char="─", style="class:separator")

        # Question bar (shown only when a confirm is pending)
        self.question_win = Window(
            content=FormattedTextControl(self._question_fragments),
            height=1,
        )

        # Status bar
        self.status_win = Window(
            content=FormattedTextControl(self._status_fragments),
            height=1,
        )

        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            text = self.input.text
            self.input.text = ""
            if self._question is not None:
                ans = text.strip().lower() or "n"
                self._answer = ans
                self._question = None
                if self._answer_event:
                    self._answer_event.set()
                event.app.invalidate()
                return
            text = text.strip()
            if text:
                self._queue.put_nowait(text)

        # Shift+Enter = newline. prompt_toolkit exposes this as ControlJ:
        # many terminals (Windows Terminal, VS Code, iTerm2, WezTerm) report
        # Shift+Enter as the same escape sequence as Ctrl+J (\x0a or ESC+[13;2u).
        # Alt+Enter and ESC-then-Enter are kept as universal fallbacks.
        @kb.add("c-j")
        @kb.add("escape", "enter")
        def _(event):
            self.input.buffer.insert_text("\n")

        @kb.add("c-c")
        def _(event):
            if self._question is not None:
                # Cancel the pending confirmation: answer "no" and continue
                self._answer = "n"
                self._question = None
                if self._answer_event:
                    self._answer_event.set()
                event.app.invalidate()
                return
            event.app.exit(exception=KeyboardInterrupt())

        layout = HSplit([
            self.transcript,
            Window(height=1, char="─", style="class:separator"),
            self.input_label,
            self.input,
            self.question_win,
            self.separator,
            self.status_win,
        ])

        app_kwargs = dict(
            layout=Layout(layout, focused_element=self.input),
            key_bindings=kb,
            full_screen=True,
            style=Style.from_dict({
                "separator": "#444444",
                "label": "#5fd7ff",
                "status": "bg:#1e6f5c #ffffff bold",
                "question": "bg:#8a6d00 #ffffff bold",
                "you": "#5fff5f bold",
                "assistant": "#5fd7ff bold",
                "dim": "#6c6c6c",
                "ok": "#5fff87",
                "error": "#ff5f5f bold",
                "add": "#5fff87",
                "del": "#ff5f5f",
                "plan": "#ff87d7",
                "title": "#5fd7ff bold",
                "warn": "#ffd75f",
            }),
            mouse_support=True,
        )
        if output is not None:
            app_kwargs["output"] = output

        self.app: Application = Application(**app_kwargs)

    # ── transcript ────────────────────────────────────────────────────

    def append(self, text: str, end: str = "\n", style: str | None = None) -> None:
        """Append text (optionally styled) to the transcript."""
        if not self._lines:
            self._lines.append([])
        self._lines[-1].append((style or "", text))
        if end == "\n":
            self._lines.append([])
        self.app.invalidate()

    def clear(self) -> None:
        self._lines = [[]]
        self.app.invalidate()

    def set_status(self, text: str) -> None:
        self._status = text
        self.app.invalidate()

    def exit(self) -> None:
        self.app.exit()

    # ── confirmation ──────────────────────────────────────────────────

    async def confirm(self, question: str) -> bool:
        """Show the question bar and wait for y/n (Enter defaults to no)."""
        self._question = question
        self._answer = None
        self._answer_event = asyncio.Event()
        self.app.invalidate()
        await self._answer_event.wait()
        self._answer_event = None
        return self._answer in ("y", "yes")

    def _question_fragments(self):
        if not self._question:
            return [("", "")]
        return [("class:question", f" ⚠ {self._question}   (type y/n + Enter)")]

    # ── status ────────────────────────────────────────────────────────

    def _status_fragments(self):
        return [("class:status", f" ◆ MForege │ {self._status} │ /help │ Enter=send │ Shift+Enter=newline │ Ctrl+C=quit ")]

    # ── transcript rendering (colored, auto-scroll tail) ─────────────

    def _fragments(self):
        frags: list[tuple[str, str]] = []
        rows = 24
        try:
            rows = self.app.output.get_size().rows
        except Exception:
            pass
        reserve = 9  # input(3) + label(1) + separators(2) + status(1) + question(1) + margin
        visible = max(1, rows - reserve)
        tail = self._lines[-visible:]
        for line in tail:
            for style, text in line:
                frags.append((style, text))
            frags.append(("", "\n"))
        return frags

    # ── main runner ───────────────────────────────────────────────────

    async def run(self, handler) -> None:
        """
        Run the UI forever, dispatching each submitted message to `handler`.
        Handler runs inside the app's event loop, so ui.append() calls from
        it (and from tool callbacks) render live.
        """
        async def worker():
            while True:
                text = await self._queue.get()
                try:
                    await handler(text)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.append(f"[!] Unexpected error: {e}", style="class:error")

        worker_task = asyncio.get_event_loop().create_task(worker())
        try:
            await self.app.run_async()
        finally:
            worker_task.cancel()
