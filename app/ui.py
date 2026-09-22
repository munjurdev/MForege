"""
Terminal UI for MForege
=======================
Agent-panel chat layout using prompt_toolkit (always-running full-screen app).
The input box is the BOTTOM-MOST element — everything else sits above it
(context/status info belongs above the composer, never under it):

    ┌─────────────────────────────────────┐
    │  transcript (colored, scrolls)      │
    │   ✻ Thinking… (live, collapsible)   │
    │   · tool(args) ✓ (0.3s)             │
    │   ✓ Applied:  -old / +new           │
    │   markdown reply (bold/code/bullet) │
    ├─────────────────────────────────────┤
    │  ✻ read_file(path=main.py)          │ activity (0–1 row)
    │  /help  /plan …                     │ slash menu (0–6 rows)
    │  ⚠ EDIT main.py? (y/n)              │ confirm (0–1 row)
    │  ◆ model │ ctx │ ⏳ 3s (Esc=stop)   │ status (1 row)
    ├─────────────────────────────────────┤
    │  > input box   ← bottom-most row    │
    └─────────────────────────────────────┘

Colors: You=green, Assistant=cyan, activity=dim, diffs=green/red,
plan=magenta, thinking=italic gray, errors=red, md-code=orange,
md-header=bold blue, md-link=green underline. The app runs continuously,
so tool activity appears live while the model works (same as an IDE panel).

Transcript model (important): the transcript is a list of SEGMENTS, not a
flat line list. Four segment kinds:
  - "text"  — ordinary transcript lines (append)
  - "think" — the Thinking block; collapsible WITHOUT losing its content
  - "block" — named live blocks (TODOS) redrawn in place
  - "md"    — a live markdown reply (begin_reply/stream_reply/end_reply);
              re-rendered in place from raw text on every streamed chunk
Because segments are variable-height objects, mutating one can never
corrupt another (e.g. collapsing Thinking no longer wipes TODOS), and
thinking text is preserved for later expansion.
"""

import asyncio
import re
import sys
import time
from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.margins import Margin
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

_THINK_HEADER = ("class:think", " ✻ Thinking…")
_THINK_SUMMARY = ("class:think", " ✻ Thinking… (Ctrl+T expands)")

_PHASE_PREFIX = "⏳"  # live phase timer in the status bar (thinking/responding)

# ── Markdown rendering (assistant replies) ─────────────────────────────
# Assistant output is markdown (bold, code, headers, lists — the same way
# coding-agent replies are formatted). The renderer turns it into styled
# terminal fragments: **bold** → bold, `x` → highlighted code, fenced
# blocks → their own style, headers/bullets/quotes → clean terminal forms.
_MD_INLINE_RE = re.compile(
    r"(`[^`\n]+`"                    # inline code
    r"|\*\*[^*\n]+\*\*"             # **bold**
    r"|\*[^*\n]+\*"                 # *italic*
    r"|\[[^\]\n]+\]\([^)\n]+\))"   # [text](url)
)


def _md_inline(text: str) -> list[tuple[str, str]]:
    """Parse inline markdown (code/bold/italic/links) into styled fragments."""
    frags: list[tuple[str, str]] = []
    pos = 0
    for m in _MD_INLINE_RE.finditer(text):
        if m.start() > pos:
            frags.append(("", text[pos:m.start()]))
        tok = m.group(0)
        if tok.startswith("`"):
            frags.append(("class:md-code", tok[1:-1]))
        elif tok.startswith("**"):
            frags.append(("class:md-bold", tok[2:-2]))
        elif tok.startswith("["):
            try:
                link_text, url = tok[1:-1].split("](", 1)
            except ValueError:
                frags.append(("", tok))
            else:
                frags.append(("class:md-link", link_text))
                frags.append(("class:md-dim", f" ({url})"))
        else:
            frags.append(("class:md-italic", tok[1:-1]))
        pos = m.end()
    if pos < len(text):
        frags.append(("", text[pos:]))
    return frags


def _render_markdown(text: str) -> list[list[tuple[str, str]]]:
    """Render a markdown document into transcript lines (styled fragments).

    Block-level pass (headers, lists, fenced code, quotes, rules) with
    inline parsing for the rest. Idempotent: same input → same lines.
    """
    out: list[list[tuple[str, str]]] = []
    in_code = False
    for raw in text.split("\n"):
        if in_code:
            if raw.strip().startswith("```"):
                in_code = False
                out.append([("class:md-dim", raw.strip())])
            else:
                out.append([("class:md-code-block", raw)])
            continue
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = True
            out.append([("class:md-dim", stripped)])
            continue
        if not stripped:
            out.append([])
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            style = "class:md-header" if len(m.group(1)) <= 2 else "class:md-subheader"
            out.append([(style, m.group(2))])
            continue
        m = re.match(r"^(\s*)[-*+]\s+(.*)$", raw)
        if m:
            out.append([("class:md-bullet", f"{m.group(1)}• "),
                        *(_md_inline(m.group(2)))])
            continue
        m = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", raw)
        if m:
            out.append([("class:md-bullet", f"{m.group(1)}{m.group(2)}. "),
                        *(_md_inline(m.group(3)))])
            continue
        m = re.match(r"^>\s?(.*)$", stripped)
        if m:
            out.append([("class:md-quote", "▌ "), ("class:md-quote", m.group(1))])
            continue
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            out.append([("class:md-dim", "─" * 32)])
            continue
        out.append(_md_inline(raw))
    return out

_WHEEL_STEP = 3  # transcript lines per mouse-wheel notch

# ── Slash-command menu (Freebuff-style) ────────────────────────────────
# Shown live as the user types "/" in the input box: filtered list with
# descriptions, arrow-key navigation, Enter to accept. Exact match → menu
# hides so the same Enter press submits the command.
COMMANDS: list[tuple[str, str]] = [
    ("/help", "Display keyboard shortcuts and tips"),
    ("/diagnostics", "Show version, model, context usage, sessions"),
    ("/interview", "AI asks questions to flesh out a request into a spec"),
    ("/plan", "Show the current mission plan"),
    ("/review", "Review changes made this conversation"),
    ("/queue", "Message queueing info"),
    ("/new", "Clear the conversation and start a new chat"),
    ("/history", "Browse and resume past conversations"),
    ("/tools", "List registered tools"),
    ("/copy", "Copy the conversation to the clipboard"),
    ("/export", "Write the full conversation to a file"),
    ("/bash", "Run a shell command with the agent's safety guards"),
    ("/theme:toggle", "Toggle between light and dark mode"),
    ("/byok", "Show where to configure your API key / model"),
    ("/model", "Switch model live (free catalog: Groq, OpenRouter, Ollama)"),
    ("/reasoning", "Set how hard the model thinks (low / high / max)"),
    ("/resume", "Resume a recent chat (default: latest)"),
    ("/sessions", "List past conversations"),
    ("/vscode-hint", "Fix Shift+Enter for VS Code's terminal"),
    ("/clear", "Reset conversation + transcript"),
    ("/exit", "Quit"),
]

_CMD_COL_WIDTH = 16  # description column starts here (screenshot layout)


def _display_width(s: str) -> int:
    """Terminal cell width of a string (wide CJK/emoji chars count 2)."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in s)


def boxed(lines: list[str]) -> str:
    """Draw a box around `lines` — the structured welcome panel.

    Uses single-line box-drawing characters (┌─┐│└┘): they have far wider
    font coverage than the rounded variants (╭╮╰╯), which render as `¦`
    in some terminal fonts (seen in VS Code on Windows).
    """
    try:
        width = max((_display_width(ln) for ln in lines), default=0) + 4
        top = "┌" + "─" * width + "┐"
        body = "".join(
            "│  " + ln + " " * max(0, width - _display_width(ln) - 2) + "│\n"
            for ln in lines
        )
        bottom = "└" + "─" * width + "┘"
        return top + "\n" + body + bottom
    except Exception:
        return "\n".join(lines)


def _patch_windows_shift_enter() -> None:
    """
    Make Shift+Enter work on Windows terminals.

    prompt_toolkit 3.0.53 has three gaps on Windows:

    1. Classic console input (`ConsoleInputReader`, plain cmd.exe/conhost):
       its shift-combine table maps Tab/arrows/Home/End — but NOT Enter, so
       Shift+Enter arrives as plain ControlM (= Enter = send). Fixed by
       post-processing the key event: ControlM + SHIFT_PRESSED (no Ctrl)
       → ControlJ, which our key bindings treat as a newline.
    2. VT escape table: the modern sequence ESC[13;2u is absent from
       ANSI_SEQUENCES, so the key is dropped entirely. Fixed by
       registering it → ControlJ.
    3. VT-console reader (`Vt100ConsoleInputReader`, used by VS Code /
       Windows Terminal when ENABLE_VIRTUAL_TERMINAL_INPUT is on — the
       path in real crash tracebacks): its `_get_keys` yields only the
       character and DISCARDS ControlKeyState, so Shift+Enter reaches the
       parser as a bare "\r" (= Enter). Fixed by re-implementing the
       generator with the same filters but "\r" → "\n" when Shift-only.

    (On Unix, Shift+Enter arrives as \x0a = ControlJ already and works.)
    All patches are idempotent, best-effort, and no-ops on non-Windows.
    """
    if sys.platform != "win32":
        return
    try:
        from prompt_toolkit.keys import Keys

        # ── gap 2: VT-escape path (ESC[13;2u = Shift+Enter, CSI-u encoding) ──
        # NOTE: ANSI_SEQUENCES keys are STR here (the parser feeds str), a
        # bytes key poisons _IS_PREFIX_OF_LONGER_MATCH_CACHE and crashes
        # every keypress ("startswith first arg must be bytes...").
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        ANSI_SEQUENCES.setdefault("\x1b[13;2u", Keys.ControlJ)

        # ── gap 1: classic Win32 console path ──
        from prompt_toolkit.input.win32 import ConsoleInputReader
        from prompt_toolkit.key_binding import KeyPress

        if not getattr(ConsoleInputReader, "_mforege_shift_patched", False):
            _orig = ConsoleInputReader._event_to_key_presses

            def _patched(self, ev):
                keys = _orig(self, ev)
                try:
                    state = ev.ControlKeyState
                    shift = bool(state & ConsoleInputReader.SHIFT_PRESSED)
                    ctrl = bool(
                        state
                        & (ConsoleInputReader.LEFT_CTRL_PRESSED
                           | ConsoleInputReader.RIGHT_CTRL_PRESSED)
                    )
                    if shift and not ctrl:
                        keys = [
                            KeyPress(Keys.ControlJ, "\n")
                            if kp.key == Keys.ControlM else kp
                            for kp in keys
                        ]
                except Exception:
                    pass
                return keys

            ConsoleInputReader._event_to_key_presses = _patched
            ConsoleInputReader._mforege_shift_patched = True

        # ── gap 3: VT-console reader path (THE one used by VS Code / Windows
        # Terminal when ENABLE_VIRTUAL_TERMINAL_INPUT is on). Its _get_keys
        # yields only u_char and DISCARDS ControlKeyState, so the parser
        # sees a bare "\r" (= Enter) for Shift+Enter. Fix: replicate its
        # exact filters, but translate "\r" → "\n" when SHIFT_PRESSED and
        # no Ctrl ("\n" parses to ControlJ = newline in our bindings).
        from prompt_toolkit.input.win32 import Vt100ConsoleInputReader
        from prompt_toolkit.win32_types import (
            EventTypes,
            KEY_EVENT_RECORD,
        )

        if not getattr(Vt100ConsoleInputReader, "_mforege_shift_patched", False):
            _SHIFT = 0x0010   # SHIFT_PRESSED
            _LCTRL = 0x0008   # LEFT_CTRL_PRESSED
            _RCTRL = 0x0100   # RIGHT_CTRL_PRESSED

            def _get_keys(self, read, input_records):
                for i in range(read.value):
                    ir = input_records[i]
                    if ir.EventType not in EventTypes:
                        continue
                    ev = getattr(ir.Event, EventTypes[ir.EventType])
                    if not (isinstance(ev, KEY_EVENT_RECORD) and ev.KeyDown):
                        continue
                    u_char = ev.uChar.UnicodeChar
                    if u_char == "\x00":
                        continue
                    state = ev.ControlKeyState
                    shift = bool(state & _SHIFT)
                    ctrl = bool(state & (_LCTRL | _RCTRL))
                    if u_char == "\r" and shift and not ctrl:
                        yield "\n"           # Shift+Enter → newline
                    else:
                        yield u_char

            Vt100ConsoleInputReader._get_keys = _get_keys
            Vt100ConsoleInputReader._mforege_shift_patched = True
    except Exception:
        pass  # input must never break the app


_patch_windows_shift_enter()


class TranscriptScrollBar(Margin):
    """One-column scrollbar on the transcript's right edge.

    Reflects MForege's OWN viewport state (_scroll_offset over the
    materialized line count) — not prompt_toolkit's built-in Window
    scroll, which never moves here because `_fragments` already windows
    the lines before the Window renders them. Hidden (width 0) while the
    content fits, so nothing is reserved on short transcripts."""

    def __init__(self, ui):
        self._ui = ui

    def get_width(self, get_ui_content) -> int:
        try:
            ui = self._ui
            if len(ui._materialized_lines()) <= ui._visible_row_count():
                return 0
            return 1
        except Exception:
            return 0

    def create_margin(self, window_render_info, width: int, height: int):
        try:
            ui = self._ui
            total = len(ui._materialized_lines())
            visible = max(1, min(height, ui._visible_row_count()))
            if total <= visible:
                return []
            max_offset = total - visible
            offset = max(0, min(ui._scroll_offset, max_offset))
            thumb = max(1, visible * visible // total)
            # Inverted mapping so it reads like a normal scrollbar: viewing
            # the live tail (offset 0 = bottom of content) → thumb at the
            # BOTTOM of the track; fully scrolled up → thumb at the top.
            pos = ((max_offset - offset) * (visible - thumb)) // max(1, max_offset)
            pos = max(0, min(pos, visible - thumb))
            frags = []
            for row in range(height):
                if row < height - visible:
                    continue  # rows above the content window stay empty
                rel = row - (height - visible)
                char, style = (("█", "class:scrollbar-thumb")
                               if pos <= rel < pos + thumb
                               else ("│", "class:scrollbar-track"))
                frags.append((style, char))
            return frags
        except Exception:
            return []


def _new_text_seg() -> dict:
    return {"kind": "text", "lines": [[]]}


class TranscriptControl(FormattedTextControl):
    """Transcript control with mouse-wheel scrolling wired to ChatUI's
    viewport. The control's mouse_handler runs BEFORE the Window's built-in
    scroll handler (Window._mouse_handler is only the fallback), so the
    wheel drives MForege's own scroll offset instead of the Window's
    independent vertical_scroll (which would clip lines rather than
    navigate history)."""

    _ui = None  # set by ChatUI.__init__

    def mouse_handler(self, mouse_event):
        try:
            if mouse_event.event_type == MouseEventType.SCROLL_UP:
                if self._ui is not None:
                    self._ui.scroll_up(_WHEEL_STEP)
                return None
            if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                if self._ui is not None:
                    self._ui.scroll_down(_WHEEL_STEP)
                return None
        except Exception:
            pass  # display must never break the chat
        return super().mouse_handler(mouse_event)


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
        self._segments: list[dict] = [_new_text_seg()]
        self._blocks: dict[str, dict] = {}  # name -> block segment (identity-stable)
        self._status = status_text
        self._question: str | None = None
        self._answer: str | None = None
        self._answer_event: asyncio.Event | None = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._injected_output = output
        self._scroll_offset = 0        # 0 = follow the live tail
        self._last_line_count = 1      # for anchored scrolling compensation
        self._rendered_once = False    # anchor baseline set on first render
        self._activity: str | None = None  # "editing foo.py" pinned indicator
        # Esc-to-stop: the UI never stops the agent itself — it records the
        # request, calls the _on_stop hook (installed by main.py, flips the
        # agent's cooperative flag) and cancels the handler task so the
        # in-flight request unwinds at its next await point.
        self._stop_requested = False
        self._on_stop = None           # Callable[[], None] | None
        self._handler_task: asyncio.Task | None = None
        # Live markdown reply segment (None = no reply streaming right now)
        self._md_seg: dict | None = None
        # Live phase timer ("⏳ thinking 3s" in the status bar)
        self._phase_label: str | None = None
        self._phase_started: float | None = None
        # Hover state of the '✕ End session' button (reverse-video while
        # the pointer is over it; MOUSE_MOVE sets it, MOUSE_UP clears it).
        self._end_session_hover = False
        # Session bar (reference-panel style): start time + optional fixed
        # label override (tests); formatted like '· 1h 2m' next to the model.
        self._session_started: float | None = None
        self._session_label: str | None = None

        # Transcript pane (colored fragments, scrollable via PgUp/PgDn
        # and the mouse wheel)
        self._transcript_ctrl = TranscriptControl(self._fragments)
        self._transcript_ctrl._ui = self
        self.transcript = Window(
            content=self._transcript_ctrl,
            wrap_lines=True,
            right_margins=[TranscriptScrollBar(self)],
        )

        # Input box — grows with content (2–5 rows), drawn inside a bordered
        # frame (IDE-style). NO title in the border — the reference panel's
        # composer has a plain border; identity lives in the session bar above.
        self.input = TextArea(height=self._input_height, multiline=True, wrap_lines=True)
        self.input_frame = Frame(self.input, style="class:inputbox")

        # Slash-command popup (visible while typing a "/" prefix)
        self._menu_items: list[tuple[str, str]] = []
        self._menu_index = 0
        self.menu_win = Window(
            content=FormattedTextControl(self._menu_fragments),
            height=self._menu_height,   # 0 when the menu is hidden
            style="class:menu",
        )
        self.input.buffer.on_text_changed += lambda buff: self._refresh_menu()
        self.separator = Window(height=1, char="─", style="class:separator")

        # Question bar (occupies a row only while a confirm is pending)
        self.question_win = Window(
            content=FormattedTextControl(self._question_fragments),
            height=lambda: 1 if self._question else 0,
        )

        # Live activity indicator (occupies a row only while a tool runs)
        self.activity_win = Window(
            content=FormattedTextControl(self._activity_fragments),
            height=lambda: 1 if self._activity else 0,
        )

        # Status ROW (reference-panel layout): ONE full-width window whose
        # fragments compute the middle padding themselves — '◆ model …' on
        # the left, '✕ End session' pinned to the right edge, and the whole
        # row reading as one continuous green (class:status) highlight.
        # (A VSplit of three windows does NOT pin right: leftover width is
        # distributed round-robin to every growable window.)
        self.status_win = Window(
            content=FormattedTextControl(self._status_fragments),
            height=1,
            char=" ",
            style="class:status",
        )

        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            if self._menu_items:
                # First Enter takes the highlighted command; a second Enter
                # (menu now hidden/exact) submits it.
                self.menu_accept()
                event.app.invalidate()
                return
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
                self.echo_submission(text)
                # While the agent is mid-turn (or a message is already
                # waiting), the message QUEUES — tell the user visibly,
                # like the reference panel's "✓ Queued" note, so a silent
                # submit never looks lost.
                if self._handler_task is not None and not self._handler_task.done():
                    self.append("  ✓ Queued — will run after the current turn.",
                                style="class:dim")
                self._queue.put_nowait(text)

        # Arrow keys navigate the popup when it is visible; otherwise they
        # keep the TextArea's normal cursor movement (filter-conditioned).
        @kb.add("up", filter=Condition(lambda: bool(self._menu_items)))
        def _(event):
            self.menu_move(-1)

        @kb.add("down", filter=Condition(lambda: bool(self._menu_items)))
        def _(event):
            self.menu_move(1)

        @kb.add("escape")
        def _(event):
            if self._menu_items:
                self._menu_items = []
                self._invalidate()
                return
            if self._question is not None:
                # Esc cancels a pending confirmation (same as answering n)
                self._answer = "n"
                self._question = None
                if self._answer_event:
                    self._answer_event.set()
                event.app.invalidate()
                return
            if self._phase_started is not None and not self.input.text:
                # Esc while the agent is working → ask it to stop
                self.request_stop()
                event.app.invalidate()
                return
            if self.input.text:
                # Standard terminal behavior: Esc clears the input line
                self.input.text = ""
                event.app.invalidate()

        # Newline in the input box:
        # - Shift+Enter: works in Windows Terminal / cmd / most Unix
        #   terminals (patched above: ESC[13;2u, console shift-table,
        #   VT-reader translation).
        # - Alt+Enter: the reliable path in VS Code's terminal — its
        #   xterm.js front-end transmits Shift+Enter as a plain \r (the
        #   Shift state never reaches any program), but Alt+Enter arrives
        #   as ESC+\r, which IS distinguishable.
        # - Ctrl+J: manual fallback (\x0a = literal newline everywhere).
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

        # Ctrl+T — expand/collapse the most recent Thinking block
        @kb.add("c-t")
        def _(event):
            self.toggle_thinking()

        # ── mouse wheel (positionless fallback) ──────────────────────
        # With mouse_support on, terminals that send wheel events without
        # a cursor position arrive here. prompt_toolkit's default for
        # these feeds an Up/Down arrow key press (which would move the
        # input cursor) — override so the wheel scrolls the transcript.
        @kb.add(Keys.ScrollUp, eager=True)
        def _(event):
            self.scroll_up(_WHEEL_STEP)

        @kb.add(Keys.ScrollDown, eager=True)
        def _(event):
            self.scroll_down(_WHEEL_STEP)

        # ── transcript scrolling (real viewport control) ─────────────
        # eager=True is REQUIRED: prompt_toolkit's basic bindings map
        # pageup/pagedown to buffer history navigation on the focused
        # input, and focused-control bindings outrank app-level ones —
        # silently eating the keys (scroll would never fire). Eager
        # bindings match before everything else.
        @kb.add("pageup", eager=True)
        def _(event):
            self.scroll_up(10)

        @kb.add("pagedown", eager=True)
        def _(event):
            self.scroll_down(10)

        @kb.add("c-up", eager=True)
        def _(event):
            self.scroll_up(1)

        @kb.add("c-down", eager=True)
        def _(event):
            self.scroll_down(1)

        @kb.add("s-up", eager=True)
        def _(event):
            self.scroll_up(1)

        @kb.add("s-down", eager=True)
        def _(event):
            self.scroll_down(1)

        @kb.add("end", eager=True)
        def _(event):
            self.scroll_bottom()

        # Bottom stack — matches my panel: EVERYTHING sits ABOVE the input;
        # the input box is the last row on screen, nothing below it
        # (status/context info lives above the composer, not under it).
        layout = HSplit([
            self.transcript,
            Window(height=1, char="─", style="class:separator"),
            self.activity_win,
            self.menu_win,
            self.question_win,   # confirm bar directly above the input
            self.status_win,     # status/context above the input too
            self.input_frame,    # input = bottom-most element
        ])

        app_kwargs = dict(
            layout=Layout(layout, focused_element=self.input),
            key_bindings=kb,
            full_screen=True,
            style=Style.from_dict({
                # md-* + core palette come from _DARK_STYLE (shared with
                # /theme:toggle — one source of truth for both palettes)
                **self._DARK_STYLE,
                "scrollbar-track": "#444444",
                "scrollbar-thumb": "bold #5fd7ff",
                "label": "#5fd7ff",
                "status": "bg:#1e6f5c #ffffff bold",
                "scroll": "bg:#1e6f5c #ffd75f bold",
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
                "think": "italic #878787",
                "inputbox": "#5fd7ff",
                "menu": "bg:#1c1c1c",
                "menu-cmd": "#5fd7ff bold",
                "menu-desc": "#6c6c6c",
                "menu-selected": "reverse",
                "session-end": "#ff5f5f bold",
                "session-hover": "reverse",
            }),
            # Mouse support: wheel scrolls the transcript (TranscriptControl
            # + ScrollUp/Down bindings). Click-to-position stays handled by
            # the input's own control.
            mouse_support=True,
            # Keeps the live phase timer ("⏳ thinking Ns") ticking even
            # when nothing streams (long tool runs, silent model calls).
            refresh_interval=1.0,
        )
        if output is not None:
            app_kwargs["output"] = output

        self.app: Application = Application(**app_kwargs)

    # ── input sizing ─────────────────────────────────────────────────

    # Freebuff-panel sizing: the composer is a roomy 2 content rows tall
    # when empty (border + 2 rows + border = 4 total), then grows with typed
    # lines up to MAX_INPUT_ROWS. MIN=1 made the box feel cramped next to
    # the reference panel.
    MIN_INPUT_ROWS = 2
    MAX_INPUT_ROWS = 5
    # ── slash-command menu (Freebuff-style popup) ────────────────────

    def _refresh_menu(self) -> None:
        """Refilter the command menu from the current input text."""
        try:
            text = self.input.text
            if text.startswith("/") and " " not in text and "\n" not in text:
                q = text.lower()
                self._menu_items = [(c, d) for c, d in COMMANDS
                                    if c.startswith(q)]
                # Exact match → nothing left to choose; Enter will submit.
                if len(self._menu_items) == 1 and self._menu_items[0][0] == text:
                    self._menu_items = []
            else:
                self._menu_items = []
            self._menu_index = 0
        except Exception:
            self._menu_items = []
        self._invalidate()

    def menu_move(self, delta: int) -> None:
        """Move the menu selection (wraps around)."""
        if not self._menu_items:
            return
        self._menu_index = (self._menu_index + delta) % len(self._menu_items)
        self._invalidate()

    def menu_accept(self) -> None:
        """Put the highlighted command into the input and close the menu."""
        if not self._menu_items:
            return
        cmd = self._menu_items[self._menu_index][0]
        self._menu_items = []
        self.input.text = cmd + " "   # trailing space keeps the menu closed
        self._invalidate()

    def _menu_height(self) -> int:
        """0 when hidden (no reserved blank row), max 6 rows when open."""
        return min(len(self._menu_items), 6)

    def _menu_fragments(self):
        frags: list[tuple[str, str]] = []
        try:
            if not self._menu_items:
                return frags
            start = max(0, min(self._menu_index - 5, len(self._menu_items) - 6))
            window = self._menu_items[start:start + 6]
            for i, (cmd, desc) in enumerate(window):
                idx = start + i
                pad = " " * max(0, _CMD_COL_WIDTH - _display_width(cmd))
                if idx == self._menu_index:
                    frags.append(("reverse class:menu-cmd", f" {cmd}"))
                    frags.append(("reverse class:menu-desc", f"{pad}{desc} "))
                else:
                    frags.append(("class:menu-cmd", f" {cmd}"))
                    frags.append(("class:menu-desc", f"{pad}{desc} "))
                frags.append(("", "\n"))
        except Exception:
            pass  # renderer must never break the chat
        return frags

    def _input_height(self) -> int:
        """Dynamic height: MIN_INPUT_ROWS base (2, Freebuff-style) + extra
        lines typed, capped at MAX_INPUT_ROWS."""
        try:
            lines = self.input.text.count("\n") + 1
        except Exception:
            return self.MIN_INPUT_ROWS
        return max(self.MIN_INPUT_ROWS, min(self.MAX_INPUT_ROWS, lines))

    # ── transcript ────────────────────────────────────────────────────

    def echo_submission(self, text: str) -> None:
        """Echo the user's message AT SUBMIT TIME (chat-app style).

        Multi-line pastes are indented so they don't render as one giant
        green wall. Echoing here — not in the message handler — means a
        message sent while the agent is still working appears immediately,
        in order, instead of surfacing only when its turn starts.
        """
        try:
            self.scroll_bottom()  # sending = want to see the fresh reply
            lines = text.split("\n")
            self.append(f"You: {lines[0]}", style="class:you")
            for extra in lines[1:]:
                self.append(f"    {extra}", style="class:you")
        except Exception:
            pass  # display must never break the chat

    # ── live markdown reply (agent-style formatted output) ───────────

    def begin_reply(self) -> None:
        """Start a live markdown reply segment.

        While active, stream_reply() appends raw markdown and the whole
        segment re-renders in place — bold/code/headers/lists appear live,
        exactly like watching an agent write its reply.
        """
        try:
            self._md_seg = {"kind": "md", "text": "", "lines": []}
            self._segments.append(self._md_seg)
            self._invalidate()
        except Exception:
            pass  # display must never break the chat

    def stream_reply(self, chunk: str) -> None:
        """Append raw markdown to the open reply and re-render it live."""
        try:
            if self._md_seg is None:
                self.begin_reply()
            self._md_seg["text"] += chunk
            self._md_seg["lines"] = _render_markdown(self._md_seg["text"])
            self._invalidate()
        except Exception:
            pass  # display must never break the chat

    def end_reply(self) -> None:
        """Close the live reply segment (final render stays in the transcript)."""
        try:
            self._md_seg = None
            self._invalidate()
        except Exception:
            pass  # display must never break the chat

    def append(self, text: str, *, end: str = "\n", style: str | None = None) -> None:
        """Append text (optionally styled) to the transcript. Never raises.

        end/style are KEYWORD-ONLY by design: ui.append("x", "class:dim")
        used to bind the style into `end`, silently dropping the newline
        (real bug seen in the wild). Now misuse raises TypeError loudly
        instead of corrupting the transcript quietly.
        """
        try:
            seg = None
            if self._segments:
                last = self._segments[-1]
                # Never merge into a live markdown segment — it re-renders
                # itself from raw text; foreign appends go to a fresh seg.
                if last["kind"] == "text":
                    seg = last
            if seg is None:
                seg = _new_text_seg()
                self._segments.append(seg)
            self._write_lines(seg["lines"], text, style or "")
            if end == "\n":
                seg["lines"].append([])
            self._invalidate()
        except Exception:
            pass  # display must never break the chat

    def clear(self) -> None:
        self._segments = [_new_text_seg()]
        self._blocks = {}
        self._md_seg = None
        self._scroll_offset = 0
        self._invalidate()

    def set_status(self, text: str) -> None:
        self._status = text
        self._invalidate()

    def set_session_start(self, started: float | None = None) -> None:
        """Start the session clock shown next to the model in the session bar.

        Call with the default (None) to anchor at 'now'. Passing a fixed
        label (set_session_label) overrides the computed clock entirely.
        """
        self._session_started = time.monotonic() if started is None else started
        self._invalidate()

    def set_session_label(self, label: str | None) -> None:
        """Force a fixed session-bar label (e.g. '1h left') — test hook."""
        self._session_label = label
        self._invalidate()

    # ── agent phase timer (live "⏳ thinking Ns") ─────────────────────

    def set_phase(self, label: str | None) -> None:
        """Mark the start/end of an agent phase ("thinking" / "responding").

        While a phase is active the status bar shows a live elapsed timer
        and Esc is armed to stop the agent. Passing another label mid-turn
        (thinking → responding) KEEPS the clock running — one timer per
        turn. Pass None when the turn ends.
        """
        if label:
            if self._phase_started is None:
                self._phase_started = time.monotonic()
            self._phase_label = label
        else:
            self._phase_label = None
            self._phase_started = None
        self._invalidate()

    @property
    def phase_active(self) -> bool:
        """True while the agent is mid-turn (timer running, Esc armed)."""
        return self._phase_started is not None

    # ── Esc-to-stop ──────────────────────────────────────────────────

    def set_stop_hook(self, hook) -> None:
        """Install the callback fired when the user presses Esc mid-turn
        (main.py wires this to Agent.request_stop)."""
        self._on_stop = hook

    def request_stop(self) -> None:
        """User pressed Esc: record it and stop the running turn.

        Two mechanisms, belt and suspenders:
        1. the _on_stop hook flips the agent's cooperative flag (graceful
           unwinding, partial answers saved by the generator's finally),
        2. the handler task is cancelled so an in-flight HTTP wait ends
           immediately instead of at the next cooperative checkpoint.
        The handler itself catches the CancelledError and prints the note,
        so the worker loop survives and the next message still runs.
        """
        self._stop_requested = True
        self.set_activity(None)
        if self._on_stop is not None:
            try:
                self._on_stop()
            except Exception:
                self.append("[!] Stop failed — Ctrl+C still works.", style="class:error")
        task = self._handler_task
        if task is not None and not task.done():
            task.cancel()
        else:
            self.set_status("stopped")

    def _invalidate(self) -> None:
        """Request a redraw — never raises (display must not break the chat)."""
        try:
            self.app.invalidate()
        except Exception:
            pass

    @staticmethod
    def _write_lines(lines: list, text: str, style: str) -> None:
        """Write text (which may contain newlines) into a segment's line list."""
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                lines.append([])
            if part:
                lines[-1].append((style, part))

    # ── scrolling ─────────────────────────────────────────────────

    def scroll_up(self, n: int = 1) -> None:
        self._scroll_offset += n
        self._invalidate()

    def scroll_down(self, n: int = 1) -> None:
        self._scroll_offset = max(0, self._scroll_offset - n)
        self._invalidate()

    def scroll_bottom(self) -> None:
        self._scroll_offset = 0
        self._invalidate()

    # ── live activity indicator (what is MForege touching right now) ──

    def set_activity(self, text: str | None) -> None:
        """Pin a one-line 'MForege is editing X' indicator above the input."""
        self._activity = text
        self._invalidate()

    # ── light/dark theme toggle (/theme:toggle) ───────────────────

    _DARK_STYLE = {
        "md-bold": "bold #ffffff",
        "md-italic": "italic #d0d0d0",
        "md-code": "#ff9d5c",
        "md-code-block": "#e8e8e8",
        "md-header": "bold #5fd7ff",
        "md-subheader": "bold #87d7ff",
        "md-bullet": "#5fd7ff",
        "md-quote": "#af87ff",
        "md-link": "#5fff87 underline",
        "md-dim": "#6c6c6c",
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
        "think": "italic #878787",
    }
    _LIGHT_STYLE = {
        "md-bold": "bold #000000",
        "md-italic": "italic #333333",
        "md-code": "#b34700",
        "md-code-block": "#222222",
        "md-header": "bold #0052a3",
        "md-subheader": "bold #3377bb",
        "md-bullet": "#0052a3",
        "md-quote": "#6633aa",
        "md-link": "#007700 underline",
        "md-dim": "#888888",
        "separator": "#bbbbbb",
        "label": "#0066cc",
        "status": "bg:#006633 #ffffff bold",
        "question": "bg:#a15c00 #ffffff bold",
        "you": "#006600 bold",
        "assistant": "#0052a3 bold",
        "dim": "#888888",
        "ok": "#007700",
        "error": "#cc0000 bold",
        "add": "#007700",
        "del": "#cc0000",
        "plan": "#aa22aa",
        "title": "#0052a3 bold",
        "warn": "#8a6d00",
        "think": "italic #666666",
    }

    def toggle_theme(self) -> None:
        """Swap between dark and light palettes (Ctrl+T unaffected)."""
        self._dark_theme = not getattr(self, "_dark_theme", True)
        try:
            self.app.style = Style.from_dict(
                self._LIGHT_STYLE if not self._dark_theme else self._DARK_STYLE
            )
        except Exception:
            pass
        self._invalidate()

    # ── live in-place blocks (Thinking, TODOS) ────────────────────

    def stream_thinking(self, token: str) -> None:
        """
        Stream reasoning tokens into the italic 'Thinking' block. Rendered
        live like Codebuff's Thinking; collapses to a one-line summary when
        the answer starts — but the full text is KEPT (Ctrl+T re-expands).
        """
        try:
            seg = None
            if self._segments:
                last = self._segments[-1]
                if last["kind"] == "think" and not last.get("collapsed"):
                    seg = last
            if seg is None:
                # header line + empty body line (tokens stream into the body)
                seg = {"kind": "think", "lines": [[_THINK_HEADER], []], "collapsed": False}
                self._segments.append(seg)
            self._write_lines(seg["lines"], token, "class:think")
            self._invalidate()
        except Exception:
            pass

    def finish_thinking(self) -> None:
        """Collapse any open Thinking block(s) to a one-line summary."""
        try:
            for seg in self._segments:
                if seg["kind"] == "think" and not seg.get("collapsed"):
                    seg["collapsed"] = True
            self._invalidate()
        except Exception:
            pass

    def thinking_is_open(self) -> bool:
        try:
            return any(
                s["kind"] == "think" and not s.get("collapsed")
                for s in self._segments
            )
        except Exception:
            return False

    def toggle_thinking(self) -> None:
        """Ctrl+T: expand the newest collapsed Thinking block (or collapse)."""
        try:
            for seg in reversed(self._segments):
                if seg["kind"] == "think":
                    seg["collapsed"] = not seg.get("collapsed", False)
                    break
            self._invalidate()
        except Exception:
            pass

    def get_thinking_text(self) -> str:
        """Full reasoning text of all Thinking blocks (debug/logging)."""
        try:
            out: list[str] = []
            for seg in self._segments:
                if seg["kind"] == "think":
                    for ln in seg["lines"]:
                        if ln == [_THINK_HEADER]:
                            continue  # the '✻ Thinking…' header is not reasoning
                        out.append("".join(t for _, t in ln))
            return "\n".join(out).strip()
        except Exception:
            return ""

    def replace_block(self, name: str, fragments: list) -> None:
        """
        Draw or redraw a named block (e.g. 'TODOS') in the transcript.
        The block segment is identity-stable, so redraws replace content in
        place and can never clobber other segments (or get clobbered).
        """
        try:
            new_lines = self._as_lines(fragments)
            seg = self._blocks.get(name)
            if seg is None:
                seg = {"kind": "block", "name": name, "lines": new_lines}
                self._blocks[name] = seg
                self._segments.append(seg)
                self._segments.append(_new_text_seg())  # separator after block
            else:
                seg["lines"] = new_lines
            self._invalidate()
        except Exception:
            pass

    @staticmethod
    def _as_lines(fragments: list) -> list:
        """Convert [(style, text-with-\\n), ...] into internal line lists."""
        lines: list[list[tuple[str, str]]] = [[]]
        for style, text in fragments:
            parts = text.split("\n")
            for i, part in enumerate(parts):
                if i > 0:
                    lines.append([])
                if part:
                    lines[-1].append((style, part))
        return lines

    def exit(self) -> None:
        self.app.exit()

    # ── confirmation ──────────────────────────────────────────────────

    async def confirm(self, question: str) -> bool:
        """Show the question bar and wait for y/n (Enter defaults to no)."""
        self._question = question
        self._answer = None
        self._answer_event = asyncio.Event()
        self._invalidate()
        try:
            await self._answer_event.wait()
        finally:
            # Clear the bar even if the wait was cancelled (Esc-stop while
            # a confirmation is pending) — never leave a dangling question.
            self._question = None
            self._answer_event = None
            self._invalidate()
        return self._answer in ("y", "yes")

    def _question_fragments(self):
        if not self._question:
            return [("", "")]
        return [("class:question", f" ⚠ {self._question}   (type y/n + Enter)")]

    def _activity_fragments(self):
        if not self._activity:
            return [("", "")]
        return [("class:activity", f" ✻ {self._activity}")]  # same glyph as the Thinking header — one agent, one icon

    # ── status ────────────────────────────────────────────────────────

    def _status_fragments(self):
        # Reference-panel session ROW: '◆ <model> · <clock> │ ctx…' on the
        # left, stretchy class:status padding in the middle, and the
        # clickable '✕ End session' pinned to the RIGHT EDGE — the whole
        # row reads as one continuous green highlight (the window's own
        # char=' '/class:status background fills any residual width).
        # The pad width is computed from the live terminal width; on any
        # failure it degrades to 1 space (never breaks the chat).
        left = self._session_bar()
        if self._phase_started is not None:
            elapsed = max(0, int(time.monotonic() - self._phase_started))
            left += f" │ {_PHASE_PREFIX} {self._phase_label} {elapsed}s (Esc=stop)"
        if self._scroll_offset > 0:
            left += f" │ ↑ {self._scroll_offset} lines · End=bottom"

        end_txt = "✕ End session "
        try:
            cols = self.app.output.get_size().columns
        except Exception:
            cols = 80
        pad = max(1, cols - _display_width(left) - _display_width(end_txt))

        frags: list[tuple[str, str]] = [("class:status", left),
                                        ("class:status", " " * pad)]
        frags.append(self._end_session_fragments()[-1])
        return frags

    def _end_session_fragments(self):
        """The clickable, hover-reactive '✕ End session' fragment.

        CLICKABLE (MOUSE_DOWN quits, same as /exit / Ctrl+C) and
        HOVER-REACTIVE (MOUSE_MOVE flips the reverse-video highlight;
        MOUSE_UP clears it). MOUSE_MOVE arrives on both event paths
        (VT100 ?1003h any-motion tracking and Windows MOUSE_MOVED
        records), so this works in VS Code, Windows Terminal and plain
        cmd.
        """
        def _end_session(mouse_event) -> None:
            try:
                if mouse_event.event_type == MouseEventType.MOUSE_DOWN:
                    self.exit()
                elif mouse_event.event_type == MouseEventType.MOUSE_MOVE:
                    if not self._end_session_hover:
                        self._end_session_hover = True
                        self._invalidate()
                elif self._end_session_hover:  # MOUSE_UP (and any other event)
                    self._end_session_hover = False
                    self._invalidate()
            except Exception:
                pass  # display must never break the chat

        style = "class:session-end class:session-hover" if self._end_session_hover \
            else "class:session-end"
        return [(style, "✕ End session ", _end_session)]

    def _session_bar(self) -> str:
        """Reference-panel session row: 'GLM 5.3 Flash · 1h left'.

        The model comes from the status text; the session clock counts up
        from UI start (set_session_start), or the fixed '1h left' style can
        be forced via _session_label for tests.
        """
        model = self._status.split("│")[0].strip() or "MForege"
        clock = self._session_label
        if clock is None and self._session_started is not None:
            mins = int((time.monotonic() - self._session_started) / 60)
            clock = f"{mins // 60}h {mins % 60}m" if mins >= 60 else f"{mins}m"
        return f" ◆ {model}" + (f" · {clock}" if clock else "")

    # ── transcript rendering (colored, scrollable viewport) ─────────

    def _materialized_lines(self) -> list:
        """Flatten segments to renderable lines (collapsed think → summary,
        md segments → live-rendered markdown)."""
        lines: list = []
        for seg in self._segments:
            if seg.get("collapsed"):
                lines.append([_THINK_SUMMARY])
                continue
            if seg["kind"] == "md":
                lines.extend(seg["lines"] or _render_markdown(seg["text"]))
                continue
            lines.extend(seg["lines"])
        # Anchored scrolling: while scrolled up, new lines must NOT slide
        # the viewport away from what the user is reading. (Skipped on the
        # first render so the baseline matches reality, not the constructor.)
        if self._scroll_offset > 0 and self._rendered_once:
            grew = len(lines) - self._last_line_count
            if grew > 0:
                self._scroll_offset += grew
        self._last_line_count = len(lines)
        self._rendered_once = True
        return lines

    def _visible_row_count(self) -> int:
        """How many transcript rows fit, given everything below the pane."""
        rows = 24
        try:
            rows = self.app.output.get_size().rows
        except Exception:
            pass
        # reserve = status(1) + question(1) + separator(1) + activity/menu
        #           (0..1) + input (2..5, dynamic) + breathing room
        reserve = 5 + self._input_height() + 2
        return max(1, rows - reserve)

    def _fragments(self):
        frags: list[tuple[str, str]] = []
        try:
            lines = self._materialized_lines()
            if self._scroll_offset == 0:
                # Follow the live tail — but ignore trailing blank
                # separator lines appended after each message, so a tiny
                # viewport still shows the newest real content.
                while lines and not lines[-1]:
                    lines.pop()
            visible = self._visible_row_count()

            if self._scroll_offset == 0:
                window = lines[-visible:]           # follow the live tail
            else:
                # Clamp so the top is reachable but never overshot.
                self._scroll_offset = min(self._scroll_offset, max(0, len(lines) - visible))
                end = len(lines) - self._scroll_offset
                start = max(0, end - visible)
                window = lines[start:end]

            for line in window:
                for style, text in line:
                    frags.append((style, text))
                frags.append(("", "\n"))
        except Exception:
            pass  # renderer must never break the chat
        return frags

    # ── main runner ───────────────────────────────────────────────────

    async def run(self, handler) -> None:
        """
        Run the UI forever, dispatching each submitted message to `handler`.
        Handler runs inside the app's event loop, so ui.append() calls from
        it (and from tool callbacks) render live.
        """
        worker_task = asyncio.get_event_loop().create_task(self._worker(handler))
        try:
            await self.app.run_async()
        finally:
            worker_task.cancel()

    async def _worker(self, handler) -> None:
        """Dispatch loop: run each queued message as a child task so Esc can
        cancel ONE message mid-flight without killing the worker loop."""
        while True:
            text = await self._queue.get()
            handler_task = asyncio.get_event_loop().create_task(handler(text))
            self._handler_task = handler_task
            try:
                await handler_task
            except asyncio.CancelledError:
                raise  # app shutdown — propagate
            except Exception as e:
                self.append(f"[!] Unexpected error: {e}", style="class:error")
            finally:
                self._handler_task = None
