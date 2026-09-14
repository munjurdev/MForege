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
plan=magenta, thinking=italic gray, errors=red. The app runs continuously,
so tool activity appears live while the model works (same as an IDE panel).

Transcript model (important): the transcript is a list of SEGMENTS, not a
flat line list. Three segment kinds:
  - "text"  — ordinary transcript lines (append)
  - "think" — the Thinking block; collapsible WITHOUT losing its content
  - "block" — named live blocks (TODOS) redrawn in place
Because segments are variable-height objects, mutating one can never
corrupt another (e.g. collapsing Thinking no longer wipes TODOS), and
thinking text is preserved for later expansion.
"""

import asyncio
import sys
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

_THINK_SUMMARY = ("class:think", " ✦ thought for a moment — Ctrl+T expands")

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

        # Transcript pane (colored fragments, scrollable via PgUp/PgDn
        # and the mouse wheel)
        self._transcript_ctrl = TranscriptControl(self._fragments)
        self._transcript_ctrl._ui = self
        self.transcript = Window(
            content=self._transcript_ctrl,
            wrap_lines=True,
            right_margins=[TranscriptScrollBar(self)],
        )

        # Input box — grows with content (1–5 rows), drawn inside a bordered
        # frame with the label embedded in the top border (IDE-style).
        self.input = TextArea(height=self._input_height, multiline=True, wrap_lines=True)
        self.input_frame = Frame(self.input, title=" message ", style="class:inputbox")

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

        # Status bar
        self.status_win = Window(
            content=FormattedTextControl(self._status_fragments),
            height=1,
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

        layout = HSplit([
            self.transcript,
            Window(height=1, char="─", style="class:separator"),
            self.activity_win,
            self.menu_win,
            self.input_frame,
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
            }),
            # Mouse support: wheel scrolls the transcript (TranscriptControl
            # + ScrollUp/Down bindings). Click-to-position stays handled by
            # the input's own control.
            mouse_support=True,
        )
        if output is not None:
            app_kwargs["output"] = output

        self.app: Application = Application(**app_kwargs)

    # ── input sizing ─────────────────────────────────────────────────

    MIN_INPUT_ROWS = 1
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
        """Dynamic height: 1 row + extra lines typed, capped at MAX_INPUT_ROWS."""
        try:
            lines = self.input.text.count("\n") + 1
        except Exception:
            return self.MIN_INPUT_ROWS
        return max(self.MIN_INPUT_ROWS, min(self.MAX_INPUT_ROWS, lines))

    # ── transcript ────────────────────────────────────────────────────

    def append(self, text: str, end: str = "\n", style: str | None = None) -> None:
        """Append text (optionally styled) to the transcript. Never raises."""
        try:
            seg = None
            if self._segments:
                last = self._segments[-1]
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
        self._scroll_offset = 0
        self._invalidate()

    def set_status(self, text: str) -> None:
        self._status = text
        self._invalidate()

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
                seg = {"kind": "think", "lines": [[]], "collapsed": False}
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
        await self._answer_event.wait()
        self._answer_event = None
        return self._answer in ("y", "yes")

    def _question_fragments(self):
        if not self._question:
            return [("", "")]
        return [("class:question", f" ⚠ {self._question}   (type y/n + Enter)")]

    def _activity_fragments(self):
        if not self._activity:
            return [("", "")]
        return [("class:activity", f" ✸ {self._activity}")]

    # ── status ────────────────────────────────────────────────────────

    def _status_fragments(self):
        # Scroll position indicator: while the viewport is away from the
        # live tail, show how far up the user is (anchored-scroll aware —
        # the number grows as new lines arrive, so it never lies).
        frags: list[tuple[str, str]] = [
            ("class:status", f" ◆ MForege │ {self._status}")
        ]
        if self._scroll_offset > 0:
            frags.append(("class:scroll",
                          f" │ ↑ {self._scroll_offset} lines · End=bottom"))
        frags.append(("class:status",
                      " │ Enter=send │ Shift/Alt+Enter=newline │ Ctrl+T=thinking │ Ctrl+C=quit "))
        return frags

    # ── transcript rendering (colored, scrollable viewport) ─────────

    def _materialized_lines(self) -> list:
        """Flatten segments to renderable lines (collapsed think → summary)."""
        lines: list = []
        for seg in self._segments:
            if seg.get("collapsed"):
                lines.append([_THINK_SUMMARY])
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
        # reserve = label(1) + separators(2) + status(1) + question(1)
        #           + input (1..5, dynamic) + breathing room
        reserve = 5 + self._input_height() + 2
        return max(1, rows - reserve)

    def _fragments(self):
        frags: list[tuple[str, str]] = []
        try:
            lines = self._materialized_lines()
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
