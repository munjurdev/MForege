"""Tests for the terminal UI (ChatUI) — logic only, no real terminal needed"""
import asyncio
import sys

import pytest
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.mouse_events import MouseEventType

from app.ui import ChatUI, COMMANDS


class _SmallDummyOutput(DummyOutput):
    """DummyOutput reporting a deterministic 10-row terminal so scroll
    math is exact in tests (DummyOutput's default is 40x80)."""
    def get_size(self):
        from types import SimpleNamespace
        return SimpleNamespace(rows=10, columns=80)


@pytest.fixture
def ui():
    return ChatUI(status_text="test", output=DummyOutput())


@pytest.fixture
def ui_small():
    return ChatUI(status_text="test", output=_SmallDummyOutput())


def _text(ui) -> str:
    return "".join(t for _, t in ui._fragments())


class TestDynamicInputHeight:
    def test_empty_is_one_row(self, ui):
        ui.input.text = ""
        assert ui._input_height() == 1

    def test_single_line_is_one_row(self, ui):
        ui.input.text = "hello"
        assert ui._input_height() == 1

    def test_grows_with_lines(self, ui):
        ui.input.text = "line1\nline2"
        assert ui._input_height() == 2
        ui.input.text = "a\nb\nc"
        assert ui._input_height() == 3

    def test_caps_at_max(self, ui):
        ui.input.text = "\n".join(f"line{i}" for i in range(20))
        assert ui._input_height() == ChatUI.MAX_INPUT_ROWS

    def test_height_is_a_callable_dimension(self, ui):
        # TextArea passes height through to its inner Window's dimension
        from prompt_toolkit.layout.containers import Window
        assert isinstance(ui.input.window, Window)
        assert callable(ui.input.window.height) or ui.input.window.height is not None


class TestTranscript:
    def test_append_line(self, ui):
        ui.append("You: hi", style="class:you")
        assert "You: hi" in _text(ui)

    def test_append_without_newline_streams_into_last_line(self, ui):
        ui.append("MForege: ", end="", style="class:assistant")
        ui.append("hello ", end="")
        ui.append("world", end="")
        ui.append("")  # finalize the line
        assert "MForege: hello world" in _text(ui)

    def test_clear(self, ui):
        ui.append("something")
        ui.clear()
        assert ui._segments == [{"kind": "text", "lines": [[]]}]

    def test_broken_ui_hook_does_not_crash_append(self, ui):
        # _fragments raising must not break append callers
        ui._segments = None  # corrupt internals on purpose
        ui.append("x")  # must not raise


class TestScrolling:
    def test_offset_zero_follows_tail(self, ui_small):
        for i in range(50):
            ui_small.append(f"line {i}")
        text = "".join(t for _, t in ui_small._fragments())
        assert "line 49" in text
        assert "line 0" not in text

    def test_scroll_up_reveals_history(self, ui_small):
        for i in range(50):
            ui_small.append(f"line {i}")
        ui_small.scroll_up(30)
        text = "".join(t for _, t in ui_small._fragments())
        assert "line 49" not in text   # live tail is out of view
        assert "line 19" in text       # history window moved up instead

    def test_scroll_bottom_snaps_back_to_tail(self, ui_small):
        for i in range(50):
            ui_small.append(f"line {i}")
        ui_small.scroll_up(30)
        ui_small.scroll_bottom()
        assert "line 49" in "".join(t for _, t in ui_small._fragments())

    def test_scroll_down_clamps_at_zero(self, ui_small):
        for i in range(50):
            ui_small.append(f"line {i}")
        ui_small.scroll_down(5)  # already at bottom
        assert ui_small._scroll_offset == 0

    def test_huge_upscroll_never_shows_blank_pane(self, ui_small):
        for i in range(60):
            ui_small.append(f"line {i}")
        ui_small.scroll_up(9999)  # way past the top
        text = "".join(t for _, t in ui_small._fragments())
        assert "line 0" in text       # top of history is reachable…
        assert "line 59" not in text  # …but not overshot into blankness

    def test_new_content_shifts_viewport_not_context(self, ui_small):
        # Scrolled up 5 lines; 5 new lines arrive → identical viewport
        # (anchored scrolling — what you're reading must not move)
        for i in range(30):
            ui_small.append(f"old {i}")
        ui_small.scroll_up(5)
        before = "".join(t for _, t in ui_small._fragments())
        for i in range(5):
            ui_small.append(f"new {i}")
        assert before == "".join(t for _, t in ui_small._fragments())
        # …and the fresh output is one PgDn/End away
        ui_small.scroll_bottom()
        assert "new 4" in "".join(t for _, t in ui_small._fragments())

    def test_status_bar_shows_scroll_hint_only_when_scrolled(self, ui):
        # scroll indicator: '↑ N lines · End=bottom' while scrolled up
        assert "lines · End" not in ui._status_fragments()[0][1]
        ui.append("x")
        ui.scroll_up(3)
        assert "↑ 3 lines · End=bottom" in ui._status_fragments()[1][1]
        ui.scroll_bottom()
        assert "lines · End" not in ui._status_fragments()[0][1]


class TestActivityIndicator:
    def test_hidden_by_default(self, ui):
        assert ui._activity_fragments() == [("", "")]

    def test_shown_while_tool_runs(self, ui):
        ui.set_activity("read_file(path=settings.py)")
        assert "read_file(path=settings.py)" in ui._activity_fragments()[0][1]

    def test_cleared_after_tool_ends(self, ui):
        ui.set_activity("read_file(path=settings.py)")
        ui.set_activity(None)
        assert ui._activity_fragments() == [("", "")]

    def test_set_activity_never_raises_when_corrupted(self, ui):
        ui.app = None  # simulate a broken app reference
        ui.set_activity("boom")  # must not raise


class TestThinkingBlock:
    def test_streaming_creates_block_then_collapses(self, ui):
        ui.stream_thinking("I need to ")
        ui.stream_thinking("check the file")
        text = _text(ui)
        assert "check the file" in text        # full reasoning visible while open
        ui.finish_thinking()
        text = _text(ui)
        assert "check the file" not in text    # hidden while collapsed…
        assert "thought for a moment" in text  # …replaced by the marker

    def test_collapsed_thinking_preserves_text_and_expands(self, ui):
        ui.stream_thinking("secret reasoning steps")
        ui.finish_thinking()
        assert "secret reasoning steps" not in _text(ui)
        ui.toggle_thinking()                    # Ctrl+T behavior
        assert "secret reasoning steps" in _text(ui)
        assert "thought for a moment" not in _text(ui)
        ui.toggle_thinking()                    # collapse again
        assert "secret reasoning steps" not in _text(ui)

    def test_get_thinking_text_keeps_reasoning(self, ui):
        ui.stream_thinking("alpha ")
        ui.stream_thinking("beta")
        ui.finish_thinking()
        assert ui.get_thinking_text() == "alpha beta"

    def test_finish_without_streaming_is_noop(self, ui):
        ui.finish_thinking()  # must not raise or add lines
        assert _text(ui).strip() == ""

    def test_new_thinking_after_finish_starts_fresh_block(self, ui):
        ui.stream_thinking("first")
        ui.finish_thinking()
        ui.stream_thinking("second")
        text = _text(ui)
        assert "thought for a moment" in text   # old block stays collapsed
        assert "second" in text                 # new block streams open

    def test_clear_resets_thinking_state(self, ui):
        ui.stream_thinking("hmm")
        ui.clear()
        ui.stream_thinking("fresh")  # starts a new block, no crash
        assert "fresh" in _text(ui)

    def test_collapse_never_wipes_other_segments(self, ui):
        """Regression: collapsing Thinking used to wipe a TODOS block drawn
        earlier in the transcript (index-based line surgery bug)."""
        ui.append("You: plan it", style="class:you")
        ui.replace_block("TODOS", [("class:plan", " TODOS\n"), ("class:plan", " ✎ step\n")])
        ui.stream_thinking("thinking hard about the plan")
        ui.append("MForege: done", style="class:assistant")
        ui.finish_thinking()
        text = _text(ui)
        assert "TODOS" in text                  # block survived ✓
        assert "✎ step" in text
        assert "MForege: done" in text          # later text survived ✓
        assert "thinking hard" not in text      # thinking collapsed ✓
        ui.toggle_thinking()
        assert "thinking hard" in _text(ui)     # and can come back


class TestLiveBlocks:
    def test_draw_and_redraw_in_place(self, ui):
        ui.replace_block("TODOS", [("class:plan", " TODOS\n"), ("class:plan", " ✎ step one\n")])
        seg_count = len(ui._segments)
        ui.replace_block("TODOS", [("class:plan", " TODOS\n"), ("class:ok", " ✓ step one\n")])
        text = _text(ui)
        assert "✓ step one" in text
        assert "✎ step one" not in text          # old render replaced
        # redraw must NOT grow the transcript with duplicate blocks
        assert len(ui._segments) == seg_count

    def test_multiple_blocks_coexist(self, ui):
        ui.replace_block("A", [("", " block a\n")])
        ui.replace_block("B", [("", " block b\n")])
        text = _text(ui)
        assert "block a" in text and "block b" in text

    def test_clear_removes_blocks(self, ui):
        ui.replace_block("TODOS", [("", " TODOS\n")])
        ui.clear()
        assert ui._blocks == {}
        assert "TODOS" not in _text(ui)

    def test_text_after_block_goes_to_new_segment(self, ui):
        ui.replace_block("TODOS", [("", " TODOS\n")])
        ui.append("after the block")
        text = _text(ui)
        assert "TODOS" in text and "after the block" in text


class TestShiftEnterWindowsPatch:
    """The Windows Shift+Enter compatibility patch (app/ui.py, applied at import)."""

    def test_vt_sequence_maps_to_controlj(self):
        pytest.importorskip("prompt_toolkit.input.ansi_escape_sequences")
        import app.ui as ui_mod  # ensure patch applied
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.keys import Keys
        if sys.platform == "win32":
            # keys must be str (parser feeds str); a bytes key crashes the
            # vt100 prefix cache at runtime (regression: real crash report)
            seq = ANSI_SEQUENCES.get("\x1b[13;2u")
            assert seq == Keys.ControlJ
            assert not any(isinstance(k, bytes) for k in ANSI_SEQUENCES), (
                "bytes key in ANSI_SEQUENCES poisons the prefix cache"
            )

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows console path")
    def test_console_shift_enter_becomes_newline(self):
        import app.ui as ui_mod  # ensure patch applied
        from prompt_toolkit.input.win32 import ConsoleInputReader
        from prompt_toolkit.win32_types import KEY_EVENT_RECORD
        from prompt_toolkit.keys import Keys

        def make_event(shift):
            ev = KEY_EVENT_RECORD()
            ev.KeyDown = 1
            ev.VirtualKeyCode = 13
            ev.uChar.UnicodeChar = "\r"
            ev.ControlKeyState = 0x0010 if shift else 0
            return ev

        reader = ConsoleInputReader.__new__(ConsoleInputReader)
        s = ConsoleInputReader._event_to_key_presses(reader, make_event(True))
        assert [str(k.key) for k in s] == [str(Keys.ControlJ)]
        p = ConsoleInputReader._event_to_key_presses(reader, make_event(False))
        assert [str(k.key) for k in p] == [str(Keys.ControlM)]

    def test_alt_enter_bound_again(self, ui):
        """Alt+Enter (escape,enter) must be bound: VS Code's terminal never
        transmits Shift+Enter to programs, so Alt+Enter is the only
        distinguishable newline path there."""
        found = False
        for b in ui.app.key_bindings.bindings:
            keys = list(b.keys)
            names = [str(k) for k in keys]
            if (
                len(keys) == 2
                and "Escape" in names[0]
                and ("ControlM" in names[1] or "Enter" in names[1])
            ):
                found = True
        assert found, f"Alt+Enter binding missing (looked through registry)"

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows VT-reader path")
    def test_vt_reader_shift_enter_becomes_newline(self):
        """Regression for the REAL VS Code/Windows Terminal path:
        Vt100ConsoleInputReader._get_keys used to discard ControlKeyState,
        so Shift+Enter reached the parser as bare \r (= send)."""
        import app.ui as ui_mod  # ensure patch applied
        from ctypes import c_long
        from prompt_toolkit.input.win32 import Vt100ConsoleInputReader
        from prompt_toolkit.win32_types import (
            KEY_EVENT_RECORD, EVENT_RECORD, INPUT_RECORD,
        )
        from prompt_toolkit.input.vt100_parser import (
            Vt100Parser, _IS_PREFIX_OF_LONGER_MATCH_CACHE,
        )
        from prompt_toolkit.keys import Keys

        _IS_PREFIX_OF_LONGER_MATCH_CACHE.clear()

        def make_record(char, state):
            ev = KEY_EVENT_RECORD()
            ev.KeyDown = 1
            ev.VirtualKeyCode = 13 if char == "\r" else 0
            ev.uChar.UnicodeChar = char
            ev.ControlKeyState = state
            er = EVENT_RECORD()
            er.KeyEvent = ev
            ir = INPUT_RECORD()
            ir.EventType = 1
            ir.Event = er
            return ir

        reader = Vt100ConsoleInputReader()
        # Shift+Enter → "\n" → parses to ControlJ (newline binding)
        chars = list(reader._get_keys(
            c_long(1), (INPUT_RECORD * 1)(make_record("\r", 0x0010))))
        assert chars == ["\n"]
        got = []
        p = Vt100Parser(lambda kp: got.append(kp))
        for ch in chars:
            p.feed(ch)
        p.flush()
        assert any(str(k.key) == str(Keys.ControlJ) for k in got)
        # plain Enter untouched
        assert list(reader._get_keys(
            c_long(1), (INPUT_RECORD * 1)(make_record("\r", 0)))) == ["\r"]


class TestStatusAndQuestion:
    def test_status_fragment(self, ui):
        ui.set_status("thinking…")
        assert "thinking…" in ui._status_fragments()[0][1]

    def test_question_hidden_by_default(self, ui):
        assert ui._question_fragments() == [("", "")]

    def test_question_shown_when_pending(self, ui):
        ui._question = "CREATE file 'x.txt'"
        assert "CREATE file 'x.txt'" in ui._question_fragments()[0][1]


class TestSlashMenu:
    """Freebuff-style slash-command popup menu."""

    def test_menu_empty_by_default(self, ui):
        assert ui._menu_items == []

    def test_slash_opens_filtered_menu(self, ui):
        ui.input.text = "/"
        ui._refresh_menu()
        assert len(ui._menu_items) == len(COMMANDS)

    def test_filter_narrows(self, ui):
        ui.input.text = "/re"
        ui._refresh_menu()
        cmds = [c for c, _ in ui._menu_items]
        assert "/review" in cmds and "/resume" in cmds
        assert "/help" not in cmds

    def test_exact_match_hides_menu(self, ui):
        ui.input.text = "/plan"
        ui._refresh_menu()
        assert ui._menu_items == []

    def test_space_or_multiline_hides_menu(self, ui):
        ui.input.text = "/reasoning low"
        ui._refresh_menu()
        assert ui._menu_items == []
        ui.input.text = "/plan\n"
        ui._refresh_menu()
        assert ui._menu_items == []

    def test_move_wraps(self, ui):
        ui.input.text = "/"
        ui._refresh_menu()
        n = len(ui._menu_items)
        ui.menu_move(-1)
        assert ui._menu_index == n - 1
        ui.menu_move(1)
        assert ui._menu_index == 0

    def test_accept_fills_input_and_closes(self, ui):
        ui.input.text = "/pla"
        ui._refresh_menu()
        ui._menu_index = 0
        ui.menu_accept()
        assert ui.input.text.startswith("/plan")
        assert ui._menu_items == []

    def test_menu_fragments_render_rows(self, ui):
        ui.input.text = "/"
        ui._refresh_menu()
        frags = ui._menu_fragments()
        text = "".join(t for _, t in frags)
        assert "/help" in text and "Display keyboard shortcuts" in text

    def test_menu_height_capped_at_six(self, ui):
        ui.input.text = "/"
        ui._refresh_menu()
        assert ui._menu_height() == 6

    def test_bindings_registered(self, ui):
        from prompt_toolkit.keys import Keys
        binding_keys = {k for b in ui.app.key_bindings.bindings for k in b.keys}
        assert Keys.Up.value in binding_keys
        assert Keys.Down.value in binding_keys
        assert Keys.Escape.value in binding_keys


class TestBoxedWelcome:
    def test_boxed_shapes(self):
        from app.ui import boxed
        out = boxed(["hello", "world is longer"])
        lines = out.split("\n")
        assert len(lines) == 4
        assert lines[0].startswith("┌") and lines[0].endswith("┐")
        assert lines[-1].startswith("└") and lines[-1].endswith("┘")
        assert all(lines[1].startswith("│") and lines[1].endswith("│")
                   for _ in [1])
        widths = {len(lines[0]), len(lines[1]), len(lines[3])}
        assert len(widths) == 1  # every border line equal width

    def test_boxed_pads_to_equal_width(self):
        from app.ui import boxed
        out = boxed(["a", "bb"])
        assert len(out.split("\n")[1]) == len(out.split("\n")[2])

    def test_boxed_handles_wide_chars(self):
        from app.ui import boxed
        out = boxed(["✦ ok"])
        assert len(out.split("\n")[0]) == len(out.split("\n")[-1])


class TestScrollBindingsEager:
    """Regression: prompt_toolkit's basic bindings map pageup/pagedown to
    buffer-history on the focused input and outrank app-level bindings,
    silently eating the scroll keys. They must be registered eager=True."""

    def _eager_for(self, ui, key):
        from prompt_toolkit.keys import Keys as _K
        key = getattr(_K, key) if isinstance(key, str) else key
        hits = [b for b in ui.app.key_bindings.bindings if key in b.keys]
        assert hits, f"no binding registered for {key}"
        return any(b.eager() for b in hits)

    def test_pageup_eager(self, ui):
        assert self._eager_for(ui, "PageUp")

    def test_pagedown_eager(self, ui):
        assert self._eager_for(ui, "PageDown")

    def test_ctrl_up_down_eager(self, ui):
        assert self._eager_for(ui, "ControlUp")
        assert self._eager_for(ui, "ControlDown")

    def test_end_eager(self, ui):
        assert self._eager_for(ui, "End")

    def test_scroll_still_moves_viewport(self, ui):
        for i in range(40):
            ui.append(f"line {i}")
        ui.scroll_up(10)
        text = _text(ui)
        # viewport left the live tail…
        assert "line 39" not in text
        # …and End/scroll_bottom returns to it
        ui.scroll_bottom()
        assert "line 39" in _text(ui)


class TestMouseWheelScroll:
    """Mouse wheel scrolls the transcript viewport (both event paths)."""

    def _wheel(self, ui, event_type):
        from prompt_toolkit.mouse_events import (
            MouseEvent, MouseButton,
        )
        from prompt_toolkit.data_structures import Point
        ev = MouseEvent(
            position=Point(0, 0),
            event_type=event_type,
            button=MouseButton.NONE,
            modifiers=frozenset(),
        )
        return ui._transcript_ctrl.mouse_handler(ev)

    def test_wheel_up_moves_viewport(self, ui):
        for i in range(40):
            ui.append(f"line {i}")
        assert self._wheel(ui, MouseEventType.SCROLL_UP) is None
        assert ui._scroll_offset == 3  # _WHEEL_STEP
        assert "line 39" not in _text(ui)

    def test_wheel_down_returns_to_tail(self, ui):
        for i in range(40):
            ui.append(f"line {i}")
        self._wheel(ui, MouseEventType.SCROLL_UP)
        self._wheel(ui, MouseEventType.SCROLL_DOWN)
        assert ui._scroll_offset == 0
        assert "line 39" in _text(ui)

    def test_click_passthrough_unhandled(self, ui):
        # Non-scroll events must stay NotImplemented so the Window (or
        # nothing) handles them — wheel override must not swallow clicks.
        assert self._wheel(ui, MouseEventType.MOUSE_DOWN) is NotImplemented

    def test_positionless_scroll_keys_bound(self, ui):
        # Terminals without cursor-position wheel events feed Keys.ScrollUp/
        # ScrollDown; they must be bound (and eager, not feeding arrows).
        from prompt_toolkit.keys import Keys
        bound = {k for b in ui.app.key_bindings.bindings for k in b.keys}
        assert Keys.ScrollUp in bound and Keys.ScrollDown in bound
        scrolls = [b for b in ui.app.key_bindings.bindings
                   if Keys.ScrollUp in b.keys]
        assert all(b.eager() for b in scrolls)


class TestScrollIndicator:
    """Status bar shows '↑ N lines · End=bottom' while scrolled up."""

    def test_hidden_at_bottom(self, ui):
        assert "lines · End" not in "".join(t for _, t in ui._status_fragments())

    def test_shows_offset_while_scrolled(self, ui):
        ui.scroll_up(12)
        status = "".join(t for _, t in ui._status_fragments())
        assert "↑ 12 lines · End=bottom" in status

    def test_updates_with_offset(self, ui):
        ui.scroll_up(1)
        assert "↑ 1 lines" in "".join(t for _, t in ui._status_fragments())
        ui.scroll_up(4)
        assert "↑ 5 lines" in "".join(t for _, t in ui._status_fragments())

    def test_hidden_again_after_bottom(self, ui):
        ui.scroll_up(7)
        ui.scroll_bottom()
        assert "lines · End" not in "".join(t for _, t in ui._status_fragments())


class TestScrollBarMargin:
    """Scrollbar margin on the transcript's right edge (own viewport state)."""

    def _sb(self, ui):
        from app.ui import TranscriptScrollBar
        return TranscriptScrollBar(ui)

    def _margin(self, ui, sb, height=12):
        return "".join(c for _, c in sb.create_margin(None, 1, height)[:height])

    def test_hidden_when_content_fits(self, ui):
        ui.append("hello")
        assert self._sb(ui).get_width(lambda: None) == 0

    def test_visible_when_content_overflows(self, ui):
        for i in range(60):
            ui.append(f"line {i}")
        assert self._sb(ui).get_width(lambda: None) == 1

    def test_thumb_at_bottom_when_viewing_tail(self, ui):
        for i in range(60):
            ui.append(f"line {i}")
        m = self._margin(ui, self._sb(ui))
        assert m.endswith("██") and m.startswith("│")

    def test_thumb_moves_up_when_scrolled(self, ui):
        for i in range(60):
            ui.append(f"line {i}")
        sb = self._sb(ui)
        at_bottom = self._margin(ui, sb)
        ui.scroll_up(20)
        scrolled = self._margin(ui, sb)
        assert scrolled != at_bottom
        assert scrolled.index("█") < at_bottom.index("█")

    def test_thumb_returns_after_bottom(self, ui):
        for i in range(60):
            ui.append(f"line {i}")
        sb = self._sb(ui)
        at_bottom = self._margin(ui, sb)
        ui.scroll_up(15)
        ui.scroll_bottom()
        assert self._margin(ui, sb) == at_bottom

    def test_window_has_right_margin(self, ui):
        from app.ui import TranscriptScrollBar
        margins = getattr(ui.transcript, "right_margins", [])
        assert any(isinstance(m, TranscriptScrollBar) for m in margins)
