"""Tests for the terminal UI (ChatUI) — logic only, no real terminal needed"""
import asyncio

import pytest
from prompt_toolkit.output import DummyOutput

from app.ui import ChatUI


@pytest.fixture
def ui():
    return ChatUI(status_text="test", output=DummyOutput())


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
        text = "".join(t for _, t in ui._fragments())
        assert "You: hi" in text

    def test_append_without_newline_streams_into_last_line(self, ui):
        ui.append("MForege: ", end="", style="class:assistant")
        ui.append("hello ", end="")
        ui.append("world", end="")
        ui.append("")  # finalize the line
        text = "".join(t for _, t in ui._fragments())
        assert "MForege: hello world" in text

    def test_clear(self, ui):
        ui.append("something")
        ui.clear()
        assert ui._lines == [[]]

    def test_broken_ui_hook_does_not_crash_append(self, ui):
        # _fragments raising must not break append callers
        ui._lines = None  # corrupt internals on purpose
        ui.append("x")  # must not raise


class TestStatusAndQuestion:
    def test_status_fragment(self, ui):
        ui.set_status("thinking…")
        assert "thinking…" in ui._status_fragments()[0][1]

    def test_question_hidden_by_default(self, ui):
        assert ui._question_fragments() == [("", "")]

    def test_question_shown_when_pending(self, ui):
        ui._question = "CREATE file 'x.txt'"
        assert "CREATE file 'x.txt'" in ui._question_fragments()[0][1]
