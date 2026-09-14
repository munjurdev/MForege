"""Tests for ConversationMemory"""
import pytest
from datetime import datetime

from app.agent.memory import ConversationMemory, Message


def make_msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


class TestMessage:
    def test_to_dict_contains_role_and_content(self):
        msg = Message(role="user", content="hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"
        assert "timestamp" in d

    def test_timestamp_defaults_to_now(self):
        before = datetime.now()
        msg = Message(role="user", content="x")
        assert before <= msg.timestamp <= datetime.now()

    def test_tool_call_id_optional(self):
        msg = Message(role="tool", content="42", tool_call_id="call_1")
        assert msg.tool_call_id == "call_1"


class TestConversationMemory:
    def test_add_and_count(self):
        mem = ConversationMemory()
        mem.add(make_msg("user", "hi"))
        mem.add(make_msg("assistant", "hello"))
        assert mem.count == 2

    def test_trim_respects_max_messages(self):
        mem = ConversationMemory(max_messages=4)
        for i in range(10):
            mem.add(make_msg("user", f"msg {i}"))
        assert mem.count == 4
        # Keeps the most recent ones
        assert mem.messages[-1].content == "msg 9"
        assert mem.messages[0].content == "msg 6"

    def test_trim_with_zero_budget_keeps_nothing(self):
        # Regression: negative slice previously kept everything
        mem = ConversationMemory(max_messages=0)
        mem.add(make_msg("user", "hi"))
        assert mem.count == 0

    def test_clear_removes_non_system(self):
        mem = ConversationMemory()
        mem.add(make_msg("user", "hi"))
        mem.add(make_msg("assistant", "hello"))
        mem.clear()
        assert mem.count == 0

    def test_get_filter_by_role_and_limit(self):
        mem = ConversationMemory()
        mem.add(make_msg("user", "a"))
        mem.add(make_msg("assistant", "b"))
        mem.add(make_msg("user", "c"))
        assert [m.content for m in mem.get(role="user")] == ["a", "c"]
        assert [m.content for m in mem.get(limit=2)] == ["b", "c"]

    def test_last_user_message(self):
        mem = ConversationMemory()
        mem.add(make_msg("user", "question"))
        mem.add(make_msg("assistant", "answer"))
        assert mem.last_user_message == "question"

    def test_last_user_message_empty(self):
        mem = ConversationMemory()
        assert mem.last_user_message is None
        assert mem.last_message() is None


class TestAutoCondense:
    """Auto-condensation: old turns fold into a summary (session-summary style)"""

    def _fill(self, n=30):
        m = ConversationMemory(max_messages=200)
        for i in range(n):
            m.add(Message(role="user", content=f"Q{i} " + "x" * 120))
            m.add(Message(role="assistant", content=f"A{i} " + "y" * 120))
        return m

    def test_no_condense_when_below_threshold(self):
        m = self._fill(3)
        assert m.maybe_condense(131072) is False
        assert m.summary == ""

    def test_condenses_when_over_threshold(self):
        m = self._fill(30)
        m.CONDENSE_AT = 0.0
        assert m.maybe_condense(2000) is True
        assert m.summary != ""
        assert len(m.messages) == ConversationMemory.KEEP_RECENT
        # recent kept verbatim
        assert m.messages[-1].content.startswith("A29")

    def test_summary_injected_into_api_messages(self):
        from app.agent.memory import ConversationMemory as CM  # noqa: F401
        m = self._fill(30)
        m.CONDENSE_AT = 0.0
        m.maybe_condense(2000)
        msgs = [
            {"role": "system", "content": "sys"},
            *m.to_openai_format(),
        ]
        # the agent injects summary separately; here just verify summary text
        assert "Q0" in m.summary and "A5" in m.summary

    def test_idempotent_no_re_fold_of_recent(self):
        m = self._fill(30)
        m.CONDENSE_AT = 0.0
        m.maybe_condense(2000)
        before = list(m.messages)
        # still over threshold (summary counts), but only KEEP_RECENT remain
        folded_again = m.maybe_condense(2000)
        # with only KEEP_RECENT messages left, nothing more to fold
        assert folded_again is False
        assert [x.content for x in m.messages] == [x.content for x in before]

    def test_token_estimate_counts_summary(self):
        m = ConversationMemory()
        m.add(Message(role="user", content="a" * 400))
        base = m.token_estimate()
        m.summary = "s" * 400
        assert m.token_estimate() > base

    def test_zero_window_is_safe(self):
        m = self._fill(5)
        assert m.maybe_condense(0) is False
