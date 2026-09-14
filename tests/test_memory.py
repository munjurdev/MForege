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
