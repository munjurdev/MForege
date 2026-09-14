"""Tests for LongTermMemory (file-backed, no network)"""
import json
import os

import pytest

from app.agent.long_term_memory import (
    LongTermMemory,
    extract_facts,
    _parse_facts,
    MAX_FACTS,
)


@pytest.fixture
def mem_path(tmp_path):
    return str(tmp_path / "memory.json")


class TestStore:
    def test_add_and_all(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        assert mem.add("User's name is Munjur")
        assert mem.all() == ["User's name is Munjur"]
        assert len(mem) == 1

    def test_persists_across_instances(self, mem_path):
        LongTermMemory(path=mem_path).add("User likes Python")
        mem2 = LongTermMemory(path=mem_path)
        assert mem2.all() == ["User likes Python"]

    def test_missing_file_starts_empty(self, mem_path):
        assert LongTermMemory(path=mem_path).all() == []

    def test_corrupt_file_starts_empty(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        mem = LongTermMemory(path=str(bad))
        assert mem.all() == []
        # and it can recover: adding still works
        assert mem.add("recovered")

    def test_duplicate_rejected(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        assert mem.add("User's name is Munjur")
        assert not mem.add("User's name is Munjur")
        # near-duplicate: punctuation/case differences
        assert not mem.add("user's Name is Munjur!")
        assert len(mem) == 1

    def test_empty_fact_rejected(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        assert not mem.add("")
        assert not mem.add("   ")
        assert not mem.add(None)

    def test_long_fact_truncated(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        fact = "x" * 1000
        assert mem.add(fact)
        assert len(mem.all()[0]) == 300

    def test_max_facts_eviction(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        for i in range(MAX_FACTS + 10):
            mem.add(f"Fact number {i} is stored")
        assert len(mem) == MAX_FACTS

    def test_clear(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        mem.add("something")
        assert mem.clear() is True
        assert mem.all() == []
        assert LongTermMemory(path=mem_path).all() == []

    def test_remove_by_index(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        mem.add("fact one")
        mem.add("fact two")
        assert mem.remove(1) is True
        assert mem.all() == ["fact two"]
        assert mem.remove(5) is False

    def test_json_file_is_human_readable(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        mem.add("User likes chai")
        with open(mem_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data == ["User likes chai"]


class TestGuessName:
    def test_finds_simple_name(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        mem.add("User's name is Munjur")
        assert mem.guess_name() == "Munjur"

    def test_finds_name_without_apostrophe(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        mem.add("Users name is Alice")
        assert mem.guess_name() == "Alice"

    def test_finds_full_name(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        mem.add("User's name is John Doe Jr")
        assert mem.guess_name() == "John Doe Jr"

    def test_finds_called_me_pattern(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        mem.add("User asked to be called Boss")
        assert mem.guess_name() == "Boss"

    def test_returns_none_when_unknown(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        mem.add("User likes tea")
        mem.add("User is building a shop")
        assert mem.guess_name() is None

    def test_returns_none_when_empty(self, mem_path):
        assert LongTermMemory(path=mem_path).guess_name() is None

    def test_ignores_ridiculously_long_match(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        mem.add("User's name is " + "x" * 100)
        assert mem.guess_name() is None

    def test_counts_property(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        assert mem.count == 0
        mem.add("a")
        mem.add("b")
        assert mem.count == 2


class TestInjection:
    def test_inject_with_no_facts_returns_original(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        prompt = "You are MForege."
        assert mem.inject_into(prompt) == prompt

    def test_inject_appends_facts(self, mem_path):
        mem = LongTermMemory(path=mem_path)
        mem.add("User's name is Munjur")
        mem.add("User is building a Django e-commerce site")
        result = mem.inject_into("You are MForege.")
        assert result.startswith("You are MForege.")
        assert "User's name is Munjur" in result
        assert "Django e-commerce" in result
        assert "past sessions" in result  # instructional framing


class TestParseFacts:
    def test_parses_plain_json_array(self):
        raw = '["User likes tea", "User knows Python"]'
        facts = _parse_facts(raw, existing=[])
        assert facts == ["User likes tea", "User knows Python"]

    def test_strips_code_fences(self):
        raw = '```json\n["User likes tea"]\n```'
        assert _parse_facts(raw, []) == ["User likes tea"]

    def test_dedupes_against_existing(self):
        raw = '["user likes tea!", "User enjoys Rust"]'
        facts = _parse_facts(raw, existing=["User likes tea"])
        assert facts == ["User enjoys Rust"]

    def test_invalid_json_returns_empty(self):
        assert _parse_facts("not json at all", []) == []
        assert _parse_facts('{"a": 1}', []) == []  # not a list

    def test_non_string_items_skipped(self):
        raw = '[42, null, "valid fact"]'
        assert _parse_facts(raw, []) == ["valid fact"]

    def test_caps_facts_per_turn(self):
        raw = json.dumps([f"fact {i}" for i in range(10)])
        assert len(_parse_facts(raw, [])) == 3


class TestExtractFacts:
    @pytest.mark.asyncio
    async def test_extracts_new_facts_via_llm(self, mem_path):
        class FakeLLM:
            async def create_chat_completion(self, **kwargs):
                msg = type("M", (), {"content": '["User is building a Django shop"]'})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

        facts = await extract_facts(
            FakeLLM(), "model", "I'm building a Django shop",
            "Great! Django is a solid choice.", [],
        )
        assert facts == ["User is building a Django shop"]

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty_not_raise(self):
        class BrokenLLM:
            async def create_chat_completion(self, **kwargs):
                raise ConnectionError("network down")

        facts = await extract_facts(BrokenLLM(), "m", "hi", "hello", [])
        assert facts == []

    @pytest.mark.asyncio
    async def test_empty_messages_skipped(self):
        class NeverCalled:
            async def create_chat_completion(self, **kwargs):
                raise AssertionError("should not be called")

        assert await extract_facts(NeverCalled(), "m", "", "reply", []) == []
        assert await extract_facts(NeverCalled(), "m", "msg", "", []) == []
