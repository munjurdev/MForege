"""Tests for 413 'request too large' handling and adaptive recovery.

Reproduces the field failure: Groq free tier, Error 413,
"Limit 8000, Requested 8161" while the meter showed ctx 1%.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest
from openai import APIStatusError

import app.agent.agent as agent_module
from app.agent.agent import Agent, AgentConfig, MIN_MAX_TOKENS
from app.agent.memory import ConversationMemory, Message
from app.llm.client import LLMClient, LLMRequestTooLargeError, LLMRateLimitError


def _make_413(limit=8000, requested=8161):
    """Build an openai APIStatusError subclass shaped like Groq's real 413."""
    from openai import BadRequestError  # concrete APIStatusError subclass
    detail = (
        f"Request too large for model 'openai/gpt-oss-20b' in organization "
        f"org_x service_tier 'on_demand' on tokens per minute (TPM): "
        f"Limit {limit}, Requested {requested}, please reduce your message size."
    )
    req = type("R", (), {"headers": {}, "method": "POST", "url": "http://x"})()
    resp = type("S", (), {
        "status_code": 413, "headers": {}, "request": req,
    })()
    body = {"error": {"message": detail, "type": "request_too_large", "code": "413"}}
    try:
        raise BadRequestError(detail, response=resp, body=body)
    except BadRequestError as e:
        return e


def _too_large_agent() -> Agent:
    agent = Agent(
        config=AgentConfig(streaming=False, max_tokens=4096),
        api_key="test-key",
        backend="openai",
    )
    # no tools → simple single-round flow
    agent.tools.tools = []
    return agent


class TestClient413Detection:
    @pytest.mark.asyncio
    async def test_413_becomes_request_too_large(self):
        client = LLMClient(backend="openai", api_key="sk-test")

        async def reject(**kwargs):
            raise _make_413(limit=8000, requested=8161)

        with patch.object(client.client.chat.completions, "create", new=AsyncMock(side_effect=reject)):
            with pytest.raises(LLMRequestTooLargeError) as exc:
                await client.create_chat_completion(model="m", messages=[])

        assert exc.value.limit_tokens == 8000
        assert exc.value.requested_tokens == 8161
        assert "too large" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_429_token_too_large_also_becomes_413(self):
        """Groq sometimes reports TPM overage as 429 with 'too large' text."""
        client = LLMClient(backend="openai", api_key="sk-test")

        def make_429():
            req = type("R", (), {"headers": {}, "method": "POST", "url": "http://x"})()
            resp = type("S", (), {"status_code": 429, "headers": {}, "request": req})()
            body = {"error": {"message": (
                "Request too large for model 'x' on tokens per minute (TPM): "
                "Limit 8000, Requested 9000."
            ), "type": "tokens", "code": "rate_limit_exceeded"}}
            return RateLimitShim(resp, body)

        class RateLimitShim(Exception):
            def __init__(self, resp, body):
                from openai import RateLimitError
                # construct through the real class so except-clauses match
                self.__class__ = RateLimitError
                RateLimitError.__init__(self, "429", response=resp, body=body)

        async def reject(**kwargs):
            raise make_429()

        with patch.object(client.client.chat.completions, "create", new=AsyncMock(side_effect=reject)):
            with pytest.raises(LLMRequestTooLargeError):
                await client.create_chat_completion(model="m", messages=[])

    @pytest.mark.asyncio
    async def test_429_normal_rate_limit_still_rate_limit(self):
        client = LLMClient(backend="openai", api_key="sk-test")

        def make_429():
            from openai import RateLimitError
            req = type("R", (), {"headers": {}, "method": "POST", "url": "http://x"})()
            resp = type("S", (), {"status_code": 429, "headers": {}, "request": req})()
            return RateLimitError("429", response=resp, body={
                "error": {"message": "Rate limit reached for requests.", "code": "429"}
            })

        async def reject(**kwargs):
            raise make_429()

        with patch.object(client.client.chat.completions, "create", new=AsyncMock(side_effect=reject)):
            with patch("app.llm.client.asyncio.sleep", new=AsyncMock()):
                with pytest.raises(LLMRateLimitError):
                    await client.create_chat_completion(model="m", messages=[])


class TestMemoryForceCondense:
    def test_folds_all_but_recent(self):
        mem = ConversationMemory()
        for i in range(12):
            mem.add(Message(role="user", content=f"question {i} " + "x" * 200))
            mem.add(Message(role="assistant", content=f"answer {i} " + "y" * 200))
        assert mem.force_condense(keep_recent=4) is True
        non_system = [m for m in mem.messages if m.role != "system"]
        assert len(non_system) == 4
        assert "question 0" in mem.summary  # oldest content survives as digest

    def test_noop_when_short(self):
        mem = ConversationMemory()
        mem.add(Message(role="user", content="hi"))
        assert mem.force_condense() is False

    def test_token_estimate_with_extra(self):
        mem = ConversationMemory()
        mem.add(Message(role="user", content="abcd" * 100))  # 400 chars → 100 tokens
        base = mem.token_estimate()
        assert mem.token_estimate(extra=400) == base + 100


class TestAgentAdaptiveRecovery:
    @pytest.mark.asyncio
    async def test_recovers_and_answers_after_413(self):
        """The field bug: first request 413s, agent adapts, answers anyway."""
        agent = _too_large_agent()
        # rich history so force_condense has something to fold
        for i in range(10):
            agent.memory.add(Message(role="user", content=f"q{i} " + "x" * 300))
            agent.memory.add(Message(role="assistant", content=f"a{i} " + "y" * 300))

        calls = {"n": 0}

        async def first_413(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise LLMRequestTooLargeError(
                    "Request too large: Limit 8000, Requested 8161",
                    limit_tokens=8000, requested_tokens=8161,
                )
            return SimpleNamespaceChoices("Adapted fine!")

        with patch.object(agent.llm, "create_chat_completion", new=AsyncMock(side_effect=first_413)):
            reply = await agent.chat("make me a project", stream=False)

        assert reply == "Adapted fine!"
        assert calls["n"] == 2
        assert agent._max_tokens_cap == 2048  # 4096 * 0.5
        assert agent.memory.summary != ""  # older turns were folded

    @pytest.mark.asyncio
    async def test_pre_shrink_next_message_no_413(self):
        """After learning a 413, the NEXT message must pre-shrink — the
        cap persists so recovery isn't pay-per-message."""
        agent = _too_large_agent()
        agent._max_tokens_cap = 1024
        agent._condense_hint = True
        agent.memory.add(Message(role="user", content="old"))
        agent.memory.add(Message(role="assistant", content="older"))

        seen = {}

        async def capture(**kwargs):
            seen["max_tokens"] = kwargs.get("max_tokens")
            seen["history_len"] = len(kwargs["messages"])
            m = AsyncMock()
            m.return_value = SimpleNamespaceChoices("ok")
            return m.return_value

        with patch.object(agent.llm, "create_chat_completion", new=AsyncMock(side_effect=capture)):
            await agent.chat("next question", stream=False)

        assert seen["max_tokens"] == 1024            # learned cap applied
        assert seen["history_len"] <= 5              # history condensed first

    @pytest.mark.asyncio
    async def test_gives_up_honestly_when_impossible(self):
        """When nothing can shrink (tiny history), the error surfaces."""
        agent = _too_large_agent()
        agent.memory.add(Message(role="user", content="tiny"))

        async def always_413(**kwargs):
            raise LLMRequestTooLargeError(
                "Request too large: Limit 8000, Requested 9000",
                limit_tokens=8000, requested_tokens=9000,
            )

        with patch.object(agent.llm, "create_chat_completion", new=AsyncMock(side_effect=always_413)):
            with pytest.raises(LLMRequestTooLargeError):
                await agent.chat("hi", stream=False)

    @pytest.mark.asyncio
    async def test_adapting_event_emitted(self):
        agent = _too_large_agent()
        agent.memory.add(Message(role="user", content="old stuff"))
        agent.memory.add(Message(role="assistant", content="older stuff"))

        calls = {"n": 0}

        async def first_413(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise LLMRequestTooLargeError(
                    "Request too large: Limit 8000, Requested 8161",
                    limit_tokens=8000, requested_tokens=8161,
                )
            return SimpleNamespaceChoices("done")

        seen_events = []
        agent.on_activity = lambda ev, dt: seen_events.append((ev, dt))

        with patch.object(agent.llm, "create_chat_completion", new=AsyncMock(side_effect=first_413)):
            await agent.chat("hello", stream=False)

        assert any(ev == "adapting" for ev, _ in seen_events)


class TestContextMeterTruth:
    def test_meter_includes_overhead(self):
        agent = _too_large_agent()
        history_only = agent.memory.token_estimate()
        used, window = agent.context_usage()
        assert used > history_only          # system prompt + tools counted
        assert used >= 2000                 # 4.2K-char system prompt ≈ 1K tokens + 1K slack

    def test_effective_max_tokens_floor(self):
        agent = _too_large_agent()
        agent._max_tokens_cap = 100         # below floor
        assert agent._effective_max_tokens() == MIN_MAX_TOKENS


class SimpleNamespaceChoices:
    """Minimal response for the non-streaming path: .choices[0].message."""
    def __init__(self, content):
        msg = type("M", (), {"content": content, "tool_calls": None})()
        self.choices = [type("C", (), {"message": msg})()]
