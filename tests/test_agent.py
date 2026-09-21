"""Tests for the Agent tool-call loop (LLM is mocked, no network)"""
import json
from types import SimpleNamespace

import pytest

from app.agent.agent import MAX_TOOL_ROUNDS, Agent, AgentConfig
from app.agent.tools import CalculatorTool


def make_response(content="", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def make_tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_agent() -> Agent:
    agent = Agent(
        config=AgentConfig(streaming=False),
        api_key="test-key",
        backend="openai",
    )
    agent.register_tools(CalculatorTool())
    agent.llm.create_chat_completion = None  # replaced per-test
    return agent


class TestNonStreamingToolLoop:
    @pytest.mark.asyncio
    async def test_tool_result_is_sent_back_to_model(self):
        agent = make_agent()
        calls = []

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                # Model requests the calculator
                return make_response(tool_calls=[
                    make_tool_call("call_1", "calculator", json.dumps({"expression": "2 + 2"}))
                ])
            # Second round: model answers using the tool result
            return make_response(content="The answer is 4")

        agent.llm.create_chat_completion = fake_completion
        result = await agent.chat("what is 2+2?")

        assert result == "The answer is 4"
        # 2 main calls, no extraction side-call (facts file removed —
        # persistence is conversation-scoped via the session store)
        assert len(calls) == 2

        # Second request must contain the tool result as a 'tool' role message
        second_messages = calls[1]["messages"]
        tool_msgs = [m for m in second_messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert "4" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_final_answer_stored_in_memory(self):
        agent = make_agent()

        async def fake_completion(**kwargs):
            return make_response(content="hello there")

        agent.llm.create_chat_completion = fake_completion
        await agent.chat("hi")

        roles = [m.role for m in agent.memory.messages]
        assert roles == ["user", "assistant"]
        assert agent.memory.messages[-1].content == "hello there"

    @pytest.mark.asyncio
    async def test_bad_json_arguments_do_not_crash(self):
        agent = make_agent()
        calls = []

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return make_response(tool_calls=[
                    make_tool_call("call_1", "calculator", "not valid json{")
                ])
            return make_response(content="recovered")

        agent.llm.create_chat_completion = fake_completion
        result = await agent.chat("use the calculator")

        assert result == "recovered"
        tool_msg = next(m for m in calls[1]["messages"] if m["role"] == "tool")
        assert "Error" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_message(self):
        agent = make_agent()
        calls = []

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return make_response(tool_calls=[
                    make_tool_call("call_1", "nonexistent", "{}")
                ])
            return make_response(content="ok")

        agent.llm.create_chat_completion = fake_completion
        await agent.chat("do a thing")
        tool_msg = next(m for m in calls[1]["messages"] if m["role"] == "tool")
        assert "unknown tool" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_tool_loop_bounded_by_max_rounds(self):
        agent = make_agent()
        calls = []

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            # Always request a tool — must not loop forever
            return make_response(tool_calls=[
                make_tool_call(f"call_{len(calls)}", "calculator", '{"expression": "1+1"}')
            ])

        agent.llm.create_chat_completion = fake_completion
        await agent.chat("loop forever")

        # MAX_TOOL_ROUNDS requests, then 1 forced final request without tools
        assert len(calls) == MAX_TOOL_ROUNDS + 1
        assert "tools" not in calls[-1] or calls[-1].get("tools") is None

    @pytest.mark.asyncio
    async def test_activity_events_emitted(self):
        agent = make_agent()
        events = []
        agent.on_activity = lambda e, d: events.append((e, d))
        calls = []

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return make_response(tool_calls=[
                    make_tool_call("c1", "calculator", json.dumps({"expression": "2+2"}))
                ])
            return make_response(content="4")

        agent.llm.create_chat_completion = fake_completion
        await agent.chat("use calculator")

        kinds = [e for e, _ in events]
        assert "tool_start" in kinds
        assert "tool_end" in kinds
        # tool_start carries the tool name + args brief
        start_detail = next(d for e, d in events if e == "tool_start")
        assert "calculator" in start_detail

    @pytest.mark.asyncio
    async def test_activity_hook_never_breaks_chat(self):
        agent = make_agent()

        def bad_hook(e, d):
            raise RuntimeError("UI bug")

        agent.on_activity = bad_hook
        calls = []

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return make_response(tool_calls=[
                    make_tool_call("c1", "calculator", json.dumps({"expression": "1+1"}))
                ])
            return make_response(content="2")

        agent.llm.create_chat_completion = fake_completion
        result = await agent.chat("hi")  # must not raise
        assert result == "2"

    @pytest.mark.asyncio
    async def test_rounds_configurable_per_agent(self):
        agent = make_agent()
        agent.config.max_tool_rounds = 3
        calls = []

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            return make_response(tool_calls=[
                make_tool_call(f"call_{len(calls)}", "calculator", '{"expression": "1+1"}')
            ])

        agent.llm.create_chat_completion = fake_completion
        await agent.chat("loop forever")

        # 3 configured rounds, then 1 forced final request without tools
        assert len(calls) == 4
        assert "tools" not in calls[-1] or calls[-1].get("tools") is None


class TestCooperativeStop:
    """Esc-to-stop: cooperative cancellation flag checked before every
    model call, after each streamed chunk, and between tool rounds."""

    @pytest.mark.asyncio
    async def test_stop_before_first_call_skips_model(self):
        agent = make_agent()
        calls = []

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            return make_response(content="never")

        agent.llm.create_chat_completion = fake_completion
        agent.request_stop()
        result = await agent.chat("hi")

        assert result == ""
        assert calls == []  # no request left the machine

    @pytest.mark.asyncio
    async def test_stop_between_tool_rounds(self):
        agent = make_agent()
        calls = []
        events = []

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return make_response(tool_calls=[
                    make_tool_call("c1", "calculator", json.dumps({"expression": "1+1"}))
                ])
            return make_response(content="should not happen")

        agent.llm.create_chat_completion = fake_completion

        def hook(event, detail):
            events.append(event)
            if event == "tool_end":
                agent.request_stop()  # user pressed Esc while tools ran

        agent.on_activity = hook
        result = await agent.chat("loop")

        assert result == ""
        assert len(calls) == 1          # round 2 never called the model
        assert "tool_end" in events
        assert "stopped" in events      # UI is told why

    @pytest.mark.asyncio
    async def test_stream_stop_mid_stream_keeps_partial_answer(self):
        agent = make_agent()

        async def fake_stream(**kwargs):
            async def gen():
                delta = SimpleNamespace(content="Hel", tool_calls=None)
                yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])
                agent._stop_requested = True  # Esc pressed mid-stream
                delta = SimpleNamespace(content="lo", tool_calls=None)
                yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])
            return gen()

        agent.llm.create_chat_completion = fake_stream
        agent.config.streaming = True

        stream = await agent.chat("hi", stream=True)
        chunks = [c async for c in stream]

        assert chunks == ["Hel"]                      # nothing after the stop
        assert agent.memory.messages[-1].content == "Hel"  # partial answer saved

    @pytest.mark.asyncio
    async def test_clear_stop_resets_flag(self):
        agent = make_agent()
        agent.request_stop()
        agent._clear_stop()

        async def fake_completion(**kwargs):
            return make_response(content="back to work")

        agent.llm.create_chat_completion = fake_completion
        assert await agent.chat("hi") == "back to work"

    @pytest.mark.asyncio
    async def test_stop_flag_clears_after_firing(self):
        """One Esc stops exactly one turn — a consumed stop must not
        swallow the NEXT message too."""
        agent = make_agent()
        calls = []

        async def fake_completion(**kwargs):
            calls.append(kwargs)
            return make_response(content="ok")

        agent.llm.create_chat_completion = fake_completion
        agent.request_stop()
        assert await agent.chat("first") == ""     # fires once
        assert await agent.chat("second") == "ok"  # flag self-cleared
        assert len(calls) == 1


class TestStreamingToolLoop:
    @pytest.mark.asyncio
    async def test_streamed_chunks_and_memory(self):
        agent = make_agent()

        async def fake_stream(**kwargs):
            async def gen():
                for piece in ["Hello", " ", "world"]:
                    delta = SimpleNamespace(content=piece, tool_calls=None)
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])
            return gen()

        agent.llm.create_chat_completion = fake_stream
        agent.config.streaming = True

        chunks = []
        stream = await agent.chat("hi", stream=True)
        async for chunk in stream:
            chunks.append(chunk)

        assert chunks == ["Hello", " ", "world"]
        assert agent.memory.messages[-1].content == "Hello world"

    @pytest.mark.asyncio
    async def test_streamed_tool_call_triggers_second_round(self):
        agent = make_agent()
        calls = []

        def delta(content=None, tool_calls=None):
            return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))])

        def tc_delta(index, call_id=None, name=None, arguments=None):
            fn = SimpleNamespace(name=name, arguments=arguments)
            return SimpleNamespace(index=index, id=call_id, function=fn)

        async def fake_stream(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                async def gen():
                    yield delta(tool_calls=[tc_delta(0, call_id="call_9", name="calculator")])
                    yield delta(tool_calls=[tc_delta(0, arguments='{"expression": "3*3"}')])
                return gen()
            else:
                async def gen():
                    yield delta(content="9")
                return gen()

        agent.llm.create_chat_completion = fake_stream
        agent.config.streaming = True

        chunks = []
        stream = await agent.chat("what is 3*3?", stream=True)
        async for chunk in stream:
            chunks.append(chunk)

        assert chunks == ["9"]
        assert len(calls) == 2
        tool_msgs = [m for m in calls[1]["messages"] if m["role"] == "tool"]
        assert tool_msgs and "9" in tool_msgs[0]["content"]
        assert agent.memory.messages[-1].content == "9"
