"""Tests for LLMClient configuration and retry behavior (no network)"""
from unittest.mock import AsyncMock, patch

import pytest
from openai import APITimeoutError

from app.llm.client import (
    LLMClient,
    LLMAuthError,
    LLMConnectionError,
    LLMResponseError,
)


class TestClientInit:
    def test_openai_without_key_raises_friendly_error(self):
        with pytest.raises(LLMAuthError, match="OPENAI_API_KEY"):
            LLMClient(backend="openai", api_key=None)

    def test_custom_without_base_url_raises(self):
        with pytest.raises(LLMConnectionError, match="base URL"):
            LLMClient(backend="custom", api_key="k", base_url=None)

    def test_custom_without_key_raises(self):
        with pytest.raises(LLMAuthError):
            LLMClient(backend="custom", api_key=None, base_url="http://x")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            LLMClient(backend="nope")

    def test_ollama_gets_default_url_and_dummy_key(self):
        client = LLMClient(backend="ollama")
        assert client.client.base_url.host == "localhost"
        assert client.client.base_url.port == 11434
        assert client.api_key is None  # attribute stays None; client uses "ollama"

    def test_openai_with_key_initializes(self):
        client = LLMClient(backend="openai", api_key="sk-test")
        assert client.client is not None


class TestRetries:
    @pytest.mark.asyncio
    async def test_retries_then_succeeds_on_timeout(self):
        client = LLMClient(backend="openai", api_key="sk-test")
        ok_response = {"id": "x"}

        call_count = {"n": 0}

        async def flaky(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise APITimeoutError(request=None)
            return ok_response

        with patch.object(client.client.chat.completions, "create", new=AsyncMock(side_effect=flaky)):
            with patch("app.llm.client.asyncio.sleep", new=AsyncMock()):
                result = await client.create_chat_completion(model="m", messages=[])

        assert result == ok_response
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self):
        client = LLMClient(backend="openai", api_key="sk-test")
        call_count = {"n": 0}

        async def always_timeout(**kwargs):
            call_count["n"] += 1
            raise APITimeoutError(request=None)

        with patch.object(client.client.chat.completions, "create", new=AsyncMock(side_effect=always_timeout)):
            with patch("app.llm.client.asyncio.sleep", new=AsyncMock()):
                with pytest.raises(LLMConnectionError, match="timed out"):
                    await client.create_chat_completion(model="m", messages=[])

        # MAX_RETRIES + 1 total attempts
        assert call_count["n"] == 3

    @pytest.mark.asyncio
    async def test_auth_error_not_retried(self):
        from openai import AuthenticationError

        client = LLMClient(backend="openai", api_key="sk-test")
        call_count = {"n": 0}

        def make_auth_error():
            req = type("R", (), {"headers": {}, "method": "POST", "url": "http://x"})()
            resp = type("S", (), {"status_code": 401, "headers": {}, "request": req})()
            try:
                raise AuthenticationError("bad key", response=resp, body=None)
            except AuthenticationError as e:
                return e

        async def reject(**kwargs):
            call_count["n"] += 1
            raise make_auth_error()

        with patch.object(client.client.chat.completions, "create", new=AsyncMock(side_effect=reject)):
            with pytest.raises(LLMAuthError):
                await client.create_chat_completion(model="m", messages=[])

        assert call_count["n"] == 1  # no retry on auth failure

    @pytest.mark.asyncio
    async def test_other_status_error_wrapped(self):
        from openai import BadRequestError

        client = LLMClient(backend="openai", api_key="sk-test")

        req = type("R", (), {"headers": {}, "method": "POST", "url": "http://x"})()
        resp = type("S", (), {"status_code": 400, "headers": {}, "request": req})()

        async def bad_request(**kwargs):
            raise BadRequestError("bad", response=resp, body=None)

        with patch.object(client.client.chat.completions, "create", new=AsyncMock(side_effect=bad_request)):
            with pytest.raises(LLMResponseError):
                await client.create_chat_completion(model="m", messages=[])
