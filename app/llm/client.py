"""
LLM Client Module
=================
Unified client supporting multiple LLM backends:

- openai: OpenAI API (requires API key)
- ollama: Local Ollama server (free, no API key)
- custom: Any OpenAI-compatible API (e.g., Groq, Together, etc.)

Adds connection timeouts, automatic retries for transient failures,
and friendly error messages for common problems (missing key, Ollama
not running, rate limits).
"""

import asyncio
from typing import Optional, Any

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)

# Seconds to wait before a request gives up
REQUEST_TIMEOUT = 60.0
# Transient failures are retried up to this many times
MAX_RETRIES = 2
# Base delay between retries (doubles each attempt)
RETRY_BASE_DELAY = 1.0


class LLMAuthError(Exception):
    """API key missing or rejected"""


class LLMConnectionError(Exception):
    """Could not reach the backend (e.g., Ollama not running)"""


class LLMRateLimitError(Exception):
    """Rate limit / quota exceeded"""


class LLMResponseError(Exception):
    """Backend returned an unexpected error response"""


class LLMClient:
    """
    Unified LLM client supporting multiple backends.

    Backends:
    - openai: OpenAI API (requires API key)
    - ollama: Local Ollama server (free, no API key)
    - custom: Any OpenAI-compatible API
    """

    def __init__(self, backend: str = "openai", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.client: Optional[AsyncOpenAI] = None
        self._init_client()

    def _init_client(self):
        """Initialize the appropriate client based on backend"""
        if self.backend == "openai":
            if not self.api_key:
                raise LLMAuthError(
                    "OpenAI backend requires an API key. "
                    "Set OPENAI_API_KEY or use --backend ollama for a free local model."
                )
            self.client = AsyncOpenAI(api_key=self.api_key, timeout=REQUEST_TIMEOUT)
        elif self.backend == "ollama":
            ollama_url = self.base_url or "http://localhost:11434/v1"
            self.client = AsyncOpenAI(base_url=ollama_url, api_key="ollama", timeout=REQUEST_TIMEOUT)
        elif self.backend == "custom":
            if not self.base_url:
                raise LLMConnectionError(
                    "Custom backend requires a base URL (--base-url or BASE_URL env var)."
                )
            if not self.api_key:
                raise LLMAuthError("Custom backend requires an API key (API_KEY env var).")
            self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=REQUEST_TIMEOUT)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    async def create_chat_completion(self, **kwargs) -> Any:
        """
        Create a chat completion using the configured backend.

        Retries transient failures (connection issues, timeouts, 5xx,
        rate limits with backoff). Raises user-friendly exceptions for
        permanent problems.
        """
        assert self.client is not None, "Client not initialized"

        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                return await self.client.chat.completions.create(**kwargs)

            except APITimeoutError:
                last_error = LLMConnectionError(
                    f"Request to {self.backend} backend timed out after {REQUEST_TIMEOUT:.0f}s. "
                    "The model may be loading — try again."
                )
            except APIConnectionError as e:
                last_error = LLMConnectionError(self._connection_message(e))
            except AuthenticationError as e:
                raise LLMAuthError(
                    f"API key rejected by {self.backend} backend. Check your key and try again."
                ) from e
            except RateLimitError as e:
                last_error = LLMRateLimitError(
                    f"Rate limit reached on {self.backend} backend: {e.message if hasattr(e, 'message') else e}"
                )
            except InternalServerError as e:
                last_error = LLMResponseError(
                    f"{self.backend} backend had an internal error (server side). Retrying may help."
                )
            except APIStatusError as e:
                raise LLMResponseError(
                    f"{self.backend} backend returned error {e.status_code}: "
                    f"{e.message if hasattr(e, 'message') else e}"
                ) from e

            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))

        assert last_error is not None
        raise last_error

    def _connection_message(self, e: Exception) -> str:
        if self.backend == "ollama":
            return (
                "Could not connect to Ollama. Is it running? Start it with `ollama serve` "
                "and make sure a model is pulled (e.g., `ollama pull llama3`)."
            )
        if self.backend == "custom":
            return f"Could not connect to custom backend at '{self.base_url}'. Check the base URL."
        return f"Could not connect to the OpenAI API: {e}"

    async def ping(self) -> None:
        """
        Verify the key/model/endpoint with a tiny real request.

        Raises the friendly LLM* errors on failure; returns None on success.
        Used by the setup wizard so a bad key never gets saved.
        """
        await self.create_chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
