"""Tests for the Exa web search tool (no network — Exa client is mocked)"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.tools.web_search import ExaSearchTool, MAX_CONTENT_CHARS


def make_result(title, url, text):
    return SimpleNamespace(title=title, url=url, text=text)


class TestUnconfigured:
    @pytest.mark.asyncio
    async def test_without_key_returns_helpful_message(self):
        tool = ExaSearchTool(api_key="")
        result = await tool.execute(query="latest AI news")
        assert result.startswith("Error: web search is not configured")
        assert "EXA_API_KEY" in result

    def test_is_configured_flag(self):
        assert ExaSearchTool(api_key="k").is_configured is True
        assert ExaSearchTool(api_key="").is_configured is False

    def test_lazy_client_not_created_without_key(self):
        tool = ExaSearchTool(api_key="")
        assert tool._get_client() is None


class TestSearch:
    @pytest.mark.asyncio
    async def test_formats_results_for_llm(self):
        tool = ExaSearchTool(api_key="test-key")
        fake_response = SimpleNamespace(results=[
            make_result("Result A", "https://a.com", "Content about A"),
            make_result("Result B", "https://b.com", "Content about B"),
        ])

        with patch("app.tools.web_search.asyncio.to_thread", new=None):
            # patch to_thread with a fake that returns our response
            async def fake_to_thread(fn, *args, **kwargs):
                return fake_response

            with patch("app.tools.web_search.asyncio.to_thread", side_effect=fake_to_thread):
                result = await tool.execute(query="test query")

        assert "[1] Result A" in result
        assert "https://a.com" in result
        assert "Content about B" in result
        assert "---" in result  # separator between results

    @pytest.mark.asyncio
    async def test_long_content_is_truncated(self):
        tool = ExaSearchTool(api_key="test-key")
        long_text = "x" * (MAX_CONTENT_CHARS * 5)
        fake_response = SimpleNamespace(results=[
            make_result("Big page", "https://big.com", long_text),
        ])

        async def fake_to_thread(fn, *args, **kwargs):
            return fake_response

        with patch("app.tools.web_search.asyncio.to_thread", side_effect=fake_to_thread):
            result = await tool.execute(query="big")

        assert "xxxxx" in result
        assert len(result) < len(long_text)  # was actually truncated

    @pytest.mark.asyncio
    async def test_no_results(self):
        tool = ExaSearchTool(api_key="test-key")
        fake_response = SimpleNamespace(results=[])

        async def fake_to_thread(fn, *args, **kwargs):
            return fake_response

        with patch("app.tools.web_search.asyncio.to_thread", side_effect=fake_to_thread):
            result = await tool.execute(query="nothing here")

        assert result.startswith("No results found")

    @pytest.mark.asyncio
    async def test_api_error_is_reported_not_raised(self):
        tool = ExaSearchTool(api_key="test-key")

        async def fake_to_thread(fn, *args, **kwargs):
            raise RuntimeError("boom")

        with patch("app.tools.web_search.asyncio.to_thread", side_effect=fake_to_thread):
            result = await tool.execute(query="q")

        assert result.startswith("Error: web search failed")

    def test_tool_schema(self):
        tool = ExaSearchTool(api_key="k")
        schema = tool.to_dict()
        assert schema["function"]["name"] == "web_search"
        assert schema["function"]["parameters"]["required"] == ["query"]
