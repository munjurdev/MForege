"""
Exa Web Search Tool
===================
Gives the agent real-time web search powered by Exa (exa.ai).

Exa returns clean, LLM-ready page content instead of raw snippets, so the
model can actually read and use the results.

Setup:
1. Get a free API key at https://dashboard.exa.ai
2. Add it to .env:  EXA_API_KEY=your_key_here
3. Restart the CLI

Without a key the tool is still registered, but returns a helpful message
instead of search results.
"""

import asyncio
from typing import Any, Optional

from decouple import config as env_config
from pydantic import PrivateAttr

from app.agent.tools import Tool

# Cap page content per result so a few searches don't blow the context window
MAX_CONTENT_CHARS = 1500
# A hung search must not wedge the agent's turn forever (Esc also works,
# but a self-recovering timeout is the friendlier failure).
SEARCH_TIMEOUT = 30.0


class ExaSearchTool(Tool):
    """Web search tool backed by the Exa API"""

    # Private attributes (Tool is a pydantic model)
    _api_key: str = PrivateAttr(default="")
    _num_results: int = PrivateAttr(default=5)
    _client: Optional[Any] = PrivateAttr(default=None)

    def __init__(self, api_key: Optional[str] = None, num_results: int = 5, **data):
        super().__init__(
            name="web_search",
            description=(
                "Search the web for current information (news, facts, docs). "
                "Returns titles, URLs and page content for the top results."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"],
            },
            execute=self._execute,
            **data,
        )
        self._api_key = api_key or env_config("EXA_API_KEY", default="")
        self._num_results = num_results
        self._client = None

    @property
    def is_configured(self) -> bool:
        """True when an API key is available"""
        return bool(self._api_key)

    def _get_client(self):
        """Lazily create the Exa client (keeps exa_py import optional)"""
        if self._client is None:
            if not self._api_key:
                return None
            from exa_py import Exa
            self._client = Exa(api_key=self._api_key)
        return self._client

    async def _execute(self, query: str) -> str:
        """Run the search and format results for the model"""
        client = self._get_client()
        if client is None:
            return (
                "Error: web search is not configured. "
                "Set EXA_API_KEY in .env (free key at https://dashboard.exa.ai) "
                "and restart the CLI."
            )

        try:
            # exa_py is synchronous — run in a thread so the event loop stays free
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.search_and_contents,
                    query,
                    num_results=self._num_results,
                    text=True,
                ),
                timeout=SEARCH_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return f"Error: web search timed out after {SEARCH_TIMEOUT:.0f}s — try again later"
        except Exception as e:
            return f"Error: web search failed: {e}"

        if not response.results:
            return f"No results found for: {query}"

        parts = []
        for i, result in enumerate(response.results, 1):
            title = getattr(result, "title", "") or "(no title)"
            url = getattr(result, "url", "") or ""
            content = (getattr(result, "text", "") or "")[:MAX_CONTENT_CHARS]
            parts.append(f"[{i}] {title}\nURL: {url}\n{content}")

        return "\n\n---\n\n".join(parts)
