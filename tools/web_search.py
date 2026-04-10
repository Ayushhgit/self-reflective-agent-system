"""
Provides a lightweight web search capability using the DuckDuckGo Search API
(no API key required).  Falls back gracefully if unavailable.
"""

import logging

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for the given query and return a summary of results.

    Parameters
    ----------
    query       : Search query string.
    max_results : Number of results to return (default 5).

    Returns
    -------
    str  Formatted search results or an error message.
    """
    logger.info("[WebSearch] Searching for: %s", query[:80])
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No results found."

        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            snippet = r.get("body", "")
            url = r.get("href", "")
            lines.append(f"{i}. **{title}**\n   {snippet}\n   Source: {url}")

        return "\n\n".join(lines)

    except ImportError:
        return (
            "Web search unavailable. Install duckduckgo-search: "
            "pip install duckduckgo-search"
        )
    except Exception as exc:
        logger.warning("[WebSearch] Error: %s", exc)
        return f"Search error: {exc}"
