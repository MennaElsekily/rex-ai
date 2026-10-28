"""
Live web search for RFP Research via the Tavily Search API.

Endpoint (Tavily API reference): POST https://api.tavily.com/search,
"Authorization: Bearer <TAVILY_API_KEY>"; response "results" items carry
"title", "url", "content", "score" and "published_date".

Web content is untrusted input: it is only stored as data (length-capped,
control characters removed), never followed as instructions, and only
http(s) URLs are kept. Tenders extracted from these pages
(agents/rfp_research/web_tenders.py) are stored as "pending" for admin
approval; nothing found here is approved automatically.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Literal, Optional
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from agents.config import get_secret

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.tavily.com/search"
SOURCE_NAME = "Tavily web search"
MAX_CONTENT_CHARS = 4000
# Full page text kept for tender-field extraction (deadline, documents...).
MAX_RAW_CONTENT_CHARS = 12000

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class WebPage(BaseModel):
    title: str
    url: str
    content: str
    score: Optional[float] = None
    published_date: Optional[str] = None
    source_domain: str
    # The full page text (Tavily "raw_content"), when requested.
    raw_content: Optional[str] = None
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class WebSearchResult(BaseModel):
    source_name: str = SOURCE_NAME
    status: Literal["success", "awaiting_credentials", "error"]
    message: str
    query: str
    pages: list[WebPage] = Field(default_factory=list)


def _clean(text: object, limit: int) -> str:
    return _CONTROL_CHARS.sub("", str(text or "")).strip()[:limit]


def _to_page(item: dict) -> Optional[WebPage]:
    url = _clean(item.get("url"), 2000)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None

    content = _clean(item.get("content"), MAX_CONTENT_CHARS)
    if not content:
        return None

    score = item.get("score")
    return WebPage(
        title=_clean(item.get("title"), 300) or parsed.netloc,
        url=url,
        content=content,
        score=float(score) if isinstance(score, (int, float)) else None,
        published_date=_clean(item.get("published_date"), 40) or None,
        source_domain=parsed.netloc.lower(),
        raw_content=_clean(item.get("raw_content"), MAX_RAW_CONTENT_CHARS) or None,
    )


def search_tavily(
    query: str,
    max_results: int = 5,
    timeout: float = 20.0,
    http_client: Optional[httpx.Client] = None,
    include_domains: Optional[list[str]] = None,
    include_raw_content: bool = False,
) -> WebSearchResult:
    """
    One Tavily search. Never raises for ordinary failures: a missing key
    is "awaiting_credentials", any request/response problem is "error",
    so a failed search is never reported as "nothing found".

    include_domains: search only these sites (e.g. official tender portals).
    include_raw_content: also return each page's full text (WebPage.raw_content).
    """
    api_key = get_secret("TAVILY_API_KEY")
    if not api_key:
        return WebSearchResult(
            status="awaiting_credentials",
            message="Web search is not configured: set TAVILY_API_KEY in .env.",
            query=query,
        )

    payload = {
        "query": query,
        "search_depth": "basic",
        "topic": "general",
        "max_results": max(1, min(int(max_results), 20)),
        "include_answer": False,
        "include_raw_content": bool(include_raw_content),
    }
    if include_domains:
        payload["include_domains"] = list(include_domains)
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        client = http_client or httpx.Client(timeout=timeout)
        try:
            response = client.post(SEARCH_URL, json=payload, headers=headers)
        finally:
            if http_client is None:
                client.close()

        if response.status_code in (401, 403):
            return WebSearchResult(
                status="error", message="Tavily rejected the API key.", query=query
            )
        if response.status_code == 429:
            return WebSearchResult(
                status="error", message="Tavily rate limit reached; try again later.",
                query=query,
            )
        response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException:
        return WebSearchResult(status="error", message="Web search timed out.", query=query)
    except Exception as exc:
        # The key is in a header, never in the message.
        logger.warning("Tavily search failed: %s", type(exc).__name__)
        return WebSearchResult(
            status="error",
            message=f"Web search failed ({type(exc).__name__}).",
            query=query,
        )

    items = data.get("results") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return WebSearchResult(
            status="error", message="Unexpected response from Tavily.", query=query
        )

    pages = [page for page in (_to_page(i) for i in items if isinstance(i, dict)) if page]
    return WebSearchResult(
        status="success",
        message=f"Retrieved {len(items)} result(s); {len(pages)} usable.",
        query=query,
        pages=pages,
    )
