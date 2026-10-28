"""
External curriculum research, restricted to approved domains.

Reuses the RFP agent's Tavily client (agents/rfp_research/api_clients/
tavily.py). Only domains listed in CURRICULUM_RESEARCH_DOMAINS (comma
separated; subdomains included) are searched, and any result from another
domain is discarded. The setting is empty by default, so research is
off until the owner approves an allowlist (see
docs/CURRICULUM_AGENT_ARCHITECTURE.md §16).

Research is evidence of *industry relevance* only: never an academy fact
and never a framework requirement. Each source keeps its title, URL,
publisher (domain), published date if given, an excerpt and when it was
retrieved; the model can only cite these source ids.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from agents.config import get_setting

DOMAINS_SETTING = "CURRICULUM_RESEARCH_DOMAINS"
MAX_RESULTS = 8
MAX_EXCERPT_CHARS = 800
_DOMAIN = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+$")


def allowed_domains(raw: Optional[str] = None) -> list[str]:
    """The configured allowlist (lower-case, valid domain names only)."""
    value = get_setting(DOMAINS_SETTING, "") if raw is None else raw
    domains = []
    for part in str(value or "").split(","):
        domain = part.strip().casefold().removeprefix("www.")
        if _DOMAIN.match(domain) and domain not in domains:
            domains.append(domain)
    return domains


def domain_allowed(url: str, domains: list[str]) -> bool:
    host = urlparse(str(url or "")).netloc.casefold().split(":")[0].removeprefix("www.")
    return any(host == d or host.endswith("." + d) for d in domains)


def source_id(url: str) -> str:
    return "SRC-" + hashlib.sha256(str(url).encode("utf-8")).hexdigest()[:10]


def search_sources(
    query: str,
    max_results: int = 5,
    domains: Optional[list[str]] = None,
    search: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """
    Search approved domains. Returns status "ok", "not_configured" (no
    approved domains), "awaiting_credentials" (no Tavily key), "error" or
    "invalid_request", with sources [{source_id, title, url, publisher,
    published, excerpt, retrieved_at}].
    """
    query = " ".join(str(query or "").split())[:300]
    if len(query) < 3:
        return {"status": "invalid_request", "message": "A search query is required.", "sources": []}
    domains = allowed_domains() if domains is None else domains
    if not domains:
        return {"status": "not_configured", "sources": [],
                "message": "External research is disabled: no approved domains are configured "
                           f"({DOMAINS_SETTING})."}
    if search is None:
        from agents.rfp_research.api_clients.tavily import search_tavily as search
    result = search(query, max_results=max(1, min(int(max_results), MAX_RESULTS)),
                    include_domains=domains, include_raw_content=False)
    if result.status != "success":
        return {"status": result.status, "message": result.message, "sources": []}
    sources = []
    for page in result.pages:
        if not domain_allowed(page.url, domains):
            continue  # never trust a result from outside the allowlist
        sources.append({
            "source_id": source_id(page.url), "title": page.title, "url": page.url,
            "publisher": page.source_domain, "published": page.published_date,
            "excerpt": page.content[:MAX_EXCERPT_CHARS], "retrieved_at": page.retrieved_at,
        })
    return {"status": "ok", "sources": sources, "domains": domains}
