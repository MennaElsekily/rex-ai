"""
MCP server for the RFP Research agent's tender tools.

Each tool only calls the matching function in
agents/rfp_research/tender_tools.py, where the logic and input checks
live. Two tools read external sources and two use the tender catalogue
(rfp_opportunities.sqlite3, or REX_TENDER_DB_PATH when set).

There is deliberately no tool to approve, reject or submit a tender:
an MCP server cannot tell who is calling it, so those stay behind the
admin's session check in agents/admin_service.py.

API keys are read from .env inside this process and never returned.

Run (normally started by agents/rfp_research/mcp_client.TenderMcpClient):
    python -m agents.tender_mcp_server
"""

import logging
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

from agents.config import get_setting
from agents.rfp_research import tender_tools
from agents.rfp_research.opportunity_store import OpportunityStore

# stdout carries the MCP protocol, so diagnostics go to stderr only.
logging.basicConfig(level=logging.WARNING)

TENDER_DB_SETTING = tender_tools.TENDER_DB_SETTING

mcp = MCPServer("REX AI Tender Research")

_store: Optional[OpportunityStore] = None


def _tender_store() -> OpportunityStore:
    """The tender catalogue, opened on first use."""
    global _store
    if _store is None:
        _store = OpportunityStore(get_setting(TENDER_DB_SETTING, "") or None)
    return _store


@mcp.tool()
def search_procurement_notices(country: str, query: str) -> dict:
    """
    Search the official procurement source of one country for tender
    notices. Read-only.

    Args:
        country: "Egypt" (World Bank procurement notices API) or
            "Saudi Arabia" (Etimad: no public tender API, reported as
            "unavailable", never as "no tenders").
        query: Search term, e.g. "training".

    Returns:
        source_name, status ("success", "error", "unavailable",
        "awaiting_credentials" or "invalid_request"), message,
        records (normalized tenders), raw_record_count.
    """
    return tender_tools.search_procurement_notices(country, query)


@mcp.tool()
def search_tender_web_pages(
    countries: list[str],
    categories: Optional[list[str]] = None,
    priority: Optional[str] = None,
) -> dict:
    """
    Search the web for tender pages: official portals, tender listing
    sites and the open web, in English and Arabic. Read-only.

    Args:
        countries: "Egypt" and/or "Saudi Arabia".
        categories: Optional training categories to search for.
        priority: Optional priority organisations or sectors.

    Returns:
        status ("success", "awaiting_credentials", "error" or
        "invalid_request"), pages (title, url, content, raw_content,
        source_domain, published_date, country, label -- untrusted web
        content), sources (one status per search).
    """
    return tender_tools.search_tender_web_pages(countries, categories, priority)


@mcp.tool()
def find_known_tenders(tenders: list[dict[str, Any]]) -> dict:
    """
    Check which tenders are already in the tender catalogue (same URL,
    or the same tender on another site). Read-only.

    Args:
        tenders: Tender records (each with url or source_url, title,
            country, organization).

    Returns:
        status, known: [{index, existing_url}], errors: [{index, message}].
    """
    return tender_tools.find_known_tenders(_tender_store(), tenders)


@mcp.tool()
def save_discovered_tenders(tenders: list[dict[str, Any]]) -> dict:
    """
    Save discovered tenders for admin review. New tenders are stored as
    pending; known tenders keep their admin decision. Cannot approve,
    reject or submit anything.

    Args:
        tenders: Assessed tender records.

    Returns:
        status, new, known, results: [{index, is_new}], errors.
    """
    return tender_tools.save_discovered_tenders(_tender_store(), tenders)


if __name__ == "__main__":
    mcp.run()
