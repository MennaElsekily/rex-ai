"""
The RFP agent's tender tools, as plain functions over the existing code.

    search_procurement_notices  -> api_clients.query_world_bank_egypt /
                                   query_etimad_saudi_arabia
    search_tender_web_pages     -> web_tenders.collect_pages (Tavily)
    find_known_tenders          -> OpportunityStore.find_duplicate
    save_discovered_tenders     -> OpportunityStore.save (pending only)

agents/tender_mcp_server.py exposes these four functions as MCP tools,
and the RFP graph calls them through agents/rfp_research/mcp_client.py.
No business logic lives in the server: it only calls these functions.

Every function validates its input, never raises for ordinary failures,
and returns one JSON-compatible dict with a "status". The functions only
read sources and save PENDING tenders: nothing here can approve, reject
or submit anything -- those are admin actions (agents/admin_service.py).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agents.rfp_research import web_tenders
from agents.rfp_research.api_clients import (
    query_etimad_saudi_arabia,
    query_world_bank_egypt,
)
from agents.rfp_research.opportunity_store import OpportunityStore
from agents.rfp_research.state import ALLOWED_COUNTRIES, TRAINING_CATEGORIES

logger = logging.getLogger(__name__)

# The setting that tells agents/tender_mcp_server.py which tender
# database to use (default: rfp_opportunities.sqlite3).
TENDER_DB_SETTING = "REX_TENDER_DB_PATH"

STATUS_OK = "ok"
STATUS_INVALID = "invalid_request"
STATUS_ERROR = "error"

# Inputs are capped: they come from the agent, but a tender record also
# carries text copied from web pages.
MAX_TENDERS_PER_CALL = 200
MAX_QUERY_CHARS = 200

def _notice_source(country: str):
    # Looked up at call time, so tests can replace a connector.
    return {"Egypt": query_world_bank_egypt, "Saudi Arabia": query_etimad_saudi_arabia}.get(country)


def _invalid(message: str, **extra: Any) -> dict[str, Any]:
    return {"status": STATUS_INVALID, "message": message, **extra}


def _clean_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()[:MAX_QUERY_CHARS]
    return text or None


# ---------------------------------------------------------------------
# Sources (read-only, external)
# ---------------------------------------------------------------------

def search_procurement_notices(country: str, query: str) -> dict[str, Any]:
    """
    The official procurement source for one country: the World Bank
    procurement notices API (Egypt) or the Etimad connector (Saudi
    Arabia, which has no public tender API and reports "unavailable").
    Returns the connector's APIQueryResult as a dict: source_name, status
    ("success" / "error" / "unavailable" / "awaiting_credentials"),
    message, records, raw_record_count.
    """
    source = _notice_source(country)
    if source is None:
        return _invalid(f"Unsupported country: {country!r}. Use one of: {', '.join(ALLOWED_COUNTRIES)}.",
                        records=[])
    query = _clean_text(query)
    if not query:
        return _invalid("A search query is required.", records=[])

    try:
        return source(query).model_dump(mode="json")
    except Exception as exc:  # the connectors report errors, but never let one escape
        logger.warning("Procurement source failed: %s", type(exc).__name__)
        return {"source_name": f"Procurement source ({country})", "status": STATUS_ERROR,
                "message": f"The source failed ({type(exc).__name__}).", "records": []}


def search_tender_web_pages(
    countries: list[str],
    categories: Optional[list[str]] = None,
    priority: Optional[str] = None,
) -> dict[str, Any]:
    """
    Web research for tenders (web_tenders.collect_pages): official
    portals, tender listing sites and the open web, per country.
    Returns status ("success" if any search succeeded, else
    "awaiting_credentials" or "error"), pages (each WebPage with its
    country and search label) and sources (one status per search).
    Page text is untrusted web content.
    """
    countries = list(countries or [])
    unknown = [c for c in countries if c not in ALLOWED_COUNTRIES]
    if not countries or unknown:
        return _invalid(f"Countries must be chosen from: {', '.join(ALLOWED_COUNTRIES)}.",
                        pages=[], sources=[])
    categories = list(categories or [])
    unknown = [c for c in categories if c not in TRAINING_CATEGORIES]
    if unknown:
        return _invalid(f"Unknown categories: {', '.join(map(str, unknown))}.", pages=[], sources=[])

    try:
        pages, sources = web_tenders.collect_pages(countries, categories, _clean_text(priority))
    except Exception as exc:
        logger.warning("Web research failed: %s", type(exc).__name__)
        return {"status": STATUS_ERROR, "message": f"Web research failed ({type(exc).__name__}).",
                "pages": [], "sources": []}

    statuses = {source["status"] for source in sources}
    status = "success" if "success" in statuses else (
        "awaiting_credentials" if "awaiting_credentials" in statuses else STATUS_ERROR)
    return {
        "status": status,
        "pages": [{**page.model_dump(), "country": country, "label": label}
                  for page, country, label in pages],
        "sources": sources,
    }


# ---------------------------------------------------------------------
# Tender catalogue (shared data)
# ---------------------------------------------------------------------

def _check_tenders(tenders: Any) -> Optional[dict[str, Any]]:
    if not isinstance(tenders, list) or not all(isinstance(t, dict) for t in tenders):
        return _invalid("tenders must be a list of tender objects.")
    if len(tenders) > MAX_TENDERS_PER_CALL:
        return _invalid(f"At most {MAX_TENDERS_PER_CALL} tenders per call.")
    return None


def find_known_tenders(store: OpportunityStore, tenders: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Which of these tenders are already in the catalogue (same URL, or
    the same tender listed on another site). Read-only.
    Returns known: [{index, existing_url}] and errors: [{index, message}].
    """
    problem = _check_tenders(tenders)
    if problem:
        return {**problem, "known": [], "errors": []}

    known, errors = [], []
    for index, tender in enumerate(tenders):
        try:
            existing = store.find_duplicate(tender)
        except ValueError as exc:  # no url / source id to identify it
            errors.append({"index": index, "message": str(exc)})
            continue
        if existing:
            known.append({"index": index, "existing_url": existing})
    return {"status": STATUS_OK, "known": known, "errors": errors}


def save_discovered_tenders(store: OpportunityStore, tenders: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Save discovered tenders. A new tender is stored as PENDING; a known
    one only has its discovered fields refreshed and keeps its admin
    decision. Approval and application fields in the input are ignored
    (OpportunityStore.save writes only the discovered fields).
    Returns new, known, results: [{index, is_new}] and errors.
    """
    problem = _check_tenders(tenders)
    if problem:
        return {**problem, "new": 0, "known": 0, "results": [], "errors": []}

    results, errors = [], []
    for index, tender in enumerate(tenders):
        try:
            results.append({"index": index, "is_new": store.save(tender)})
        except ValueError as exc:
            errors.append({"index": index, "message": str(exc)})
    new = sum(1 for r in results if r["is_new"])
    return {"status": STATUS_OK, "new": new, "known": len(results) - new,
            "results": results, "errors": errors}


class LocalTenderTools:
    """
    The same four tools, called in-process instead of over MCP, with the
    same methods as agents/rfp_research/mcp_client.TenderMcpClient. The
    tests use it (so they need no server subprocess); the application
    uses TenderMcpClient.
    """

    def __init__(self, store: OpportunityStore) -> None:
        self.store = store

    def search_procurement_notices(self, country: str, query: str) -> dict[str, Any]:
        return search_procurement_notices(country, query)

    def search_tender_web_pages(self, countries, categories=None, priority=None) -> dict[str, Any]:
        return search_tender_web_pages(countries, categories, priority)

    def find_known_tenders(self, tenders: list[dict[str, Any]]) -> dict[str, Any]:
        return find_known_tenders(self.store, tenders)

    def save_discovered_tenders(self, tenders: list[dict[str, Any]]) -> dict[str, Any]:
        return save_discovered_tenders(self.store, tenders)

    def close(self) -> None:
        pass
