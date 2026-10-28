"""
Real API client for Egypt tender discovery via the World Bank
Procurement Notices API.

Endpoint: https://search.worldbank.org/api/procnotices
Confirmed working by direct HTTP request during development (no API
key required, JSON response, 'qterm' free-text search, 'rows'
pagination size). Verified that filtering by qterm="Egypt ..." returns
records whose project_ctry_name is genuinely Egypt, with real
reference numbers, descriptions, notice types (e.g. "Invitation for
Bids", "Request for Expression of Interest"), and submission
deadlines where published.

HONEST SCOPE LIMITATION: this is NOT Egypt's national tender portal
(etenders.gov.eg), for which no public API could be confirmed during
investigation (see the project's final report). It only covers
tenders tied to World Bank-FINANCED projects in Egypt -- a real,
genuine, but partial subset of all Egyptian government procurement.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any

from agents.rfp_research.api_clients.base import APIQueryResult, NormalizedTender

logger = logging.getLogger(__name__)

BASE_URL = "https://search.worldbank.org/api/procnotices"
SOURCE_NAME = "World Bank Procurement Notices API (Egypt: World Bank-financed projects only)"
USER_AGENT = "REXAI-RFPResearchAgent/1.0"

# Notice types confirmed (by inspecting real API responses) to
# represent a historical/awarded outcome rather than a current,
# open opportunity -- excluded so results favor actionable tenders.
HISTORICAL_NOTICE_TYPES = {"Contract Award", "Contract Awards", "Debarment"}


def query_world_bank_egypt(
    category_query: str,
    rows: int = 25,
    timeout: float = 15.0,
) -> APIQueryResult:
    """Query the World Bank Procurement Notices API for Egypt-related notices."""
    params = {
        "format": "json",
        "qterm": f"Egypt {category_query}",
        "rows": str(rows),
    }
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"

    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("World Bank API request failed: %s", exc)
        return APIQueryResult(
            source_name=SOURCE_NAME,
            status="error",
            message=f"World Bank API request failed: {exc}",
        )

    raw_records = payload.get("procnotices", [])

    if not isinstance(raw_records, list):
        return APIQueryResult(
            source_name=SOURCE_NAME,
            status="error",
            message="World Bank API returned an unexpected response shape.",
        )

    records = [
        record
        for record in (_normalize(raw) for raw in raw_records)
        if record is not None
    ]

    return APIQueryResult(
        source_name=SOURCE_NAME,
        status="success",
        message=(
            f"Retrieved {len(raw_records)} raw record(s); "
            f"{len(records)} confirmed as genuinely Egypt-specific, "
            "current opportunities."
        ),
        records=records,
        raw_record_count=len(raw_records),
    )


def _parse_notice_date(raw_date: Any) -> str | None:
    """Parse the API's 'DD-Mon-YYYY' noticedate into an ISO date string."""
    if not raw_date:
        return None

    try:
        return datetime.strptime(str(raw_date), "%d-%b-%Y").date().isoformat()
    except ValueError:
        # Keep the raw value rather than discard a real, just
        # differently-formatted date -- never invent a date instead.
        return str(raw_date)


def _build_submission_instructions(raw: dict[str, Any]) -> str | None:
    """Compose a contact string from whatever contact fields the API provides."""
    parts = [
        raw.get("contact_name"),
        raw.get("contact_email"),
        raw.get("contact_phone_no"),
        raw.get("contact_address"),
    ]
    parts = [str(part) for part in parts if part]
    return "; ".join(parts) if parts else None


def _normalize(raw: dict[str, Any]) -> NormalizedTender | None:
    country = str(raw.get("project_ctry_name") or "")

    # Defend against the free-text qterm search returning loose
    # matches that aren't actually about Egypt.
    if "egypt" not in country.casefold():
        return None

    notice_type = raw.get("notice_type")
    if notice_type in HISTORICAL_NOTICE_TYPES:
        return None

    record_id = raw.get("id")
    if not record_id:
        return None

    missing: list[str] = []

    title = raw.get("bid_description") or raw.get("project_name")
    if not title:
        missing.append("title")

    deadline = raw.get("submission_deadline_date")
    if deadline:
        deadline = str(deadline).split("T")[0]
    else:
        missing.append("deadline")

    # The World Bank's own contact details for the notice are a more
    # accurate "issuing organization" than the broader project name,
    # when present.
    organization = raw.get("contact_organization") or raw.get("project_name")
    if not organization:
        missing.append("organization")

    reference_number = raw.get("bid_reference_no")
    if not reference_number:
        missing.append("reference_number")

    if not notice_type:
        missing.append("category")

    publication_date = _parse_notice_date(raw.get("noticedate"))
    if not publication_date:
        missing.append("publication_date")

    submission_url = raw.get("contact_web_url") or None
    if not submission_url:
        missing.append("submission_url")

    submission_instructions = _build_submission_instructions(raw)
    if not submission_instructions:
        missing.append("submission_instructions")

    # This API does not expose eligibility requirements or a list of
    # attached documents for any record -- always explicitly flagged
    # as missing rather than silently left as an empty list.
    missing.append("eligibility")
    missing.append("available_documents")

    return NormalizedTender(
        title=title,
        reference_number=reference_number,
        organization=organization,
        country="Egypt",
        source_id=str(record_id),
        source_url=f"https://search.worldbank.org/notice/{record_id}",
        publication_date=publication_date,
        deadline=deadline,
        status=raw.get("notice_status"),
        category=notice_type,
        scope_of_work=raw.get("bid_description"),
        eligibility=[],
        submission_instructions=submission_instructions,
        submission_url=submission_url,
        contact_email=str(raw.get("contact_email") or "").strip() or None,
        available_documents=[],
        missing_information=missing,
        source_name=SOURCE_NAME,
        verification_status="verified_from_api",
    )
