"""
Shared data model for tender API connectors (agents/rfp_research/api_clients/).

Every country connector returns an APIQueryResult, which distinguishes
four outcomes explicitly -- an API error must never be presented the
same way as "the API worked and found nothing":

- "success": the API was called and responded (records may be empty).
- "unavailable": no accessible API exists/was confirmed for this
  country (architecture is ready, but there is nothing to call yet).
- "awaiting_credentials": a real, documented endpoint is known, but
  this environment has no credentials configured for it.
- "error": the API call was attempted and failed (network, HTTP
  error, malformed response, etc.).

NormalizedTender is the common shape every connector's records are
converted into, so agents/rfp_research/graph.py can treat all
countries' data the same way regardless of source API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class NormalizedTender(BaseModel):
    """One tender record, normalized from whatever a source API returned."""

    title: Optional[str] = None
    reference_number: Optional[str] = None
    organization: Optional[str] = None
    country: str
    source_id: str = Field(
        description="Stable identifier from the source API, used for deduplication."
    )
    source_url: Optional[str] = None
    publication_date: Optional[str] = None
    deadline: Optional[str] = None
    status: Optional[str] = Field(
        default=None,
        description="The tender's status AS REPORTED BY THE SOURCE API (e.g. 'Published', 'Draft'), not inferred.",
    )
    category: Optional[str] = None
    scope_of_work: Optional[str] = None
    eligibility: list[str] = Field(default_factory=list)
    submission_instructions: Optional[str] = None
    submission_url: Optional[str] = None
    # The buyer's contact email exactly as the official API gives it (the
    # only address an application may ever be prepared for).
    contact_email: Optional[str] = None
    available_documents: list[str] = Field(default_factory=list)
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_name: str
    verification_status: str = "verified_from_api"
    missing_information: list[str] = Field(default_factory=list)


class APIQueryResult(BaseModel):
    """The outcome of one connector's attempt to query its source API."""

    source_name: str
    status: Literal["success", "unavailable", "error", "awaiting_credentials"]
    message: str
    records: list[NormalizedTender] = Field(default_factory=list)
    raw_record_count: int = 0
