"""
Saudi Arabia tender API connector -- architecture ready, NOT
operational. Does not fall back to web search or scraping.

Investigation findings (see the project's final report for full
detail of what was checked):

- The official Etimad Developer Portal (https://apiportal.etimad.sa)
  exists and lists API products, but the ones found during this
  investigation ("Contracts Plus", "Bank Contracts Inquiry") are
  historical/awarded CONTRACT lookups requiring an already-known
  contract or beneficiary ID -- not a tender-discovery/listing feed --
  and appear to require a commercial agreement with per-volume fees.
- No public, documented, open tender-LISTING API was confirmed for
  Saudi Arabia during this investigation.
- Third-party wrappers exist (e.g. "Tenders Alerts") but require
  contacting the vendor directly for an API key and are not an
  official government source, so they are not integrated here.

This module exists so a real connector can be dropped in later with
the exact same APIQueryResult/NormalizedTender interface as
world_bank.py, without redesigning the rest of the agent.

To complete this integration once real access exists:
1. Confirm the documented endpoint, auth method, and response schema
   from Etimad's Developer Portal (apiportal.etimad.sa) or whichever
   product actually provides open tender listings.
2. Set ETIMAD_API_KEY (and any other required credential) as an
   environment variable -- never hardcode it.
3. Replace the body of query_etimad_saudi_arabia() with a real HTTP
   call against the confirmed endpoint, normalizing its response into
   NormalizedTender the same way world_bank.py does.
"""

from __future__ import annotations

import os

from agents.rfp_research.api_clients.base import APIQueryResult

SOURCE_NAME = "Etimad (Saudi Arabia) -- no verified tender-listing API"


def query_etimad_saudi_arabia(category_query: str) -> APIQueryResult:
    """
    Always returns status="unavailable" today -- see module docstring.
    Reads ETIMAD_API_KEY from the environment (for future use only;
    it currently has no effect, since no confirmed endpoint exists to
    call with it).
    """
    has_key = bool(os.environ.get("ETIMAD_API_KEY"))

    credential_note = (
        "ETIMAD_API_KEY is set in this environment, but no verified "
        "tender-listing endpoint has been confirmed to use it with yet."
        if has_key
        else "No ETIMAD_API_KEY is configured. Set it once a documented "
        "tender-listing endpoint and credentials are obtained from "
        "Etimad to enable this connector."
    )

    return APIQueryResult(
        source_name=SOURCE_NAME,
        status="unavailable",
        message=(
            "No documented, publicly accessible API for listing open "
            "Saudi Arabia tenders was confirmed. The official Etimad "
            "Developer Portal offers contract-history lookup products "
            "requiring an already-known contract/beneficiary ID, not "
            "tender discovery, and appears to require a commercial "
            f"agreement. {credential_note}"
        ),
    )
