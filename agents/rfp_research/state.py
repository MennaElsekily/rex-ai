"""
State for the RFP/Tender Research Agent's internal LangGraph workflow.

Separate from, and invisible to, the top-level REX AI orchestrator's
OrchestratorState (agents/orchestrator.py). The orchestrator only ever
sees this agent's final normalized response (see agent.py).
"""

from typing import Any, Optional, TypedDict

# Brief Section "Geographic Scope": the only two countries this agent
# is allowed to return opportunities for.
ALLOWED_COUNTRIES = ("Egypt", "Saudi Arabia")

CATEGORY_KEYWORDS = (
    "education", "training", "technical training",
    "professional development", "digital transformation",
    "software", "it training", "corporate learning", "e-learning",
    "curriculum", "capacity building", "skills",
)


class RFPResearchState(TypedDict, total=False):
    # Set by the agent before invoking the graph.
    request_text: str
    conversation_id: Optional[str]

    # Populated by validate_request.
    requested_countries: list[str]
    out_of_scope_country: Optional[str]

    # Populated by query_apis: one APIQueryResult (as a dict) per
    # requested country's connector, so the UI can show exactly which
    # source was used and whether it actually succeeded.
    api_results: list[dict[str, Any]]
    candidate_records_raw: list[dict[str, Any]]

    # Populated by filter_relevant: NormalizedTender dicts that pass a
    # lightweight keyword pre-check (saves LLM calls on obviously
    # unrelated API records before the real relevance judgment).
    candidate_records: list[dict[str, Any]]

    # Populated by detect_duplicates (against this conversation's
    # previously seen source_ids, via ConversationStore).
    new_records: list[dict[str, Any]]
    duplicate_count: int

    # Populated by assess_and_filter (LLM relevance + qualification).
    assessed_results: list[dict[str, Any]]

    # Populated by apply_business_rule and store_opportunities.
    final_results: list[dict[str, Any]]
    newly_discovered_count: int

    # Populated by compose_response.
    message: str
    response: dict[str, Any]
