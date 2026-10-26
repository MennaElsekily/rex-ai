"""
LangGraph workflow for the RFP/Tender Research Agent (Agent 2).

Nodes (brief Section "Agent 2"):
    validate_request -> query_apis (real country-specific API
        connectors -- agents/rfp_research/api_clients/)
        -> [out of scope: compose_response, early exit]
        -> filter_relevant (lightweight keyword pre-check)
        -> [nothing candidate: compose_response, early exit]
        -> detect_duplicates (against this conversation's history, by
           the API's own stable source_id)
        -> [all duplicates: compose_response, early exit]
        -> assess_and_filter (LLM relevance + qualification judgment
           on already-structured API records -- no page scraping)
        -> apply_business_rule -> store_opportunities
        -> compose_response -> END

This agent no longer uses web search or HTML scraping as its normal
discovery mechanism (see agents/rfp_research/api_clients/ for the real
API connectors). An API connector reporting "unavailable" or "error"
is never silently treated as "no tenders found" -- see
query_apis/compose_response below.

This graph is internal to the RFP agent. The top-level REX AI
orchestrator (agents/orchestrator.py) never sees these nodes or this
state -- it only calls RFPResearchAgent.process_request() (agent.py).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from agents.conversation_store import ConversationStore
from agents.rfp_research.api_clients import (
    query_etimad_saudi_arabia,
    query_world_bank_egypt,
)
from agents.rfp_research.opportunity_store import OpportunityStore
from agents.rfp_research.state import (
    ALLOWED_COUNTRIES,
    CATEGORY_KEYWORDS,
    RFPResearchState,
)
from agents.rfp_research.tools import build_assessment_chain

RFP_AGENT_NAME = "rfp_research"

CATEGORY_QUERY = "training"
# Confirmed via a direct API check during development: the World Bank
# API's qterm appears to match ALL space-separated terms together
# (AND semantics), so a long compound query like "education training
# technical IT software digital skills professional development"
# returns zero results even though each term alone matches real
# records. A short, broad term plus client-side keyword relevance
# filtering (_mentions_relevant_category below) and the LLM relevance
# judgment in tools.py together do the real category narrowing.

_DISALLOWED_COUNTRY_HINTS = (
    "usa", "united states", "u.s.", "europe", "european union", "eu ",
    "uk", "united kingdom", "britain", "germany", "france", "canada",
    "australia", "spain", "italy",
)

_COUNTRY_KEYWORDS = {
    "Egypt": ("egypt", ".eg", "مصر"),
    "Saudi Arabia": ("saudi", "ksa", ".sa", "السعودية", "سعودية"),
}


def _detect_requested_countries(text: str) -> list[str]:
    normalized = text.casefold()

    return [
        country
        for country, keywords in _COUNTRY_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    ]


def _detect_out_of_scope_country(text: str) -> Optional[str]:
    normalized = text.casefold()

    for hint in _DISALLOWED_COUNTRY_HINTS:
        if hint in normalized:
            return hint.strip()

    return None


def _mentions_relevant_category(record: dict[str, Any]) -> bool:
    haystack = " ".join([
        str(record.get("title") or ""),
        str(record.get("scope_of_work") or ""),
        str(record.get("category") or ""),
    ]).casefold()

    return any(keyword in haystack for keyword in CATEGORY_KEYWORDS)


def _is_expired(deadline: Optional[str]) -> bool:
    if not deadline:
        return False

    try:
        deadline_date = datetime.fromisoformat(str(deadline)).date()
    except ValueError:
        return False

    return deadline_date < date.today()


def make_validate_request_node():
    def validate_request_node(state: RFPResearchState) -> dict[str, Any]:
        text = state.get("request_text", "")

        return {
            "requested_countries": (
                _detect_requested_countries(text) or list(ALLOWED_COUNTRIES)
            ),
            "out_of_scope_country": _detect_out_of_scope_country(text),
        }

    return validate_request_node


def route_after_validation(state: RFPResearchState) -> str:
    return "out_of_scope" if state.get("out_of_scope_country") else "query"


def make_query_apis_node():
    def query_apis_node(state: RFPResearchState) -> dict[str, Any]:
        countries = state.get("requested_countries", list(ALLOWED_COUNTRIES))
        api_results: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []

        if "Egypt" in countries:
            result = query_world_bank_egypt(CATEGORY_QUERY)
            api_results.append(result.model_dump())

            if result.status == "success":
                candidates.extend(
                    record.model_dump() for record in result.records
                )

        if "Saudi Arabia" in countries:
            result = query_etimad_saudi_arabia(CATEGORY_QUERY)
            api_results.append(result.model_dump())
            # status == "unavailable": no records to add, by design --
            # never silently reported as "zero tenders found."

        return {"api_results": api_results, "candidate_records_raw": candidates}

    return query_apis_node


def make_filter_relevant_node():
    def filter_relevant_node(state: RFPResearchState) -> dict[str, Any]:
        raw = state.get("candidate_records_raw", [])
        candidates = [
            record
            for record in raw
            if not _is_expired(record.get("deadline"))
            and _mentions_relevant_category(record)
        ]
        return {"candidate_records": candidates}

    return filter_relevant_node


def route_after_filter(state: RFPResearchState) -> str:
    any_api_succeeded = any(
        result.get("status") == "success"
        for result in state.get("api_results", [])
    )

    if not any_api_succeeded:
        return "no_api_data"

    return "dedup" if state.get("candidate_records") else "no_results"


def make_detect_duplicates_node(
    conversation_store: Optional[ConversationStore],
):
    def detect_duplicates_node(state: RFPResearchState) -> dict[str, Any]:
        candidates = state.get("candidate_records", [])
        conversation_id = state.get("conversation_id")

        seen_ids: set[str] = set()

        if conversation_store is not None and conversation_id:
            previous = conversation_store.get(conversation_id, RFP_AGENT_NAME)

            if previous and isinstance(previous.get("seen_source_ids"), list):
                seen_ids = set(previous["seen_source_ids"])

        new_records = []
        duplicate_count = 0
        batch_seen: set[str] = set()

        for record in candidates:
            key = f"{record.get('source_name')}::{record.get('source_id')}"

            if key in seen_ids or key in batch_seen:
                duplicate_count += 1
                continue

            batch_seen.add(key)
            new_records.append(record)

        if conversation_store is not None and conversation_id and new_records:
            conversation_store.set(
                conversation_id,
                RFP_AGENT_NAME,
                {"seen_source_ids": sorted(seen_ids | batch_seen)},
            )

        return {"new_records": new_records, "duplicate_count": duplicate_count}

    return detect_duplicates_node


def route_after_dedup(state: RFPResearchState) -> str:
    return "assess" if state.get("new_records") else "no_new_results"


def make_assess_and_filter_node(model: str):
    def assess_and_filter_node(state: RFPResearchState) -> dict[str, Any]:
        chain = build_assessment_chain(model)
        assessed: list[dict[str, Any]] = []

        for record in state.get("new_records", []):
            try:
                assessment = chain.invoke({
                    "title": record.get("title") or "",
                    "organization": record.get("organization") or "",
                    "country": record.get("country") or "",
                    "category": record.get("category") or "",
                    "scope_of_work": record.get("scope_of_work") or "",
                    "eligibility": "; ".join(record.get("eligibility") or []) or "Not specified",
                    "deadline": record.get("deadline") or "Not specified",
                })
            except Exception as exc:
                assessed.append({
                    **record,
                    "ai_summary": None,
                    "qualification_status": "unknown",
                    "qualification_reason": f"Assessment unavailable: {exc}",
                })
                continue

            if not assessment.is_relevant:
                continue

            assessed.append({
                **record,
                "ai_summary": assessment.summary,
                "qualification_status": assessment.qualification_status,
                "qualification_reason": assessment.qualification_reason,
            })

        return {"assessed_results": assessed}

    return assess_and_filter_node


def make_apply_business_rule_node():
    def apply_business_rule_node(state: RFPResearchState) -> dict[str, Any]:
        results = []

        for record in state.get("assessed_results", []):
            # "Important Business Rule" (brief, Agent 2): Egypt
            # opportunities are flagged immediately for review; Saudi
            # Arabian opportunities enter the monitored pipeline. This
            # label describes REVIEW PRIORITY only -- it does not mean
            # the tender has been verified as still open or that the
            # academy qualifies (see qualification_status for that).
            review_status = (
                "flagged_for_immediate_review"
                if record.get("country") == "Egypt"
                else "monitored_pipeline"
            )
            results.append({**record, "review_status": review_status})

        return {"final_results": results}

    return apply_business_rule_node


def make_store_opportunities_node(
    opportunity_store: Optional[OpportunityStore],
):
    def store_opportunities_node(state: RFPResearchState) -> dict[str, Any]:
        results = state.get("final_results", [])

        if opportunity_store is None:
            return {"newly_discovered_count": len(results)}

        newly_discovered = 0

        for record in results:
            if opportunity_store.save(record):
                newly_discovered += 1

        return {"newly_discovered_count": newly_discovered}

    return store_opportunities_node


def make_compose_response_node():
    def compose_response_node(state: RFPResearchState) -> dict[str, Any]:
        api_results = state.get("api_results", [])

        if state.get("out_of_scope_country"):
            country = state["out_of_scope_country"]
            message = (
                f"This agent only covers tenders in Egypt and Saudi "
                f"Arabia. '{country}' is outside the supported scope. "
                "هذا النظام يغطي فرص المناقصات في مصر والسعودية فقط."
            )
            return {
                "message": message,
                "response": {
                    "message": message, "opportunities": [],
                    "api_results": [], "duplicate_count": 0,
                    "newly_discovered_count": 0,
                },
            }

        any_api_succeeded = any(
            result.get("status") == "success" for result in api_results
        )

        if not any_api_succeeded:
            # An API error/unavailability must never read like "no
            # tenders found" -- state explicitly what happened.
            problems = "; ".join(
                f"{result['source_name']}: {result['status']} -- {result['message']}"
                for result in api_results
            )
            message = (
                "No tender data could be retrieved -- this is NOT the "
                f"same as finding zero relevant tenders. {problems}"
            )
            return {
                "message": message,
                "response": {
                    "message": message, "opportunities": [],
                    "api_results": api_results, "duplicate_count": 0,
                    "newly_discovered_count": 0,
                },
            }

        if "final_results" not in state:
            if "new_records" in state:
                count = state.get("duplicate_count", 0)
                message = (
                    f"No new opportunities -- all {count} relevant "
                    f"match{'es' if count != 1 else ''} were already "
                    "found in an earlier search in this conversation."
                )
            else:
                message = (
                    "The configured APIs responded successfully, but "
                    "no relevant education/training opportunities were "
                    "found for Egypt or Saudi Arabia. "
                    "لم يتم العثور على فرص مناقصات ذات صلة."
                )

            return {
                "message": message,
                "response": {
                    "message": message, "opportunities": [],
                    "api_results": api_results,
                    "duplicate_count": state.get("duplicate_count", 0),
                    "newly_discovered_count": 0,
                },
            }

        results = state["final_results"]
        new_count = state.get("newly_discovered_count", len(results))

        message = (
            f"Found {len(results)} relevant tender"
            f"{'s' if len(results) != 1 else ''} via API "
            f"({new_count} newly discovered, "
            f"{state.get('duplicate_count', 0)} duplicate(s) skipped)."
        )

        return {
            "message": message,
            "response": {
                "message": message,
                "opportunities": results,
                "api_results": api_results,
                "duplicate_count": state.get("duplicate_count", 0),
                "newly_discovered_count": new_count,
            },
        }

    return compose_response_node


def build_graph(
    conversation_store: Optional[ConversationStore] = None,
    opportunity_store: Optional[OpportunityStore] = None,
    model: str = "qwen2.5:3b",
):
    """Build and compile the RFP Research Agent's internal LangGraph workflow."""
    graph = StateGraph(RFPResearchState)

    graph.add_node("validate_request", make_validate_request_node())
    graph.add_node("query_apis", make_query_apis_node())
    graph.add_node("filter_relevant", make_filter_relevant_node())
    graph.add_node(
        "detect_duplicates", make_detect_duplicates_node(conversation_store)
    )
    graph.add_node("assess_and_filter", make_assess_and_filter_node(model))
    graph.add_node("apply_business_rule", make_apply_business_rule_node())
    graph.add_node(
        "store_opportunities", make_store_opportunities_node(opportunity_store)
    )
    graph.add_node("compose_response", make_compose_response_node())

    graph.set_entry_point("validate_request")

    graph.add_conditional_edges(
        "validate_request",
        route_after_validation,
        {"out_of_scope": "compose_response", "query": "query_apis"},
    )

    graph.add_edge("query_apis", "filter_relevant")

    graph.add_conditional_edges(
        "filter_relevant",
        route_after_filter,
        {
            "no_api_data": "compose_response",
            "no_results": "compose_response",
            "dedup": "detect_duplicates",
        },
    )

    graph.add_conditional_edges(
        "detect_duplicates",
        route_after_dedup,
        {"no_new_results": "compose_response", "assess": "assess_and_filter"},
    )

    graph.add_edge("assess_and_filter", "apply_business_rule")
    graph.add_edge("apply_business_rule", "store_opportunities")
    graph.add_edge("store_opportunities", "compose_response")
    graph.add_edge("compose_response", END)

    return graph.compile()
