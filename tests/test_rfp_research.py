"""
Tests for agents.rfp_research (RFP/Tender Research Agent, Agent 2).

This agent is API-first: it queries agents.rfp_research.api_clients
(World Bank Procurement Notices API for Egypt; an explicitly
"unavailable" Etimad connector for Saudi Arabia -- see that module's
docstring for why). It no longer uses web search or HTML scraping.

For automated testing, the two external boundaries are mocked so the
suite is deterministic, fast, and does not depend on network access or
a running Ollama server:

- agents.rfp_research.graph.query_world_bank_egypt
- agents.rfp_research.graph.query_etimad_saudi_arabia
- agents.rfp_research.graph.build_assessment_chain

A separate, manually-run live verification against the real World
Bank API is reported in the final report, not in this file -- pytest
must not depend on external services to pass.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agents.conversation_store import ConversationStore
from agents.rfp_research import RFPResearchAgent
from agents.rfp_research.api_clients.base import APIQueryResult, NormalizedTender
from agents.rfp_research.opportunity_store import OpportunityStore


def _tender(
    source_id="OP001",
    country="Egypt",
    title="Digital Skills Training Tender",
    deadline=None,
    source_name="World Bank Procurement Notices API (Egypt-financed)",
):
    return NormalizedTender(
        title=title,
        reference_number="REF-001",
        organization="Example Ministry",
        country=country,
        source_id=source_id,
        source_url=f"https://example.com/notice/{source_id}",
        deadline=deadline,
        status="Published",
        category="Request for Expression of Interest",
        scope_of_work="Deliver digital skills training to public-sector staff.",
        eligibility=["Relevant training capability"],
        source_name=source_name,
        verification_status="verified_from_api",
    )


def _success(records, raw_count=None):
    return APIQueryResult(
        source_name="World Bank Procurement Notices API (Egypt-financed)",
        status="success",
        message=f"Retrieved {raw_count or len(records)} record(s).",
        records=records,
        raw_record_count=raw_count if raw_count is not None else len(records),
    )


def _assessment(is_relevant=True, qualification_status="qualified"):
    return SimpleNamespace(
        is_relevant=is_relevant,
        summary="Short factual summary.",
        qualification_status=qualification_status,
        qualification_reason="Matches stated eligibility.",
    )


@pytest.fixture
def stores(tmp_path):
    conv_store = ConversationStore(db_path=tmp_path / "conv.sqlite3")
    opp_store = OpportunityStore(db_path=tmp_path / "opportunities.sqlite3")
    return conv_store, opp_store


def _make_agent(conv_store, opp_store):
    return RFPResearchAgent(conversation_store=conv_store, opportunity_store=opp_store)


def test_successful_api_response_returns_assessed_opportunities(stores):
    conv_store, opp_store = stores

    with patch(
        "agents.rfp_research.graph.query_world_bank_egypt",
        return_value=_success([_tender()]),
    ):
        with patch(
            "agents.rfp_research.graph.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )

            with patch(
                "agents.rfp_research.graph.build_assessment_chain"
            ) as mock_chain:
                mock_chain.return_value.invoke.return_value = _assessment()
                agent = _make_agent(conv_store, opp_store)

                response = agent.process_request({
                    "request_id": "r1",
                    "user_input": "Find education tenders in Egypt",
                    "context": {"conversation_id": "conv-1"},
                })

    assert response["status"] == "success"
    opportunities = response["result"]["opportunities"]
    assert len(opportunities) == 1
    assert opportunities[0]["qualification_status"] == "qualified"
    assert opportunities[0]["review_status"] == "flagged_for_immediate_review"


def test_api_error_is_never_reported_as_zero_tenders_found(stores):
    """
    Regression test for the explicit instruction: an API error must
    never be presented the same way as "no tenders found."
    """
    conv_store, opp_store = stores

    with patch(
        "agents.rfp_research.graph.query_world_bank_egypt",
        return_value=APIQueryResult(
            source_name="World Bank Procurement Notices API (Egypt-financed)",
            status="error",
            message="World Bank API request failed: connection timed out",
        ),
    ):
        with patch(
            "agents.rfp_research.graph.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            agent = _make_agent(conv_store, opp_store)

            response = agent.process_request({
                "request_id": "r1",
                "user_input": "Find education tenders in Egypt",
                "context": {"conversation_id": "conv-1"},
            })

    assert response["status"] == "success"  # the agent itself didn't crash
    result = response["result"]
    assert result["opportunities"] == []
    assert "NOT the same as finding zero" in result["message"]
    assert result["api_results"][0]["status"] == "error"


def test_saudi_arabia_reports_unavailable_not_silently_empty(stores):
    conv_store, opp_store = stores

    with patch(
        "agents.rfp_research.graph.query_etimad_saudi_arabia"
    ) as mock_etimad:
        mock_etimad.return_value = APIQueryResult(
            source_name="Etimad (Saudi Arabia) -- no verified tender-listing API",
            status="unavailable",
            message="No documented, publicly accessible API was confirmed.",
        )
        agent = _make_agent(conv_store, opp_store)

        response = agent.process_request({
            "request_id": "r1",
            "user_input": "Find education tenders in Saudi Arabia",
            "context": {"conversation_id": "conv-1"},
        })

    result = response["result"]
    assert result["opportunities"] == []
    assert result["api_results"][0]["status"] == "unavailable"
    assert "No API data could be retrieved" not in result["message"]  # sanity
    assert "NOT the same as finding zero" in result["message"]


def test_no_automatic_fallback_to_web_search():
    """
    Confirms the architecture does not import or use any web-search/
    scraping module -- a direct check against the instruction not to
    silently fall back to DuckDuckGo/HTML scraping. The old
    agents/rfp_research/search.py (DuckDuckGo + BeautifulSoup) was
    deleted; this import must fail.
    """
    import agents.rfp_research.graph as graph_module

    assert not hasattr(graph_module, "search_opportunities_tool")
    assert not hasattr(graph_module, "fetch_page")

    with pytest.raises(ModuleNotFoundError):
        import agents.rfp_research.search  # noqa: F401


def test_irrelevant_tender_is_filtered_by_keyword_before_llm_call(stores):
    conv_store, opp_store = stores
    irrelevant = _tender(
        source_id="OP999",
        title="Bridge Construction Supervision Contract",
    )
    irrelevant.scope_of_work = "Civil engineering supervision for a bridge project."
    irrelevant.category = "Contract Award"

    with patch(
        "agents.rfp_research.graph.query_world_bank_egypt",
        return_value=_success([irrelevant]),
    ):
        with patch(
            "agents.rfp_research.graph.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_assessment_chain"
            ) as mock_chain:
                agent = _make_agent(conv_store, opp_store)

                response = agent.process_request({
                    "request_id": "r1",
                    "user_input": "Find education tenders in Egypt",
                    "context": {"conversation_id": "conv-1"},
                })

    assert response["result"]["opportunities"] == []
    mock_chain.return_value.invoke.assert_not_called()


def test_llm_relevance_judgment_can_still_exclude_a_record(stores):
    conv_store, opp_store = stores

    with patch(
        "agents.rfp_research.graph.query_world_bank_egypt",
        return_value=_success([_tender()]),
    ):
        with patch(
            "agents.rfp_research.graph.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_assessment_chain"
            ) as mock_chain:
                mock_chain.return_value.invoke.return_value = _assessment(
                    is_relevant=False
                )
                agent = _make_agent(conv_store, opp_store)

                response = agent.process_request({
                    "request_id": "r1",
                    "user_input": "Find education tenders in Egypt",
                    "context": {"conversation_id": "conv-1"},
                })

    assert response["result"]["opportunities"] == []


def test_expired_opportunity_is_excluded(stores):
    conv_store, opp_store = stores
    expired = _tender(deadline="2020-01-01")

    with patch(
        "agents.rfp_research.graph.query_world_bank_egypt",
        return_value=_success([expired]),
    ):
        with patch(
            "agents.rfp_research.graph.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            agent = _make_agent(conv_store, opp_store)

            response = agent.process_request({
                "request_id": "r1",
                "user_input": "Find education tenders in Egypt",
                "context": {"conversation_id": "conv-1"},
            })

    assert response["result"]["opportunities"] == []


def test_out_of_scope_country_is_rejected_without_calling_apis(stores):
    conv_store, opp_store = stores

    with patch("agents.rfp_research.graph.query_world_bank_egypt") as mock_wb:
        agent = _make_agent(conv_store, opp_store)

        response = agent.process_request({
            "request_id": "r1",
            "user_input": "Find tenders in Germany",
            "context": {"conversation_id": "conv-1"},
        })

    assert response["result"]["opportunities"] == []
    mock_wb.assert_not_called()


def test_duplicate_detection_by_stable_source_id(stores):
    conv_store, opp_store = stores

    with patch(
        "agents.rfp_research.graph.query_world_bank_egypt",
        return_value=_success([_tender()]),
    ):
        with patch(
            "agents.rfp_research.graph.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_assessment_chain"
            ) as mock_chain:
                mock_chain.return_value.invoke.return_value = _assessment()
                agent = _make_agent(conv_store, opp_store)

                first = agent.process_request({
                    "request_id": "r1",
                    "user_input": "Find education tenders in Egypt",
                    "context": {"conversation_id": "conv-dedup"},
                })
                second = agent.process_request({
                    "request_id": "r2",
                    "user_input": "Find education tenders in Egypt",
                    "context": {"conversation_id": "conv-dedup"},
                })

    assert len(first["result"]["opportunities"]) == 1
    assert second["result"]["opportunities"] == []
    assert second["result"]["duplicate_count"] == 1


def test_qualification_unknown_when_eligibility_not_stated(stores):
    conv_store, opp_store = stores
    tender = _tender()
    tender.eligibility = []

    with patch(
        "agents.rfp_research.graph.query_world_bank_egypt",
        return_value=_success([tender]),
    ):
        with patch(
            "agents.rfp_research.graph.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_assessment_chain"
            ) as mock_chain:
                mock_chain.return_value.invoke.return_value = _assessment(
                    qualification_status="unknown"
                )
                agent = _make_agent(conv_store, opp_store)

                response = agent.process_request({
                    "request_id": "r1",
                    "user_input": "Find education tenders in Egypt",
                    "context": {"conversation_id": "conv-1"},
                })

    assert response["result"]["opportunities"][0]["qualification_status"] == "unknown"


def test_opportunities_are_persisted_to_the_searchable_store(stores):
    conv_store, opp_store = stores

    with patch(
        "agents.rfp_research.graph.query_world_bank_egypt",
        return_value=_success([_tender()]),
    ):
        with patch(
            "agents.rfp_research.graph.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_assessment_chain"
            ) as mock_chain:
                mock_chain.return_value.invoke.return_value = _assessment()
                agent = _make_agent(conv_store, opp_store)

                agent.process_request({
                    "request_id": "r1",
                    "user_input": "Find education tenders in Egypt",
                    "context": {"conversation_id": "conv-1"},
                })

    stored = opp_store.search(country="Egypt")
    assert len(stored) == 1
    assert stored[0]["source_id"] == "OP001"


def test_assessment_failure_does_not_crash_or_fabricate(stores):
    conv_store, opp_store = stores

    with patch(
        "agents.rfp_research.graph.query_world_bank_egypt",
        return_value=_success([_tender()]),
    ):
        with patch(
            "agents.rfp_research.graph.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_assessment_chain"
            ) as mock_chain:
                mock_chain.return_value.invoke.side_effect = RuntimeError(
                    "model unavailable"
                )
                agent = _make_agent(conv_store, opp_store)

                response = agent.process_request({
                    "request_id": "r1",
                    "user_input": "Find education tenders in Egypt",
                    "context": {"conversation_id": "conv-1"},
                })

    assert response["status"] == "success"
    opportunities = response["result"]["opportunities"]
    assert len(opportunities) == 1
    assert opportunities[0]["qualification_status"] == "unknown"
    assert "unavailable" in opportunities[0]["qualification_reason"]


def test_empty_user_input_is_rejected(stores):
    conv_store, opp_store = stores
    agent = _make_agent(conv_store, opp_store)

    response = agent.process_request({
        "request_id": "r1", "user_input": "   ", "context": {},
    })

    assert response["status"] == "error"
