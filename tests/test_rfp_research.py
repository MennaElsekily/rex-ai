"""
Tests for agents.rfp_research (RFP/Tender Research Agent, Agent 2).

This file covers the tender-API path: agents.rfp_research.api_clients
(World Bank Procurement Notices API for Egypt; an explicitly
"unavailable" Etimad connector for Saudi Arabia -- see that module's
docstring for why). Web research (both countries) is covered in
test_rfp_web_research.py; here the web search is offline (conftest.py),
so it only adds "awaiting_credentials" source reports.

For automated testing, the external boundaries are mocked so the
suite is deterministic, fast, and does not depend on network access or
a running Ollama server:

- agents.rfp_research.tender_tools.query_world_bank_egypt
- agents.rfp_research.tender_tools.query_etimad_saudi_arabia
  (the graph reaches them through the tender tools; conftest.py makes
  the agent use those tools in-process)
- agents.rfp_research.graph.build_relevance_chain
- agents.rfp_research.graph.build_qualification_chain

Relevance and qualification are TWO separate LLM calls/chains (see
agents/rfp_research/tools.py's module docstring): a live run against
the real World Bank API showed a combined call entangling "is this
relevant" with "do we have enough eligibility info to qualify",
causing a genuinely on-topic tender ("Training and Capacity Building
Consultancy Services") to be rejected as not relevant merely because
eligibility was unstated. Several tests below specifically guard
against that regression.

A separate, manually-run live verification against the real World
Bank API and a real Ollama model is reported in the final report, not
in this file -- pytest must not depend on external services to pass.
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
    eligibility=("Relevant training capability",),
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
        eligibility=list(eligibility),
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


def _relevance(is_relevant=True):
    return SimpleNamespace(
        is_relevant=is_relevant,
        summary="Short factual summary.",
        relevance_reason="On-topic training tender.",
    )


def _qualification(qualification_status="qualified"):
    return SimpleNamespace(
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
        "agents.rfp_research.tender_tools.query_world_bank_egypt",
        return_value=_success([_tender()]),
    ):
        with patch(
            "agents.rfp_research.tender_tools.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )

            with patch(
                "agents.rfp_research.graph.build_relevance_chain"
            ) as mock_relevance, patch(
                "agents.rfp_research.graph.build_qualification_chain"
            ) as mock_qualification:
                mock_relevance.return_value.invoke.return_value = _relevance()
                mock_qualification.return_value.invoke.return_value = _qualification()
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
        "agents.rfp_research.tender_tools.query_world_bank_egypt",
        return_value=APIQueryResult(
            source_name="World Bank Procurement Notices API (Egypt-financed)",
            status="error",
            message="World Bank API request failed: connection timed out",
        ),
    ):
        with patch(
            "agents.rfp_research.tender_tools.query_etimad_saudi_arabia"
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
        "agents.rfp_research.tender_tools.query_etimad_saudi_arabia"
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


def test_no_html_scraper_module():
    """
    Web research goes through the Tavily search API (web_tenders.py),
    which returns each page's text; the old DuckDuckGo + BeautifulSoup
    scraper (agents/rfp_research/search.py) stays deleted.
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
        "agents.rfp_research.tender_tools.query_world_bank_egypt",
        return_value=_success([irrelevant]),
    ):
        with patch(
            "agents.rfp_research.tender_tools.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_relevance_chain"
            ) as mock_relevance, patch(
                "agents.rfp_research.graph.build_qualification_chain"
            ) as mock_qualification:
                agent = _make_agent(conv_store, opp_store)

                response = agent.process_request({
                    "request_id": "r1",
                    "user_input": "Find education tenders in Egypt",
                    "context": {"conversation_id": "conv-1"},
                })

    assert response["result"]["opportunities"] == []
    mock_relevance.return_value.invoke.assert_not_called()
    mock_qualification.return_value.invoke.assert_not_called()


def test_llm_relevance_judgment_can_still_exclude_a_record(stores):
    """
    The relevance LLM call can still reject a genuinely unrelated
    tender that slipped past the keyword pre-filter -- and, since
    qualification is a separate call, it must never even be reached
    for a record the relevance call rejected.
    """
    conv_store, opp_store = stores

    with patch(
        "agents.rfp_research.tender_tools.query_world_bank_egypt",
        return_value=_success([_tender()]),
    ):
        with patch(
            "agents.rfp_research.tender_tools.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_relevance_chain"
            ) as mock_relevance, patch(
                "agents.rfp_research.graph.build_qualification_chain"
            ) as mock_qualification:
                mock_relevance.return_value.invoke.return_value = _relevance(
                    is_relevant=False
                )
                agent = _make_agent(conv_store, opp_store)

                response = agent.process_request({
                    "request_id": "r1",
                    "user_input": "Find education tenders in Egypt",
                    "context": {"conversation_id": "conv-1"},
                })

    assert response["result"]["opportunities"] == []
    mock_qualification.return_value.invoke.assert_not_called()


def test_relevance_chain_is_never_given_eligibility_information(stores):
    """
    Architectural regression guard for the Egypt relevance-filtering
    bug: the relevance call's input must not even contain an
    eligibility field, so missing/unstated eligibility cannot
    influence is_relevant.
    """
    conv_store, opp_store = stores

    with patch(
        "agents.rfp_research.tender_tools.query_world_bank_egypt",
        return_value=_success([_tender(eligibility=())]),
    ):
        with patch(
            "agents.rfp_research.tender_tools.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_relevance_chain"
            ) as mock_relevance, patch(
                "agents.rfp_research.graph.build_qualification_chain"
            ) as mock_qualification:
                mock_relevance.return_value.invoke.return_value = _relevance()
                mock_qualification.return_value.invoke.return_value = _qualification(
                    qualification_status="unknown"
                )
                agent = _make_agent(conv_store, opp_store)

                agent.process_request({
                    "request_id": "r1",
                    "user_input": "Find education tenders in Egypt",
                    "context": {"conversation_id": "conv-1"},
                })

    relevance_call_kwargs = mock_relevance.return_value.invoke.call_args[0][0]
    assert "eligibility" not in relevance_call_kwargs


def test_expired_opportunity_is_excluded(stores):
    conv_store, opp_store = stores
    expired = _tender(deadline="2020-01-01")

    with patch(
        "agents.rfp_research.tender_tools.query_world_bank_egypt",
        return_value=_success([expired]),
    ):
        with patch(
            "agents.rfp_research.tender_tools.query_etimad_saudi_arabia"
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

    with patch("agents.rfp_research.tender_tools.query_world_bank_egypt") as mock_wb:
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
        "agents.rfp_research.tender_tools.query_world_bank_egypt",
        return_value=_success([_tender()]),
    ):
        with patch(
            "agents.rfp_research.tender_tools.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_relevance_chain"
            ) as mock_relevance, patch(
                "agents.rfp_research.graph.build_qualification_chain"
            ) as mock_qualification:
                mock_relevance.return_value.invoke.return_value = _relevance()
                mock_qualification.return_value.invoke.return_value = _qualification()
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
    """
    Brief requirement: "A relevant opportunity must not be rejected
    solely because eligibility requirements are missing or
    insufficient" -- a record with no stated eligibility must still be
    RETAINED (relevant), just with qualification_status "unknown".

    This is decided in CODE, without calling the qualification LLM at
    all: a live run showed qwen2.5:3b, even in its own dedicated
    qualification-only call, sometimes answering "not_qualified" with
    reasoning like "no information provided about eligibility" --
    exactly the entanglement the brief's business rule forbids ("never
    default to 'not_qualified' merely because eligibility was not
    stated"). Since "was eligibility stated at all" is a plain fact,
    deciding it in code instead of asking the LLM removes that failure
    mode by construction.
    """
    conv_store, opp_store = stores
    tender = _tender(eligibility=())

    with patch(
        "agents.rfp_research.tender_tools.query_world_bank_egypt",
        return_value=_success([tender]),
    ):
        with patch(
            "agents.rfp_research.tender_tools.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_relevance_chain"
            ) as mock_relevance, patch(
                "agents.rfp_research.graph.build_qualification_chain"
            ) as mock_qualification:
                mock_relevance.return_value.invoke.return_value = _relevance(
                    is_relevant=True
                )
                # Deliberately mocked to return something OTHER than
                # "unknown" -- if this chain were called at all here,
                # the test would be able to see that leak through. It
                # must not be called for a record with no eligibility.
                mock_qualification.return_value.invoke.return_value = _qualification(
                    qualification_status="not_qualified"
                )
                agent = _make_agent(conv_store, opp_store)

                response = agent.process_request({
                    "request_id": "r1",
                    "user_input": "Find education tenders in Egypt",
                    "context": {"conversation_id": "conv-1"},
                })

    opportunities = response["result"]["opportunities"]
    assert len(opportunities) == 1
    assert opportunities[0]["qualification_status"] == "unknown"
    mock_qualification.return_value.invoke.assert_not_called()


def test_qualification_not_qualified_when_requirement_clearly_unmet(stores):
    """Correct assignment of the third qualification state."""
    conv_store, opp_store = stores
    tender = _tender(eligibility=("Must hold a Saudi commercial registration",))

    with patch(
        "agents.rfp_research.tender_tools.query_world_bank_egypt",
        return_value=_success([tender]),
    ):
        with patch(
            "agents.rfp_research.tender_tools.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_relevance_chain"
            ) as mock_relevance, patch(
                "agents.rfp_research.graph.build_qualification_chain"
            ) as mock_qualification:
                mock_relevance.return_value.invoke.return_value = _relevance(
                    is_relevant=True
                )
                mock_qualification.return_value.invoke.return_value = _qualification(
                    qualification_status="not_qualified"
                )
                agent = _make_agent(conv_store, opp_store)

                response = agent.process_request({
                    "request_id": "r1",
                    "user_input": "Find education tenders in Egypt",
                    "context": {"conversation_id": "conv-1"},
                })

    opportunities = response["result"]["opportunities"]
    assert len(opportunities) == 1
    assert opportunities[0]["qualification_status"] == "not_qualified"


def test_opportunities_are_persisted_to_the_searchable_store(stores):
    conv_store, opp_store = stores

    with patch(
        "agents.rfp_research.tender_tools.query_world_bank_egypt",
        return_value=_success([_tender()]),
    ):
        with patch(
            "agents.rfp_research.tender_tools.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_relevance_chain"
            ) as mock_relevance, patch(
                "agents.rfp_research.graph.build_qualification_chain"
            ) as mock_qualification:
                mock_relevance.return_value.invoke.return_value = _relevance()
                mock_qualification.return_value.invoke.return_value = _qualification()
                agent = _make_agent(conv_store, opp_store)

                agent.process_request({
                    "request_id": "r1",
                    "user_input": "Find education tenders in Egypt",
                    "context": {"conversation_id": "conv-1"},
                })

    stored = opp_store.search(country="Egypt")
    assert len(stored) == 1
    assert stored[0]["source_id"] == "OP001"


def test_relevance_assessment_failure_does_not_crash_or_fabricate(stores):
    conv_store, opp_store = stores

    with patch(
        "agents.rfp_research.tender_tools.query_world_bank_egypt",
        return_value=_success([_tender()]),
    ):
        with patch(
            "agents.rfp_research.tender_tools.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_relevance_chain"
            ) as mock_relevance, patch(
                "agents.rfp_research.graph.build_qualification_chain"
            ) as mock_qualification:
                mock_relevance.return_value.invoke.side_effect = RuntimeError(
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
    mock_qualification.return_value.invoke.assert_not_called()


def test_qualification_assessment_failure_does_not_crash_or_fabricate(stores):
    """
    A relevance call that succeeds but a qualification call that
    fails must still keep the (confirmed-relevant) record, with
    qualification_status "unknown" rather than crashing or guessing.
    """
    conv_store, opp_store = stores

    with patch(
        "agents.rfp_research.tender_tools.query_world_bank_egypt",
        return_value=_success([_tender()]),
    ):
        with patch(
            "agents.rfp_research.tender_tools.query_etimad_saudi_arabia"
        ) as mock_etimad:
            mock_etimad.return_value = APIQueryResult(
                source_name="Etimad", status="unavailable", message="No API."
            )
            with patch(
                "agents.rfp_research.graph.build_relevance_chain"
            ) as mock_relevance, patch(
                "agents.rfp_research.graph.build_qualification_chain"
            ) as mock_qualification:
                mock_relevance.return_value.invoke.return_value = _relevance(
                    is_relevant=True
                )
                mock_qualification.return_value.invoke.side_effect = RuntimeError(
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
    assert opportunities[0]["ai_summary"]  # relevance's summary was still kept


def test_empty_user_input_is_rejected(stores):
    conv_store, opp_store = stores
    agent = _make_agent(conv_store, opp_store)

    response = agent.process_request({
        "request_id": "r1", "user_input": "   ", "context": {},
    })

    assert response["status"] == "error"


@pytest.mark.parametrize("text, expected", [
    # Country names hidden inside ordinary words are not countries.
    ("Find training tenders in Egypt for causal inference", None),
    ("Find tenders in Egypt worth a thousand dollars", None),
    ("Find tenders in Egypt for the Duke programme", None),
    ("Egyptian ministry training tenders", None),
    ("مناقصات تدريب في مصر", None),
    # Real out-of-scope countries are still rejected.
    ("Find training tenders in Germany", "germany"),
    ("Tenders in the USA", "usa"),
    ("Tenders in the U.S.A.", "usa"),
    ("EU funded training tenders", "europe"),
    ("European Union training", "europe"),
    ("UK tenders", "uk"),
    ("مناقصات بألمانيا", "germany"),
])
def test_out_of_scope_country_uses_whole_words(text, expected):
    from agents.rfp_research.graph import _detect_out_of_scope_country

    assert _detect_out_of_scope_country(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Egyptian ministry training tenders", ["Egypt"]),
    ("KSA tenders", ["Saudi Arabia"]),
    ("Saudi Arabian e-learning RFPs", ["Saudi Arabia"]),
    ("مناقصات في السعودية", ["Saudi Arabia"]),
    ("training tenders", []),
])
def test_requested_countries(text, expected):
    from agents.rfp_research.graph import _detect_requested_countries

    assert _detect_requested_countries(text) == expected


def test_in_scope_request_with_country_like_substring_reaches_the_api(stores):
    conv_store, opp_store = stores

    with patch(
        "agents.rfp_research.tender_tools.query_world_bank_egypt",
        return_value=_success([]),
    ) as mock_wb:
        agent = _make_agent(conv_store, opp_store)
        response = agent.process_request({
            "request_id": "r1",
            "user_input": "Find training tenders in Egypt for causal inference",
            "context": {"conversation_id": "conv-1"},
        })

    mock_wb.assert_called_once()
    assert "outside the supported scope" not in response["result"]["message"]
