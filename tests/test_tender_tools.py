"""
The RFP agent's tender tools (agents/rfp_research/tender_tools.py), called
in-process. Mocked: the procurement connectors and the web search
(conftest.py keeps the web search offline). Temp SQLite files.
"""

import pytest

from agents.rfp_research import tender_tools, web_tenders
from agents.rfp_research.api_clients.base import APIQueryResult
from agents.rfp_research.api_clients.tavily import WebPage, WebSearchResult
from agents.rfp_research.opportunity_store import OpportunityStore


@pytest.fixture
def store(tmp_path):
    s = OpportunityStore(db_path=tmp_path / "tenders.sqlite3")
    yield s
    s.close()


def tender(url="https://etenders.gov.eg/t/1", title="Teacher Training Services Tender", **extra):
    return {"url": url, "title": title, "country": "Egypt", "organization": "Ministry of Education", **extra}


# --------------------------------------------------------------------
# search_procurement_notices
# --------------------------------------------------------------------

def test_egypt_uses_the_world_bank_connector(monkeypatch):
    queries = []

    def world_bank(query):
        queries.append(query)
        return APIQueryResult(source_name="World Bank", status="success", message="1 record", records=[])

    monkeypatch.setattr(tender_tools, "query_world_bank_egypt", world_bank)

    result = tender_tools.search_procurement_notices("Egypt", "training")

    assert queries == ["training"]
    assert result["status"] == "success" and result["source_name"] == "World Bank"


def test_saudi_arabia_reports_the_etimad_connector_as_unavailable():
    result = tender_tools.search_procurement_notices("Saudi Arabia", "training")

    assert result["status"] == "unavailable" and result["records"] == []


@pytest.mark.parametrize("country, query", [("Germany", "training"), ("Egypt", "   ")])
def test_invalid_procurement_requests(country, query):
    assert tender_tools.search_procurement_notices(country, query)["status"] == "invalid_request"


def test_a_connector_crash_is_an_error_result(monkeypatch):
    def broken(query):
        raise RuntimeError("boom")

    monkeypatch.setattr(tender_tools, "query_world_bank_egypt", broken)

    result = tender_tools.search_procurement_notices("Egypt", "training")

    assert result["status"] == "error" and "RuntimeError" in result["message"]


# --------------------------------------------------------------------
# search_tender_web_pages
# --------------------------------------------------------------------

def test_web_pages_carry_their_country_and_search_label(monkeypatch):
    page = WebPage(title="Teacher training tender", url="https://etenders.gov.eg/t/1",
                   content="Training.", source_domain="etenders.gov.eg")
    monkeypatch.setattr(web_tenders, "search_tavily", lambda query, **_: WebSearchResult(
        status="success", message="ok", query=query, pages=[page]))

    result = tender_tools.search_tender_web_pages(["Egypt"], ["Technical Training"], "Ministry")

    assert result["status"] == "success"
    assert [(p["url"], p["country"], p["label"]) for p in result["pages"]] == [
        ("https://etenders.gov.eg/t/1", "Egypt", "official portal")]
    assert len(result["sources"]) == 4


def test_unconfigured_web_search_is_reported():
    # conftest.py: every web search reports awaiting_credentials.
    result = tender_tools.search_tender_web_pages(["Saudi Arabia"])

    assert result["status"] == "awaiting_credentials" and result["pages"] == []


@pytest.mark.parametrize("countries, categories", [
    ([], None), (["Kenya"], None), (["Egypt"], ["Cooking"]),
])
def test_invalid_web_requests(countries, categories):
    assert tender_tools.search_tender_web_pages(countries, categories)["status"] == "invalid_request"


# --------------------------------------------------------------------
# find_known_tenders / save_discovered_tenders
# --------------------------------------------------------------------

def test_save_then_find(store):
    saved = tender_tools.save_discovered_tenders(store, [tender(), tender("https://b.example/2", "Corporate Learning Platform Tender")])
    again = tender_tools.save_discovered_tenders(store, [tender()])
    found = tender_tools.find_known_tenders(store, [
        tender(), tender("https://tendersinfo.com/x", "Teacher Training Services Tender"),
        tender("https://c.example/3", "Digital Skills Programme Tender"),
    ])

    assert (saved["new"], saved["known"]) == (2, 0)
    assert (again["new"], again["known"]) == (0, 1)
    assert [k["index"] for k in found["known"]] == [0, 1]  # the second one: same tender, other site


def test_saving_can_never_approve_or_change_a_decision(store):
    tender_tools.save_discovered_tenders(store, [tender(approval_status="approved", decided_by="web page")])
    row = store.search()[0]
    assert row["approval_status"] == "pending" and row["decided_by"] is None

    store.decide(row["opportunity_id"], "rejected", "admin-session-1")
    tender_tools.save_discovered_tenders(store, [tender(approval_status="approved")])

    assert store.get(row["opportunity_id"])["approval_status"] == "rejected"


def test_a_tender_without_an_identity_is_reported_not_saved(store):
    result = tender_tools.save_discovered_tenders(store, [{"title": "No link at all"}])

    assert result["new"] == 0 and result["errors"][0]["index"] == 0
    assert store.search() == []


@pytest.mark.parametrize("tenders", ["not a list", [1, 2], [{}] * (tender_tools.MAX_TENDERS_PER_CALL + 1)])
def test_invalid_tender_lists(store, tenders):
    assert tender_tools.save_discovered_tenders(store, tenders)["status"] == "invalid_request"
    assert tender_tools.find_known_tenders(store, tenders)["status"] == "invalid_request"
