"""
agents/tender_mcp_server.py through a real MCP subprocess
(TenderMcpClient): real stdio transport, real tool schemas, a temporary
tender database (REX_TENDER_DB_PATH) and an empty Tavily key, so no call
reaches the network.
"""

from pathlib import Path

import pytest

from agents.rfp_research.mcp_client import TenderMcpClient
from agents.rfp_research.opportunity_store import OpportunityStore

TOOLS = {"search_procurement_notices", "search_tender_web_pages",
         "find_known_tenders", "save_discovered_tenders"}
REAL_DB = Path(__file__).resolve().parents[1] / "agents" / "rfp_research" / "rfp_opportunities.sqlite3"


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("tender_server") / "tenders.sqlite3"


@pytest.fixture(scope="module")
def client(db_path):
    client = TenderMcpClient(env={"REX_TENDER_DB_PATH": str(db_path), "TAVILY_API_KEY": ""},
                             startup_timeout=60)
    yield client
    client.close()


def test_exactly_the_four_tender_tools_with_schemas(client):
    tools = {tool["name"]: tool for tool in client.list_tools()}

    assert set(tools) == TOOLS  # no approve / reject / submit tool
    for tool in tools.values():
        assert tool["description"] and tool["input_schema"]["type"] == "object"
    assert tools["search_procurement_notices"]["input_schema"]["required"] == ["country", "query"]
    assert tools["save_discovered_tenders"]["input_schema"]["required"] == ["tenders"]


def test_saved_tenders_are_pending_in_the_configured_database(client, db_path):
    tender = {"url": "https://etenders.gov.eg/t/1", "title": "Teacher Training Services Tender",
              "country": "Egypt", "approval_status": "approved"}

    saved = client.save_discovered_tenders([tender])
    found = client.find_known_tenders([tender])

    assert (saved["status"], saved["new"]) == ("ok", 1)
    assert found["known"] == [{"index": 0, "existing_url": "https://etenders.gov.eg/t/1"}]
    rows = OpportunityStore(db_path=db_path).search()
    assert [(r["url"], r["approval_status"]) for r in rows] == [("https://etenders.gov.eg/t/1", "pending")]


def test_saudi_procurement_source_is_reported_unavailable(client):
    result = client.search_procurement_notices("Saudi Arabia", "training")

    assert result["status"] == "unavailable"


def test_errors_come_back_as_results(client):
    assert client.search_procurement_notices("Germany", "training")["status"] == "invalid_request"
    assert client.find_known_tenders([])["status"] == "ok"


def test_arguments_that_break_the_schema_raise_a_clear_error(client):
    from agents.mcp_tool_client import McpToolError

    with pytest.raises(McpToolError, match="reported an error"):
        client.find_known_tenders(["not a tender object"])
    # The session survives: the next call works.
    assert client.find_known_tenders([])["status"] == "ok"


def test_web_search_without_a_key_is_reported_not_hidden(client):
    result = client.search_tender_web_pages(["Egypt"])

    assert result["status"] == "awaiting_credentials" and result["pages"] == []
    assert {s["status"] for s in result["sources"]} == {"awaiting_credentials"}


def test_the_real_tender_database_is_not_used(client, db_path):
    assert db_path != REAL_DB and db_path.exists()


def test_the_rfp_agent_runs_end_to_end_through_the_server(client, db_path):
    """The real graph over the real server: no network (Etimad stub, no web key)."""
    from agents.rfp_research import RFPResearchAgent

    agent = RFPResearchAgent(opportunity_store=OpportunityStore(db_path=db_path), model="test", tools=client)
    result = agent.process_request({"request_id": "r", "user_input": "Find training tenders in Saudi Arabia"})

    assert result["status"] == "success"
    body = result["result"]
    assert [r["status"] for r in body["api_results"]] == ["unavailable"] + ["awaiting_credentials"] * 4
    assert "NOT the same as finding zero" in body["message"]
