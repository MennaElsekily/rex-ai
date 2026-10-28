"""
Shared test setup.

RFP Research searches the web on every request (web_tenders.py). Tests
never reach the real Tavily API: by default every web search reports
"awaiting_credentials" with no pages. Tests that exercise web research
replace agents.rfp_research.web_tenders.search_tavily themselves.

Tests never open the real tender database: an OpportunityStore created
without a path uses a temporary file instead of
agents/rfp_research/rfp_opportunities.sqlite3.

The RFP agent normally reaches its tender tools through the tender MCP
server (a subprocess, which these patches could not reach). In tests it
uses the same tool functions in-process (tender_tools.LocalTenderTools);
tests/test_tender_mcp_server.py covers the real server.
"""

import pytest

from agents.rfp_research import agent as rfp_agent_module
from agents.rfp_research import application_profile, opportunity_store, web_tenders
from agents.rfp_research.api_clients.tavily import WebSearchResult
from agents.rfp_research.tender_tools import LocalTenderTools


@pytest.fixture(autouse=True)
def temporary_curriculum_storage(tmp_path, monkeypatch):
    """Agent 3's store and PDF folder default to temporary paths in tests."""
    from agents.curriculum_research import curriculum_store, curriculum_tools

    monkeypatch.setattr(curriculum_store, "DEFAULT_DB_PATH", tmp_path / "default_curricula.sqlite3")
    monkeypatch.setattr(curriculum_tools, "DEFAULT_DOCUMENTS_DIR", tmp_path / "curriculum_documents")
    monkeypatch.delenv("CURRICULUM_RESEARCH_DOMAINS", raising=False)


@pytest.fixture(autouse=True)
def in_process_curriculum_agent(tmp_path, monkeypatch):
    """
    Agent 3 by default uses the in-process toolbox on a temporary index,
    database and folders (never the real index), and a model that fails
    if called: tests that generate pass a FakeLLM (tests/curriculum_fakes.py).
    tests/test_curriculum_mcp_server.py covers the real server.
    """
    from agents.curriculum_research import curriculum_agent
    from tests.curriculum_fakes import NoModel, in_process_tools

    monkeypatch.setattr(curriculum_agent, "default_curriculum_tools",
                        lambda: in_process_tools(tmp_path / "curriculum_agent"))
    monkeypatch.setattr(curriculum_agent, "default_llm", lambda model=None: NoModel())


@pytest.fixture(autouse=True)
def temporary_tender_database(tmp_path, monkeypatch):
    monkeypatch.setattr(opportunity_store, "DEFAULT_DB_PATH", tmp_path / "default_tenders.sqlite3")
    # The academy's uploaded documents, likewise.
    monkeypatch.setattr(application_profile, "DEFAULT_DOCUMENTS_DIR", tmp_path / "application_documents")


# Saved before any test patches it, so a test can check the real default
# (it only builds the client; the server starts on the first tool call).
_REAL_DEFAULT_TENDER_TOOLS = rfp_agent_module.default_tender_tools


@pytest.fixture(autouse=True)
def in_process_tender_tools(monkeypatch):
    monkeypatch.setattr(rfp_agent_module, "default_tender_tools", LocalTenderTools)


@pytest.fixture
def real_default_tender_tools():
    return _REAL_DEFAULT_TENDER_TOOLS


def _offline_search(query, **_):
    return WebSearchResult(status="awaiting_credentials",
                           message="Web search is disabled in tests.", query=query)


@pytest.fixture(autouse=True)
def offline_web_search(monkeypatch):
    monkeypatch.setattr(web_tenders, "search_tavily", _offline_search)
