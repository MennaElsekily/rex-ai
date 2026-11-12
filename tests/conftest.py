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
def release_chroma_indexes():
    """
    Chroma keeps one open "system" (its SQLite and index files) per index
    path for the life of the process. Tests build many throw-away indexes
    in their own temp folders, so without this the suite slowly runs out of
    file descriptors (Windows' C runtime allows 512) and an unrelated later
    test fails with "Too many open files".
    """
    yield
    from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()


@pytest.fixture(autouse=True)
def temporary_knowledge_base_index(tmp_path, monkeypatch):
    """
    A store opened without an explicit path (VectorStore(), the RAG MCP
    server, Agent 3's AcademyCatalogue) gets an empty temporary index,
    never knowledge_base/vector_db. Tests that need a catalogue build one.
    """
    monkeypatch.setenv("REX_VECTOR_DB_DIR", str(tmp_path / "default_vector_db"))


@pytest.fixture(autouse=True)
def no_real_email(monkeypatch):
    """
    No test can send a real email: without a Resend API key every send is
    refused before any network call. (Tests that exercise the provider pass
    their own key and replace the network call.)
    """
    monkeypatch.setenv("RESEND_API_KEY", "")


@pytest.fixture(autouse=True)
def temporary_conversation_and_lead_storage(tmp_path, monkeypatch):
    """Conversation state and learner leads (real user data) are never written by tests."""
    from agents import conversation_store
    from agents.rex_assistant.lead_store import JSONLLeadStore

    monkeypatch.setattr(conversation_store, "DEFAULT_DB_PATH", tmp_path / "default_conversations.sqlite3")
    monkeypatch.setattr(JSONLLeadStore.__init__, "__defaults__", (str(tmp_path / "default_leads.jsonl"),))


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
def temporary_document_storage(tmp_path, monkeypatch):
    """Agent 4's database and PDF folder default to temporary paths in tests."""
    from agents.document_generation import document_store, document_tools

    monkeypatch.setattr(document_store, "DEFAULT_DB_PATH", tmp_path / "default_documents.sqlite3")
    monkeypatch.setattr(document_tools, "DEFAULT_DOCUMENTS_DIR", tmp_path / "default_document_pdfs")


@pytest.fixture(autouse=True)
def in_process_document_agent(tmp_path, monkeypatch):
    """
    Agent 4 by default uses the in-process toolbox on temporary storage and a
    model that fails if called: tests that generate pass a fake LLM
    (tests/document_fakes.py). tests/test_document_mcp_server.py covers the
    real server.
    """
    from agents.document_generation import document_agent
    from tests.curriculum_fakes import NoModel
    from tests.document_fakes import in_process_document_tools

    monkeypatch.setattr(document_agent, "default_document_tools",
                        lambda: in_process_document_tools(tmp_path / "document_agent"))
    monkeypatch.setattr(document_agent, "default_llm", lambda model=None: NoModel())


@pytest.fixture(autouse=True)
def temporary_marketing_storage(tmp_path, monkeypatch):
    """Agent 5's database defaults to a temporary path in tests."""
    from agents.marketing import marketing_store

    monkeypatch.setattr(marketing_store, "DEFAULT_DB_PATH", tmp_path / "default_marketing.sqlite3")


@pytest.fixture(autouse=True)
def in_process_marketing_agent(tmp_path, monkeypatch):
    """
    Agent 5 by default uses the in-process toolbox on temporary storage and a model that fails if called: tests that
    generate pass a fake LLM (tests/marketing_fakes.py). tests/test_marketing_mcp_server.py covers the real server.
    """
    from agents.marketing import marketing_agent
    from tests.curriculum_fakes import NoModel
    from tests.marketing_fakes import in_process_marketing_tools

    monkeypatch.setattr(marketing_agent, "default_marketing_tools",
                        lambda: in_process_marketing_tools(tmp_path / "marketing_agent"))
    monkeypatch.setattr(marketing_agent, "default_llm", lambda model=None: NoModel())


@pytest.fixture(autouse=True)
def temporary_tender_database(tmp_path, monkeypatch):
    monkeypatch.setattr(opportunity_store, "DEFAULT_DB_PATH", tmp_path / "default_tenders.sqlite3")
    # The academy's uploaded documents, likewise.
    monkeypatch.setattr(application_profile, "DEFAULT_DOCUMENTS_DIR", tmp_path / "application_documents")


@pytest.fixture(autouse=True)
def temporary_pending_workflow_store(tmp_path, monkeypatch):
    """A paused-then-resumed Ask Agents request (agents/pending_workflows.py) never touches the real database."""
    from agents import pending_workflows

    monkeypatch.setattr(pending_workflows, "DEFAULT_DB_PATH", tmp_path / "default_pending_workflows.sqlite3")


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
