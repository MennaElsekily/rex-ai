"""
The curriculum-approval pause/resume mechanism (agents/pending_workflows.py)
against the REAL Curriculum and Document agents (fake models, temporary
storage) -- not the simplified fixed handlers in
tests/test_pending_workflows.py, which cover the mechanism itself.

This caught two real bugs while it was being built:
1. Document Generation could not identify which curriculum a chained
   request ("create a curriculum ... and generate a document") meant,
   because a curriculum just drafted is never in the approved list its
   own text-matching searches -- it needed to read the curriculum
   Agent 3 had just produced earlier in the same orchestrator plan
   (previous_results), the same way the Marketing agent already could.
2. Resuming after approval re-submitted the ORIGINAL free text, which
   could not tell two generically-titled curricula apart and asked a
   clarifying question instead of finishing the document -- resuming
   now replays the exact document/marketing request the agent had
   already resolved before it paused.
"""

import pytest

from agents.admin_service import AdminService
from agents.curriculum_research.curriculum_agent import CurriculumAgent
from agents.document_generation.document_agent import DocumentAgent
from agents.document_generation.document_store import DocumentStore
from agents.document_generation.document_tools import DocumentToolbox
from agents.orchestrator import Orchestrator
from tests.curriculum_fakes import FakeLLM, bilingual_answers, in_process_tools
from tests.document_fakes import fake_llm as document_fake_llm

PASSCODE = "pass-123"
REQUEST_TEXT = "Create a beginner curriculum for data analysis, 40 hours, and generate a course description document."


@pytest.fixture
def curriculum_tools(tmp_path):
    tools = in_process_tools(tmp_path / "curriculum")
    yield tools
    tools.close()


@pytest.fixture
def document_tools(tmp_path, curriculum_tools):
    tools = DocumentToolbox(store=DocumentStore(tmp_path / "documents.sqlite3"), curricula=curriculum_tools.store,
                            documents_dir=tmp_path / "pdfs")
    yield tools
    tools.store.close()


@pytest.fixture
def orchestrator(curriculum_tools, document_tools):
    router = Orchestrator()
    curriculum_agent = CurriculumAgent(tools=curriculum_tools, llm=FakeLLM(bilingual_answers("en")))
    router.register_agent("curriculum_research", curriculum_agent.process_request)
    document_agent = DocumentAgent(tools=document_tools, llm=document_fake_llm("en"))
    router.register_agent("document_generation", document_agent.process_request)
    return router


@pytest.fixture
def admin(orchestrator, curriculum_tools, document_tools):
    return AdminService(orchestrator_factory=lambda: orchestrator, passcode=PASSCODE,
                        curriculum_store=curriculum_tools.store, document_store=document_tools.store)


@pytest.fixture
def token(admin):
    return admin.login(PASSCODE)


def test_the_chained_request_identifies_the_just_drafted_curriculum(admin, token):
    """Bug 1: document_generation used to say "no curriculum is approved yet" without naming one at all."""
    response = admin.ask(token, REQUEST_TEXT, context={"conversation_id": "c1"})

    assert response["status"] == "success"
    curriculum_result, document_result = response["result"]
    assert curriculum_result["agent_name"] == "curriculum_research" and curriculum_result["status"] == "success"
    assert document_result["agent_name"] == "document_generation" and document_result["status"] == "paused"
    # It named the exact curriculum it is waiting on, not a generic "nothing is approved" message.
    assert curriculum_result["result"]["curriculum_id"] in curriculum_result["result"]["message"]


def test_approving_resumes_with_the_exact_document_already_resolved(admin, token, curriculum_tools):
    """Bug 2: resuming used to re-submit the free text and ask the admin to pick a curriculum all over again."""
    response = admin.ask(token, REQUEST_TEXT, context={"conversation_id": "c1"})
    curriculum_id = response["result"][0]["result"]["curriculum_id"]

    admin.approve_curriculum(token, curriculum_id, 1)

    update = admin.pending_chat_update_for(token, "c1")
    assert update is not None and update["status"] == "success"
    document_result = update["result"][0]
    assert document_result["agent_name"] == "document_generation" and document_result["status"] == "success"
    assert document_result["result"]["kind"] == "document"
    assert "clarify" not in str(document_result["result"].get("kind"))
    # The document really was generated and saved against the approved curriculum.
    saved_id = document_result["result"]["document_id"]
    assert saved_id and saved_id.startswith("GDOC-")


def test_without_the_chain_a_bare_document_request_still_asks_which_curriculum(admin, token, document_tools):
    """The previous_results fallback must never fire outside a real chained plan."""
    from tests.document_fakes import save_curriculum

    save_curriculum(document_tools)   # an unrelated draft, not mentioned by this request and never approved
    response = admin.ask(token, "Generate a course description document", context={"conversation_id": "c2"})

    assert response["status"] == "error"
    # Not resolved to that draft's id: with an empty previous_results (no chained curriculum step ran),
    # this is unchanged from the original behaviour -- nothing is approved, so nothing can be used.
    assert response["result"][0]["result"]["kind"] == "not_approved"
