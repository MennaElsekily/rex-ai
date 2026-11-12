"""
A chained Ask Agents request ("create a curriculum and a document/
campaign from it") pauses when the curriculum it just created needs
approval first, and automatically resumes once the admin approves it --
through the SAME orchestrator, never a second one (agents/
pending_workflows.py, wired into admin_service.py's ask()/
approve_curriculum()/request_curriculum_changes()).

The curriculum and document agents are kept deliberately simple here
(fixed handlers, not the real LLM-driven graphs) so these tests cover
only the NEW pause/resume mechanism; the real agents' own behaviour is
covered by their own test files.
"""

import pytest

from agents.admin_service import AdminService
from agents.curriculum_research.curriculum_store import CurriculumStore
from agents.orchestrator import Orchestrator
from agents.pending_workflows import PendingWorkflowStore
from tests.curriculum_fixtures import valid_curriculum

PASSCODE = "pass-123"
REQUEST_TEXT = "Create a curriculum for data analysis and generate a professional course description document."


def curriculum_handler(curriculum_id):
    def handler(request):
        return {"status": "success", "result": {
            "kind": "curriculum", "curriculum_id": curriculum_id, "version": 1,
            "review_status": "pending_review", "message": "Curriculum drafted; pending review."}}
    return handler


def document_handler(calls, document_id="DOC-0123456789ab"):
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return {"status": "error", "result": {"kind": "not_approved", "message": "Not approved yet."},
                   "error_message": "Not approved yet."}
        return {"status": "success", "result": {
            "kind": "document", "document_id": document_id, "message": "Course Description generated."}}
    return handler


@pytest.fixture
def curriculum_store(tmp_path):
    store = CurriculumStore(tmp_path / "curricula.sqlite3")
    yield store
    store.close()


@pytest.fixture
def pending_store(tmp_path):
    store = PendingWorkflowStore(tmp_path / "pending.sqlite3")
    yield store
    store.close()


@pytest.fixture
def seeded_curriculum_id(curriculum_store):
    saved = curriculum_store.save_version(valid_curriculum(), {"valid": True, "errors": []}, [])
    return saved["curriculum_id"]


@pytest.fixture
def document_calls():
    return []


@pytest.fixture
def orchestrator(seeded_curriculum_id, document_calls):
    router = Orchestrator()
    router.register_agent("curriculum_research", curriculum_handler(seeded_curriculum_id))
    router.register_agent("document_generation", document_handler(document_calls))
    return router


@pytest.fixture
def admin(orchestrator, curriculum_store, pending_store):
    return AdminService(orchestrator_factory=lambda: orchestrator, passcode=PASSCODE,
                        curriculum_store=curriculum_store, pending_workflow_store=pending_store)


@pytest.fixture
def token(admin):
    return admin.login(PASSCODE)


# --------------------------------------------------------------------
# Pausing
# --------------------------------------------------------------------

def test_a_chained_request_pauses_instead_of_just_reporting_the_failure(admin, token, seeded_curriculum_id):
    response = admin.ask(token, REQUEST_TEXT, context={"conversation_id": "c1"})

    assert response["status"] == "success"   # not reported as a failure
    results = response["result"]
    assert results[0]["agent_name"] == "curriculum_research" and results[0]["status"] == "success"
    assert results[1]["agent_name"] == "document_generation" and results[1]["status"] == "paused"
    assert seeded_curriculum_id in results[0]["result"]["message"]
    assert "ready for your review" in results[0]["result"]["message"]


def test_nothing_is_paused_without_a_conversation_id(admin, token, pending_store, seeded_curriculum_id):
    admin.ask(token, REQUEST_TEXT, context={})   # no conversation_id: cannot be resumed anyway

    assert pending_store.pending_for_curriculum(seeded_curriculum_id) is None


def test_an_ordinary_single_agent_request_is_never_paused(admin, token):
    response = admin.ask(token, "Find education tenders in Egypt", context={"conversation_id": "c1"})
    assert response["status"] == "error"   # document_generation/curriculum_research are not even registered
    assert "paused" not in str(response).lower() and "awaiting_approval" not in str(response)


# --------------------------------------------------------------------
# Resuming on approval
# --------------------------------------------------------------------

def test_approving_the_curriculum_resumes_through_the_same_orchestrator(
    admin, token, seeded_curriculum_id, document_calls, pending_store,
):
    admin.ask(token, REQUEST_TEXT, context={"conversation_id": "c1"})
    assert pending_store.pending_for_curriculum(seeded_curriculum_id) is not None

    admin.approve_curriculum(token, seeded_curriculum_id, 1)

    assert pending_store.pending_for_curriculum(seeded_curriculum_id) is None   # consumed
    assert len(document_calls) == 2   # the first (paused) attempt, then the resumed one
    update = admin.pending_chat_update_for(token, "c1")
    assert update is not None and update["status"] == "success"
    assert update["result"][0]["result"]["kind"] == "document"


def test_the_curriculum_step_is_not_repeated_on_resume(admin, token, seeded_curriculum_id, orchestrator):
    calls = []
    real_handler = orchestrator.agents["curriculum_research"]
    orchestrator.register_agent("curriculum_research", lambda r: (calls.append(r), real_handler(r))[1])

    admin.ask(token, REQUEST_TEXT, context={"conversation_id": "c1"})
    admin.approve_curriculum(token, seeded_curriculum_id, 1)

    assert len(calls) == 1   # curriculum_research ran only for the first (paused) attempt, never again


def test_resuming_without_a_pending_workflow_does_nothing(admin, token, seeded_curriculum_id):
    admin.approve_curriculum(token, seeded_curriculum_id, 1)   # nothing was ever paused for it

    assert admin.pending_chat_update_for(token, "c1") is None


def test_pending_chat_update_is_consumed_only_once(admin, token, seeded_curriculum_id):
    admin.ask(token, REQUEST_TEXT, context={"conversation_id": "c1"})
    admin.approve_curriculum(token, seeded_curriculum_id, 1)

    first = admin.pending_chat_update_for(token, "c1")
    second = admin.pending_chat_update_for(token, "c1")

    assert first is not None and second is None


def test_a_different_conversation_never_sees_someone_elses_resumed_result(admin, token, seeded_curriculum_id):
    admin.ask(token, REQUEST_TEXT, context={"conversation_id": "c1"})
    admin.approve_curriculum(token, seeded_curriculum_id, 1)

    assert admin.pending_chat_update_for(token, "c2") is None
    assert admin.pending_chat_update_for(token, "c1") is not None


# --------------------------------------------------------------------
# Rejecting instead of approving
# --------------------------------------------------------------------

def test_requesting_changes_cancels_the_pending_workflow_instead_of_resuming(
    admin, token, seeded_curriculum_id, document_calls, pending_store,
):
    admin.ask(token, REQUEST_TEXT, context={"conversation_id": "c1"})

    admin.request_curriculum_changes(token, seeded_curriculum_id, 1, "Please add more detail.")

    assert pending_store.pending_for_curriculum(seeded_curriculum_id) is None
    assert len(document_calls) == 1   # never resumed: document_generation was not asked again
    update = admin.pending_chat_update_for(token, "c1")
    assert update is not None and update["status"] == "error"
    assert "cancelled" in update["error_message"].lower()


# --------------------------------------------------------------------
# Multilingual pause note
# --------------------------------------------------------------------

def test_the_pause_note_is_in_arabic_when_the_conversation_is(admin, token):
    response = admin.ask(token, REQUEST_TEXT, context={"conversation_id": "c1", "preferred_language": "ar"})

    assert "جاهز لمراجعتك" in response["result"][0]["result"]["message"]
