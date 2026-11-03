"""
Agent 4 inside the system: orchestrator routing and registration, the admin
session rules (generate, edit, approve, download), and the whole workflow

    approved curriculum -> Document agent -> generate -> validate -> PDF
    -> review -> edit -> approve -> download

with the real orchestrator, agent, tools, stores and PDF renderer on
temporary storage and a fake language model.
"""

from unittest.mock import MagicMock

import pytest

from agents.admin_service import AdminAuthError, AdminService
from agents.conversation_store import ConversationStore
from agents.document_generation.document_agent import DocumentAgent
from agents.document_generation.document_store import DocumentStoreError
from agents.document_generation.registration import register_document_generation
from agents.orchestrator import Orchestrator
from agents.register_agents import create_admin_orchestrator
from tests.curriculum_fixtures import valid_curriculum
from tests.document_fakes import approved_curriculum, fake_llm, in_process_document_tools, save_curriculum

PASSCODE = "pass-123"


@pytest.fixture
def system(tmp_path):
    toolbox = in_process_document_tools(tmp_path / "system")
    orchestrator = Orchestrator(excluded_agents=("rex_assistant",))
    agent = register_document_generation(orchestrator, ConversationStore(db_path=":memory:"))
    agent._tools, agent._llm = toolbox, fake_llm()
    admin = AdminService(orchestrator_factory=lambda: orchestrator, passcode=PASSCODE,
                         curriculum_store=toolbox.curricula, curriculum_documents_dir=str(tmp_path / "curriculum_pdfs"),
                         document_store=toolbox.store, document_output_dir=str(toolbox.documents_dir))

    class System:
        pass

    result = System()
    result.toolbox, result.orchestrator, result.agent, result.admin = toolbox, orchestrator, agent, admin
    result.token = admin.login(PASSCODE)
    return result


def generate(system, curriculum_id, doc_type="course_description", **kwargs):
    response = system.admin.generate_document(system.token, curriculum_id, doc_type, **kwargs)
    return response, response["result"][0]


def editable_blocks(content):
    return [(s["key"], i) for s in content["sections"] for i, b in enumerate(s["blocks"])
            if b["source"] in ("generated", "admin") and b["type"] in ("paragraph", "bullets")]


# --------------------------------------------------------------------
# Registration and routing
# --------------------------------------------------------------------

def test_the_agent_is_registered_with_the_admin_orchestrator(tmp_path):
    rfp = MagicMock()
    rfp.AGENT_NAME = "rfp_research"
    orchestrator = create_admin_orchestrator(
        rfp_research_agent=rfp, conversation_store=ConversationStore(db_path=tmp_path / "c.sqlite3"))

    agent = orchestrator.agents["document_generation"].__self__
    assert isinstance(agent, DocumentAgent) and orchestrator.document_generation_agent is agent
    assert "document_generation" in orchestrator._continuation_checks
    assert agent._tools is None and agent._llm is None   # registering starts no server and loads no model


@pytest.mark.parametrize("text", [
    "Generate a course description for curriculum 123.",
    "Create a participant guide for the approved SQL curriculum.",
    "Generate a course proposal using the selected curriculum.",
    "Generate a course participant guide",
    "Create a course question bank",
    "Write the capstone project brief for CUR-0123456789ab",
    "أنشئ دليل المتدرب لمنهج Python",
])
def test_document_requests_route_to_agent_4_even_when_they_name_a_curriculum(text):
    assert Orchestrator()._make_plan(text) == ["document_generation"]


@pytest.mark.parametrize("text, expected", [
    ("Create a curriculum about SQL for business analysts, 24 hours", ["curriculum_research"]),
    ("Make a PDF of the curriculum", ["curriculum_research"]),
    ("Find education tenders in Egypt", ["rfp_research"]),
    ("Recommend a course for beginners", ["rex_assistant"]),
    ("Create a curriculum and a marketing campaign", ["curriculum_research", "marketing"]),
    ("Write a marketing campaign", ["marketing"]),
    ("Suggest new courses for our data track", ["curriculum_research"]),
])
def test_other_routes_keep_their_requests(text, expected):
    assert Orchestrator()._make_plan(text) == expected


def test_a_form_request_goes_to_agent_4_whatever_the_text_says(system):
    toolbox = system.toolbox
    cid = approved_curriculum(toolbox)

    response = system.orchestrator.handle_request(
        "Find education tenders in Egypt", {"conversation_id": "a", "document_request": {
            "curriculum_id": cid, "document_type": "course_description"}})

    assert response["status"] == "success" and response["result"][0]["agent_name"] == "document_generation"


def test_a_question_about_the_curriculum_continues_through_the_orchestrator(system):
    cid = approved_curriculum(system.toolbox)
    first = system.orchestrator.handle_request("Generate a participant guide", {"conversation_id": "z"})
    assert first["result"][0]["result"]["kind"] == "clarify"

    second = system.orchestrator.handle_request(cid, {"conversation_id": "z"})   # no keyword, only the open question

    assert second["status"] == "success" and second["result"][0]["result"]["document_type"] == "participant_guide"


# --------------------------------------------------------------------
# Admin session rules
# --------------------------------------------------------------------

@pytest.mark.parametrize("method, args", [
    ("list_approved_curricula", ()), ("generate_document", ("CUR-0123456789ab", "course_description")),
    ("list_documents", ()), ("document_counts", ()), ("get_document", ("GDOC-0123456789ab",)),
    ("edit_document", ("GDOC-0123456789ab", [])), ("approve_document", ("GDOC-0123456789ab", 1)),
    ("document_pdf", ("GDOC-0123456789ab",)), ("document_audit_trail", ()),
])
def test_every_document_action_needs_an_admin_session(system, method, args):
    cid = approved_curriculum(system.toolbox)
    for bad in (None, "", "forged-token"):
        with pytest.raises(AdminAuthError):
            getattr(system.admin, method)(bad, *args)
    assert system.toolbox.store.list() == []
    assert system.toolbox.curricula.get(cid)["approved_version"] == 1


def test_without_a_passcode_documents_are_unavailable(system, tmp_path):
    admin = AdminService(orchestrator_factory=lambda: None, passcode="", document_store=system.toolbox.store)
    with pytest.raises(AdminAuthError):
        admin.list_documents("anything")


# --------------------------------------------------------------------
# The whole workflow
# --------------------------------------------------------------------

def test_end_to_end_generate_review_edit_approve_download(system):
    admin, token = system.admin, system.token
    cid = approved_curriculum(system.toolbox)
    assert [c["curriculum_id"] for c in admin.list_approved_curricula(token)["curricula"]] == [cid]

    # 1. select curriculum + type, generate
    response, result = generate(system, cid, "participant_guide")
    assert response["status"] == "success" and result["status"] == "success", result
    document_id = result["result"]["document_id"]
    assert admin.document_counts(token) == {"draft": 0, "pending_review": 1, "approved": 0}

    # 2. review
    detail = admin.get_document(token, document_id)
    document = detail["document"]
    assert document["version_status"] == "pending_review" and document["filename"]
    assert detail["curriculum_changed"] is False and detail["audit"][0]["event"] == "version_saved"
    assert admin.document_pdf(token, document_id, 1)["data"].startswith(b"%PDF")

    # 3. edit text the model wrote -> a new version that needs review again
    section, index = editable_blocks(document["content"])[0]
    saved = admin.edit_document(token, document_id, [{"section": section, "block": index, "value": "Welcome, everyone!"}])
    assert (saved["version"], saved["review_status"]) == (2, "pending_review")
    edited = admin.get_document(token, document_id)["document"]
    block = next(s for s in edited["content"]["sections"] if s["key"] == section)["blocks"][index]
    assert block["text"] == "Welcome, everyone!" and block["source"] == "admin"
    assert edited["filename"] and edited["filename"].endswith("_v2_en.pdf")

    # 4. approve (admin only) and download
    approved = admin.approve_document(token, document_id, 2, "Looks good")
    assert approved["version_status"] == "approved" and approved["approved_version"] == 2
    pdf = admin.document_pdf(token, document_id)
    assert pdf["filename"] == f"{document_id}_v2_en_approved.pdf" and pdf["approved"] is True
    assert pdf["data"].startswith(b"%PDF")
    trail = admin.document_audit_trail(token, document_id)
    assert [(a["event"], a["actor"].startswith("admin-session-")) for a in trail if a["event"] == "review"] == [("review", True)]
    assert admin.document_counts(token)["approved"] == 1

    # 5. the older version was kept and cannot be approved any more
    assert admin.get_document(token, document_id, 1)["document"]["version_status"] == "pending_review"
    with pytest.raises(DocumentStoreError):
        admin.approve_document(token, document_id, 1)


def test_the_admin_cannot_edit_text_taken_from_the_curriculum(system):
    cid = approved_curriculum(system.toolbox)
    document_id = generate(system, cid)[1]["result"]["document_id"]
    content = system.admin.get_document(system.token, document_id)["document"]["content"]
    locked = next((s["key"], i) for s in content["sections"] for i, b in enumerate(s["blocks"])
                  if b["source"] == "curriculum" and b["type"] in ("paragraph", "bullets"))

    with pytest.raises(ValueError, match="approved curriculum cannot be edited"):
        system.admin.edit_document(system.token, document_id, [{"section": locked[0], "block": locked[1], "value": "Changed"}])
    with pytest.raises(ValueError, match="does not exist"):
        system.admin.edit_document(system.token, document_id, [{"section": "nope", "block": 0, "value": "x"}])
    assert system.toolbox.store.get(document_id)["current_version"] == 1


def test_an_edit_that_breaks_the_rules_stays_a_draft_and_cannot_be_approved(system):
    cid = approved_curriculum(system.toolbox)
    document_id = generate(system, cid)[1]["result"]["document_id"]
    section, index = editable_blocks(system.admin.get_document(system.token, document_id)["document"]["content"])[0]

    saved = system.admin.edit_document(system.token, document_id, [
        {"section": section, "block": index, "value": "This course is accredited by the ministry."}])

    assert saved["review_status"] == "draft" and saved["validation"]["errors"][0]["code"] == "accreditation_claim"
    with pytest.raises(DocumentStoreError, match="draft"):
        system.admin.approve_document(system.token, document_id, saved["version"])
    assert system.toolbox.store.get(document_id)["approved_version"] is None


def test_an_approved_document_is_untouched_when_the_curriculum_is_revised(system):
    cid = approved_curriculum(system.toolbox)
    document_id = generate(system, cid)[1]["result"]["document_id"]
    system.admin.approve_document(system.token, document_id, 1)
    before = system.admin.document_pdf(system.token, document_id, 1)["data"]

    revised = valid_curriculum()
    revised["title"] = "Machine Learning Foundations v2"
    saved = system.toolbox.curricula.save_version(revised, {"valid": True, "errors": []}, [], curriculum_id=cid)
    system.toolbox.curricula.approve(cid, saved["version"], "admin-session-other")

    detail = system.admin.get_document(system.token, document_id)
    assert detail["curriculum_changed"] is True   # flagged, not silently rewritten
    assert detail["document"]["content"]["curriculum_version"] == 1
    assert system.admin.document_pdf(system.token, document_id, 1)["data"] == before


def test_regenerating_through_the_admin_creates_a_new_version(system):
    cid = approved_curriculum(system.toolbox)
    first = generate(system, cid)[1]["result"]
    second = generate(system, cid, document_id=first["document_id"])[1]["result"]

    assert second["document_id"] == first["document_id"] and second["version"] == 2
    assert [v["version"] for v in system.admin.get_document(system.token, first["document_id"])["versions"]] == [1, 2]


def test_generation_refuses_unapproved_curricula_through_the_admin(system):
    pending = save_curriculum(system.toolbox)["curriculum_id"]

    response, result = generate(system, pending)

    assert response["status"] == "error" and result["status"] == "error"
    assert "needs to be approved" in response["error_message"]
    assert system.admin.list_documents(system.token) == []
    assert system.admin.list_approved_curricula(system.token)["curricula"] == []


def test_document_lists_filter_and_reject_unknown_statuses(system):
    cid = approved_curriculum(system.toolbox)
    document_id = generate(system, cid)[1]["result"]["document_id"]

    assert [d["document_id"] for d in system.admin.list_documents(system.token, "pending_review")] == [document_id]
    assert system.admin.list_documents(system.token, "approved") == []
    assert [d["document_id"] for d in system.admin.list_documents(system.token, curriculum_id=cid)] == [document_id]
    with pytest.raises(ValueError):
        system.admin.list_documents(system.token, "bogus")
    assert system.admin.get_document(system.token, "GDOC-0123456789ab") is None
    assert system.admin.document_pdf(system.token, "GDOC-0123456789ab") is None
