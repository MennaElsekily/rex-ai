"""
Opt-in end-to-end admin workflows on the REAL runtime path, skipped by
default: set REX_REAL_MODEL_TESTS=1 (needs Ollama; the tender search also
needs network access; tens of minutes).

AdminService -> Central Orchestrator -> the real agents with their real
MCP servers (curriculum, documents, marketing, tenders) and the real local
model, on temporary databases and folders (tests/conftest.py) and a
temporary copy of the knowledge-base index built from
knowledge_base/documents. The project's databases and index are never
opened. Email: the real send path runs with a recording provider in place
of Resend, so nothing is sent.

Workflow 1: curriculum -> approval -> resumed document generation.
Workflow 2: curriculum -> approval -> resumed marketing campaign.
Workflow 3: tender search (Egypt) -> approval -> tailored draft -> edit ->
            send to the configured recipient (recorded, not sent) -> audit.
Workflow 4: one request needing two agents (tenders + a curriculum).
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import pytest

from agents.curriculum_research import curriculum_agent
from agents.curriculum_research.curriculum_agent import default_curriculum_tools as REAL_CURRICULUM_TOOLS
from agents.curriculum_research.curriculum_agent import default_llm as REAL_CURRICULUM_LLM
from agents.document_generation import document_agent
from agents.document_generation.document_agent import default_document_tools as REAL_DOCUMENT_TOOLS
from agents.document_generation.document_agent import default_llm as REAL_DOCUMENT_LLM
from agents.marketing import marketing_agent
from agents.marketing.marketing_agent import default_llm as REAL_MARKETING_LLM
from agents.marketing.marketing_agent import default_marketing_tools as REAL_MARKETING_TOOLS
from agents.rfp_research import agent as rfp_agent_module, web_tenders
from agents.rfp_research.web_tenders import search_tavily as REAL_SEARCH_TAVILY

pytestmark = pytest.mark.skipif(os.environ.get("REX_REAL_MODEL_TESTS") != "1",
                                reason="Real-model test: set REX_REAL_MODEL_TESTS=1 (needs Ollama and network).")

PASSCODE = "acceptance-1234"
RECIPIENT = "acceptance-recipient@example.com"
DOCUMENTS = Path(__file__).resolve().parents[1] / "knowledge_base" / "documents"
COURSE_LIST = dict(re.findall(r"^- ([A-Z]+-\d+) (.+?):", (DOCUMENTS / "course_list.md").read_text("utf-8"), re.MULTILINE))
OBSOLETE = re.compile(r"(?i)nilebridge|DA-XL-101|DA-SQL-201|MK-ANA-301|MK-STR-101|AI-FND-101")


@pytest.fixture(scope="module")
def andalusia_index(tmp_path_factory):
    from knowledge_base import pipeline
    from knowledge_base.vector_store import VectorStore

    directory = tmp_path_factory.mktemp("andalusia_index")
    store = VectorStore(persist_directory=directory / "db")
    summary = pipeline.ingest_directory(DOCUMENTS, vector_store=store, metadata_path=directory / "meta.json",
                                        manifest_path=directory / "manifest.json")
    assert not summary["files_failed"]
    return store.persist_directory


@pytest.fixture
def admin(andalusia_index, real_default_tender_tools, monkeypatch, tmp_path):
    """The real AdminService and admin orchestrator on temporary storage, with the real agents' servers and model."""
    from agents.admin_service import AdminService
    from agents.conversation_store import ConversationStore
    from agents.register_agents import create_admin_orchestrator
    from agents.rfp_research.application_profile import ApplicationProfile
    from agents.rfp_research.opportunity_store import OpportunityStore

    monkeypatch.setenv("REX_VECTOR_DB_DIR", andalusia_index)
    monkeypatch.setenv("CURRICULUM_INDEX_DIR", andalusia_index)
    monkeypatch.setenv("REX_EMAIL_TEST_RECIPIENT", RECIPIENT)
    monkeypatch.setenv("REX_EMAIL_MODE", "test")
    for module, name, real in ((curriculum_agent, "default_curriculum_tools", REAL_CURRICULUM_TOOLS),
                               (curriculum_agent, "default_llm", REAL_CURRICULUM_LLM),
                               (document_agent, "default_document_tools", REAL_DOCUMENT_TOOLS),
                               (document_agent, "default_llm", REAL_DOCUMENT_LLM),
                               (marketing_agent, "default_marketing_tools", REAL_MARKETING_TOOLS),
                               (marketing_agent, "default_llm", REAL_MARKETING_LLM),
                               (rfp_agent_module, "default_tender_tools", real_default_tender_tools),
                               (web_tenders, "search_tavily", REAL_SEARCH_TAVILY)):
        monkeypatch.setattr(module, name, real)

    tenders = OpportunityStore(tmp_path / "tenders.sqlite3")
    profile = ApplicationProfile(db_path=tenders.db_path, documents_dir=tmp_path / "application_documents")
    profile.update_fields({"legal_name": "Andalusia Academy LLC", "contact_email": "tenders@andalusia.academy",
                           "country": "Egypt", "overview": "A training academy in Cairo offering courses in "
                           "data analytics, artificial intelligence, software and mobile development and digital "
                           "marketing."}, "acceptance-test")
    orchestrator_holder = {}

    def factory():
        if "o" not in orchestrator_holder:
            orchestrator_holder["o"] = create_admin_orchestrator(
                conversation_store=ConversationStore(tmp_path / "conversations.sqlite3"), opportunity_store=tenders)
        return orchestrator_holder["o"]

    service = AdminService(orchestrator_factory=factory, opportunity_store=tenders, application_profile=profile,
                           passcode=PASSCODE)
    yield service, service.login(PASSCODE), tenders
    for agent in vars(orchestrator_holder.get("o", object())).values():
        if hasattr(agent, "close"):
            try:
                agent.close()
            except Exception:
                pass


def _report(title, response):
    print(f"\n===== {title} =====\nstatus: {response.get('status')}\nsummary:\n{response.get('summary')}")
    for result in response.get("result") or []:
        body = result.get("result") if isinstance(result.get("result"), dict) else {}
        print(f"  - {result.get('agent_name')}: {result.get('status')} "
              f"{ {k: body.get(k) for k in ('curriculum_id', 'review_status', 'bilingual_companion_id', 'document_id', 'campaign_id', 'kind') if body.get(k)} } "
              f"{result.get('error_message') or ''}")


def _first(response, agent_name):
    return next(r for r in response["result"] if r["agent_name"] == agent_name)


def _approve_both_languages(service, token, curriculum_id):
    english = service.get_curriculum(token, curriculum_id)["curriculum"]
    assert english["status"] == "pending_review", english["validation"]["errors"]
    service.approve_curriculum(token, curriculum_id, english["version"])
    companion = service.curriculum_translation(token, curriculum_id)
    if companion:
        arabic = service.get_curriculum(token, companion)["curriculum"]
        if arabic["status"] == "pending_review":
            service.approve_curriculum(token, companion, arabic["version"])
    return english, companion


def test_workflow_1_curriculum_then_document_after_approval(admin):
    service, token, _ = admin
    request = "Create a 30-hour Excel for Business Analysis curriculum for beginners, then a course proposal document from it"
    response = service.ask(token, request, {"conversation_id": "wf1"})
    _report("Workflow 1 (before approval)", response)

    assert [r["agent_name"] for r in response["result"]] == ["curriculum_research", "document_generation"]
    curriculum = _first(response, "curriculum_research")["result"]
    assert _first(response, "document_generation")["status"] == "paused"          # waits for the approval
    english, companion = _approve_both_languages(service, token, curriculum["curriculum_id"])

    resumed = service.pending_chat_update_for(token, "wf1")                        # resumed by the approval
    _report("Workflow 1 (resumed after approval)", resumed)
    document = _first(resumed, "document_generation")
    assert document["status"] == "success", document.get("error_message")
    record = service.get_document(token, document["result"]["document_id"])["document"]
    assert record["curriculum_id"] == curriculum["curriculum_id"]
    assert record["curriculum_version"] == english["version"]                     # the approved version
    pdf = service.approved_curriculum_pdf(token, curriculum["curriculum_id"])
    assert pdf["data"][:4] == b"%PDF"
    if companion and service.get_curriculum(token, companion)["curriculum"]["status"] == "approved":
        assert service.curricula_pdf_zip(token, curriculum["curriculum_id"])["data"][:2] == b"PK"
    assert not OBSOLETE.search(str(record))


def test_workflow_2_curriculum_then_marketing_after_approval(admin):
    service, token, _ = admin
    response = service.ask(token, "Create a 24-hour beginner curriculum about SQL for data analysts and then a marketing campaign for it",
                           {"conversation_id": "wf2"})
    _report("Workflow 2 (before approval)", response)
    curriculum = _first(response, "curriculum_research")["result"]
    assert _first(response, "marketing")["status"] == "paused"
    english, _ = _approve_both_languages(service, token, curriculum["curriculum_id"])

    resumed = service.pending_chat_update_for(token, "wf2")
    _report("Workflow 2 (resumed after approval)", resumed)
    campaign = _first(resumed, "marketing")
    assert campaign["status"] == "success", campaign.get("error_message")
    record = service.get_campaign(token, campaign["result"]["campaign_id"])["campaign"]
    assert record["curriculum_id"] == curriculum["curriculum_id"]
    text = str(record)
    assert english["content"]["title"] in text                                    # the right course
    assert not OBSOLETE.search(text)
    other_courses = [name for cid, name in COURSE_LIST.items() if name in text and name != english["content"]["title"]]
    print("other catalogue course names mentioned:", other_courses)


def test_workflow_3_tender_search_approval_draft_and_send_to_the_configured_recipient(admin):
    from agents.rfp_research.email_sending import EmailSendingService

    service, token, tenders = admin
    response = service.search_tenders(token, {"countries": ["Egypt"], "categories": ["training"]})
    _report("Workflow 3 (tender search, Egypt)", response)
    found = service.list_tenders(token)
    print(f"tenders stored: {len(found)}; e.g. " + "; ".join(
        f"{t['opportunity_id']} {t.get('country')} {str(t.get('source_url') or t.get('link') or '')[:60]}" for t in found[:3]))
    if not found:
        pytest.skip("The live tender sources returned nothing for Egypt today.")
    assert all(t.get("country") in ("Egypt", "EG", None) for t in found)
    tender = found[0]
    service.decide(token, tender["opportunity_id"], "approved", authorize_submission=True)
    draft = service.current_email_draft(token, tender["opportunity_id"])
    assert draft and draft["subject"] and "Andalusia Academy" in draft["body"]
    service.save_email_draft(token, tender["opportunity_id"], draft["subject"], draft["body"] + "\n\n(Reviewed.)")

    shown = service.email_sending_mode(token)["recipient"]                        # what the admin sees
    sent_calls = []
    service._email_sending_service = EmailSendingService(
        tenders, send_fn=lambda to, *a, **k: sent_calls.append(to) or "recorded-not-sent")
    result = service.send_application_email(token, tender["opportunity_id"])
    audit = tenders.email_sends(tender["opportunity_id"])[0]
    print("shown recipient:", shown, "| provider called with:", sent_calls, "| audit recipient:", audit["recipient"],
          "| tender's own contact (never used):", audit.get("intended_recipient"))
    assert shown == RECIPIENT == sent_calls[0] == audit["recipient"] == result["to"]
    assert audit["status"] == "sent" and audit["provider_reference"] == "recorded-not-sent"


def test_workflow_4_one_request_runs_two_agents_and_synthesizes_both(admin):
    service, token, _ = admin
    response = service.ask(token, "Find training tenders in Egypt and create a curriculum for a data analytics course",
                           {"conversation_id": "wf4"})
    _report("Workflow 4 (two agents)", response)

    assert [r["agent_name"] for r in response["result"]] == ["rfp_research", "curriculum_research"]
    assert all(r["status"] in ("success", "error") for r in response["result"])
    assert "1. Tender research:" in response["summary"] and "2. Curriculum research:" in response["summary"]
    curriculum = _first(response, "curriculum_research")
    if curriculum["status"] == "success" and curriculum["result"].get("curriculum_id"):
        title = service.get_curriculum(token, curriculum["result"]["curriculum_id"])["curriculum"]["content"]["title"]
        assert "tender" not in title.casefold() and "egypt" not in title.casefold()   # its own part of the request
