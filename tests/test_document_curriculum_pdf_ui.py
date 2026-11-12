"""
Curriculum PDF (Agent 3's own PDF of an approved curriculum, offered in the Documents tab), the progress feedback
while a document is written, and the document-type selector with all ten choices.

The Curriculum PDF is read-only: it uses the curriculum's approved version, Agent 3's renderer, and writes nothing.
Fake model, temporary storage; the UI is Streamlit's AppTest (it cannot click a download button, so the PDF itself is
checked through the toolbox and AdminService that the button calls).
"""

import pytest
from pypdf import PdfReader

from agents.admin_service import AdminAuthError, AdminService
from agents.conversation_store import ConversationStore
from agents.curriculum_research import pdf as curriculum_pdf
from agents.curriculum_research.curriculum_store import CurriculumStore
from agents.document_generation import templates
from agents.document_generation.document_store import DocumentStore
from agents.document_generation.document_tools import DocumentToolbox
from tests.curriculum_fixtures import valid_curriculum
from tests.document_fakes import ADMIN, approved_curriculum, in_process_document_tools, save_curriculum
from tests.test_document_final_fixes import has_word
from tests.test_document_new_types import fake
from tests.test_rex_ai_app import env, start, unlock  # noqa: F401  (env is a fixture)

PASSCODE = "pass-123"


@pytest.fixture
def toolbox(tmp_path):
    return in_process_document_tools(tmp_path / "cpdf")


@pytest.fixture
def admin(toolbox, tmp_path):
    return AdminService(orchestrator_factory=lambda: None, passcode=PASSCODE, curriculum_store=toolbox.curricula,
                        curriculum_documents_dir=str(tmp_path / "cpdfs"), document_store=toolbox.store,
                        document_output_dir=str(toolbox.documents_dir))


def first_page(data: bytes) -> str:
    import io
    return PdfReader(io.BytesIO(data)).pages[0].extract_text()


# --------------------------------------------------------------------
# The download
# --------------------------------------------------------------------

def test_an_approved_curriculum_gives_agent_3s_pdf_of_its_approved_version(toolbox):
    cid = approved_curriculum(toolbox)

    result = toolbox.approved_curriculum_pdf(cid)

    assert result["filename"] == f"{cid}_v1_en.pdf" and result["version"] == 1 and result["language"] == "en"
    assert result["data"].startswith(b"%PDF") and result["pages"] >= 2
    cover = first_page(result["data"])
    assert "Machine Learning Foundations" in cover and "Approved by the academy" in cover and cid in cover


def test_the_pdf_is_in_the_curriculums_language(toolbox):
    cid = approved_curriculum(toolbox, valid_curriculum("ar"))

    result = toolbox.approved_curriculum_pdf(cid)

    assert result["filename"] == f"{cid}_v1_ar.pdf" and result["language"] == "ar"
    assert has_word(first_page(result["data"]), "معتمد")                 # the Arabic "approved" status on the cover


def test_it_reuses_agent_3s_renderer_and_does_not_draw_a_pdf_of_its_own(toolbox, monkeypatch):
    cid = approved_curriculum(toolbox)
    seen = {}

    def spy(record):
        seen["record"] = record
        return b"%PDF-from-agent-3", 7

    monkeypatch.setattr(curriculum_pdf, "render_pdf", spy)
    result = toolbox.approved_curriculum_pdf(cid)

    assert result["data"] == b"%PDF-from-agent-3" and result["pages"] == 7
    assert seen["record"]["curriculum_id"] == cid and seen["record"]["version_status"] == "approved"


def test_the_approved_version_is_used_even_when_a_newer_version_exists(toolbox):
    cid = approved_curriculum(toolbox)
    revised = valid_curriculum()
    revised["title"] = "A Revised Title Not Yet Approved"
    toolbox.curricula.save_version(revised, {"valid": True, "errors": []}, [], curriculum_id=cid)

    result = toolbox.approved_curriculum_pdf(cid)

    cover = first_page(result["data"])
    assert result["version"] == 1 and "Machine Learning Foundations" in cover and "Revised" not in cover


@pytest.mark.parametrize("make", [lambda tb: save_curriculum(tb)["curriculum_id"],
                                  lambda tb: save_curriculum(tb, valid=False)["curriculum_id"]])
def test_an_unapproved_curriculum_is_refused_with_the_reason(toolbox, make):
    cid = make(toolbox)
    with pytest.raises(ValueError, match="needs to be approved"):
        toolbox.approved_curriculum_pdf(cid)


def test_changes_requested_unknown_and_malformed_ids_are_refused(toolbox):
    saved = save_curriculum(toolbox)
    toolbox.curricula.request_changes(saved["curriculum_id"], 1, ADMIN, "fix it")
    with pytest.raises(ValueError, match="needs to be approved"):
        toolbox.approved_curriculum_pdf(saved["curriculum_id"])
    with pytest.raises(ValueError, match="No curriculum"):
        toolbox.approved_curriculum_pdf("CUR-0123456789ab")
    for bad in ("nope", "../x", "", None):
        with pytest.raises(ValueError):
            toolbox.approved_curriculum_pdf(bad)


def test_it_writes_nothing_and_never_modifies_the_curriculum(toolbox, tmp_path):
    cid = approved_curriculum(toolbox)
    before = (toolbox.curricula.get(cid), toolbox.curricula.documents(cid), toolbox.curricula.audit_trail(cid),
              toolbox.curricula.versions(cid))

    for _ in range(2):
        toolbox.approved_curriculum_pdf(cid)

    after = (toolbox.curricula.get(cid), toolbox.curricula.documents(cid), toolbox.curricula.audit_trail(cid),
             toolbox.curricula.versions(cid))
    assert after == before                                         # the curriculum, its versions, audit and PDF table
    assert toolbox.curricula.documents(cid) == []                   # not even Agent 3's own PDF table is written
    assert toolbox.store.list() == [] and not toolbox.documents_dir.exists()
    assert not list(tmp_path.rglob("*.pdf"))


def test_the_admin_service_checks_the_session_and_returns_the_download(admin, toolbox):
    cid = approved_curriculum(toolbox)
    for bad in (None, "", "forged"):
        with pytest.raises(AdminAuthError):
            admin.approved_curriculum_pdf(bad, cid)

    result = admin.approved_curriculum_pdf(admin.login(PASSCODE), cid)

    assert result["data"].startswith(b"%PDF") and result["filename"] == f"{cid}_v1_en.pdf"
    pending = save_curriculum(toolbox)["curriculum_id"]
    with pytest.raises(ValueError, match="needs to be approved"):
        admin.approved_curriculum_pdf(admin.login(PASSCODE), pending)


def test_the_curriculum_pdf_is_the_same_file_agent_3_renders_for_that_version(toolbox):
    cid = approved_curriculum(toolbox)
    record = toolbox.curricula.get(cid, 1)

    data, pages = curriculum_pdf.render_pdf(record)
    ours = toolbox.approved_curriculum_pdf(cid)

    assert (ours["pages"], first_page(ours["data"])[:80]) == (pages, first_page(data)[:80])


# --------------------------------------------------------------------
# Progress feedback while a document is written
# --------------------------------------------------------------------

@pytest.fixture
def system(toolbox, tmp_path):
    from agents.document_generation.registration import register_document_generation
    from agents.orchestrator import Orchestrator

    orchestrator = Orchestrator(excluded_agents=("rex_assistant",))
    agent = register_document_generation(orchestrator, ConversationStore(db_path=":memory:"))
    agent._tools, agent._llm = toolbox, fake()
    admin = AdminService(orchestrator_factory=lambda: orchestrator, passcode=PASSCODE, curriculum_store=toolbox.curricula,
                         curriculum_documents_dir=str(tmp_path / "cpdfs"), document_store=toolbox.store,
                         document_output_dir=str(toolbox.documents_dir))

    class System:
        pass

    result = System()
    result.toolbox, result.admin, result.token = toolbox, admin, admin.login(PASSCODE)
    return result


def test_the_admin_page_is_told_each_step_through_the_whole_orchestration(system):
    cid = approved_curriculum(system.toolbox)
    steps = []

    response = system.admin.generate_document(system.token, cid, "question_bank",
                                              context={"progress": lambda *step: steps.append(step)})

    assert response["status"] == "success"
    assert steps == [(0, 2, "M1"), (1, 2, "M2"), (2, 2, "")]        # before each module, then "saving"


def test_a_failing_progress_callback_never_stops_the_generation(system):
    cid = approved_curriculum(system.toolbox)

    def broken(*step):
        raise RuntimeError("the page went away")

    response = system.admin.generate_document(system.token, cid, "presentation_outline", context={"progress": broken})

    assert response["status"] == "success" and len(system.toolbox.store.list()) == 1


def test_documents_made_without_a_callback_are_unchanged(system):
    cid = approved_curriculum(system.toolbox)
    assert system.admin.generate_document(system.token, cid, "instructor_guide")["status"] == "success"


# --------------------------------------------------------------------
# The Documents tab: all ten choices
# --------------------------------------------------------------------

def default_toolbox():
    return DocumentToolbox(store=DocumentStore(), curricula=CurriculumStore())


def seed_approved(language="en"):
    toolbox = default_toolbox()
    cid = toolbox.curricula.save_version(valid_curriculum(language), {"valid": True, "errors": []}, [])["curriculum_id"]
    toolbox.curricula.approve(cid, 1, ADMIN)
    return toolbox, cid


def test_every_choice_has_an_english_and_an_arabic_name():
    choices = [*templates.TEMPLATES, templates.CURRICULUM_PDF]
    assert len(choices) == 10
    for language in ("en", "ar"):
        names = [templates.type_label(c, language) for c in choices]
        assert all(names) and len(set(names)) == 10 and not any(n in choices for n in names)


def test_the_selector_lists_all_ten_document_choices(env):
    seed_approved()
    at = unlock(start(), section="documents")

    box = at.selectbox(key="doc_type")

    assert box.options == ["Course Description", "Course Proposal", "Participant Guide", "Capstone Project Brief",
                           "Learning Outcomes and Assessment Plan", "Facilitator / Instructor Guide", "Presentation Outline",
                           "Question Bank", "Assessment Rubrics", "Curriculum PDF"]


@pytest.mark.parametrize("doc_type", ["instructor_guide", "presentation_outline", "question_bank", "assessment_rubric"])
def test_each_new_type_is_requested_through_the_same_form(env, doc_type):
    _, cid = seed_approved()
    env["orchestrator"].handle_request.return_value = {"status": "success", "result": [{
        "agent_name": "document_generation", "status": "success", "result": {"kind": "document", "message": "Generated."}}]}
    at = unlock(start(), section="documents")
    at.selectbox(key="doc_type").select(doc_type).run()

    next(b for b in at.button if b.label == "Generate document").click().run()

    assert not at.exception, at.exception
    context = env["orchestrator"].handle_request.call_args.kwargs["context"]
    assert context["document_request"] == {"curriculum_id": cid, "document_type": doc_type, "language": None,
                                           "extra": {}, "document_id": None}
    assert callable(context["progress"])                       # the page asks to be told how far the writing has got


def test_choosing_curriculum_pdf_shows_the_download_not_the_generate_form(env):
    seed_approved()
    at = unlock(start(), section="documents")

    at.selectbox(key="doc_type").select("curriculum_pdf").run()

    assert not at.exception, at.exception
    assert any("Curriculum Agent's own PDF" in i.value for i in at.info)
    assert not any(b.label == "Generate document" for b in at.button)       # nothing is generated for this choice
    assert not at.error


def test_the_curriculum_pdf_choice_calls_the_same_toolbox_method_the_download_uses(env, monkeypatch):
    _, cid = seed_approved()
    calls = []
    real = DocumentToolbox.approved_curriculum_pdf

    def spy(self, curriculum_id):
        calls.append(curriculum_id)
        return real(self, curriculum_id)

    monkeypatch.setattr(DocumentToolbox, "approved_curriculum_pdf", spy)
    at = unlock(start(), section="documents")
    at.selectbox(key="doc_type").select("curriculum_pdf").run()

    assert calls == [cid] and not at.exception


def test_a_curriculum_pdf_that_cannot_be_made_is_shown_as_an_error(env, monkeypatch):
    seed_approved()

    def refuse(self, curriculum_id):
        raise ValueError("The PDF could not be made: disk full")

    monkeypatch.setattr(DocumentToolbox, "approved_curriculum_pdf", refuse)
    at = unlock(start(), section="documents")
    at.selectbox(key="doc_type").select("curriculum_pdf").run()

    assert not at.exception and any("disk full" in e.value for e in at.error)


def test_the_language_note_names_the_curriculums_language(env):
    seed_approved("ar")
    english = unlock(start(), section="documents")
    assert any("Language: Arabic" in c.value for c in english.caption)

    arabic = start()
    arabic.session_state["lang"] = "ar"
    unlock(arabic, section="documents")
    assert any("اللغة: العربية" in c.value for c in arabic.caption)
