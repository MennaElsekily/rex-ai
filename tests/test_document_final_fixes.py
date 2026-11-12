"""
Final Agent 4 fixes:

1. a version approved before approved PDFs existed is downloaded from its review PDF (read-only fallback);
2. Arabic documents use an Arabic label for the missing "training format" and show the level in Arabic, and
   documents made before that still validate and can be edited;
3. a leftover review PDF (crash between writing the file and recording it) no longer blocks that version for ever,
   and a failed cleanup can never delete a file another process has recorded.

Temporary storage, fake model.
"""

import hashlib
import sqlite3

import pytest

from agents.admin_service import AdminService
from agents.conversation_store import ConversationStore
from agents.document_generation import pdf as pdf_module
from agents.document_generation import templates
from agents.document_generation.composer import compose
from agents.document_generation.document_agent import DocumentAgent
from agents.document_generation.document_store import DocumentStoreError
from agents.document_generation.validation import validate_content
from tests.curriculum_fixtures import valid_curriculum
from tests.document_fakes import ADMIN, approved_curriculum, fake_llm, in_process_document_tools, parts_answers
from tests.test_document_fixes import cover_text, has_word, make_document
from tests.test_rex_ai_app import env, start, unlock  # noqa: F401  (env is a fixture)

PASSCODE = "pass-123"


@pytest.fixture
def toolbox(tmp_path):
    return in_process_document_tools(tmp_path / "final")


@pytest.fixture
def admin(toolbox, tmp_path):
    return AdminService(orchestrator_factory=lambda: None, passcode=PASSCODE, curriculum_store=toolbox.curricula,
                        curriculum_documents_dir=str(tmp_path / "cpdfs"), document_store=toolbox.store,
                        document_output_dir=str(toolbox.documents_dir))


def pdf_files(toolbox):
    return sorted(p.name for p in toolbox.documents_dir.iterdir())


def make_legacy_approved(toolbox):
    """A version approved by the earlier release: approved, with a review PDF and no approved PDF."""
    _, document_id = make_document(toolbox)
    toolbox.approve_document(document_id, 1, ADMIN)
    (toolbox.documents_dir / f"{document_id}_v1_en_approved.pdf").unlink()
    with sqlite3.connect(toolbox.store.db_path) as db:
        db.execute("UPDATE document_versions SET approved_filename=NULL, approved_sha256=NULL, approved_pages=NULL, "
                   "approved_size_bytes=NULL")
    return document_id


# --------------------------------------------------------------------
# 1. Legacy approved documents can be downloaded
# --------------------------------------------------------------------

def test_a_legacy_approved_version_is_downloaded_from_its_review_pdf(toolbox, admin):
    document_id = make_legacy_approved(toolbox)
    token = admin.login(PASSCODE)
    review_bytes = (toolbox.documents_dir / f"{document_id}_v1_en.pdf").read_bytes()
    record_before, files_before = toolbox.store.get(document_id, 1), pdf_files(toolbox)

    download = admin.document_pdf(token, document_id, 1)

    assert download["data"] == review_bytes and download["filename"] == f"{document_id}_v1_en.pdf"
    assert download["approved"] is False   # it is the review copy, and is reported as one
    assert toolbox.document_bytes(document_id, 1) == review_bytes
    # read-only: the approved record, its history and the folder are exactly as they were
    assert toolbox.store.get(document_id, 1) == record_before and pdf_files(toolbox) == files_before
    assert record_before["version_status"] == "approved" and record_before["approved_filename"] is None
    assert len(toolbox.store.audit_trail(document_id)) == 3   # version_saved, pdf_rendered, review: nothing added


def test_a_newer_version_of_a_legacy_document_gets_a_real_approved_pdf(toolbox, admin):
    document_id = make_legacy_approved(toolbox)
    token = admin.login(PASSCODE)
    first = toolbox.store.get(document_id, 1)["content"]
    admin.edit_document(token, document_id, [])   # no edits: a new version of the same content
    admin.approve_document(token, document_id, 2)

    new = admin.document_pdf(token, document_id, 2)
    old = admin.document_pdf(token, document_id, 1)

    assert new["approved"] is True and new["filename"].endswith("_v2_en_approved.pdf")
    assert old["approved"] is False and old["filename"] == f"{document_id}_v1_en.pdf"   # v1 still served, unchanged
    assert toolbox.store.get(document_id, 1)["content"] == first


def test_the_fallback_does_not_apply_to_versions_that_are_not_approved(toolbox):
    _, document_id = make_document(toolbox)
    assert toolbox.document_bytes(document_id, 1) == (toolbox.documents_dir / f"{document_id}_v1_en.pdf").read_bytes()
    toolbox.approve_document(document_id, 1, ADMIN)
    assert toolbox.document_bytes(document_id, 1) == (toolbox.documents_dir / f"{document_id}_v1_en_approved.pdf").read_bytes()


def test_a_legacy_version_with_no_file_at_all_still_reports_it_missing(toolbox, admin):
    document_id = make_legacy_approved(toolbox)
    (toolbox.documents_dir / f"{document_id}_v1_en.pdf").unlink()
    with pytest.raises(ValueError, match="missing"):
        admin.document_pdf(admin.login(PASSCODE), document_id, 1)


# --------------------------------------------------------------------
# 2. Arabic localization
# --------------------------------------------------------------------

def test_the_arabic_cover_names_the_missing_field_in_arabic(tmp_path):
    toolbox = in_process_document_tools(tmp_path / "ar")
    _, document_id = make_document(toolbox, "ar")

    cover = cover_text(toolbox.documents_dir / f"{document_id}_v1_ar.pdf")

    assert "training_format" not in cover and has_word(cover, "نمط") and has_word(cover, "التدريب")


def test_the_english_cover_names_the_missing_field_by_its_label(toolbox):
    _, document_id = make_document(toolbox)
    cover = cover_text(toolbox.documents_dir / f"{document_id}_v1_en.pdf")
    assert "Training format" in cover and "training_format" not in cover


@pytest.mark.parametrize("level, arabic", [("Beginner", "مبتدئ"), ("Intermediate", "متوسط"), ("Advanced", "متقدم")])
def test_arabic_documents_show_the_level_in_arabic(toolbox, level, arabic):
    content = valid_curriculum("ar")
    content["level"] = level
    cid = approved_curriculum(toolbox, content)
    record = toolbox.get_approved_curriculum(cid)["curriculum"]
    document = compose(templates.TEMPLATES["outcomes_assessment_plan"], record, "ar")
    trace = next(s for s in document["sections"] if s["key"] == "traceability")["blocks"][1]["text"]

    assert arabic in trace and level not in trace
    assert validate_content(document, record)["valid"]


def test_english_documents_keep_the_english_level(toolbox):
    cid = approved_curriculum(toolbox)
    record = toolbox.get_approved_curriculum(cid)["curriculum"]
    document = compose(templates.TEMPLATES["outcomes_assessment_plan"], record, "en")
    assert "Level: Beginner" in next(s for s in document["sections"] if s["key"] == "traceability")["blocks"][1]["text"]


def test_the_arabic_pdf_shows_the_level_in_arabic_on_the_traceability_page(tmp_path):
    toolbox = in_process_document_tools(tmp_path / "ar2")
    cid = approved_curriculum(toolbox, valid_curriculum("ar"))
    _, document_id = make_document(toolbox, "ar", curriculum_id=cid, doc_type="outcomes_assessment_plan")
    from pypdf import PdfReader
    pages = PdfReader(str(toolbox.documents_dir / f"{document_id}_v1_ar.pdf")).pages

    last_page = pages[-1].extract_text()
    assert has_word(last_page, "مبتدئ") and "Beginner" not in last_page


def test_a_document_made_before_the_level_was_localized_still_validates_and_can_be_edited(toolbox):
    cid = approved_curriculum(toolbox, valid_curriculum("ar"))
    record = toolbox.get_approved_curriculum(cid)["curriculum"]
    template = templates.TEMPLATES["course_description"]
    parts = template.parts_schema.model_validate(parts_answers("ar")["DescriptionParts"])
    old_style = compose(template, record, "ar", parts, localize_level=False)
    assert "Beginner" in str(old_style)

    assert validate_content(old_style, record)["valid"]                       # the earlier wording is accepted
    saved = toolbox.save_document_draft(old_style)
    assert saved["review_status"] == "pending_review"
    edit = toolbox.save_admin_edit(old_style, saved["document_id"], ADMIN)   # and it can be edited and saved again
    assert edit["review_status"] == "pending_review"


def test_the_legacy_wording_does_not_open_a_way_to_change_curriculum_text(toolbox):
    cid = approved_curriculum(toolbox, valid_curriculum("ar"))
    record = toolbox.get_approved_curriculum(cid)["curriculum"]
    template = templates.TEMPLATES["course_description"]
    parts = template.parts_schema.model_validate(parts_answers("ar")["DescriptionParts"])
    old_style = compose(template, record, "ar", parts, localize_level=False)
    objectives = next(s for s in old_style["sections"] if s["key"] == "objectives")
    objectives["blocks"][0]["items"][0] = "Something else"

    result = validate_content(old_style, record)

    assert not result["valid"] and result["errors"][0]["code"] == "curriculum_changed"


def test_the_agent_reply_names_the_missing_field_in_arabic(tmp_path):
    toolbox = in_process_document_tools(tmp_path / "ar3")
    cid = approved_curriculum(toolbox, valid_curriculum("ar"))
    agent = DocumentAgent(tools=toolbox, llm=fake_llm("ar"))

    result = agent.process_request({"request_id": "r", "user_input": "x", "context": {
        "language": "ar", "document_request": {"curriculum_id": cid, "document_type": "course_description"}}})

    message = result["result"]["message"]
    assert result["status"] == "success" and "نمط التدريب" in message and "training_format" not in message


def test_the_admin_warning_uses_the_label_in_both_languages(env):
    from agents.curriculum_research.curriculum_store import CurriculumStore
    from agents.document_generation.document_store import DocumentStore
    from agents.document_generation.document_tools import DocumentToolbox

    toolbox = DocumentToolbox(store=DocumentStore(), curricula=CurriculumStore())
    make_document(toolbox)

    english = unlock(start(), section="documents")
    assert any("Training format" in w.value and "training_format" not in w.value for w in english.warning)

    arabic = start()
    arabic.session_state["lang"] = "ar"
    unlock(arabic, section="documents")
    assert any("نمط التدريب" in w.value and "training_format" not in w.value for w in arabic.warning)


# --------------------------------------------------------------------
# 3. Review-PDF crash recovery
# --------------------------------------------------------------------

def make_unrendered_document(toolbox):
    record = None
    cid = approved_curriculum(toolbox)
    record = toolbox.get_approved_curriculum(cid)["curriculum"]
    template = templates.TEMPLATES["course_description"]
    content = compose(template, record, "en", template.parts_schema.model_validate(parts_answers()["DescriptionParts"]))
    return toolbox.save_document_draft(content)["document_id"]


def test_a_leftover_review_pdf_no_longer_blocks_the_version(toolbox):
    document_id = make_unrendered_document(toolbox)
    toolbox.documents_dir.mkdir(parents=True, exist_ok=True)
    leftover = toolbox.documents_dir / f"{document_id}_v1_en.pdf"
    leftover.write_bytes(b"left by a crash before the file was recorded")

    result = toolbox.render_document_pdf(document_id)

    assert result["status"] == "ok", result
    record = result["document"]
    data = leftover.read_bytes()
    assert data.startswith(b"%PDF") and hashlib.sha256(data).hexdigest() == record["sha256"]   # a real PDF, recorded
    assert record["filename"] == leftover.name and record["pages"] >= 2


def test_rendering_still_never_replaces_a_recorded_pdf(toolbox):
    _, document_id = make_document(toolbox)
    path = toolbox.documents_dir / f"{document_id}_v1_en.pdf"
    recorded = toolbox.store.get(document_id)
    stamp = path.stat().st_mtime_ns

    again = toolbox.render_document_pdf(document_id)["document"]

    assert again["sha256"] == recorded["sha256"] and path.stat().st_mtime_ns == stamp


def test_a_failed_record_removes_the_file_this_call_wrote(toolbox, monkeypatch):
    document_id = make_unrendered_document(toolbox)

    def failing(*args, **kwargs):
        raise DocumentStoreError("disk full")

    monkeypatch.setattr(toolbox.store, "attach_pdf", failing)
    result = toolbox.render_document_pdf(document_id)

    assert result["status"] == "invalid_request" and "disk full" in result["message"]
    assert not (toolbox.documents_dir / f"{document_id}_v1_en.pdf").exists()
    assert toolbox.store.get(document_id)["filename"] is None


def test_cleanup_never_deletes_a_file_another_process_recorded(toolbox, monkeypatch):
    document_id = make_unrendered_document(toolbox)
    real_attach = toolbox.store.attach_pdf

    def lost_the_race(doc, version, filename, data, pages):
        real_attach(doc, version, filename, data, pages)        # "another process" recorded this file first
        raise DocumentStoreError("This version already has a PDF.")

    monkeypatch.setattr(toolbox.store, "attach_pdf", lost_the_race)
    result = toolbox.render_document_pdf(document_id)

    path = toolbox.documents_dir / f"{document_id}_v1_en.pdf"
    assert result["status"] == "invalid_request"
    assert path.is_file()                                       # the recorded file survives our failed attempt
    assert hashlib.sha256(path.read_bytes()).hexdigest() == toolbox.store.get(document_id)["sha256"]


def test_a_pdf_recorded_by_another_process_while_rendering_is_left_alone(toolbox, monkeypatch):
    document_id = make_unrendered_document(toolbox)
    real_render = pdf_module.render_document_pdf
    other = {}

    def render_while_another_process_finishes(record, approval=None):
        data, pages = real_render(record, approval)
        name = f"{document_id}_v1_en.pdf"
        toolbox.documents_dir.mkdir(parents=True, exist_ok=True)
        (toolbox.documents_dir / name).write_bytes(data + b"\n% other process")
        toolbox.store.attach_pdf(document_id, 1, name, data + b"\n% other process", pages)
        other["bytes"] = data + b"\n% other process"
        return data, pages

    monkeypatch.setattr(pdf_module, "render_document_pdf", render_while_another_process_finishes)
    result = toolbox.render_document_pdf(document_id)

    path = toolbox.documents_dir / f"{document_id}_v1_en.pdf"
    assert result["status"] == "ok" and path.read_bytes() == other["bytes"]   # their file and record were not touched
    assert hashlib.sha256(path.read_bytes()).hexdigest() == toolbox.store.get(document_id)["sha256"]


def test_an_approved_pdf_written_by_another_approval_is_never_replaced(toolbox, monkeypatch):
    _, document_id = make_document(toolbox)
    real_render = pdf_module.render_document_pdf
    theirs = {}

    def approve_in_the_meantime(record, approval=None):
        data, pages = real_render(record, approval)
        if approval:   # another administrator's approval completes while this one is rendering
            name = f"{document_id}_v1_en_approved.pdf"
            (toolbox.documents_dir / name).write_bytes(data)
            toolbox.store.approve(document_id, 1, "admin-session-other", approved_pdf=(name, data, pages))
            theirs["bytes"] = data
        return data, pages

    monkeypatch.setattr(pdf_module, "render_document_pdf", approve_in_the_meantime)
    with pytest.raises(DocumentStoreError, match="approved"):
        toolbox.approve_document(document_id, 1, ADMIN)

    path = toolbox.documents_dir / f"{document_id}_v1_en_approved.pdf"
    assert path.read_bytes() == theirs["bytes"]
    stored = toolbox.store.get(document_id, 1)
    assert stored["version_status"] == "approved" and hashlib.sha256(theirs["bytes"]).hexdigest() == stored["approved_sha256"]
    assert [a["actor"] for a in toolbox.store.audit_trail(document_id) if a["event"] == "review"] == ["admin-session-other"]
