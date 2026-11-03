"""
Agent 4 fixes after the audit: routing against curriculum creation, the
approved PDF (a separate file made before the approval is recorded), the
additive database migration, approval failures, paged approved-curriculum
lists, server-side checks of the extra fields, editing approved documents
as new versions, and the UI refresh. Temporary storage, fake model.
"""

import re

import arabic_reshaper
import sqlite3
from datetime import datetime, timezone

import pytest
from pypdf import PdfReader

from agents.admin_service import AdminService
from agents.conversation_store import ConversationStore
from agents.curriculum_research.curriculum_store import CurriculumStore
from agents.document_generation import templates
from agents.document_generation import pdf as pdf_module
from agents.document_generation.composer import compose
from agents.document_generation.document_agent import DocumentAgent
from agents.document_generation.document_store import DocumentStore, DocumentStoreError
from agents.document_generation.document_tools import DocumentToolbox
from agents.document_generation.registration import register_document_generation
from agents.document_generation.validation import validate_content
from agents.orchestrator import Orchestrator
from tests.curriculum_fixtures import valid_curriculum
from tests.document_fakes import ADMIN, approved_curriculum, fake_llm, in_process_document_tools, parts_answers
from tests.test_rex_ai_app import env, start, unlock  # noqa: F401  (env is a fixture)

PASSCODE = "pass-123"


@pytest.fixture
def toolbox(tmp_path):
    return in_process_document_tools(tmp_path / "fixes")


def make_document(toolbox, language="en", curriculum_id=None, doc_type="course_description"):
    curriculum_id = curriculum_id or approved_curriculum(toolbox, valid_curriculum(language))
    record = toolbox.get_approved_curriculum(curriculum_id)["curriculum"]
    template = templates.TEMPLATES[doc_type]
    schema = template.parts_schema
    parts = schema.model_validate(parts_answers(language)[schema.__name__]) if schema else None
    content = compose(template, record, language, parts)
    saved = toolbox.store.save_version(content, validate_content(content, record))
    toolbox.render_document_pdf(saved["document_id"])
    return curriculum_id, saved["document_id"]


def cover_text(path):
    return PdfReader(str(path)).pages[0].extract_text()


# --------------------------------------------------------------------
# 1. Routing: a new curriculum is Agent 3's; a document from a curriculum is Agent 4's
# --------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Create a curriculum about SQL with an assessment plan", "curriculum_research"),
    ("Design a 40-hour curriculum on writing a course proposal", "curriculum_research"),
    ("Build a syllabus for Python including a participant guide", "curriculum_research"),
    ("Generate a curriculum with a capstone project brief", "curriculum_research"),
    ("انشئ منهج عن SQL مع خطة التقييم", "curriculum_research"),
    ("Generate a course description for curriculum CUR-123", "document_generation"),
    ("Using the approved SQL curriculum, create a participant guide", "document_generation"),
    ("Create a participant guide for the approved SQL curriculum", "document_generation"),
    ("Generate a course proposal using the selected curriculum", "document_generation"),
    ("Write the capstone project brief for CUR-0123456789ab", "document_generation"),
    ("أنشئ دليل المتدرب لمنهج Python", "document_generation"),
    ("Generate a course participant guide", "document_generation"),
])
def test_curriculum_creation_and_document_requests_are_told_apart(text, expected):
    assert Orchestrator()._make_plan(text) == [expected]


@pytest.mark.parametrize("text, expected", [
    ("Create a curriculum about SQL, 30 hours", ["curriculum_research"]),
    ("Make a PDF of the curriculum", ["curriculum_research"]),
    ("Find education tenders in Egypt", ["rfp_research"]),
    ("Recommend a course for beginners", ["rex_assistant"]),
    ("Create a curriculum and a marketing campaign", ["curriculum_research", "marketing"]),
    ("Suggest new courses for our data track", ["curriculum_research"]),
])
def test_the_other_routes_are_unchanged(text, expected):
    assert Orchestrator()._make_plan(text) == expected


# --------------------------------------------------------------------
# 2. The approved PDF is a separate file whose cover says "approved" (and when)
# --------------------------------------------------------------------

def test_approval_makes_a_separate_approved_pdf_with_its_own_cover(toolbox):
    _, document_id = make_document(toolbox)
    review_path = toolbox.documents_dir / f"{document_id}_v1_en.pdf"
    review_bytes = review_path.read_bytes()
    assert "Pending review (not approved)" in cover_text(review_path)

    approved = toolbox.approve_document(document_id, 1, ADMIN, "ok")

    approved_path = toolbox.documents_dir / f"{document_id}_v1_en_approved.pdf"
    assert approved["approved_filename"] == approved_path.name and approved_path.is_file()
    text = cover_text(approved_path)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert "Approved by the academy" in text and "Pending review" not in text
    assert re.search(rf"Approved on\s+{today}", text)
    assert review_path.read_bytes() == review_bytes                      # the review file is untouched
    assert approved_path.read_bytes() != review_bytes
    assert approved["approved_sha256"] and approved["approved_pages"] >= 2
    stored = toolbox.store.get(document_id)
    assert stored["approved_at"][:10] == today
    assert toolbox.document_bytes(document_id, 1) == approved_path.read_bytes()   # downloads give the approved file


def has_word(text, word):
    """An Arabic word in extracted PDF text: shaped into joined letter forms, in either letter order."""
    shaped = arabic_reshaper.reshape(word)
    return shaped in text or shaped[::-1] in text


def test_the_arabic_approved_cover_says_approved_too(tmp_path):
    toolbox = in_process_document_tools(tmp_path / "ar")
    _, document_id = make_document(toolbox, "ar")
    toolbox.approve_document(document_id, 1, ADMIN)

    review = cover_text(toolbox.documents_dir / f"{document_id}_v1_ar.pdf")
    approved = cover_text(toolbox.documents_dir / f"{document_id}_v1_ar_approved.pdf")

    assert has_word(review, "بانتظار") and not has_word(review, "الاعتماد")     # pending review, no approval date
    assert has_word(approved, "الاعتماد") and not has_word(approved, "بانتظار")  # "approval date" label, no "pending"
    assert datetime.now(timezone.utc).strftime("%Y-%m-%d") in approved


def test_if_the_approved_pdf_cannot_be_made_nothing_is_approved(toolbox, monkeypatch):
    _, document_id = make_document(toolbox)
    real = pdf_module.render_document_pdf

    def failing(record, approval=None):
        if approval:
            raise RuntimeError("font missing")
        return real(record, approval)

    monkeypatch.setattr(pdf_module, "render_document_pdf", failing)
    with pytest.raises(RuntimeError, match="font missing"):
        toolbox.approve_document(document_id, 1, ADMIN)

    stored = toolbox.store.get(document_id)
    assert stored["approved_version"] is None and stored["version_status"] == "pending_review"
    assert stored["approved_filename"] is None
    assert not list(toolbox.documents_dir.glob("*_approved.pdf"))
    assert [a["event"] for a in toolbox.store.audit_trail(document_id)].count("review") == 0


def test_if_recording_the_approval_fails_the_approved_file_is_removed(toolbox, monkeypatch):
    _, document_id = make_document(toolbox)

    def failing(*args, **kwargs):
        raise DocumentStoreError("disk full")

    monkeypatch.setattr(toolbox.store, "approve", failing)
    with pytest.raises(DocumentStoreError, match="disk full"):
        toolbox.approve_document(document_id, 1, ADMIN)

    assert not list(toolbox.documents_dir.glob("*_approved.pdf"))
    assert toolbox.store.get(document_id)["version_status"] == "pending_review"


def test_a_leftover_approved_file_from_a_failed_attempt_is_replaced(toolbox):
    _, document_id = make_document(toolbox)
    (toolbox.documents_dir / f"{document_id}_v1_en_approved.pdf").write_bytes(b"leftover")

    approved = toolbox.approve_document(document_id, 1, ADMIN)

    assert toolbox.document_bytes(document_id, 1).startswith(b"%PDF") and approved["approved_pages"] >= 2


def test_the_store_itself_will_not_approve_without_its_approved_pdf_or_twice(toolbox):
    _, document_id = make_document(toolbox)
    with pytest.raises(TypeError):
        toolbox.store.approve(document_id, 1, ADMIN)   # approved_pdf is required
    toolbox.approve_document(document_id, 1, ADMIN)
    with pytest.raises(DocumentStoreError):
        toolbox.store.approve(document_id, 1, ADMIN, approved_pdf=("x.pdf", b"x", 1))


def test_an_older_approved_version_keeps_its_own_approved_pdf(toolbox):
    curriculum_id, document_id = make_document(toolbox)
    toolbox.approve_document(document_id, 1, ADMIN)
    first = toolbox.document_bytes(document_id, 1)
    record = toolbox.get_approved_curriculum(curriculum_id)["curriculum"]
    content = toolbox.store.get(document_id)["content"]
    toolbox.save_admin_edit(content, document_id, ADMIN)
    toolbox.render_document_pdf(document_id, 2)
    toolbox.approve_document(document_id, 2, ADMIN)

    assert toolbox.document_bytes(document_id, 1) == first
    assert toolbox.document_bytes(document_id, 2) != first
    assert record and sorted(p.name for p in toolbox.documents_dir.glob("*_approved.pdf")) == [
        f"{document_id}_v1_en_approved.pdf", f"{document_id}_v2_en_approved.pdf"]


# --------------------------------------------------------------------
# 3. Backward-compatible migration of an existing database
# --------------------------------------------------------------------

OLD_SCHEMA = """
CREATE TABLE documents (document_id TEXT PRIMARY KEY, curriculum_id TEXT NOT NULL, curriculum_version INTEGER NOT NULL,
    document_type TEXT NOT NULL, language TEXT NOT NULL, title TEXT NOT NULL, status TEXT NOT NULL,
    current_version INTEGER NOT NULL, approved_version INTEGER, approved_at TEXT, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL);
CREATE TABLE document_versions (document_id TEXT NOT NULL, version INTEGER NOT NULL, status TEXT NOT NULL,
    curriculum_version INTEGER NOT NULL, content_json TEXT NOT NULL, validation_json TEXT NOT NULL,
    change_summary TEXT, filename TEXT, sha256 TEXT, pages INTEGER, size_bytes INTEGER, created_at TEXT NOT NULL,
    created_by TEXT NOT NULL, PRIMARY KEY (document_id, version));
CREATE TABLE document_audit (audit_id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL, version INTEGER,
    event TEXT NOT NULL, previous_status TEXT, new_status TEXT, actor TEXT NOT NULL, notes TEXT, at TEXT NOT NULL);
"""


def test_an_old_database_is_migrated_without_losing_anything(tmp_path):
    path = tmp_path / "old.sqlite3"
    content = {"document_type": "course_description", "title": "T", "curriculum_id": "CUR-0123456789ab",
               "curriculum_version": 1, "language": "en", "sections": []}
    with sqlite3.connect(path) as db:
        db.executescript(OLD_SCHEMA)
        db.execute("INSERT INTO documents VALUES ('GDOC-0123456789ab','CUR-0123456789ab',1,'course_description',"
                   "'en','T','approved',1,1,'2026-01-01','2026-01-01','2026-01-01')")
        db.execute("INSERT INTO document_versions VALUES ('GDOC-0123456789ab',1,'approved',1,?,'{\"valid\": true}',"
                   "NULL,'GDOC-0123456789ab_v1_en.pdf','abc',3,100,'2026-01-01','system:document_agent')",
                   (__import__("json").dumps(content),))
        db.execute("INSERT INTO document_audit (document_id, version, event, actor, at) VALUES "
                   "('GDOC-0123456789ab',1,'version_saved','system:document_agent','2026-01-01')")

    store = DocumentStore(path)
    DocumentStore(path).close()   # opening again is harmless

    columns = {row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(document_versions)")}
    assert {"approved_filename", "approved_sha256", "approved_pages", "approved_size_bytes"} <= columns
    record = store.get("GDOC-0123456789ab")
    assert record["version_status"] == "approved" and record["filename"] == "GDOC-0123456789ab_v1_en.pdf"
    assert record["approved_filename"] is None   # approved before this release: no approved copy yet
    assert store.get_approved("GDOC-0123456789ab")["version"] == 1
    assert len(store.audit_trail("GDOC-0123456789ab")) == 1 and store.counts()["approved"] == 1


def test_a_migrated_old_document_can_still_get_new_versions_and_approvals(tmp_path):
    toolbox = in_process_document_tools(tmp_path / "mig")
    curriculum_id, document_id = make_document(toolbox)
    with sqlite3.connect(toolbox.store.db_path) as db:   # make this database look like the old schema
        db.executescript("""
            ALTER TABLE document_versions DROP COLUMN approved_filename;
            ALTER TABLE document_versions DROP COLUMN approved_sha256;
            ALTER TABLE document_versions DROP COLUMN approved_pages;
            ALTER TABLE document_versions DROP COLUMN approved_size_bytes;""")
    toolbox.store.close()
    migrated = DocumentToolbox(store=DocumentStore(toolbox.store.db_path), curricula=toolbox.curricula,
                               documents_dir=toolbox.documents_dir)

    approved = migrated.approve_document(document_id, 1, ADMIN)

    assert approved["version_status"] == "approved" and approved["approved_filename"]
    assert curriculum_id


# --------------------------------------------------------------------
# 4. The 200-curriculum limit: paging and SQL counts
# --------------------------------------------------------------------

def many_curricula(toolbox, total=205, approved_every=1):
    ids = []
    for index in range(total):
        content = valid_curriculum()
        content["title"] = f"Course {index:03d}"
        saved = toolbox.curricula.save_version(content, {"valid": True, "errors": []}, [])
        if index % approved_every == 0:
            toolbox.curricula.approve(saved["curriculum_id"], 1, ADMIN)
        ids.append(saved["curriculum_id"])
    return ids


def test_the_curriculum_store_lists_only_approved_curricula_in_pages(toolbox):
    ids = many_curricula(toolbox, 12, approved_every=3)   # approved: 0, 3, 6, 9
    first = [r["curriculum_id"] for r in toolbox.curricula.list_approved(3, 0)]
    second = [r["curriculum_id"] for r in toolbox.curricula.list_approved(3, 3)]

    assert toolbox.curricula.count_approved() == 4
    assert len(first) == 3 and len(second) == 1 and not set(first) & set(second)
    assert set(first + second) == {ids[0], ids[3], ids[6], ids[9]}


def test_an_old_approved_curriculum_is_found_among_more_than_200_others(toolbox):
    oldest = approved_curriculum(toolbox)
    many_curricula(toolbox, 230, approved_every=1000)   # 230 newer curricula, of which only the first is approved
    page = toolbox.list_approved_curricula(limit=200)

    # Scanning the 200 most recent curricula (the old behaviour) would have missed both approved ones.
    assert page["total"] == 2 and oldest in {c["curriculum_id"] for c in page["curricula"]}


def test_the_tool_pages_through_more_than_200_approved_curricula(toolbox):
    many_curricula(toolbox, 205)
    first = toolbox.list_approved_curricula(limit=200, offset=0)
    second = toolbox.list_approved_curricula(limit=200, offset=200)

    assert (first["total"], len(first["curricula"]), first["has_more"]) == (205, 200, True)
    assert (len(second["curricula"]), second["has_more"]) == (5, False)
    ids = [c["curriculum_id"] for c in first["curricula"] + second["curricula"]]
    assert len(ids) == len(set(ids)) == 205
    assert toolbox.list_approved_curricula(limit=0, offset=-5)["status"] == "ok"   # bad paging is clamped


def test_the_chat_finds_a_curriculum_by_name_beyond_the_first_page(toolbox):
    zebra = valid_curriculum()
    zebra["title"] = "Zebra Studies"
    saved = toolbox.curricula.save_version(zebra, {"valid": True, "errors": []}, [])
    toolbox.curricula.approve(saved["curriculum_id"], 1, ADMIN)
    many_curricula(toolbox, 205)   # the zebra is now the oldest of 206 approved

    agent = DocumentAgent(tools=toolbox, llm=fake_llm())
    result = agent.process_request({"request_id": "r", "user_input": "Generate a course description for the Zebra curriculum",
                                    "context": {}})

    assert result["status"] == "success" and result["result"]["curriculum_id"] == saved["curriculum_id"]


def test_a_long_choice_list_is_shortened_and_still_usable_by_id(toolbox):
    many_curricula(toolbox, 30)
    agent = DocumentAgent(tools=toolbox, llm=fake_llm(), conversation_store=ConversationStore(db_path=":memory:"))
    result = agent.process_request({"request_id": "r", "user_input": "Generate a participant guide",
                                    "context": {"conversation_id": "c"}})

    lines = result["result"]["message"].splitlines()
    assert result["result"]["kind"] == "clarify" and "… and 10 more" in lines[-1]
    assert sum(1 for line in lines if re.match(r"^\d+\. ", line)) == 20


def test_document_counts_and_paging_do_not_depend_on_a_row_limit(toolbox):
    curriculum_id, _ = make_document(toolbox)
    for _ in range(29):
        make_document(toolbox, curriculum_id=curriculum_id)

    assert toolbox.store.counts() == {"draft": 0, "pending_review": 30, "approved": 0}
    pages = [toolbox.store.list(limit=12, offset=offset) for offset in (0, 12, 24)]
    ids = [row["document_id"] for page in pages for row in page]
    assert [len(p) for p in pages] == [12, 12, 6] and len(ids) == len(set(ids)) == 30


def test_the_mcp_tool_reports_total_and_has_more(toolbox):
    many_curricula(toolbox, 3)
    result = toolbox.list_approved_curricula(limit=2)

    assert (result["total"], result["has_more"], result["offset"], len(result["curricula"])) == (3, True, 0, 2)


# --------------------------------------------------------------------
# 5. The extra fields are checked by the saving tool itself
# --------------------------------------------------------------------

@pytest.mark.parametrize("extra, code", [
    ({"price": "100"}, "extra_unknown"),
    ({"organization": "x" * 500}, "extra_too_long"),
    ("not an object", "extra_invalid"),
    ({"training_format": "  Blended  "}, "extra_invalid"),
    ({"training_format": "Line one\nline two"}, "extra_invalid"),
])
def test_bad_extra_information_makes_a_draft_with_an_error(toolbox, extra, code):
    curriculum_id, document_id = make_document(toolbox)
    content = toolbox.store.get(document_id)["content"]
    content["extra"] = extra

    result = toolbox.save_document_draft(content)

    assert result["status"] == "ok" and result["review_status"] == "draft"
    assert code in [e["code"] for e in result["validation"]["errors"]]
    edit = toolbox.save_admin_edit(content, document_id, ADMIN)   # the administrators' path checks it too
    assert edit["review_status"] == "draft" and code in [e["code"] for e in edit["validation"]["errors"]]
    assert curriculum_id


def test_good_extra_information_is_accepted(toolbox):
    curriculum_id, document_id = make_document(toolbox)
    record = toolbox.get_approved_curriculum(curriculum_id)["curriculum"]
    template = templates.TEMPLATES["course_description"]
    content = compose(template, record, "en", template.parts_schema.model_validate(parts_answers()["DescriptionParts"]),
                      {"training_format": "Blended", "organization": "Acme"})

    assert toolbox.save_document_draft(content)["review_status"] == "pending_review"


# --------------------------------------------------------------------
# 6 and 7. Admin service: errors are raised, approved documents are edited as new versions
# --------------------------------------------------------------------

@pytest.fixture
def system(tmp_path):
    toolbox = in_process_document_tools(tmp_path / "svc")
    orchestrator = Orchestrator(excluded_agents=("rex_assistant",))
    agent = register_document_generation(orchestrator, ConversationStore(db_path=":memory:"))
    agent._tools, agent._llm = toolbox, fake_llm()
    admin = AdminService(orchestrator_factory=lambda: orchestrator, passcode=PASSCODE, curriculum_store=toolbox.curricula,
                         curriculum_documents_dir=str(tmp_path / "cpdfs"), document_store=toolbox.store,
                         document_output_dir=str(toolbox.documents_dir))

    class System:
        pass

    result = System()
    result.toolbox, result.admin, result.token = toolbox, admin, admin.login(PASSCODE)
    return result


def test_listing_approved_curricula_raises_instead_of_reporting_none(system, monkeypatch):
    approved_curriculum(system.toolbox)
    assert system.admin.list_approved_curricula(system.token)["total"] == 1

    monkeypatch.setattr(DocumentToolbox, "list_approved_curricula",
                        lambda self, limit=50, offset=0: {"status": "error", "message": "database is locked"})

    with pytest.raises(ValueError, match="database is locked"):
        system.admin.list_approved_curricula(system.token)


def first_editable(content):
    return next((s["key"], i) for s in content["sections"] for i, b in enumerate(s["blocks"])
                if b["source"] in ("generated", "admin") and b["type"] in ("paragraph", "bullets"))


def test_editing_an_approved_document_makes_a_new_version_and_keeps_the_approved_one(system):
    admin, token, toolbox = system.admin, system.token, system.toolbox
    curriculum_id, document_id = make_document(toolbox)
    admin.approve_document(token, document_id, 1)
    approved_v1 = admin.document_pdf(token, document_id, 1)
    content_v1 = admin.get_document(token, document_id, 1)["document"]["content"]
    section, index = first_editable(content_v1)

    saved = admin.edit_document(token, document_id, [{"section": section, "block": index, "value": "A better summary."}])

    assert (saved["version"], saved["review_status"]) == (2, "pending_review")
    head = toolbox.store.get(document_id)
    assert (head["status"], head["current_version"], head["approved_version"]) == ("pending_review", 2, 1)
    assert toolbox.store.get_approved(document_id)["version"] == 1               # consumers still get v1
    kept = admin.get_document(token, document_id, 1)["document"]
    assert kept["version_status"] == "approved" and kept["content"] == content_v1
    assert admin.document_pdf(token, document_id, 1) == approved_v1             # same approved file and bytes
    assert "A better summary." in str(admin.get_document(token, document_id, 2)["document"]["content"])
    assert admin.document_pdf(token, document_id, 2)["approved"] is False        # v2 only has a review copy

    admin.approve_document(token, document_id, 2, "ok")
    assert toolbox.store.get_approved(document_id)["version"] == 2
    assert admin.document_pdf(token, document_id, 1) == approved_v1             # v1 is still preserved
    assert admin.document_pdf(token, document_id, 2)["approved"] is True
    assert [v["status"] for v in toolbox.store.versions(document_id)] == ["approved", "approved"]
    assert curriculum_id


def test_an_approved_edit_still_cannot_touch_curriculum_text(system):
    _, document_id = make_document(system.toolbox)
    system.admin.approve_document(system.token, document_id, 1)
    content = system.admin.get_document(system.token, document_id)["document"]["content"]
    locked = next((s["key"], i) for s in content["sections"] for i, b in enumerate(s["blocks"])
                  if b["source"] == "curriculum" and b["type"] == "paragraph")

    with pytest.raises(ValueError, match="cannot be edited"):
        system.admin.edit_document(system.token, document_id, [{"section": locked[0], "block": locked[1], "value": "x"}])
    assert system.toolbox.store.get(document_id)["current_version"] == 1


def test_get_approved_is_none_until_a_version_is_approved(system):
    _, document_id = make_document(system.toolbox)
    assert system.toolbox.store.get_approved(document_id) is None
    system.admin.approve_document(system.token, document_id, 1)
    assert system.toolbox.store.get_approved(document_id)["version_status"] == "approved"


# --------------------------------------------------------------------
# 6. Streamlit: the page redraws after (re)generation; errors are shown; approved documents can be edited
# --------------------------------------------------------------------

def default_toolbox():
    return DocumentToolbox(store=DocumentStore(), curricula=CurriculumStore())


def test_regenerating_redraws_the_page_with_the_new_version(env):
    toolbox = default_toolbox()
    curriculum_id, document_id = make_document(toolbox)

    def regenerate(message, context=None):
        request = context["document_request"]
        content = toolbox.store.get(request["document_id"])["content"]
        saved = toolbox.save_document_draft(content, document_id=request["document_id"])
        toolbox.render_document_pdf(document_id, saved["version"])
        return {"status": "success", "result": [{"agent_name": "document_generation", "status": "success",
                                                 "result": {"kind": "document", "message": "Regenerated as version 2."}}]}

    env["orchestrator"].handle_request.side_effect = regenerate
    at = unlock(start())
    assert not [s for s in at.selectbox if s.key == f"doc_version_{document_id}_2"]

    at.button(key=f"doc_regen_{document_id}").click().run()

    assert not at.exception, at.exception
    version_box = next(s for s in at.selectbox if s.key == f"doc_version_{document_id}_2")   # key follows the version
    assert version_box.value == 2
    assert any("Regenerated as version 2" in s.value for s in at.success)
    assert curriculum_id


def test_a_failure_listing_curricula_is_shown_as_an_error(env, monkeypatch):
    approved_curriculum(default_toolbox())
    monkeypatch.setattr(DocumentToolbox, "list_approved_curricula",
                        lambda self, limit=50, offset=0: {"status": "error", "message": "database is locked"})

    at = unlock(start())

    assert not at.exception
    assert any("could not be listed" in e.value and "database is locked" in e.value for e in at.error)
    assert not any("No curriculum is approved yet" in i.value for i in at.info)


def test_an_approved_current_version_can_be_edited_in_the_tab(env):
    toolbox = default_toolbox()
    _, document_id = make_document(toolbox)
    at = unlock(start())
    at.button(key=f"doc_approve_{document_id}").click().run()
    assert toolbox.store.get(document_id)["approved_version"] == 1

    assert any("stays the approved one" in i.value for i in at.info)
    key = f"doc_{document_id}_1_summary_0"
    at.text_area(key=key).input("Edited after approval.")
    at.button(key=f"doc_save_{document_id}").click().run()

    assert not at.exception, at.exception
    head = toolbox.store.get(document_id)
    assert (head["current_version"], head["status"], head["approved_version"]) == (2, "pending_review", 1)
    assert "Edited after approval." in str(toolbox.store.get(document_id, 2)["content"])
    assert "Edited after approval." not in str(toolbox.store.get(document_id, 1)["content"])


def test_the_document_list_shows_that_an_older_version_is_approved(env):
    toolbox = default_toolbox()
    _, document_id = make_document(toolbox)
    toolbox.approve_document(document_id, 1, ADMIN)
    toolbox.save_admin_edit(toolbox.store.get(document_id)["content"], document_id, ADMIN)

    at = unlock(start())

    assert not at.exception
    assert any("approved v1" in e.label for e in at.expander)
