"""
Agent 4 (Educational Content & Document Generation): templates, request
reading, validation, composition, storage, tools, PDFs and the generation
workflow. Everything uses temporary databases and PDF folders and a fake
language model (tests/document_fakes.py); no Ollama, no network.
"""

import copy
import sqlite3

import pytest

from agents.conversation_store import ConversationStore
from agents.curriculum_research.llm import CurriculumModelError
from agents.document_generation import schemas, templates, understanding as rules
from agents.document_generation.composer import compose
from agents.document_generation.document_agent import DocumentAgent
from agents.document_generation.document_store import DocumentStore, DocumentStoreError
from agents.document_generation.pdf import render_document_pdf
from agents.document_generation.validation import (
    check_parts,
    clean_extra,
    missing_curriculum_fields,
    placeholder_parts,
    text_problems,
    validate_content,
)
from tests.curriculum_fixtures import valid_curriculum
from tests.document_fakes import (
    ADMIN,
    approved_curriculum,
    fake_llm,
    in_process_document_tools,
    parts_answers,
    save_curriculum,
)

PARTS_SCHEMAS = {"course_description": schemas.DescriptionParts, "course_proposal": schemas.ProposalParts,
                 "participant_guide": schemas.GuideParts, "capstone_brief": schemas.CapstoneParts}
SUPPORTED = ["course_description", "course_proposal", "participant_guide", "capstone_brief",
             "outcomes_assessment_plan"]
# Added in phase 2 (their own tests are in tests/test_document_new_types.py).
NEW_TYPES = ["instructor_guide", "presentation_outline", "question_bank", "assessment_rubric"]


@pytest.fixture
def toolbox(tmp_path):
    return in_process_document_tools(tmp_path / "docs")


@pytest.fixture
def curriculum_id(toolbox):
    return approved_curriculum(toolbox)


def build_content(toolbox, curriculum_id, doc_type, language="en", extra=None):
    """Compose a document from the approved curriculum with the canned narrative parts."""
    record = toolbox.get_approved_curriculum(curriculum_id)["curriculum"]
    template = templates.TEMPLATES[doc_type]
    schema = template.parts_schema
    parts = schema.model_validate(parts_answers(language)[schema.__name__]) if schema else None
    return compose(template, record, language, parts, extra)


def make_agent(toolbox, llm=None, store=None):
    return DocumentAgent(conversation_store=store, tools=toolbox, llm=llm or fake_llm())


def ask(agent, text, conversation="c1", **context):
    return agent.process_request({"request_id": "r1", "user_input": text,
                                  "context": {"conversation_id": conversation, **context}})


# --------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------

def test_the_supported_types_and_their_templates():
    assert templates.supported_types() == SUPPORTED + NEW_TYPES
    for doc_type, template in templates.TEMPLATES.items():
        assert set(template.sections) <= set(templates.SECTION_HEADINGS)
        assert template.sections[-1] == "traceability"
        assert template.label["en"] and template.label["ar"]


def test_unsupported_types_are_listed_with_a_reason_and_not_as_templates():
    # only the curriculum PDF is outside the generated documents: it is Agent 3's own PDF, offered as a download
    assert set(templates.UNSUPPORTED) == {"curriculum_pdf"}
    assert not set(templates.UNSUPPORTED) & set(templates.TEMPLATES)
    assert all(v["reason"] for v in templates.UNSUPPORTED.values())


# --------------------------------------------------------------------
# Reading chat requests
# --------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Generate a course description for curriculum CUR-0123456789ab", "course_description"),
    ("Create a participant guide for the approved SQL curriculum", "participant_guide"),
    ("Generate a course proposal using the selected curriculum", "course_proposal"),
    ("Write the capstone project brief", "capstone_brief"),
    ("Prepare the learning outcomes and assessment plan", "outcomes_assessment_plan"),
    ("Create a question bank", "question_bank"),
    ("أنشئ دليل المتدرب", "participant_guide"),
    ("اكتب وصف الدورة", "course_description"),
])
def test_document_requests_are_recognised(text, expected):
    assert rules.find_document_type(text) == expected
    assert rules.is_document_request(text)


@pytest.mark.parametrize("text", [
    "Create a curriculum about SQL, 30 hours",
    "Make a PDF of the curriculum",
    "Find education tenders in Egypt",
    "What is the course description of Python?",   # a question, no request verb
    "Write a marketing campaign",
])
def test_other_requests_are_not_document_requests(text):
    assert not rules.is_document_request(text)


def test_curriculum_ids_languages_and_bad_references():
    assert rules.find_curriculum_id("for cur-0123456789AB please") == "CUR-0123456789ab"
    assert rules.find_curriculum_id("for curriculum 123") is None
    assert rules.find_bad_reference("course description for curriculum 123") == "123"
    assert rules.find_bad_reference("for CUR-0123456789ab") is None
    assert rules.find_language("participant guide in Arabic") == "ar"
    assert rules.find_language("دليل بالانجليزية") == "en"
    assert rules.find_language("participant guide") is None


def test_a_curriculum_is_picked_by_the_words_of_the_request():
    rows = [{"title": "SQL for Analysts", "curriculum_id": "A"}, {"title": "Machine Learning Foundations", "curriculum_id": "B"}]
    assert [r["curriculum_id"] for r in rules.match_curricula("guide for the approved SQL curriculum", rows)] == ["A"]
    assert rules.match_curricula("guide for the selected curriculum", rows) == []
    assert rules.choose_from_options("2", ["A", "B"]) == "B"
    assert rules.choose_from_options("3", ["A", "B"]) is None


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------

def test_missing_curriculum_fields_are_identified():
    content = valid_curriculum()
    content["objectives"] = []
    content["total_hours"] = None
    assert missing_curriculum_fields(content, templates.BASE_REQUIRED) == ["objectives", "total_hours"]
    assert missing_curriculum_fields(valid_curriculum(), templates.BASE_REQUIRED) == []


def test_extra_information_is_cleaned_and_checked():
    assert clean_extra({"training_format": "  Online   live "}) == ({"training_format": "Online live"}, [])
    assert clean_extra({"price": "5"})[1][0]["code"] == "extra_unknown"
    assert clean_extra({"organization": "x" * 500})[1][0]["code"] == "extra_too_long"
    assert clean_extra("nope")[1][0]["code"] == "extra_invalid"


def test_free_text_may_not_claim_accreditation_or_invent_hours():
    hours = {40.0, 20.0}
    assert text_problems("A 20 hours practical lab.", hours) == []
    assert text_problems("The course is accredited by a national body.", hours)[0]["code"] == "accreditation_claim"
    assert text_problems("The course runs for 35 hours.", hours)[0]["code"] == "hours_mismatch"
    assert text_problems("الدورة مدتها ٣٥ ساعة", hours)[0]["code"] == "hours_mismatch"
    assert text_problems("The course is not accredited.", hours) == []


def test_generated_guide_parts_must_name_every_module_once():
    template = templates.TEMPLATES["participant_guide"]
    curriculum = valid_curriculum()
    good = schemas.GuideParts.model_validate(parts_answers()["GuideParts"])
    assert check_parts(good, template, curriculum) == []
    bad = copy.deepcopy(parts_answers()["GuideParts"])
    bad["modules"] = bad["modules"][:1]
    assert check_parts(schemas.GuideParts.model_validate(bad), template, curriculum)[0]["code"] == "module_ids"


@pytest.mark.parametrize("doc_type", SUPPORTED)
def test_every_composed_document_validates_against_its_curriculum(toolbox, curriculum_id, doc_type):
    record = toolbox.get_approved_curriculum(curriculum_id)["curriculum"]
    result = validate_content(build_content(toolbox, curriculum_id, doc_type), record)

    assert result["valid"], result["errors"]


def test_text_copied_from_the_curriculum_cannot_be_changed(toolbox, curriculum_id):
    record = toolbox.get_approved_curriculum(curriculum_id)["curriculum"]
    content = build_content(toolbox, curriculum_id, "course_description")
    objectives = next(s for s in content["sections"] if s["key"] == "objectives")
    objectives["blocks"][0]["items"][0] = "Something the curriculum never said"

    result = validate_content(content, record)

    assert not result["valid"] and result["errors"][0]["code"] == "curriculum_changed"


def test_a_document_must_name_the_curriculum_version_and_language(toolbox, curriculum_id):
    record = toolbox.get_approved_curriculum(curriculum_id)["curriculum"]
    content = build_content(toolbox, curriculum_id, "course_description")
    content["curriculum_version"] = 7
    assert validate_content(content, record)["errors"][0]["code"] == "traceability"
    content = build_content(toolbox, curriculum_id, "course_description")
    content["language"] = "ar"
    assert validate_content(content, record)["errors"][0]["code"] == "language"
    assert validate_content({"document_type": "nonsense"}, record)["errors"][0]["code"] == "document_type"
    assert not validate_content("not an object", record)["valid"]


def test_removing_or_adding_blocks_is_refused(toolbox, curriculum_id):
    record = toolbox.get_approved_curriculum(curriculum_id)["curriculum"]
    content = build_content(toolbox, curriculum_id, "course_description")
    content["sections"][1]["blocks"].append({"type": "paragraph", "text": "extra", "source": "admin"})
    assert validate_content(content, record)["errors"][0]["code"] == "sections"


def test_placeholder_parts_follow_the_curriculum_modules():
    parts = placeholder_parts(templates.TEMPLATES["participant_guide"], valid_curriculum())
    assert [m.module_id for m in parts.modules] == ["M1", "M2"]
    assert placeholder_parts(templates.TEMPLATES["outcomes_assessment_plan"], valid_curriculum()) is None


# --------------------------------------------------------------------
# Composition: nothing is invented
# --------------------------------------------------------------------

def test_a_field_the_curriculum_lacks_is_reported_not_invented(toolbox, curriculum_id):
    content = build_content(toolbox, curriculum_id, "course_description")

    assert content["missing"] == [{"section": "format", "field": "training_format"}]
    fmt = next(s for s in content["sections"] if s["key"] == "format")["blocks"][0]
    assert fmt["source"] == "missing" and fmt["text"] == templates.STRINGS["en"]["not_specified"]


def test_extra_information_from_the_admin_fills_the_gap(toolbox, curriculum_id):
    content = build_content(toolbox, curriculum_id, "course_description", extra={"training_format": "Blended"})

    assert content["missing"] == []
    fmt = next(s for s in content["sections"] if s["key"] == "format")["blocks"][0]
    assert (fmt["text"], fmt["source"]) == ("Blended", "admin")


def test_the_document_copies_the_curriculum_and_keeps_traceability(toolbox, curriculum_id):
    content = build_content(toolbox, curriculum_id, "participant_guide")
    text = str(content)

    assert content["curriculum_id"] == curriculum_id and content["curriculum_version"] == 1
    for fact in ("Prepare a dataset for modelling", "Data preparation", "Models and evaluation", "Notebook"):
        assert fact in text
    trace = next(s for s in content["sections"] if s["key"] == "traceability")
    assert curriculum_id in trace["blocks"][0]["text"] and "version 1" in trace["blocks"][0]["text"]


def test_the_outcome_matrix_links_outcomes_to_modules_and_assessments(toolbox, curriculum_id):
    content = build_content(toolbox, curriculum_id, "outcomes_assessment_plan")
    matrix = next(s for s in content["sections"] if s["key"] == "outcome_matrix")["blocks"][0]

    assert matrix["rows"][0][1:] == ["M1", "A1"] and matrix["rows"][1][1:] == ["M2", "A1"]


# --------------------------------------------------------------------
# PDFs
# --------------------------------------------------------------------

@pytest.mark.parametrize("language", ["en", "ar"])
@pytest.mark.parametrize("doc_type", SUPPORTED)
def test_each_document_renders_as_a_pdf_in_both_languages(tmp_path, doc_type, language):
    toolbox = in_process_document_tools(tmp_path / language)
    cid = approved_curriculum(toolbox, valid_curriculum(language))
    content = build_content(toolbox, cid, doc_type, language)
    record = toolbox.get_approved_curriculum(cid)["curriculum"]
    saved = toolbox.store.save_version(content, validate_content(content, record))

    data, pages = render_document_pdf(toolbox.store.get(saved["document_id"]))

    assert data.startswith(b"%PDF") and pages >= 2  # cover + content


def test_a_long_document_breaks_across_pages(toolbox):
    content = valid_curriculum()
    content["modules"] = [{**content["modules"][0], "id": f"M{i}", "hours": 4, "outcome_ids": ["LO1"]} for i in range(1, 11)]
    content["total_hours"] = 40
    cid = approved_curriculum(toolbox, content)
    parts = copy.deepcopy(parts_answers())
    parts["GuideParts"]["modules"] = [{**parts["GuideParts"]["modules"][0], "module_id": f"M{i}"} for i in range(1, 11)]
    agent = make_agent(toolbox, fake_llm(**parts))

    result = ask(agent, "Generate a participant guide", document_request={"curriculum_id": cid, "document_type": "participant_guide"})

    assert result["status"] == "success" and result["result"]["pdf"]["pages"] >= 3


# --------------------------------------------------------------------
# Store: lifecycle, versions, files
# --------------------------------------------------------------------

def saved_document(toolbox, cid, doc_type="course_description"):
    return toolbox.save_document_draft(build_content(toolbox, cid, doc_type))


def test_a_valid_document_waits_for_review_and_is_never_saved_as_approved(toolbox, curriculum_id):
    saved = saved_document(toolbox, curriculum_id)

    assert saved["status"] == "ok" and saved["review_status"] == "pending_review"
    assert toolbox.store.get(saved["document_id"])["approved_version"] is None


def test_approval_needs_an_admin_identity_and_a_rendered_pdf(toolbox, curriculum_id):
    saved = saved_document(toolbox, curriculum_id)
    for reviewer in ("", "system:document_agent", None):
        with pytest.raises(DocumentStoreError):
            toolbox.approve_document(saved["document_id"], 1, reviewer)
    with pytest.raises(DocumentStoreError, match="Render the PDF"):
        toolbox.approve_document(saved["document_id"], 1, ADMIN)
    assert toolbox.store.get(saved["document_id"])["approved_version"] is None

    toolbox.render_document_pdf(saved["document_id"], 1)
    approved = toolbox.approve_document(saved["document_id"], 1, ADMIN, "ok")

    assert approved["version_status"] == "approved" and approved["approved_version"] == 1
    assert [(a["event"], a["actor"]) for a in toolbox.store.audit_trail(saved["document_id"])][-1] == ("review", ADMIN)
    with pytest.raises(DocumentStoreError):  # approving again
        toolbox.approve_document(saved["document_id"], 1, ADMIN)


def test_a_new_version_never_replaces_the_approved_one_or_its_pdf(toolbox, curriculum_id):
    saved = saved_document(toolbox, curriculum_id)
    document_id = saved["document_id"]
    toolbox.render_document_pdf(document_id, 1)
    toolbox.approve_document(document_id, 1, ADMIN)
    first_file = toolbox.document_bytes(document_id, 1)

    second = toolbox.save_document_draft(build_content(toolbox, curriculum_id, "course_description"), document_id=document_id)
    toolbox.render_document_pdf(document_id, 2)

    head = toolbox.store.get(document_id)
    assert (second["version"], head["status"], head["approved_version"]) == (2, "pending_review", 1)
    assert toolbox.store.get(document_id, 1)["version_status"] == "approved"
    assert toolbox.document_bytes(document_id, 1) == first_file
    assert toolbox.document_bytes(document_id, 2) is not None
    with pytest.raises(DocumentStoreError):   # a version keeps the one PDF it has
        toolbox.store.attach_pdf(document_id, 1, "x.pdf", b"x", 1)


def test_rendering_twice_returns_the_same_file_and_never_overwrites(toolbox, curriculum_id):
    document_id = saved_document(toolbox, curriculum_id)["document_id"]
    first = toolbox.render_document_pdf(document_id)["document"]
    path = toolbox.documents_dir / first["filename"]
    stamp = path.stat().st_mtime_ns
    second = toolbox.render_document_pdf(document_id)["document"]

    assert second["sha256"] == first["sha256"] and path.stat().st_mtime_ns == stamp
    assert first["filename"] == f"{document_id}_v1_en.pdf"


def test_a_new_version_must_match_the_document(toolbox, curriculum_id):
    document_id = saved_document(toolbox, curriculum_id)["document_id"]
    other = build_content(toolbox, curriculum_id, "course_proposal")
    result = toolbox.save_document_draft(other, document_id=document_id)

    assert result["status"] == "invalid_request" and "same curriculum, document type" in result["message"]


def test_invalid_document_ids_and_unknown_documents(toolbox):
    for bad in ("../../etc/passwd", "GDOC-xyz", "", None, 7):
        assert toolbox.get_document(bad)["status"] == "invalid_request"
    assert toolbox.get_document("GDOC-0123456789ab")["status"] == "not_found"
    assert toolbox.render_document_pdf("GDOC-0123456789ab")["status"] == "not_found"
    with pytest.raises(DocumentStoreError):
        toolbox.document_bytes("../x")


def test_the_stored_file_name_is_built_from_ids_only(toolbox, curriculum_id):
    document_id = saved_document(toolbox, curriculum_id)["document_id"]
    toolbox.render_document_pdf(document_id)
    with sqlite3.connect(toolbox.store.db_path) as db:   # a tampered row cannot make a download leave the folder
        db.execute("UPDATE document_versions SET filename = '../../secret.pdf'")
    assert toolbox.document_bytes(document_id) is None


def test_listing_filters_by_status_and_rejects_unknown_ones(toolbox, curriculum_id):
    saved_document(toolbox, curriculum_id)
    assert len(toolbox.list_documents(status="pending_review")["documents"]) == 1
    assert toolbox.list_documents(status="approved")["documents"] == []
    assert toolbox.list_documents(status="bogus")["status"] == "invalid_request"


# --------------------------------------------------------------------
# Curriculum approval is the gate
# --------------------------------------------------------------------

def test_only_approved_curricula_are_offered_and_used(toolbox, curriculum_id):
    pending = save_curriculum(toolbox)["curriculum_id"]
    draft = save_curriculum(toolbox, valid=False)["curriculum_id"]
    changes = save_curriculum(toolbox)["curriculum_id"]
    toolbox.curricula.request_changes(changes, 1, ADMIN, "fix")

    assert [c["curriculum_id"] for c in toolbox.list_approved_curricula()["curricula"]] == [curriculum_id]
    for unapproved in (pending, draft, changes):
        result = toolbox.get_approved_curriculum(unapproved)
        assert result["status"] == "not_approved" and "needs to be approved" in result["message"]
        assert toolbox.save_document_draft(build_content(toolbox, curriculum_id, "course_description") | {
            "curriculum_id": unapproved})["status"] == "not_approved"
    assert toolbox.get_approved_curriculum("CUR-0123456789ab")["status"] == "not_found"
    assert toolbox.get_approved_curriculum("nope")["status"] == "invalid_request"
    assert toolbox.store.list() == []


def test_a_newer_unapproved_revision_does_not_change_the_approved_source(toolbox, curriculum_id):
    revised = valid_curriculum()
    revised["title"] = "Machine Learning Foundations, revised"
    toolbox.curricula.save_version(revised, {"valid": True, "errors": []}, [], curriculum_id=curriculum_id)

    result = toolbox.get_approved_curriculum(curriculum_id)

    assert result["status"] == "ok" and result["curriculum"]["version"] == 1
    assert result["curriculum"]["content"]["title"] == "Machine Learning Foundations"
    assert result["newer_unapproved_version"] is True


# --------------------------------------------------------------------
# The workflow (agent)
# --------------------------------------------------------------------

@pytest.mark.parametrize("doc_type", SUPPORTED)
def test_generate_every_type_from_the_form(toolbox, curriculum_id, doc_type):
    agent = make_agent(toolbox)

    result = ask(agent, "x", document_request={"curriculum_id": curriculum_id, "document_type": doc_type})

    assert result["status"] == "success", result
    body = result["result"]
    assert body["kind"] == "document" and body["review_status"] == "pending_review"
    assert body["curriculum_id"] == curriculum_id and body["curriculum_version"] == 1
    assert body["pdf"]["pages"] >= 2
    stored = toolbox.store.get(body["document_id"])
    assert stored["filename"] and toolbox.document_bytes(body["document_id"]).startswith(b"%PDF")


def test_the_form_choice_beats_keywords_in_the_text(toolbox, curriculum_id):
    result = ask(make_agent(toolbox), "Generate a participant guide",
                 document_request={"curriculum_id": curriculum_id, "document_type": "course_proposal"})

    assert result["result"]["document_type"] == "course_proposal"


def test_chat_request_with_a_curriculum_id(toolbox, curriculum_id):
    result = ask(make_agent(toolbox), f"Generate a course description for {curriculum_id}")

    assert result["status"] == "success" and result["result"]["document_type"] == "course_description"
    assert "Training format" in result["result"]["message"]   # the gap is called out, by its name
    assert "training_format" not in result["result"]["message"]
    assert result["result"]["missing"] == [{"section": "format", "field": "training_format"}]


def test_chat_request_names_the_curriculum_by_topic(toolbox, curriculum_id):
    result = ask(make_agent(toolbox), "Create a participant guide for the approved machine learning curriculum")

    assert result["status"] == "success" and result["result"]["curriculum_id"] == curriculum_id


def test_chat_request_without_a_curriculum_offers_the_approved_ones_and_continues(toolbox, curriculum_id):
    other = approved_curriculum(toolbox, valid_curriculum("en", 30) | {"title": "Cloud Basics"})
    store = ConversationStore(db_path=":memory:")
    agent = make_agent(toolbox, store=store)

    first = ask(agent, "Generate a course proposal using the selected curriculum")
    assert first["status"] == "success" and first["result"]["kind"] == "clarify"
    assert curriculum_id in first["result"]["message"] and other in first["result"]["message"]
    assert toolbox.store.list() == []   # nothing generated yet
    assert agent.wants_follow_up({"user_input": "2", "context": {"conversation_id": "c1"}})
    assert not agent.wants_follow_up({"user_input": "thanks", "context": {"conversation_id": "c1"}})

    second = ask(agent, "2")
    assert second["status"] == "success" and second["result"]["document_type"] == "course_proposal"
    assert second["result"]["curriculum_id"] in (curriculum_id, other)
    assert not agent.wants_follow_up({"user_input": "2", "context": {"conversation_id": "c1"}})


def test_a_fake_curriculum_number_is_explained(toolbox, curriculum_id):
    result = ask(make_agent(toolbox), "Generate a course description for curriculum 123")

    assert result["result"]["kind"] == "clarify" and "not a curriculum id" in result["result"]["message"]


def test_an_unapproved_curriculum_is_refused_with_the_reason(toolbox):
    pending = save_curriculum(toolbox)["curriculum_id"]

    result = ask(make_agent(toolbox), f"Generate a course description for {pending}")

    assert result["status"] == "error" and result["result"]["kind"] == "not_approved"
    assert "needs to be approved" in result["error_message"] and "pending_review" in result["error_message"]
    assert toolbox.store.list() == []


def test_no_approved_curriculum_at_all(toolbox):
    save_curriculum(toolbox)

    result = ask(make_agent(toolbox), "Generate a participant guide")

    assert result["status"] == "error" and "No curriculum is approved" in result["error_message"]


def test_unknown_curriculum_and_bad_input_are_errors(toolbox, curriculum_id):
    agent = make_agent(toolbox)
    assert ask(agent, "Generate a course description for CUR-0123456789ab")["result"]["kind"] == "not_found"
    for form in ({"curriculum_id": curriculum_id, "document_type": "course_description", "language": "fr"},
                 {"curriculum_id": curriculum_id, "document_type": "course_description", "extra": {"price": "1"}},
                 {"curriculum_id": curriculum_id, "document_type": "course_description", "document_id": "../x"},
                 {"curriculum_id": curriculum_id, "document_type": "nonsense"}):
        assert ask(agent, "x", document_request=form)["status"] == "error", form
    assert ask(agent, "x", document_request="not a dict")["status"] == "error"
    assert ask(agent, "")["status"] == "error"
    assert toolbox.store.list() == []


@pytest.mark.parametrize("doc_type", sorted(templates.UNSUPPORTED))
def test_unsupported_types_are_explained_not_pretended(toolbox, curriculum_id, doc_type):
    result = ask(make_agent(toolbox), "x", document_request={"curriculum_id": curriculum_id, "document_type": doc_type})

    assert result["status"] == "error" and result["result"]["kind"] == "unsupported"
    assert result["error_message"] == templates.UNSUPPORTED[doc_type]["reason"]


def test_which_document_is_asked_when_the_type_is_missing(toolbox, curriculum_id):
    agent = make_agent(toolbox)
    first = ask(agent, "Generate a document")  # routed here by the orchestrator's document words
    assert first["result"]["kind"] == "clarify" and "Course Description" in first["result"]["message"]


def test_the_document_language_is_the_curriculums(toolbox, curriculum_id):
    result = ask(make_agent(toolbox), "x", document_request={
        "curriculum_id": curriculum_id, "document_type": "course_description", "language": "ar"})

    assert result["status"] == "error" and "curriculum's own language" in result["error_message"]


def test_an_arabic_curriculum_gives_an_arabic_document(tmp_path):
    toolbox = in_process_document_tools(tmp_path / "ar")
    cid = approved_curriculum(toolbox, valid_curriculum("ar"))

    result = ask(make_agent(toolbox, fake_llm("ar")), "x", ui="ar", document_request={
        "curriculum_id": cid, "document_type": "participant_guide"}, language="ar")

    assert result["status"] == "success", result
    stored = toolbox.store.get(result["result"]["document_id"])
    assert stored["language"] == "ar" and "تجهيز البيانات" in str(stored["content"])
    assert "تم إنشاء" in result["result"]["message"]


def test_a_curriculum_missing_required_information_is_not_filled_in(toolbox):
    content = valid_curriculum()
    content["objectives"] = []
    cid = approved_curriculum(toolbox, content)

    result = ask(make_agent(toolbox), "x", document_request={"curriculum_id": cid, "document_type": "course_description"})

    assert result["status"] == "error" and result["result"]["kind"] == "missing_information"
    assert "objectives" in result["error_message"] and toolbox.store.list() == []


def test_a_capstone_brief_needs_a_capstone(toolbox):
    content = valid_curriculum()
    content["capstone"], content["capstone_omitted_reason"] = None, "Not requested"
    cid = approved_curriculum(toolbox, content)

    result = ask(make_agent(toolbox), "x", document_request={"curriculum_id": cid, "document_type": "capstone_brief"})

    assert result["result"]["kind"] == "missing_information" and "capstone" in result["error_message"]


def test_extra_information_reaches_the_document_and_clears_the_gap(toolbox, curriculum_id):
    result = ask(make_agent(toolbox), "x", document_request={
        "curriculum_id": curriculum_id, "document_type": "course_proposal",
        "extra": {"organization": "Acme Training", "training_format": "Blended"}})

    body = result["result"]
    stored = toolbox.store.get(body["document_id"])["content"]
    assert "Acme Training" in str(stored) and stored["missing"] == []


def test_a_revised_curriculum_is_mentioned(toolbox, curriculum_id):
    toolbox.curricula.save_version(valid_curriculum(), {"valid": True, "errors": []}, [], curriculum_id=curriculum_id)

    result = ask(make_agent(toolbox), "x", document_request={"curriculum_id": curriculum_id, "document_type": "course_description"})

    assert result["status"] == "success" and "newer version" in result["result"]["message"]


def test_regenerating_makes_a_new_version_of_the_same_document(toolbox, curriculum_id):
    agent = make_agent(toolbox)
    form = {"curriculum_id": curriculum_id, "document_type": "course_description"}
    first = ask(agent, "x", document_request=form)["result"]
    second = ask(agent, "x", document_request={**form, "document_id": first["document_id"]})["result"]

    assert second["document_id"] == first["document_id"] and second["version"] == 2
    assert len(toolbox.store.versions(first["document_id"])) == 2


# -- failures are reported, not hidden ------------------------------------

def test_a_model_failure_saves_nothing_and_says_so(toolbox, curriculum_id):
    llm = fake_llm(DescriptionParts=[RuntimeError("down")] * 4)

    result = ask(make_agent(toolbox, llm), "x", document_request={"curriculum_id": curriculum_id, "document_type": "course_description"})

    assert result["status"] == "error" and result["result"]["kind"] == "generation_failed"
    assert "Nothing was saved" in result["error_message"] and toolbox.store.list() == []


def test_one_repair_attempt_fixes_an_unsupported_claim(toolbox, curriculum_id):
    bad = {**parts_answers()["DescriptionParts"], "summary": "This course is accredited by a national body."}
    llm = fake_llm(DescriptionParts=[bad, parts_answers()["DescriptionParts"]])

    result = ask(make_agent(toolbox, llm), "x", document_request={"curriculum_id": curriculum_id, "document_type": "course_description"})

    assert result["status"] == "success"
    assert len(llm.prompts("DescriptionParts")) == 2 and "previous answer had these problems" in llm.prompts("DescriptionParts")[1]
    assert "accredited" not in str(toolbox.store.get(result["result"]["document_id"])["content"])


def test_parts_that_keep_failing_are_not_saved(toolbox, curriculum_id):
    bad = {**parts_answers()["DescriptionParts"], "summary": "A course of 99 hours."}
    llm = fake_llm(DescriptionParts=[bad, bad])

    result = ask(make_agent(toolbox, llm), "x", document_request={"curriculum_id": curriculum_id, "document_type": "course_description"})

    assert result["status"] == "error" and result["result"]["kind"] == "validation_failed"
    assert "99 hours" in result["error_message"] and toolbox.store.list() == []


def test_wrong_module_ids_trigger_the_repair_and_then_pass(toolbox, curriculum_id):
    wrong = copy.deepcopy(parts_answers()["GuideParts"])
    wrong["modules"][0]["module_id"] = "M9"
    llm = fake_llm(GuideParts=[wrong, parts_answers()["GuideParts"]])

    result = ask(make_agent(toolbox, llm), "x", document_request={"curriculum_id": curriculum_id, "document_type": "participant_guide"})

    assert result["status"] == "success" and "module id" in llm.prompts("GuideParts")[1]


def test_a_pdf_failure_is_an_error_but_the_saved_draft_is_kept(toolbox, curriculum_id, monkeypatch):
    import agents.document_generation.pdf as pdf_module

    def broken(record):
        raise RuntimeError("font missing")

    monkeypatch.setattr(pdf_module, "render_document_pdf", broken)
    result = ask(make_agent(toolbox), "x", document_request={"curriculum_id": curriculum_id, "document_type": "course_description"})

    assert result["status"] == "error" and result["result"]["kind"] == "render_failed"
    assert "PDF" in result["error_message"] and "was saved" in result["error_message"]
    assert len(toolbox.store.list()) == 1 and toolbox.store.list()[0]["status"] == "pending_review"


def test_a_tool_server_failure_becomes_an_error_answer(toolbox, curriculum_id):
    class Broken:
        def __getattr__(self, name):
            raise ConnectionError("server down")

    result = ask(DocumentAgent(tools=Broken(), llm=fake_llm()), "x",
                 document_request={"curriculum_id": curriculum_id, "document_type": "course_description"})

    assert result["status"] == "error" and "ConnectionError" in result["error_message"]


def test_the_agent_is_built_lazily_and_follows_the_response_contract(toolbox, curriculum_id):
    agent = DocumentAgent(tools=toolbox, llm=fake_llm())
    assert agent._graph is None
    result = ask(agent, "x", document_request={"curriculum_id": curriculum_id, "document_type": "capstone_brief"})

    assert set(result) == {"request_id", "agent_name", "status", "result", "error_message"}
    assert result["agent_name"] == "document_generation" and result["request_id"] == "r1"
