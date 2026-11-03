"""
Phase 2 document types: Instructor Guide, Presentation Outline, Question Bank and Assessment Rubrics.

Each type is checked for composition (curriculum text locked, model text editable), validation of the model's
output, English and Arabic, PDF rendering, and the whole lifecycle (generate -> edit -> new version -> approve ->
download). A fake model answers from the prompt (tests/document_fakes.py); no Ollama, no network, temporary storage.
"""

import copy

import pytest
from pypdf import PdfReader

from agents.admin_service import AdminService
from agents.conversation_store import ConversationStore
from agents.document_generation import templates, units
from agents.document_generation.composer import compose
from agents.document_generation.document_agent import DocumentAgent
from agents.document_generation.document_store import DocumentStoreError
from agents.document_generation.pdf import render_document_pdf
from agents.document_generation.validation import check_parts, check_unit, validate_content
from tests.curriculum_fakes import FakeLLM
from tests.curriculum_fixtures import valid_curriculum
from tests.document_fakes import (
    ADMIN,
    NEW_TYPES,
    approved_curriculum,
    container_answer,
    in_process_document_tools,
    new_type_answers,
)

PASSCODE = "pass-123"


@pytest.fixture
def toolbox(tmp_path):
    return in_process_document_tools(tmp_path / "new")


def curriculum_record(toolbox, language="en", content=None):
    cid = approved_curriculum(toolbox, content or valid_curriculum(language))
    return cid, toolbox.get_approved_curriculum(cid)["curriculum"]


def composed(toolbox, doc_type, language="en", content=None, extra=None):
    cid, record = curriculum_record(toolbox, language, content)
    template = templates.TEMPLATES[doc_type]
    parts = template.parts_schema.model_validate(container_answer(doc_type, language, record["content"]))
    return cid, record, compose(template, record, language, parts, extra)


def fake(language="en", **overrides):
    answers = {**new_type_answers(language), **overrides}
    return FakeLLM({name: value if isinstance(value, list) else [value] * 20 for name, value in answers.items()})


def agent_for(toolbox, llm, store=None):
    return DocumentAgent(conversation_store=store, tools=toolbox, llm=llm)


def generate(agent, cid, doc_type, **context):
    return agent.process_request({"request_id": "r", "user_input": "x", "context": {
        "conversation_id": "c", **context, "document_request": {"curriculum_id": cid, "document_type": doc_type}}})


def blocks_of(content, key):
    return next(s for s in content["sections"] if s["key"] == key)["blocks"]


def headings(content, key):
    return [b["text"] for b in blocks_of(content, key) if b["type"] == "heading"]


def first_editable(content, role=None):
    return next((s["key"], i) for s in content["sections"] for i, b in enumerate(s["blocks"])
                if b["source"] in ("generated", "admin") and b["type"] in ("paragraph", "bullets") and (role is None or b.get("role") == role))


@pytest.fixture
def admin_system(toolbox, tmp_path):
    from agents.document_generation.registration import register_document_generation
    from agents.orchestrator import Orchestrator

    orchestrator = Orchestrator(excluded_agents=("rex_assistant",))
    agent = register_document_generation(orchestrator, ConversationStore(db_path=":memory:"))
    agent._tools = toolbox
    admin = AdminService(orchestrator_factory=lambda: orchestrator, passcode=PASSCODE, curriculum_store=toolbox.curricula,
                         curriculum_documents_dir=str(tmp_path / "cpdfs"), document_store=toolbox.store,
                         document_output_dir=str(toolbox.documents_dir))

    class System:
        pass

    system = System()
    system.toolbox, system.agent, system.admin, system.token = toolbox, agent, admin, admin.login(PASSCODE)
    return system


def lifecycle(system, doc_type, language="en", llm=None):
    """Generate through the real orchestrator and agent, then edit, approve and download (the whole lifecycle)."""
    cid = approved_curriculum(system.toolbox, valid_curriculum(language))
    system.agent._llm = llm or fake(language)
    response = system.admin.generate_document(system.token, cid, doc_type, context={"preferred_language": language})
    result = response["result"][0]
    assert result["status"] == "success", result
    document_id = result["result"]["document_id"]
    admin, token = system.admin, system.token
    document = admin.get_document(token, document_id)["document"]
    assert document["version_status"] == "pending_review" and not document["validation"]["errors"]
    section, index = first_editable(document["content"])
    edited = admin.edit_document(token, document_id, [{"section": section, "block": index, "value": "Edited by the administrator."}])
    assert (edited["version"], edited["review_status"]) == (2, "pending_review")
    admin.approve_document(token, document_id, 2, "ok")
    pdf = admin.document_pdf(token, document_id)
    assert pdf["approved"] and pdf["data"].startswith(b"%PDF") and pdf["filename"].endswith(f"_v2_{language}_approved.pdf")
    assert admin.get_document(token, document_id, 1)["document"]["version_status"] == "pending_review"   # v1 preserved
    return document_id


# ====================================================================
# Engine: documents written one unit at a time
# ====================================================================

def test_the_four_new_types_are_registered_with_their_unit_schemas():
    for doc_type in NEW_TYPES:
        template = templates.TEMPLATES[doc_type]
        assert template.unit_schema is not None and template.forbid_minutes
        assert template.sections[-1] == "traceability" and set(template.sections) <= set(templates.SECTION_HEADINGS)
    assert templates.TEMPLATES["assessment_rubric"].unit_field == "rubrics"
    assert templates.UNSUPPORTED.keys() == {"curriculum_pdf"}


def test_minute_timings_are_rejected_only_for_the_new_types():
    from agents.document_generation.validation import text_problems

    assert text_problems("Spend 45 minutes here", {40.0}, minutes=True)[0]["code"] == "minutes_unsupported"
    assert text_problems("خصص 20 دقيقة", {40.0}, minutes=True)[0]["code"] == "minutes_unsupported"
    assert text_problems("Spend 45 minutes here", {40.0}) == []          # the five earlier types are unchanged
    assert text_problems("Spend the module's 20 hours here", {40.0, 20.0}, minutes=True) == []


# ====================================================================
# Instructor Guide
# ====================================================================

@pytest.mark.parametrize("language", ["en", "ar"])
def test_the_instructor_guide_composes_and_validates(toolbox, language):
    _, record, content = composed(toolbox, "instructor_guide", language)

    assert validate_content(content, record)["valid"]
    assert [s["key"] for s in content["sections"]] == list(templates.TEMPLATES["instructor_guide"].sections)


def test_instructor_only_guidance_is_separated_from_participant_content(toolbox):
    _, record, content = composed(toolbox, "instructor_guide")
    s = templates.STRINGS["en"]
    teaching = blocks_of(content, "module_teaching")
    by_heading = {}
    current = None
    for block in teaching:
        if block["type"] == "heading":
            current = block["text"]
        else:
            by_heading.setdefault(current, []).append(block)

    assert blocks_of(content, "instructor_note")[0]["text"] == s["instructor_only_notice"]
    # participant-facing activities are the curriculum's own words (locked); instructor guidance is model text (editable)
    assert all(b["source"] == "curriculum" for b in by_heading[s["participant_activities"]])
    for label in ("teaching_methods", "instructor_activities", "instructor_notes", "discussion_questions"):
        assert all(b["source"] == "generated" for b in by_heading[s[label]]), label
    assert "(instructor only)" in s["instructor_notes"] and "(instructor guidance)" in s["teaching_methods"]


def test_the_session_timing_comes_from_the_curriculum_hours_never_the_model(toolbox):
    _, record, content = composed(toolbox, "instructor_guide")
    text = " ".join(b.get("text", "") for b in blocks_of(content, "module_teaching"))

    assert "Allow the module's 20 hours" in text and "minute" not in text.lower()
    tampered = copy.deepcopy(content)
    block = next(b for b in blocks_of(tampered, "module_teaching") if b.get("text", "").startswith("Allow the module's"))
    block["text"] = "Allow the module's 3 hours. Cover the topics in this order: Cleaning data."
    assert validate_content(tampered, record)["errors"][0]["code"] == "curriculum_changed"


def test_materials_come_from_the_curriculum_resources(toolbox):
    _, _, content = composed(toolbox, "instructor_guide")
    block = blocks_of(content, "materials")[0]
    assert block["source"] == "curriculum" and block["items"] == ["Python", "scikit-learn"]


def test_the_instructor_guide_is_written_one_module_at_a_time(toolbox):
    cid, _ = curriculum_record(toolbox)
    llm = fake()
    progress = []

    result = generate(agent_for(toolbox, llm), cid, "instructor_guide", progress=lambda *step: progress.append(step))

    assert result["status"] == "success", result
    assert llm.steps() == ["InstructorFrame", "ModuleTeaching", "ModuleTeaching"]           # one call per module
    assert [p[:2] for p in progress] == [(0, 3), (1, 3), (2, 3), (3, 3)] and progress[1][2] == "M1" and progress[2][2] == "M2"
    stored = toolbox.store.get(result["result"]["document_id"])
    assert stored["version_status"] == "pending_review" and stored["filename"]


def test_a_wrong_module_id_or_minutes_trigger_one_repair_for_that_module_only(toolbox):
    cid, _ = curriculum_record(toolbox)
    good = new_type_answers()["ModuleTeaching"][0]
    seen = {"M1": 0}

    def teaching(prompt):
        mid = "M1" if "Module id: M1" in prompt else "M2"
        answer = good(prompt)
        if mid == "M1" and seen["M1"] == 0:
            seen["M1"] += 1
            answer["instructor_notes"] = "Spend 45 minutes on this."
        return answer

    llm = fake(ModuleTeaching=teaching)
    result = generate(agent_for(toolbox, llm), cid, "instructor_guide")

    assert result["status"] == "success"
    assert llm.steps() == ["InstructorFrame", "ModuleTeaching", "ModuleTeaching", "ModuleTeaching"]   # M1 twice, M2 once
    assert "previous answer had these problems" in llm.prompts("ModuleTeaching")[1] and "minutes" in llm.prompts("ModuleTeaching")[1]


def test_a_module_that_still_fails_after_the_repair_saves_nothing(toolbox):
    cid, _ = curriculum_record(toolbox)
    good = new_type_answers()["ModuleTeaching"][0]

    def always_minutes(prompt):
        answer = good(prompt)
        answer["discussion_questions"] = ["How do you spend 30 minutes on this?", "Why?"]
        return answer

    result = generate(agent_for(toolbox, fake(ModuleTeaching=always_minutes)), cid, "instructor_guide")

    assert result["status"] == "error" and result["result"]["kind"] == "validation_failed"
    assert "minutes" in result["error_message"] and "Nothing was saved" in result["error_message"]
    assert toolbox.store.list() == []


def test_the_model_must_name_each_module_exactly(toolbox):
    cid, record = curriculum_record(toolbox)
    template = templates.TEMPLATES["instructor_guide"]
    data = container_answer("instructor_guide", "en", record["content"])
    data["modules"][1]["module_id"] = "M1"
    parts = template.parts_schema.model_validate(data)

    assert check_parts(parts, template, record["content"])[0]["code"] == "unit_ids"


def test_an_accreditation_claim_or_a_wrong_hour_figure_is_rejected_in_a_unit(toolbox):
    _, record = curriculum_record(toolbox)
    template = templates.TEMPLATES["instructor_guide"]
    base = container_answer("instructor_guide", "en", record["content"])["modules"][0]
    for field, text, code in (("instructor_notes", "This course is accredited by a body.", "accreditation_claim"),
                              ("instructor_notes", "Allow 99 hours for this.", "hours_mismatch"),
                              ("instructor_notes", "  ", "empty_text")):
        unit = template.unit_schema.model_validate({**base, field: text})
        assert code in [p["code"] for p in check_unit(unit, template, record["content"])]


@pytest.mark.parametrize("language", ["en", "ar"])
def test_the_instructor_guide_renders_a_pdf(toolbox, language):
    _, record, content = composed(toolbox, "instructor_guide", language)
    saved = toolbox.store.save_version(content, validate_content(content, record))

    data, pages = render_document_pdf(toolbox.store.get(saved["document_id"]))

    assert data.startswith(b"%PDF") and pages >= 2


@pytest.mark.parametrize("language", ["en", "ar"])
def test_the_instructor_guide_lifecycle(admin_system, language):
    lifecycle(admin_system, "instructor_guide", language)


def test_edited_instructor_text_is_checked_for_minutes(toolbox):
    _, record, content = composed(toolbox, "instructor_guide")
    section, index = first_editable(content)
    block = blocks_of(content, section)[index]
    block["source"] = "admin"
    block["text" if block["type"] == "paragraph" else "items"] = "Plan 25 minutes." if block["type"] == "paragraph" else ["Plan 25 minutes."]

    result = validate_content(content, record)

    assert not result["valid"] and result["errors"][0]["code"] == "minutes_unsupported"
