"""
Wording fixes found in the inspected Instructor Guide / Question Bank PDFs: the instructor notice no longer repeats the
section heading or mentions answers, hours read as "20 hours" / "20 ساعة", question headings no longer repeat the type.
Documents made with the earlier wording keep validating against it (they are never rewritten); new ones use the new wording.
"""

import copy

import pytest

from agents.document_generation import templates
from agents.document_generation.composer import compose
from agents.document_generation.templates import STRINGS, hours_text
from agents.document_generation.validation import validate_content
from tests.document_fakes import container_answer
from tests.test_document_new_types import blocks_of, composed, curriculum_record, toolbox  # noqa: F401
from tests.test_document_question_bank import question_groups


# ---- hours ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("hours, expected", [(1, "1 hour"), (2, "2 hours"), (20, "20 hours"), (1.5, "1.5 hours"), (20.0, "20 hours")])
def test_english_hours(hours, expected):
    assert hours_text(hours, "en") == expected


@pytest.mark.parametrize("hours, expected", [
    (1, "ساعة واحدة"), (2, "ساعتان"), (3, "3 ساعات"), (10, "10 ساعات"), (11, "11 ساعة"), (20, "20 ساعة"), (40.0, "40 ساعة"),
    (1.5, "1.5 ساعة")])
def test_arabic_hours_follow_arabic_number_rules(hours, expected):
    assert hours_text(hours, "ar") == expected


def test_the_column_label_is_not_used_as_a_unit():
    assert "الساعات" not in hours_text(20, "ar") and "Hours" not in hours_text(20, "en")
    assert hours_text(None, "en") == "None" and hours_text("x", "ar") == "x"          # a non-number is shown as it is, never crashes


@pytest.mark.parametrize("language, expected", [("en", "(20 hours)"), ("ar", "(20 ساعة)")])
def test_the_instructor_guide_module_headings_show_the_hours_as_a_phrase(toolbox, language, expected):
    _, _, content = composed(toolbox, "instructor_guide", language)
    headings = [b["text"] for b in blocks_of(content, "module_teaching") if b["type"] == "heading" and b["text"].startswith("M")]

    assert len(headings) == 2 and all(h.endswith(expected) for h in headings)
    assert not any("Hours" in h or "الساعات" in h for h in headings)                   # the inspected PDFs showed "(20 Hours)" / "(20 الساعات)"


# ---- the instructor notice ---------------------------------------------------------------------------

@pytest.mark.parametrize("language", ["en", "ar"])
def test_the_notice_does_not_repeat_the_section_heading_or_mention_answers(toolbox, language):
    _, _, content = composed(toolbox, "instructor_guide", language)
    heading = templates.SECTION_HEADINGS["instructor_note"][language]
    notice = blocks_of(content, "instructor_note")[0]["text"]

    assert not notice.startswith(heading) and heading not in notice
    assert "answers" not in notice.lower() and "rationale" not in notice.lower() and "تفسير الإجابات" not in notice
    assert ("Participant Guide" in notice) or ("دليل المتدرب" in notice)               # the pointer to the participant material stays
    assert ("instructor" in notice) or ("المدرب" in notice)


# ---- question headings ----------------------------------------------------------------------------------

@pytest.mark.parametrize("language", ["en", "ar"])
def test_question_headings_do_not_repeat_the_question_type(toolbox, language):
    _, _, content = composed(toolbox, "question_bank", language)
    titles = [g["title"] for g in question_groups(content)]

    assert len(titles) == 8 and all(t.count(" · ") == 3 for t in titles)                # number, type, difficulty, outcomes
    assert not any(tag in t for t in titles for tag in ("Quiz", "Short-answer question", "اختبار قصير", "سؤال بإجابة قصيرة"))


# ---- documents made with the earlier wording ----------------------------------------------------------

def legacy_document(toolbox, doc_type, language="en"):
    cid, record = curriculum_record(toolbox, language)
    template = templates.TEMPLATES[doc_type]
    parts = template.parts_schema.model_validate(container_answer(doc_type, language, record["content"]))
    return record, compose(template, record, language, parts, legacy=True), compose(template, record, language, parts)


@pytest.mark.parametrize("language", ["en", "ar"])
def test_the_earlier_wording_is_still_composed_for_old_documents(toolbox, language):
    _, old, new = legacy_document(toolbox, "instructor_guide", language)

    assert blocks_of(old, "instructor_note")[0]["text"] == STRINGS[language]["instructor_only_notice_legacy"]
    assert blocks_of(new, "instructor_note")[0]["text"] == STRINGS[language]["instructor_only_notice"]
    old_heading = next(b["text"] for b in blocks_of(old, "module_teaching") if b["type"] == "heading")
    assert old_heading.endswith(f"({20} {STRINGS[language]['hours']})")                  # "(20 Hours)" / "(20 الساعات)" as before
    assert old != new


@pytest.mark.parametrize("language", ["en", "ar"])
@pytest.mark.parametrize("doc_type", ["instructor_guide", "question_bank"])
def test_a_document_made_with_the_earlier_wording_still_validates_and_is_not_rewritten(toolbox, doc_type, language):
    record, old, new = legacy_document(toolbox, doc_type, language)
    before = copy.deepcopy(old)

    assert validate_content(old, record)["valid"], validate_content(old, record)["errors"]
    assert validate_content(new, record)["valid"]                                        # the new wording validates too
    assert old == before                                                                 # validating never changes a document


def test_the_earlier_question_headings_carry_the_old_type_tag(toolbox):
    _, old, _ = legacy_document(toolbox, "question_bank")
    assert question_groups(old)[0]["title"].endswith("Quiz (multiple choice)")


@pytest.mark.parametrize("doc_type", ["instructor_guide", "question_bank"])
def test_changed_curriculum_text_is_still_rejected_whichever_wording_a_document_has(toolbox, doc_type):
    record, old, _ = legacy_document(toolbox, doc_type)
    heading = next(b for s in old["sections"] for b in s["blocks"] if b["type"] == "heading" and b["source"] == "curriculum")
    heading["text"] += " (changed)"

    result = validate_content(old, record)

    assert not result["valid"] and result["errors"][0]["code"] == "curriculum_changed"


def test_a_mixture_of_old_and_new_curriculum_text_is_rejected(toolbox):
    record, old, new = legacy_document(toolbox, "instructor_guide")
    mixed = copy.deepcopy(new)
    mixed["sections"][0] = old["sections"][0]                       # the old notice with the new module headings

    assert validate_content(mixed, record)["errors"][0]["code"] == "curriculum_changed"


@pytest.mark.parametrize("doc_type", ["course_description", "course_proposal", "capstone_brief",
                                      "presentation_outline", "assessment_rubric"])
def test_other_document_types_compose_identically_in_both_wordings(toolbox, doc_type):
    from tests.document_fakes import parts_answers

    cid, record = curriculum_record(toolbox)
    template = templates.TEMPLATES[doc_type]
    if template.unit_schema is not None:
        parts = template.parts_schema.model_validate(container_answer(doc_type, "en", record["content"]))
    else:
        parts = template.parts_schema.model_validate(parts_answers("en")[template.parts_schema.__name__])
    assert compose(template, record, "en", parts, legacy=True) == compose(template, record, "en", parts)


# ---- the Participant Guide uses the same hours wording ------------------------------------------------------------------

def _guide_parts(language, record):
    from tests.document_fakes import parts_answers

    template = templates.TEMPLATES["participant_guide"]
    return template, template.parts_schema.model_validate(parts_answers(language)["GuideParts"])


@pytest.mark.parametrize("language, expected", [("en", "(20 hours)"), ("ar", "(20 ساعة)")])
def test_the_participant_guide_module_headings_show_the_hours_as_a_phrase(toolbox, language, expected):
    cid, record = curriculum_record(toolbox, language)
    template, parts = _guide_parts(language, record)
    content = compose(template, record, language, parts)
    headings = [b["text"] for s in content["sections"] for b in s["blocks"] if b["type"] == "heading" and b["text"].startswith("M")]

    assert len(headings) == 2 and all(h.endswith(expected) for h in headings)
    assert not any("Hours" in h or "الساعات" in h for h in headings)


@pytest.mark.parametrize("language", ["en", "ar"])
def test_a_participant_guide_made_with_the_earlier_hours_wording_still_validates_and_is_not_rewritten(toolbox, language):
    cid, record = curriculum_record(toolbox, language)
    template, parts = _guide_parts(language, record)
    old = compose(template, record, language, parts, legacy=True)
    new = compose(template, record, language, parts)
    before = copy.deepcopy(old)
    old_heading = next(b["text"] for s in old["sections"] for b in s["blocks"] if b["type"] == "heading" and b["text"].startswith("M1"))

    assert old_heading.endswith(f"(20 {STRINGS[language]['hours']})")                  # "(20 Hours)" / "(20 الساعات)" as it was made
    assert old != new and validate_content(old, record)["valid"] and validate_content(new, record)["valid"]
    assert old == before


def test_changed_curriculum_text_in_a_participant_guide_is_still_rejected(toolbox):
    cid, record = curriculum_record(toolbox)
    template, parts = _guide_parts("en", record)
    old = compose(template, record, "en", parts, legacy=True)
    next(b for s in old["sections"] for b in s["blocks"] if b["type"] == "heading" and b["text"].startswith("M1"))["text"] = "M1 Data preparation (3 Hours)"

    assert validate_content(old, record)["errors"][0]["code"] == "curriculum_changed"
