"""
Presentation Outline (a structured, editable outline exported to PDF; no PowerPoint file is made) and Assessment
Rubrics (weights and points are calculated by code, never written by the model).

Helpers and fixtures are shared with tests/test_document_new_types.py. Fake model, temporary storage.
"""

import copy
from fractions import Fraction

import pytest
from pypdf import PdfReader

from agents.document_generation import templates, units
from agents.document_generation.pdf import render_document_pdf
from agents.document_generation.validation import check_unit, validate_content
from tests.curriculum_fixtures import valid_curriculum
from tests.document_fakes import approved_curriculum, container_answer, new_type_answers
from tests.test_document_new_types import (  # noqa: F401  (fixtures are used by name)
    admin_system,
    agent_for,
    blocks_of,
    composed,
    curriculum_record,
    fake,
    generate,
    headings,
    lifecycle,
    toolbox,
)


# ====================================================================
# Presentation Outline
# ====================================================================

def slide_headings(content):
    words = (templates.STRINGS["en"]["slide"], templates.STRINGS["ar"]["slide"])
    return [b["text"] for s in content["sections"] for b in s["blocks"] if b["type"] == "heading" and b["text"].split(" ")[0] in words]


@pytest.mark.parametrize("language", ["en", "ar"])
def test_the_presentation_outline_composes_and_validates(toolbox, language):
    _, record, content = composed(toolbox, "presentation_outline", language)
    assert validate_content(content, record)["valid"]


def test_slides_are_numbered_in_sequence_and_each_has_a_purpose(toolbox):
    _, _, content = composed(toolbox, "presentation_outline")
    slides = slide_headings(content)

    # title, overview, objectives, outcomes, 2 module slides, assessment, capstone, closing
    assert len(slides) == 9
    assert [h.split(" ")[1] for h in slides] == [str(n) for n in range(1, 10)]
    assert slides[4].endswith("M1 Data preparation") and slides[5].endswith("M2 Models and evaluation")
    purposes = [b for s in content["sections"] for b in s["blocks"] if b["type"] == "paragraph" and b["text"].startswith("Purpose: ")]
    assert len(purposes) == 9 and all(b["source"] == "curriculum" for b in purposes)   # one per slide, locked wording


def test_slide_content_follows_the_curriculum_and_only_the_slide_text_is_editable(toolbox):
    _, record, content = composed(toolbox, "presentation_outline")
    s = templates.STRINGS["en"]
    module_blocks = blocks_of(content, "module_slides")
    labels = [b["text"] for b in module_blocks if b["type"] == "heading"]

    for label in (s["topics"], s["key_points"], s["example"], s["discussion_prompt"], s["visual"], s["curriculum_activities"]):
        assert label in labels
    topics = next(b for b in module_blocks if b["type"] == "bullets" and b["items"] == ["Cleaning data"])
    assert topics["source"] == "curriculum"                                   # the curriculum's topics are locked
    assert blocks_of(content, "slides_objectives")[-1]["items"] == record["content"]["objectives"]
    generated = [b for sec in content["sections"] for b in sec["blocks"] if b["source"] == "generated"]
    assert generated and all(b["type"] in ("paragraph", "bullets") for b in generated)   # every free text is editable


def test_the_closing_slide_summary_is_model_text_and_the_outcomes_recap_is_locked(toolbox):
    _, _, content = composed(toolbox, "presentation_outline")
    closing = blocks_of(content, "slides_closing")
    assert [b["source"] for b in closing if b["type"] != "heading"] == ["curriculum", "generated", "curriculum"]


def test_the_outline_is_a_pdf_outline_and_never_claims_a_powerpoint_file(toolbox):
    assert templates.TEMPLATES["presentation_outline"].label["en"] == "Presentation Outline"
    _, record, content = composed(toolbox, "presentation_outline")
    saved = toolbox.store.save_version(content, validate_content(content, record))
    toolbox.render_document_pdf(saved["document_id"])
    assert toolbox.store.get(saved["document_id"])["filename"].endswith(".pdf")
    assert not list(toolbox.documents_dir.glob("*.pptx"))


def test_the_presentation_outline_is_written_one_module_at_a_time(toolbox):
    cid, _ = curriculum_record(toolbox)
    llm = fake()

    result = generate(agent_for(toolbox, llm), cid, "presentation_outline")

    assert result["status"] == "success", result
    assert llm.steps() == ["SlidesFrame", "ModuleSlides", "ModuleSlides"]


def test_slide_key_points_need_three_to_five():
    template = templates.TEMPLATES["presentation_outline"]
    base = container_answer("presentation_outline", "en", valid_curriculum())["modules"][0]
    with pytest.raises(ValueError):
        template.unit_schema.model_validate({**base, "key_points": ["only", "two"]})
    with pytest.raises(ValueError):
        template.unit_schema.model_validate({**base, "key_points": ["a", "b", "c", "d", "e", "f"]})


def test_slide_text_with_minutes_is_rejected_and_repaired(toolbox):
    cid, _ = curriculum_record(toolbox)
    good = new_type_answers()["ModuleSlides"][0]
    calls = {"n": 0}

    def slides(prompt):
        answer = good(prompt)
        if calls["n"] == 0:
            answer["example"] = "A 10 minute demo of cleaning data."
        calls["n"] += 1
        return answer

    llm = fake(ModuleSlides=slides)
    result = generate(agent_for(toolbox, llm), cid, "presentation_outline")

    assert result["status"] == "success" and llm.steps().count("ModuleSlides") == 3


@pytest.mark.parametrize("language", ["en", "ar"])
def test_the_presentation_outline_renders_a_pdf(toolbox, language):
    _, record, content = composed(toolbox, "presentation_outline", language)
    saved = toolbox.store.save_version(content, validate_content(content, record))
    data, pages = render_document_pdf(toolbox.store.get(saved["document_id"]))
    assert data.startswith(b"%PDF") and pages >= 2


@pytest.mark.parametrize("language", ["en", "ar"])
def test_the_presentation_outline_lifecycle(admin_system, language):
    lifecycle(admin_system, "presentation_outline", language)


# ====================================================================
# Assessment Rubrics
# ====================================================================

@pytest.mark.parametrize("criteria", list(range(1, 13)) + [15, 20, 33, 100])
def test_rubric_weights_always_add_up_to_exactly_100(criteria):
    weights = units.rubric_weights(criteria)

    assert len(weights) == criteria and sum(weights) == 100
    assert all(isinstance(w, int) and w >= 0 for w in weights)
    assert max(weights) - min(weights) <= 1                      # shared as equally as whole points allow
    assert weights == sorted(weights, reverse=True)              # the remainder goes to the first criteria


@pytest.mark.parametrize("criteria, expected", [(1, [100]), (2, [50, 50]), (3, [34, 33, 33]), (6, [17, 17, 17, 17, 16, 16]),
                                                (7, [15, 15, 14, 14, 14, 14, 14]), (8, [13, 13, 13, 13, 12, 12, 12, 12])])
def test_the_rounding_cases(criteria, expected):
    assert units.rubric_weights(criteria) == expected


@pytest.mark.parametrize("weight", range(1, 101))
def test_level_points_are_exact_and_the_top_level_earns_the_whole_weight(weight):
    points = units.level_points(weight)

    assert points[-1] == weight and points == sorted(points)
    assert points == [Fraction(weight * k, 4) for k in (1, 2, 3, 4)]
    assert float(units.format_points(points[0])) == weight / 4   # quarter steps are exact: no float drift


def test_formatting_of_points():
    assert [units.format_points(p) for p in units.level_points(33)] == ["8.25", "16.5", "24.75", "33"]
    assert [units.format_points(p) for p in units.level_points(34)] == ["8.5", "17", "25.5", "34"]


def curriculum_with_outcomes(count, language="en"):
    content = valid_curriculum(language)
    content["learning_outcomes"] = [{"id": f"LO{i}", "statement": f"Apply skill number {i}" if language == "en" else f"يطبق المهارة {i}"}
                                    for i in range(1, count + 1)]
    ids = [o["id"] for o in content["learning_outcomes"]]
    content["assessments"][0]["outcome_ids"] = ids
    content["capstone"]["outcome_ids"] = ids[:2]
    for module in content["modules"]:
        module["outcome_ids"] = ids[:1]
    return content


def rubric_table(content, index=0):
    return [b for b in blocks_of(content, "rubrics") if b["type"] == "table"][index]


@pytest.mark.parametrize("count, weights", [(2, ["50", "50"]), (3, ["34", "33", "33"]),
                                            (7, ["15", "15", "14", "14", "14", "14", "14"])])
def test_the_scoring_table_totals_exactly_100(toolbox, count, weights):
    _, record, content = composed(toolbox, "assessment_rubric", content=curriculum_with_outcomes(count))
    table = rubric_table(content, 0)

    assert [row[1] for row in table["rows"][:-1]] == weights
    assert table["rows"][-1] == ["Total possible score", "100", "25", "50", "75", "100"]   # every column adds up exactly
    assert table["header"][1:] == ["Weight", "Beginning", "Developing", "Proficient", "Advanced"]
    assert validate_content(content, record)["valid"]


def test_points_in_the_table_match_the_level_headings(toolbox):
    _, _, content = composed(toolbox, "assessment_rubric", content=curriculum_with_outcomes(3))
    table = rubric_table(content, 0)
    texts = [b["text"] for b in blocks_of(content, "rubrics") if b["type"] == "heading"]

    assert table["rows"][0][2:] == ["8.5", "17", "25.5", "34"] and table["rows"][1][2:] == ["8.25", "16.5", "24.75", "33"]
    assert "Beginning — 8.5 points" in texts and "Beginning — 8.25 points" in texts and "Advanced — 33 points" in texts


def test_there_is_a_rubric_for_each_assessment_and_the_capstone(toolbox):
    _, _, content = composed(toolbox, "assessment_rubric")
    titles = [h for h in headings(content, "rubrics") if h.startswith("Rubric: ")]
    assert titles == ["Rubric: A1 (practical test)", "Rubric: Capstone project"]


def test_criteria_are_exactly_the_linked_learning_outcomes(toolbox):
    content = curriculum_with_outcomes(4)
    content["capstone"]["outcome_ids"] = ["LO1", "LO9"]      # LO9 is not defined: it cannot become a criterion
    _, record, doc = composed(toolbox, "assessment_rubric", content=content)
    criteria = [h for h in headings(doc, "rubrics") if h.startswith("Criterion ")]

    assert [c.split(": ")[1].split(" ")[0] for c in criteria] == ["LO1", "LO2", "LO3", "LO4", "LO1"]
    assert not any("LO9" in str(b) for b in blocks_of(doc, "rubrics"))
    assert [r["id"] for r in units.rubric_specs(record["content"])] == ["A1", "CAP"]


def test_an_assessment_with_no_defined_outcome_is_reported_not_invented(toolbox):
    content = valid_curriculum()
    content["assessments"][0]["outcome_ids"] = ["LO9"]
    _, _, doc = composed(toolbox, "assessment_rubric", content=content)

    assert {"section": "rubrics", "field": "assessment_outcomes"} in doc["missing"]
    assert templates.field_label("assessment_outcomes", "ar") != "assessment_outcomes"


def test_level_descriptions_are_model_text_while_labels_points_and_weights_are_locked(toolbox):
    _, record, content = composed(toolbox, "assessment_rubric")
    descriptors = [b for b in blocks_of(content, "rubrics") if b.get("role") == "descriptor"]

    assert len(descriptors) == 4 * (2 + 2) and all(b["source"] == "generated" for b in descriptors)   # 4 levels x 4 criteria
    tampered = copy.deepcopy(content)
    next(b for b in blocks_of(tampered, "rubrics") if b["type"] == "heading" and "Beginning —" in b["text"])["text"] = "Beginning — 99 points"
    assert validate_content(tampered, record)["errors"][0]["code"] == "curriculum_changed"
    tampered = copy.deepcopy(content)
    rubric_table(tampered)["rows"][0][1] = "60"
    assert validate_content(tampered, record)["errors"][0]["code"] == "curriculum_changed"


def test_the_model_must_write_one_criterion_per_outcome_and_distinct_levels(toolbox):
    _, record = curriculum_record(toolbox)
    template = templates.TEMPLATES["assessment_rubric"]
    base = container_answer("assessment_rubric", "en", record["content"])["rubrics"][0]

    missing_one = template.unit_schema.model_validate({**base, "criteria": base["criteria"][:1]})
    assert check_unit(missing_one, template, record["content"])[0]["code"] == "criteria_ids"
    invented = copy.deepcopy(base)
    invented["criteria"][0]["outcome_id"] = "LO7"
    assert check_unit(template.unit_schema.model_validate(invented), template, record["content"])[0]["code"] == "criteria_ids"
    same = copy.deepcopy(base)
    for level in ("developing", "proficient", "advanced"):
        same["criteria"][0][level] = same["criteria"][0]["beginning"]
    codes = [p["code"] for p in check_unit(template.unit_schema.model_validate(same), template, record["content"])]
    assert "descriptors_similar" in codes


def test_an_edit_that_makes_two_levels_the_same_is_caught_on_the_stored_document(toolbox):
    _, record, content = composed(toolbox, "assessment_rubric")
    descriptors = [b for b in blocks_of(content, "rubrics") if b.get("role") == "descriptor"]
    descriptors[1]["text"], descriptors[1]["source"] = descriptors[0]["text"], "admin"

    result = validate_content(content, record)

    assert not result["valid"] and result["errors"][0]["code"] == "descriptors_similar"


def test_the_rubrics_are_written_one_assessment_at_a_time(toolbox):
    cid, _ = curriculum_record(toolbox)
    llm = fake()

    result = generate(agent_for(toolbox, llm), cid, "assessment_rubric")

    assert result["status"] == "success", result
    assert llm.steps() == ["AssessmentRubric", "AssessmentRubric"]        # A1 and the capstone
    assert result["result"]["review_status"] == "pending_review"


def test_rubrics_cannot_be_generated_without_assessments(toolbox):
    content = valid_curriculum()
    content["assessments"] = []
    cid = approved_curriculum(toolbox, content)

    result = generate(agent_for(toolbox, fake()), cid, "assessment_rubric")

    assert result["status"] == "error" and result["result"]["kind"] == "missing_information"
    assert "assessments" in result["error_message"] and toolbox.store.list() == []


@pytest.mark.parametrize("language", ["en", "ar"])
def test_the_rubrics_render_a_pdf_with_the_wide_table(toolbox, language):
    _, record, content = composed(toolbox, "assessment_rubric", language, content=curriculum_with_outcomes(3, language))
    saved = toolbox.store.save_version(content, validate_content(content, record))
    toolbox.render_document_pdf(saved["document_id"])
    path = toolbox.documents_dir / toolbox.store.get(saved["document_id"])["filename"]
    pages = PdfReader(str(path)).pages
    text = "\n".join(page.extract_text() for page in pages)

    assert "100" in text and "34" in text and "33" in text and len(pages) >= 3


@pytest.mark.parametrize("language", ["en", "ar"])
def test_the_rubrics_lifecycle(admin_system, language):
    lifecycle(admin_system, "assessment_rubric", language)
