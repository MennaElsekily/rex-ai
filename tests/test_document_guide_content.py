"""
Agent 4 writes educational content from Agent 3's approved curriculum: the participant guide teaches each module (an
explanation and an example in the model's own words) instead of only repeating the curriculum's topics. Fake model, no Ollama;
what the real model writes is checked separately by hand and is not established by these tests.
"""

import copy

import pytest

from agents.document_generation import prompts, schemas, templates
from agents.document_generation.validation import check_parts
from tests.curriculum_fixtures import valid_curriculum
from tests.document_fakes import approved_curriculum, fake_llm, guide_teaching, in_process_document_tools, parts_answers, save_curriculum
from tests.test_document_generation import ask, make_agent

REQUEST = {"document_type": "participant_guide"}
SHORT = "Cleaning data."                                                 # a restated topic, not an explanation
TOPIC_LIST = "Cleaning data, classification, charts, tables, models, labels, features, rows, columns, values, scores, tests."


@pytest.fixture
def toolbox(tmp_path):
    return in_process_document_tools(tmp_path / "docs")


@pytest.fixture
def curriculum_id(toolbox):
    return approved_curriculum(toolbox)


def generate(toolbox, cid, llm, **request):
    return ask(make_agent(toolbox, llm), "x", document_request={"curriculum_id": cid, **REQUEST, **request})


def guide_answer(**changes):
    answer = copy.deepcopy(parts_answers()["GuideParts"])
    answer["modules"][0].update(changes)
    return answer


def saved_blocks(toolbox, response):
    document = toolbox.store.get(response["result"]["document_id"])
    return [b for s in document["content"]["sections"] for b in s["blocks"]]


# ---- what the model is asked and given -------------------------------------------------------------------------------------------

def test_the_model_receives_the_approved_topics_and_is_told_to_teach(toolbox, curriculum_id):
    llm = fake_llm()

    response = generate(toolbox, curriculum_id, llm)

    assert response["status"] == "success"
    prompt = llm.prompts("GuideParts")[0]
    record = toolbox.get_approved_curriculum(curriculum_id)["curriculum"]
    for module in record["content"]["modules"]:
        assert module["id"] in prompt and "; ".join(module["topics"]) in prompt          # the topics reach the generator
    assert "TEACHES the course" in prompt and "explanation: 4 to 7 sentences" in prompt
    assert "never add topics" in prompt and "well-established general knowledge" in prompt   # may explain, may not invent topics


def test_only_the_participant_guide_may_explain_beyond_the_curriculum_text(toolbox, curriculum_id):
    llm = fake_llm()

    generate(toolbox, curriculum_id, llm, document_type="course_description")

    description = llm.prompts("DescriptionParts")[0]
    assert "Never add facts" in description and "well-established general knowledge" not in description


def test_the_approved_version_is_the_one_the_model_sees_not_a_newer_unapproved_revision(toolbox, curriculum_id):
    revised = valid_curriculum()
    revised["modules"][0]["topics"] = ["Unapproved revision topic"]
    save_curriculum(toolbox, revised, curriculum_id=curriculum_id)
    llm = fake_llm()

    response = generate(toolbox, curriculum_id, llm)

    assert response["status"] == "success" and response["result"]["curriculum_version"] == 1
    assert "Unapproved revision topic" not in llm.prompts("GuideParts")[0] and "Cleaning data" in llm.prompts("GuideParts")[0]


@pytest.mark.parametrize("make", ["pending", "draft"])
def test_an_unapproved_curriculum_never_reaches_the_model(toolbox, make):
    saved = save_curriculum(toolbox, valid=(make == "pending"))
    llm = fake_llm()

    response = generate(toolbox, saved["curriculum_id"], llm)

    assert response["status"] == "error" and response["result"]["kind"] == "not_approved"
    assert llm.calls == [] and toolbox.store.list() == []


# ---- the generated teaching ends up in the document ------------------------------------------------------------------------------

def test_the_generated_explanations_are_in_the_saved_document_next_to_the_unchanged_curriculum_facts(toolbox, curriculum_id):
    response = generate(toolbox, curriculum_id, fake_llm())

    assert response["status"] == "success" and response["result"]["review_status"] == "pending_review"
    blocks = saved_blocks(toolbox, response)
    generated = [b["text"] for b in blocks if b["type"] == "paragraph" and b["source"] == "generated"]
    assert guide_teaching("en", "M1")["explanation"] in generated and guide_teaching("en", "M2")["explanation"] in generated
    assert guide_teaching("en", "M1")["example"] in generated
    headings = [b["text"] for b in blocks if b["type"] == "heading"]
    assert {"Explanation", "Example", "Key concepts"} <= set(headings)
    curriculum_text = [b for b in blocks if b["source"] == "curriculum"]
    assert any(b["type"] == "bullets" and b["items"] == ["Cleaning data"] for b in curriculum_text)     # topics kept exactly
    assert any(b["type"] == "heading" and b["text"].startswith("M1 Data preparation") for b in curriculum_text)
    assert all(b["items"] != [guide_teaching("en", "M1")["explanation"]] for b in blocks if b["type"] == "bullets")
    assert response["result"]["curriculum_id"] == curriculum_id and response["result"]["curriculum_version"] == 1


def test_the_arabic_guide_teaches_in_arabic(toolbox):
    cid = approved_curriculum(toolbox, valid_curriculum("ar"))

    response = generate(toolbox, cid, fake_llm("ar"), language="ar")

    assert response["status"] == "success"
    generated = [b["text"] for b in saved_blocks(toolbox, response) if b["type"] == "paragraph" and b["source"] == "generated"]
    assert guide_teaching("ar", "M1")["explanation"] in generated


def test_the_pdf_contains_the_generated_explanation(toolbox, curriculum_id):
    from agents.document_generation.pdf import render_document_pdf

    response = generate(toolbox, curriculum_id, fake_llm())

    data, pages = render_document_pdf(toolbox.store.get(response["result"]["document_id"]))
    assert data.startswith(b"%PDF") and pages >= 2
    try:
        from pypdf import PdfReader
    except ImportError:
        pytest.skip("pypdf is not installed: the PDF text cannot be read back")
    import io

    text = " ".join(" ".join(page.extract_text().split()) for page in PdfReader(io.BytesIO(data)).pages)
    assert "step by step" in text and "Worked example" in text


# ---- the checks on the generated teaching -----------------------------------------------------------------------------------------

def check(answer, curriculum=None):
    return check_parts(schemas.GuideParts.model_validate(answer), templates.TEMPLATES["participant_guide"], curriculum or valid_curriculum())


def test_the_old_shape_without_an_explanation_is_not_accepted():
    answer = copy.deepcopy(parts_answers()["GuideParts"])
    for module in answer["modules"]:
        del module["explanation"], module["example"]
    with pytest.raises(ValueError):
        schemas.GuideParts.model_validate(answer)


def test_a_good_explanation_passes():
    assert check(parts_answers()["GuideParts"]) == []


def test_a_restated_topic_is_not_an_explanation():
    problems = check(guide_answer(explanation=SHORT))

    assert [p["code"] for p in problems] == ["explanation_too_short"] and problems[0]["where"] == "M1"
    assert "at least 40 words" in problems[0]["message"]


def test_an_explanation_that_ignores_the_modules_topics_is_rejected():
    off_topic = ("Photosynthesis happens in leaves, where light energy is turned into stored sugar for the whole plant. "
                 "Roots take up water from the soil while the stem carries it upward to every branch. Flowers attract insects "
                 "that move pollen, and seeds later grow into new plants that repeat the cycle each season.")
    problems = check(guide_answer(explanation=off_topic))

    assert [p["code"] for p in problems] == ["explanation_off_topic"] and "Cleaning data" in problems[0]["message"]


def test_an_unsupported_hour_figure_or_an_accreditation_claim_in_the_explanation_is_rejected():
    teaching = guide_teaching("en", "M1")["explanation"]

    assert "hours_mismatch" in {p["code"] for p in check(guide_answer(explanation=teaching + " It takes 99 hours."))}
    assert "accreditation_claim" in {p["code"] for p in check(guide_answer(example="The course is officially accredited and certified."))}


def test_a_looping_explanation_is_rejected():
    loop = guide_teaching("en", "M1")["explanation"] + " " + guide_teaching("en", "M1")["explanation"]

    assert "repeated_text" in {p["code"] for p in check(guide_answer(explanation=loop))}


def test_the_checks_do_not_lengthen_or_shorten_the_curriculum_facts():
    # the explanation limits are on the generated text only; topics, titles and hours come from the curriculum
    assert prompts.curriculum_block(valid_curriculum()).count("Cleaning data") == 1


# ---- repair and persistence --------------------------------------------------------------------------------------------------------

def test_a_too_short_explanation_is_repaired_with_the_problem_named(toolbox, curriculum_id):
    llm = fake_llm(GuideParts=[guide_answer(explanation=SHORT), parts_answers()["GuideParts"]])

    response = generate(toolbox, curriculum_id, llm)

    assert response["status"] == "success" and llm.steps().count("GuideParts") == 2
    repair = llm.prompts("GuideParts")[1]
    assert "previous answer had these problems" in repair and "The explanation of M1 has 2 words" in repair
    assert guide_teaching("en", "M1")["explanation"] in [b.get("text") for b in saved_blocks(toolbox, response)]


def test_an_explanation_that_stays_invalid_after_the_repair_saves_nothing(toolbox, curriculum_id):
    bad = guide_answer(explanation=SHORT)
    llm = fake_llm(GuideParts=[bad, bad])

    response = generate(toolbox, curriculum_id, llm)

    assert response["status"] == "error" and response["result"]["kind"] == "validation_failed"
    assert "Nothing was saved" in response["error_message"] and "explanation of M1" in response["error_message"]
    assert toolbox.store.list() == [] and llm.steps().count("GuideParts") == 2


def test_generating_a_document_does_not_change_the_approved_curriculum(toolbox, curriculum_id):
    before = toolbox.curricula.get(curriculum_id)

    generate(toolbox, curriculum_id, fake_llm())

    after = toolbox.curricula.get(curriculum_id)
    assert after["content"] == before["content"] and after["approved_version"] == before["approved_version"]
    assert after["status"] == before["status"]
