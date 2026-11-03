"""
Question Bank: per module, 2 multiple-choice + 1 true/false + 1 short-answer question, each mapped to the module's
learning outcomes. Exactly one correct option per multiple-choice question, every part editable on its own, and a
bank that is checked (options, answers, duplicates, topic grounding) before it is ever saved.

Helpers and fixtures are shared with tests/test_document_new_types.py. Fake model, temporary storage.
"""

import copy
import re

import pytest

from agents.document_generation import templates, units
from agents.document_generation.document_store import DocumentStoreError
from agents.document_generation.validation import check_unit, validate_content
from tests.curriculum_fixtures import valid_curriculum
from tests.document_fakes import ADMIN, container_answer, new_type_answers
from tests.test_document_new_types import (  # noqa: F401  (fixtures are used by name)
    admin_system,
    agent_for,
    blocks_of,
    composed,
    curriculum_record,
    fake,
    generate,
    lifecycle,
    toolbox,
)

TEMPLATE = templates.TEMPLATES["question_bank"]


def question_groups(content):
    """The question blocks of a bank, grouped per question: [{"title", "blocks": [...]}] (in document order)."""
    groups, current = [], None
    for block in blocks_of(content, "questions"):
        if block["type"] == "heading" and (block["text"].startswith("Question ") or block["text"].startswith("السؤال ")):
            current = {"title": block["text"], "blocks": []}
            groups.append(current)
        elif current is not None:
            current["blocks"].append(block)
    return groups


def role(group, name):
    return [b for b in group["blocks"] if b.get("role") == name]


@pytest.mark.parametrize("language", ["en", "ar"])
def test_the_question_bank_composes_and_validates(toolbox, language):
    _, record, content = composed(toolbox, "question_bank", language)
    assert validate_content(content, record)["valid"]


def test_each_module_has_two_multiple_choice_one_true_false_and_one_short_answer(toolbox):
    _, _, content = composed(toolbox, "question_bank")
    groups = question_groups(content)

    assert len(groups) == 8                                                       # 2 modules x 4 questions
    kinds = [g["title"].split(" · ")[1] for g in groups]
    assert kinds == ["Multiple choice", "Multiple choice", "True or false", "Short answer"] * 2
    assert [g["title"].split(" · ")[2] for g in groups[:4]] == ["Easy", "Medium", "Easy", "Hard"]
    # (the heading no longer repeats the type as a trailing "Quiz (...)" tag: number, type, difficulty, outcomes)
    assert all(len(g["title"].split(" · ")) == 4 and "Quiz" not in g["title"] for g in groups)


def test_questions_are_grouped_by_module_and_mapped_to_that_modules_outcomes(toolbox):
    _, record, content = composed(toolbox, "question_bank")
    blocks = blocks_of(content, "questions")
    module_headings = [b["text"] for b in blocks if b["type"] == "heading" and re.match(r"^M\d+ ", b["text"])]
    groups = question_groups(content)

    assert module_headings == ["M1 Data preparation", "M2 Models and evaluation"]
    assert "Outcomes: LO1" in groups[0]["title"] and "Outcomes: LO2" in groups[4]["title"]   # from the curriculum, not the model
    assert all(b["source"] == "curriculum" for b in blocks if b["type"] == "heading")


def test_arabic_question_banks_use_arabic_labels_only(toolbox):
    _, _, content = composed(toolbox, "question_bank", "ar")
    titles = [g["title"] for g in question_groups(content)]

    assert titles[0] == "السؤال 1 · اختيار من متعدد · سهل · المخرجات: LO1"          # (no trailing "اختبار قصير (...)" tag)
    assert not any(word in " ".join(titles) for word in ("easy", "medium", "hard", "Multiple", "True"))
    tf_answer = role(question_groups(content)[2], "tf_answer")[0]
    assert tf_answer["text"] == "صحيح"


def test_every_multiple_choice_question_has_exactly_one_correct_option(toolbox):
    _, record, content = composed(toolbox, "question_bank")
    multiple_choice = [g for g in question_groups(content) if role(g, "q_options")]

    assert len(multiple_choice) == 4
    for group in multiple_choice:
        options = role(group, "q_options")[0]["items"]
        answer = role(group, "q_answer")[0]["text"]
        normalized = [units.normalized_text(o) for o in options]
        assert len(options) == 4 and len(set(normalized)) == 4          # four different options
        assert normalized.count(units.normalized_text(answer)) == 1     # the answer is exactly one of them
    assert validate_content(content, record)["valid"]


def test_every_part_of_a_question_is_individually_editable(toolbox):
    _, _, content = composed(toolbox, "question_bank")
    group = question_groups(content)[0]

    for name in ("q_stem", "q_options", "q_answer"):
        assert role(group, name)[0]["source"] == "generated" and role(group, name)[0]["type"] in ("paragraph", "bullets")
    rationale = [b for b in group["blocks"] if b["type"] == "paragraph" and not b.get("role")]
    assert rationale and rationale[0]["source"] == "generated"                   # the explanation, too
    assert group["blocks"][0]["type"] == "paragraph"


def test_editing_one_question_leaves_the_others_and_the_locked_text_alone(admin_system):
    toolbox, admin, token = admin_system.toolbox, admin_system.admin, admin_system.token
    from tests.curriculum_fixtures import valid_curriculum as vc
    from tests.document_fakes import approved_curriculum

    cid = approved_curriculum(toolbox, vc("en"))
    admin_system.agent._llm = fake()
    document_id = admin.generate_document(token, cid, "question_bank")["result"][0]["result"]["document_id"]
    before = admin.get_document(token, document_id)["document"]["content"]
    section = "questions"
    stem_index = next(i for i, b in enumerate(blocks_of(before, section)) if b.get("role") == "q_stem")

    admin.edit_document(token, document_id, [{"section": section, "block": stem_index, "value": "What does cleaning data involve?"}])
    after = admin.get_document(token, document_id)["document"]

    assert after["version_status"] == "pending_review" and not after["validation"]["errors"]
    old, new = blocks_of(before, section), blocks_of(after["content"], section)
    assert [i for i, (a, b) in enumerate(zip(old, new)) if a != b] == [stem_index]       # exactly that one block changed


def test_changing_an_option_without_the_answer_breaks_the_single_correct_answer_rule(admin_system):
    toolbox, admin, token = admin_system.toolbox, admin_system.admin, admin_system.token
    from tests.document_fakes import approved_curriculum

    cid = approved_curriculum(toolbox, valid_curriculum("en"))
    admin_system.agent._llm = fake()
    document_id = admin.generate_document(token, cid, "question_bank")["result"][0]["result"]["document_id"]
    content = admin.get_document(token, document_id)["document"]["content"]
    blocks = blocks_of(content, "questions")
    options_index = next(i for i, b in enumerate(blocks) if b.get("role") == "q_options")
    answer_index = options_index + 2
    options = blocks[options_index]["items"]

    # rewording the correct option only: the answer no longer matches an option -> a draft that cannot be approved
    broken = admin.edit_document(token, document_id, [{"section": "questions", "block": options_index,
                                                       "value": ["A completely different option", *options[1:]]}])
    assert broken["review_status"] == "draft" and broken["validation"]["errors"][0]["code"] == "question_answer"
    with pytest.raises(DocumentStoreError, match="draft"):
        admin.approve_document(token, document_id, broken["version"])
    # changing the answer too makes it consistent again
    fixed = admin.edit_document(token, document_id, [{"section": "questions", "block": answer_index, "value": "A completely different option"}])
    assert fixed["review_status"] == "pending_review" and not fixed["validation"]["errors"]


@pytest.mark.parametrize("options", [["a one", "a two", "a three"], ["a one", "a two", "a three", "a four", "a five"],
                                     ["same", "same", "other one", "another"]])
def test_an_edit_that_leaves_the_wrong_number_of_options_or_duplicates_is_rejected(toolbox, options):
    _, record, content = composed(toolbox, "question_bank")
    group = question_groups(content)[0]
    role(group, "q_options")[0]["items"] = options
    role(group, "q_answer")[0]["text"] = options[0]

    result = validate_content(content, record)

    assert not result["valid"] and result["errors"][0]["code"] == "question_options"


def test_true_false_answers_must_be_true_or_false(toolbox):
    for language, valid, invalid in (("en", "False", "Maybe"), ("ar", "خطأ", "ربما")):
        _, record, content = composed(toolbox, "question_bank", language)
        answer = role(question_groups(content)[2], "tf_answer")[0]
        answer["text"] = valid
        assert validate_content(content, record)["valid"], language
        answer["text"] = invalid
        assert validate_content(content, record)["errors"][0]["code"] == "question_answer", language


def test_a_repeated_question_is_found_in_the_stored_bank(toolbox):
    _, record, content = composed(toolbox, "question_bank")
    groups = question_groups(content)
    role(groups[5], "q_stem")[0]["text"] = role(groups[0], "q_stem")[0]["text"]

    assert validate_content(content, record)["errors"][0]["code"] == "question_duplicate"


def test_locked_labels_difficulty_and_outcomes_cannot_be_changed(toolbox):
    _, record, content = composed(toolbox, "question_bank")
    tampered = copy.deepcopy(content)
    first = next(b for b in blocks_of(tampered, "questions") if b["type"] == "heading" and b["text"].startswith("Question 1"))
    first["text"] = first["text"].replace("Easy", "Hard").replace("LO1", "LO2")

    assert validate_content(tampered, record)["errors"][0]["code"] == "curriculum_changed"


# ---- invalid model output -------------------------------------------------------

def module_answers(**changes):
    """A ModuleQuestions answer maker that applies `changes` (a function of the answer) to every module."""
    good = new_type_answers()["ModuleQuestions"][0]

    def answer(prompt):
        data = good(prompt)
        changes.get("edit", lambda d, p: None)(data, prompt)
        return data

    return answer


def unit_of(toolbox, module_id="M1", mutate=None):
    _, record = curriculum_record(toolbox)
    data = container_answer("question_bank", "en", record["content"])["modules"]
    unit = copy.deepcopy(next(u for u in data if u["module_id"] == module_id))
    if mutate:
        mutate(unit)
    return TEMPLATE.unit_schema.model_validate(unit), record["content"]


def codes(problems):
    return [p["code"] for p in problems]


def test_a_good_unit_passes_every_check(toolbox):
    unit, content = unit_of(toolbox)
    assert check_unit(unit, TEMPLATE, content) == []


def test_duplicate_options_are_rejected(toolbox):
    unit, content = unit_of(toolbox, mutate=lambda u: u["multiple_choice"][0].update(options=["Dup A", "dup a", "Other one", "Another one"]))
    assert "question_options" in codes(check_unit(unit, TEMPLATE, content))


@pytest.mark.parametrize("option", ["All of the above", "None of the above", "all of the above."])
def test_all_or_none_of_the_above_is_rejected_because_exactly_one_option_must_be_correct(toolbox, option):
    unit, content = unit_of(toolbox, mutate=lambda u: u["multiple_choice"][0]["options"].__setitem__(3, option))
    assert "question_options" in codes(check_unit(unit, TEMPLATE, content))


def test_a_question_that_is_not_about_the_modules_topics_is_rejected(toolbox):
    unit, content = unit_of(toolbox, mutate=lambda u: u["short_answer"][0].update(question="How do you bake sourdough bread at home?"))
    problems = check_unit(unit, TEMPLATE, content)
    assert codes(problems) == ["question_ungrounded"] and "short-answer" in problems[0]["message"]


def test_a_question_about_another_modules_topic_is_not_grounded_in_this_one(toolbox):
    unit, content = unit_of(toolbox, "M1", mutate=lambda u: u["true_false"][0].update(statement="A classifier assigns items to categories."))
    assert "question_ungrounded" in codes(check_unit(unit, TEMPLATE, content))


def test_duplicate_questions_inside_a_module_and_across_modules_are_rejected(toolbox):
    inside, content = unit_of(toolbox, mutate=lambda u: u["short_answer"][0].update(question=u["multiple_choice"][0]["question"]))
    assert "question_duplicate" in codes(check_unit(inside, TEMPLATE, content))
    first, _ = unit_of(toolbox, "M1")
    second, content = unit_of(toolbox, "M2", mutate=lambda u: u["multiple_choice"][0].update(
        question="Which step cleans a dataset before modelling?"))        # M1's first question, repeated in M2
    assert "question_duplicate" in codes(check_unit(second, TEMPLATE, content, (first,)))


def test_empty_fields_minutes_and_unsupported_claims_are_rejected(toolbox):
    unit, content = unit_of(toolbox, mutate=lambda u: u["multiple_choice"][0].update(rationale="  "))
    assert "empty_text" in codes(check_unit(unit, TEMPLATE, content))
    unit, content = unit_of(toolbox, mutate=lambda u: u["short_answer"][0].update(model_answer="Cleaning data takes 30 minutes."))
    assert "minutes_unsupported" in codes(check_unit(unit, TEMPLATE, content))
    unit, content = unit_of(toolbox, mutate=lambda u: u["true_false"][0].update(rationale="The course is accredited by the ministry."))
    assert "accreditation_claim" in codes(check_unit(unit, TEMPLATE, content))


@pytest.mark.parametrize("mutate", [
    lambda u: u["multiple_choice"][0].update(options=["only", "three", "options"]),                 # not four options
    lambda u: u["multiple_choice"][0].update(correct_index=4),                                      # no such option
    lambda u: u["multiple_choice"][0].update(correct_index=-1),
    lambda u: u.update(multiple_choice=u["multiple_choice"][:1]),                                   # not two questions
    lambda u: u.update(true_false=[]),
    lambda u: u["true_false"][0].update(answer="maybe"),                                            # not a true/false answer
])
def test_output_that_does_not_fit_the_schema_is_refused_before_any_check(toolbox, mutate):
    with pytest.raises(ValueError):
        unit_of(toolbox, mutate=mutate)


# ---- generation ------------------------------------------------------------------

def test_the_question_bank_is_written_one_module_at_a_time(toolbox):
    cid, _ = curriculum_record(toolbox)
    llm = fake()
    progress = []

    result = generate(agent_for(toolbox, llm), cid, "question_bank", progress=lambda *step: progress.append(step))

    assert result["status"] == "success", result
    assert llm.steps() == ["ModuleQuestions", "ModuleQuestions"] and [p[:2] for p in progress] == [(0, 2), (1, 2), (2, 2)]
    assert toolbox.store.get(result["result"]["document_id"])["version_status"] == "pending_review"


def test_a_bad_first_answer_is_repaired_once(toolbox):
    cid, _ = curriculum_record(toolbox)
    calls = {"n": 0}

    def first_has_duplicate_options(data, prompt):
        if calls["n"] == 0:
            data["multiple_choice"][0]["options"] = ["Same", "Same", "Other one", "Another one"]
        calls["n"] += 1

    llm = fake(ModuleQuestions=module_answers(edit=first_has_duplicate_options))
    result = generate(agent_for(toolbox, llm), cid, "question_bank")

    assert result["status"] == "success" and llm.steps().count("ModuleQuestions") == 3
    assert "the same" in llm.prompts("ModuleQuestions")[1] and "previous answer had these problems" in llm.prompts("ModuleQuestions")[1]


def test_a_question_that_stays_invalid_saves_no_document_at_all(toolbox):
    cid, _ = curriculum_record(toolbox)

    def always_ungrounded(data, prompt):
        data["short_answer"][0]["question"] = "How do you bake sourdough bread at home?"

    # (the second, targeted repair rewrites that one question; it is still not about the module, so the run still fails)
    still_ungrounded = {"question": "How do you bake sourdough bread at home?", "model_answer": "With flour.", "rationale": "Bread."}
    result = generate(agent_for(toolbox, fake(ModuleQuestions=module_answers(edit=always_ungrounded), SaQuestion=still_ungrounded)),
                      cid, "question_bank")

    assert result["status"] == "error" and result["result"]["kind"] == "validation_failed"
    assert "Nothing was saved" in result["error_message"] and "[M1]" in result["error_message"]   # names the module
    assert toolbox.store.list() == []


def test_unusable_model_output_is_a_generation_failure_and_saves_nothing(toolbox):
    cid, _ = curriculum_record(toolbox)

    def three_options(data, prompt):
        data["multiple_choice"][0]["options"] = ["a", "b", "c"]

    result = generate(agent_for(toolbox, fake(ModuleQuestions=module_answers(edit=three_options))), cid, "question_bank")

    assert result["status"] == "error" and result["result"]["kind"] == "generation_failed" and toolbox.store.list() == []


def test_a_module_covered_twice_or_missing_is_caught_when_the_bank_is_assembled(toolbox):
    from agents.document_generation.validation import check_parts

    _, record = curriculum_record(toolbox)
    data = container_answer("question_bank", "en", record["content"])
    data["modules"][1] = data["modules"][0]
    parts = TEMPLATE.parts_schema.model_validate(data)

    assert check_parts(parts, TEMPLATE, record["content"])[0]["code"] == "unit_ids"


@pytest.mark.parametrize("language", ["en", "ar"])
def test_the_question_bank_renders_a_pdf(toolbox, language):
    from agents.document_generation.pdf import render_document_pdf

    _, record, content = composed(toolbox, "question_bank", language)
    saved = toolbox.store.save_version(content, validate_content(content, record))
    data, pages = render_document_pdf(toolbox.store.get(saved["document_id"]))
    assert data.startswith(b"%PDF") and pages >= 3


@pytest.mark.parametrize("language", ["en", "ar"])
def test_the_question_bank_lifecycle(admin_system, language):
    lifecycle(admin_system, "question_bank", language)
