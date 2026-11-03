"""
Question Bank quality rules found with the real model: a strengthened prompt, a repair message that names the exact
question and options, no position indexes in options, no foreign-script contamination, no repeated sentences. The
existing strict rules (distinct options, no "all/none of the above") are unchanged and still tested elsewhere.
"""

import pytest

from agents.document_generation import prompts
from agents.document_generation.validation import check_unit, repeated_sentence, text_problems
from tests.test_document_new_types import (  # noqa: F401  (fixtures are used by name)
    agent_for,
    curriculum_record,
    fake,
    generate,
    toolbox,
)
from tests.test_document_question_bank import TEMPLATE, codes, module_answers, unit_of


def options_of(toolbox, options, which=0):
    unit, content = unit_of(toolbox, mutate=lambda u: u["multiple_choice"][which].update(options=options))
    return check_unit(unit, TEMPLATE, content)


# ---- the prompt ----------------------------------------------------------------------

def test_the_question_prompt_demands_distinct_complete_options_without_position_indexes():
    text = prompts.MODULE_QUESTIONS.format_messages(
        language_name="English", curriculum_block="c", unit_block="u", problems_block="")[0].content
    assert "complete, meaningful statement" in text and "different from one another" in text
    assert "never repeat an option" in text and "never refer to another option" in text
    assert "Do not number or label the options" in text and "position or index" in text
    assert 'Never use "all of the above" or "none of the above"' in text          # the earlier rule is kept
    assert "do not repeat a sentence" in text


# ---- the repair message -------------------------------------------------------------

def test_duplicate_options_are_reported_with_the_question_the_option_numbers_and_the_text(toolbox):
    problems = options_of(toolbox, ["Remove outliers", "Handle gaps", "Remove outliers", "Fix types"], which=1)

    assert codes(problems) == ["question_options"]
    message = problems[0]["message"]
    assert "Multiple-choice question 2" in message and "option 1 and option 3" in message
    assert '"Remove outliers"' in message and "genuinely different" in message and "counting from 1" in message


def test_every_repeated_option_is_reported_separately(toolbox):
    problems = options_of(toolbox, ["Same one", "Same one", "Same one", "Other"])
    assert [("option 1 and option 2" in p["message"], "option 1 and option 3" in p["message"]) for p in problems] \
        == [(True, False), (False, True)]


def test_all_of_the_above_is_still_rejected_and_the_option_is_named(toolbox):
    problems = options_of(toolbox, ["Remove outliers", "Handle gaps", "Fix types", "All of the above"])
    assert codes(problems) == ["question_options"]
    assert "question 1, option 4" in problems[0]["message"] and '"All of the above"' in problems[0]["message"]


def test_the_repair_prompt_carries_the_detailed_message(toolbox):
    cid, _ = curriculum_record(toolbox)
    calls = {"n": 0}

    def duplicate_first(data, prompt):
        if calls["n"] == 0:
            data["multiple_choice"][1]["options"] = ["Same text", "Different one", "Same text", "Another one"]
        calls["n"] += 1

    llm = fake(ModuleQuestions=module_answers(edit=duplicate_first))
    result = generate(agent_for(toolbox, llm), cid, "question_bank")

    repair = llm.prompts("ModuleQuestions")[1]
    assert result["status"] == "success"
    assert "question 2 has options that are the same: option 1 and option 3" in repair and '"Same text"' in repair


# ---- position indexes in options -----------------------------------------------------

@pytest.mark.parametrize("option", [
    "Identify the unsuitable items, 0",              # the real model's output
    "تحديد العناصر الغير مناسبة، 0",                 # the real model's Arabic output (Arabic comma)
    "Replace missing values - 2",
    "Replace missing values (3)",
    "A) Replace missing values",
    "1. Replace missing values",
    "أ) تحديد العناصر",
])
def test_an_option_that_carries_a_position_index_is_rejected(toolbox, option):
    problems = options_of(toolbox, [option, "Handle gaps", "Fix types", "Drop rows"])
    assert codes(problems) == ["question_options"] and "labels its position" in problems[0]["message"]


@pytest.mark.parametrize("option", [
    "Use Python 3", "Scale values to 0.5", "Keep 10 percent of rows", "Version 2 of the file", "SQL", "تحديد 3 أعمدة",
    "Handle missing values, then outliers", "Remove outliers", "Cleaning order: first nulls",
])
def test_ordinary_options_with_numbers_or_commas_are_not_rejected(toolbox, option):
    assert "labels its position" not in " ".join(p["message"] for p in options_of(toolbox, [option, "Handle gaps", "Fix types", "Drop rows"]))


# ---- foreign script ------------------------------------------------------------------

@pytest.mark.parametrize("text", ["الخطوات هي تحديد العناصر، و最后是 تطبيق التحويلات", "Clean the data 最后", "Очистка данных", "データを洗う"])
def test_foreign_script_is_rejected(text):
    problems = text_problems(text, set())
    assert [p["code"] for p in problems] == ["foreign_script"] and "another writing system" in problems[0]["message"]


@pytest.mark.parametrize("text", [
    "تعلم الآلة باستخدام Python و SQL", "Use pandas and scikit-learn (Python 3)", "Clean the data", "نظف البيانات قبل النمذجة",
    "Résumé of the café data", "Prepare the dataset, 20 hours",
])
def test_arabic_and_latin_text_and_technical_terms_are_not_rejected(text):
    assert [p["code"] for p in text_problems(text, {20.0})] == []


def test_a_generated_answer_with_chinese_text_is_repaired_or_rejected(toolbox):
    cid, _ = curriculum_record(toolbox)

    def contaminated(data, prompt):
        data["short_answer"][0]["model_answer"] = "Clean the data, 最后 check it."

    still_contaminated = {"question": "Why is data cleaned before modelling a dataset?", "model_answer": "To check 最后 it.",
                          "rationale": "Because it is right."}      # (the second, targeted repair did not remove it either)
    result = generate(agent_for(toolbox, fake(ModuleQuestions=module_answers(edit=contaminated), SaQuestion=still_contaminated)),
                      cid, "question_bank")

    assert result["status"] == "error" and "another writing system" in result["error_message"] and toolbox.store.list() == []


# ---- repeated sentences ---------------------------------------------------------------

def test_a_repeated_sentence_is_found():
    looping = "Check the data. You should also verify the data again. You should also verify the data again. Done."
    assert repeated_sentence(looping) == "You should also verify the data again"
    assert repeated_sentence("يجب أيضًا التحقق من صحة البيانات. يجب أيضا التحقق من صحة البيانات.") is not None   # diacritics ignored


@pytest.mark.parametrize("text", ["Clean the data. Then check it.", "Yes. Yes. Yes.", "Check it; check it", "", "One sentence only"])
def test_text_without_a_repeated_sentence_is_fine(text):
    assert repeated_sentence(text) is None                          # (very short sentences are not counted)


@pytest.mark.parametrize("field", ["model_answer", "rationale"])
def test_repeated_sentences_in_short_answers_and_rationales_are_rejected(toolbox, field):
    loop = "You should check the data again. You should check the data again."
    unit, content = unit_of(toolbox, mutate=lambda u: u["short_answer"][0].update({field: loop}))
    problems = check_unit(unit, TEMPLATE, content)
    assert codes(problems) == ["repeated_text"] and "repeated within the same text" in problems[0]["message"]


def test_invalid_output_that_cannot_be_repaired_is_never_saved_or_corrected(toolbox):
    cid, _ = curriculum_record(toolbox)

    def indexed(data, prompt):
        data["multiple_choice"][0]["options"] = ["First choice, 0", "Second choice, 1", "Third choice, 2", "Fourth choice, 3"]

    still_indexed = {"question": "What is the purpose of cleaning data in a dataset?", "correct_index": 0, "rationale": "Because.",
                     "options": ["First choice, 0", "Second choice, 1", "Third choice, 2", "Fourth choice, 3"]}
    llm = fake(ModuleQuestions=module_answers(edit=indexed), McQuestion=still_indexed)
    result = generate(agent_for(toolbox, llm), cid, "question_bank")

    assert result["status"] == "error" and result["result"]["kind"] == "validation_failed"
    # first answer + first repair of the whole module + one targeted rewrite of the question, nothing saved
    assert llm.steps().count("ModuleQuestions") == 2 and llm.steps().count("McQuestion") == 1 and toolbox.store.list() == []
