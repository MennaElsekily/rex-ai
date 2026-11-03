"""
Question Bank: the second, targeted repair. Order of events for a module: write -> check -> first repair (the whole module
again, with the problems listed) -> check -> second repair (ONLY the questions still flagged are rewritten, one small call
each; every other question is kept) -> full check -> otherwise stop and save nothing. Other document types are unchanged.
Deterministic: a fake model, temporary storage.
"""

import copy
import json
from collections import defaultdict

import pytest

from agents.document_generation import generator, templates
from agents.document_generation.validation import check_unit, issue
from tests.curriculum_fakes import FakeLLM
from tests.document_fakes import _unit_id, new_type_answers
from tests.test_document_new_types import (  # noqa: F401  (fixtures are used by name)
    admin_system,
    agent_for,
    blocks_of,
    curriculum_record,
    fake,
    generate,
    lifecycle,
    toolbox,
)
from tests.test_document_question_bank import TEMPLATE, codes, unit_of

ITEMS = {
    "en": {"mc": "What is the purpose of cleaning data in a dataset?", "tf": "Cleaning data prepares a dataset for modelling.",
           "sa": "Why is data cleaned before modelling a dataset?", "answer": "To remove errors."},
    "ar": {"mc": "ما الغرض من تنظيف البيانات في مجموعة البيانات؟", "tf": "تنظيف البيانات يجهز البيانات للنمذجة.",
           "sa": "لماذا تُنظَّف البيانات قبل النمذجة؟", "answer": "لإزالة الأخطاء."},
}


def good_item(kind, language="en"):
    """A valid replacement question (about module M1's topics, different from the fake model's usual questions)."""
    t, why = ITEMS[language], "لأنه الصحيح" if language == "ar" else "Because it is the right one."
    if kind == "multiple_choice":
        options = ["الإجابة الصحيحة هنا", "خيار آخر مختلف", "خيار ثالث مختلف", "خيار رابع مختلف"] if language == "ar" \
            else ["It removes errors from the data", "It adds new rows", "It renames the course", "It deletes the dataset"]
        return {"question": t["mc"], "options": options, "correct_index": 0, "rationale": why}
    if kind == "true_false":
        return {"statement": t["tf"], "answer": True, "rationale": why}
    return {"question": t["sa"], "model_answer": t["answer"], "rationale": why}


def scripted(language, plans):
    """
    ModuleQuestions answers: plans = {"M1": [edit_for_attempt_0, edit_for_attempt_1, ...]}; an attempt without an edit
    (or a module without a plan) is the fake model's normal, valid answer.
    """
    good = new_type_answers(language)["ModuleQuestions"][0]
    counts = defaultdict(int)

    def answer(prompt):
        module_id = _unit_id(prompt, "Module id")
        attempt, counts[module_id] = counts[module_id], counts[module_id] + 1
        data = good(prompt)
        edits = plans.get(module_id, [])
        if attempt < len(edits) and edits[attempt] is not None:
            edits[attempt](data)
        return data

    return answer


def all_of_the_above(data):
    data["multiple_choice"][0]["options"][3] = "All of the above"


def duplicate_options(data):
    data["multiple_choice"][0]["options"][2] = data["multiple_choice"][0]["options"][0]


def llm_for(language="en", plans=None, **items):
    answers = {"ModuleQuestions": scripted(language, plans or {})}
    answers.update({name: [value] * 5 for name, value in items.items()})
    return fake(language, **answers)


def steps_of(llm, name):
    return llm.steps().count(name)


ITEM_STEPS = ("McQuestion", "TfQuestion", "SaQuestion")


# ---- the workflow: who is called, in which order ---------------------------------------

@pytest.mark.parametrize("language", ["en", "ar"])
def test_when_the_first_answer_passes_no_repair_is_called(toolbox, language):
    cid, _ = curriculum_record(toolbox, language)
    llm = llm_for(language)

    result = generate(agent_for(toolbox, llm), cid, "question_bank")

    assert result["status"] == "success" and llm.steps() == ["ModuleQuestions", "ModuleQuestions"]


@pytest.mark.parametrize("language", ["en", "ar"])
def test_when_the_first_repair_passes_the_second_is_not_called(toolbox, language):
    cid, _ = curriculum_record(toolbox, language)
    llm = llm_for(language, {"M1": [all_of_the_above]})

    result = generate(agent_for(toolbox, llm), cid, "question_bank")

    assert result["status"] == "success"
    assert llm.steps() == ["ModuleQuestions"] * 3 and not any(steps_of(llm, s) for s in ITEM_STEPS)     # M1, M1 again, M2


@pytest.mark.parametrize("language", ["en", "ar"])
def test_when_the_first_repair_fails_the_second_repair_saves_a_valid_document(toolbox, language):
    cid, _ = curriculum_record(toolbox, language)
    llm = llm_for(language, {"M1": [all_of_the_above, duplicate_options]}, McQuestion=good_item("multiple_choice", language))

    result = generate(agent_for(toolbox, llm), cid, "question_bank")

    assert result["status"] == "success", result
    assert llm.steps() == ["ModuleQuestions", "ModuleQuestions", "McQuestion", "ModuleQuestions"]       # M1, repair 1, repair 2, M2
    stored = toolbox.store.get(result["result"]["document_id"])
    assert stored["version_status"] == "pending_review" and not stored["validation"]["errors"]
    options = [b for b in blocks_of(stored["content"], "questions") if b.get("role") == "q_options"]
    assert options[0]["items"] == good_item("multiple_choice", language)["options"]               # the rewritten question is in


@pytest.mark.parametrize("language", ["en", "ar"])
def test_when_both_repairs_fail_the_run_stops_and_nothing_is_saved(toolbox, language):
    cid, _ = curriculum_record(toolbox, language)
    still_bad = {**good_item("multiple_choice", language), "options": ["A thing", "Another thing", "A third thing", "All of the above"]}
    if language == "ar":
        still_bad["options"] = ["شيء أول", "شيء ثان", "شيء ثالث", "جميع ما سبق"]
    llm = llm_for(language, {"M1": [all_of_the_above, duplicate_options]}, McQuestion=still_bad)

    result = generate(agent_for(toolbox, llm), cid, "question_bank")

    assert result["status"] == "error" and result["result"]["kind"] == "validation_failed"
    assert toolbox.store.list() == []
    assert steps_of(llm, "McQuestion") == 1 and steps_of(llm, "ModuleQuestions") == 2          # no third attempt, M2 never written
    assert "[M1]" in result["error_message"] and "Nothing was saved" in result["error_message"]
    assert ("two repair attempts" in result["error_message"]) or ("محاولتي إصلاح" in result["error_message"])


def test_a_replacement_that_is_still_invalid_is_not_corrected_silently(toolbox):
    cid, _ = curriculum_record(toolbox)
    indexed = {**good_item("multiple_choice"), "options": ["First, 0", "Second, 1", "Third, 2", "Fourth, 3"]}
    llm = llm_for("en", {"M1": [all_of_the_above, duplicate_options]}, McQuestion=indexed)

    result = generate(agent_for(toolbox, llm), cid, "question_bank")

    assert result["status"] == "error" and "labels its position" in result["error_message"] and toolbox.store.list() == []


# ---- only the invalid question is rewritten -------------------------------------------------

def test_only_the_flagged_question_is_rewritten_and_the_others_are_kept_exactly(toolbox):
    unit, content = unit_of(toolbox, mutate=lambda u: u["multiple_choice"][0]["options"].__setitem__(2, u["multiple_choice"][0]["options"][0]))
    problems = check_unit(unit, TEMPLATE, content)
    assert [p["target"] for p in problems] == ["multiple_choice:0"]
    llm = FakeLLM({"McQuestion": [good_item("multiple_choice")]})

    repaired = generator._targeted_repair(llm, TEMPLATE, content, "en", "M1", unit, problems, unit, problems)

    assert llm.steps() == ["McQuestion"]                                             # one small call, not the whole module
    assert repaired.multiple_choice[0].options == good_item("multiple_choice")["options"]
    assert repaired.multiple_choice[1] == unit.multiple_choice[1]
    assert repaired.true_false == unit.true_false and repaired.short_answer == unit.short_answer
    assert repaired.module_id == unit.module_id == "M1"
    assert check_unit(repaired, TEMPLATE, content) == []


def test_every_flagged_question_gets_its_own_call_and_valid_ones_are_untouched(toolbox):
    def break_two(u):
        u["multiple_choice"][1]["options"][3] = "None of the above"
        u["short_answer"][0]["model_answer"] = "Check the data again. Check the data again."

    unit, content = unit_of(toolbox, mutate=break_two)
    problems = check_unit(unit, TEMPLATE, content)
    assert sorted({p["target"] for p in problems}) == ["multiple_choice:1", "short_answer:0"]
    llm = FakeLLM({"McQuestion": [good_item("multiple_choice")], "SaQuestion": [good_item("short_answer")]})

    repaired = generator._targeted_repair(llm, TEMPLATE, content, "en", "M1", unit, problems, unit, problems)

    assert sorted(llm.steps()) == ["McQuestion", "SaQuestion"]
    assert repaired.multiple_choice[0] == unit.multiple_choice[0] and repaired.true_false == unit.true_false
    assert repaired.multiple_choice[1].question == ITEMS["en"]["mc"] and repaired.short_answer[0].question == ITEMS["en"]["sa"]
    assert check_unit(repaired, TEMPLATE, content) == []


def test_a_problem_that_points_at_no_question_falls_back_to_writing_the_module_again(toolbox):
    unit, content = unit_of(toolbox)
    llm = FakeLLM({"ModuleQuestions": [unit.model_dump()]})

    repaired = generator._targeted_repair(llm, TEMPLATE, content, "en", "M1", unit, [issue("x", "Something general.", "M1")],
                                          unit, [issue("x", "Something general.", "M1")])

    assert llm.steps() == ["ModuleQuestions"] and "Something general." in llm.prompts("ModuleQuestions")[0] and repaired == unit


# ---- what the second repair is told ----------------------------------------------------------

def test_the_second_repair_prompt_has_the_module_both_problem_lists_and_both_versions(toolbox):
    first, content = unit_of(toolbox, mutate=lambda u: u["multiple_choice"][0]["options"].__setitem__(3, "All of the above"))
    current, _ = unit_of(toolbox, mutate=lambda u: u["multiple_choice"][0].update(options=["Same text", "Different one", "Same text", "Another one"]))
    llm = FakeLLM({"McQuestion": [good_item("multiple_choice")]})

    generator._targeted_repair(llm, TEMPLATE, content, "en", "M1", first, check_unit(first, TEMPLATE, content),
                               current, check_unit(current, TEMPLATE, content))
    prompt = llm.prompts("McQuestion")[0]

    assert "Module id: M1" in prompt and "Module: Data preparation" in prompt                      # the exact module
    assert 'must not use "all/none of the above"' in prompt and "option 1 and option 3" in prompt     # first and current errors
    assert "All of the above" in prompt and "Same text" in prompt                                    # first and current content
    assert "What must change" in prompt.split("Problems that remain")[1]                             # after both problem lists
    assert "different, complete and meaningful statements" in prompt.split("What must change")[1]    # the explicit instruction
    assert current.multiple_choice[1].question in prompt and "Do not number or label options" in prompt   # the others, for contrast


def test_duplicate_questions_are_repaired_by_rewriting_the_repeated_one(toolbox):
    def repeat(u):
        u["short_answer"][0]["question"] = u["multiple_choice"][0]["question"]

    cid, _ = curriculum_record(toolbox)
    llm = llm_for("en", {"M1": [repeat, repeat]}, SaQuestion=good_item("short_answer"))

    result = generate(agent_for(toolbox, llm), cid, "question_bank")

    assert result["status"] == "success" and steps_of(llm, "SaQuestion") == 1 and steps_of(llm, "McQuestion") == 0
    assert "A question is repeated" in llm.prompts("SaQuestion")[0] and "genuinely different question" in llm.prompts("SaQuestion")[0]


@pytest.mark.parametrize("language", ["en", "ar"])
def test_repeated_sentences_are_repaired_by_rewriting_that_answer_only(toolbox, language):
    loop = "تحقق من البيانات مرة أخرى مباشرة. تحقق من البيانات مرة أخرى مباشرة." if language == "ar" \
        else "Check the data again carefully. Check the data again carefully."

    def looping(data):
        data["short_answer"][0]["model_answer"] = loop

    cid, _ = curriculum_record(toolbox, language)
    llm = llm_for(language, {"M1": [looping, looping]}, SaQuestion=good_item("short_answer", language))

    result = generate(agent_for(toolbox, llm), cid, "question_bank")

    assert result["status"] == "success" and steps_of(llm, "SaQuestion") == 1
    assert "A sentence is repeated" in llm.prompts("SaQuestion")[0] and "Say each thing once" in llm.prompts("SaQuestion")[0]
    answers = [b for b in blocks_of(toolbox.store.get(result["result"]["document_id"])["content"], "questions") if b.get("role") == "q_model_answer"]
    assert answers[0]["text"] == ITEMS[language]["answer"] and loop not in json.dumps(answers, ensure_ascii=False)


def test_foreign_script_is_repaired_by_rewriting_the_question_that_has_it(toolbox):
    def chinese(data):
        data["true_false"][0]["rationale"] = "Because 最后 it is right."

    cid, _ = curriculum_record(toolbox)
    llm = llm_for("en", {"M1": [chinese, chinese]}, TfQuestion=good_item("true_false"))

    result = generate(agent_for(toolbox, llm), cid, "question_bank")

    assert result["status"] == "success" and steps_of(llm, "TfQuestion") == 1 and "another writing system" in llm.prompts("TfQuestion")[0]


# ---- the lifecycle is the same in both languages, and the rest is unchanged ----------------------

@pytest.mark.parametrize("language", ["en", "ar"])
def test_a_bank_that_needed_the_second_repair_still_completes_the_whole_lifecycle(admin_system, language):
    llm = llm_for(language, {"M1": [all_of_the_above, duplicate_options]}, McQuestion=good_item("multiple_choice", language))
    lifecycle(admin_system, "question_bank", language, llm=llm)                 # generate -> edit -> v2 -> approve -> PDF -> v1 kept
    assert steps_of(llm, "McQuestion") == 1


def test_only_the_question_bank_has_the_second_repair_and_other_types_get_no_extra_calls(toolbox):
    assert [t for t, template in templates.TEMPLATES.items() if template.targeted_repair] == ["question_bank"]
    cid, _ = curriculum_record(toolbox)
    good = new_type_answers()["ModuleTeaching"][0]

    def always_minutes(prompt):
        answer = good(prompt)
        answer["instructor_notes"] = "Spend 45 minutes on this."
        return answer

    llm = fake(ModuleTeaching=always_minutes)
    result = generate(agent_for(toolbox, llm), cid, "instructor_guide")

    assert result["status"] == "error" and "one repair attempt" in result["error_message"]
    assert steps_of(llm, "ModuleTeaching") == 2 and not any(steps_of(llm, s) for s in ITEM_STEPS)     # first answer + one repair only
    assert "target" not in result["error_message"]


def test_problems_of_other_types_carry_no_target(toolbox):
    _, record = curriculum_record(toolbox)
    template = templates.TEMPLATES["instructor_guide"]
    base = copy.deepcopy(new_type_answers()["ModuleTeaching"][0]("Module id: M1"))
    base["instructor_notes"] = "Spend 45 minutes on this."
    problems = check_unit(template.unit_schema.model_validate(base), template, record["content"])
    assert problems and all("target" not in p for p in problems)


def test_the_existing_strict_checks_still_apply_to_the_second_repairs_result(toolbox):
    unit, content = unit_of(toolbox)
    # a replacement with a duplicated option would be reported again, exactly as before
    bad = copy.deepcopy(unit.model_dump())
    bad["multiple_choice"][0]["options"][1] = bad["multiple_choice"][0]["options"][0]
    problems = check_unit(type(unit).model_validate(bad), TEMPLATE, content)
    assert codes(problems) == ["question_options"] and problems[0]["target"] == "multiple_choice:0"
    assert "options that are the same" in problems[0]["message"]
