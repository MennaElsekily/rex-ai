"""
The bilingual companion a freshly generated curriculum gets automatically
(agents/curriculum_research/bilingual.py): a translation of the already-
generated, already-saved curriculum, never a second research-and-generate
run, so both languages describe exactly the same curriculum. Covers
translate_curriculum() directly, generate_companion() end to end against
the real toolbox on temporary storage, and the curriculum_agent.py hook
that calls it only for a fresh generation, never a revision.
"""

import re

import pytest

from agents.curriculum_research.bilingual import (
    COMPANION_LANGUAGE,
    MAX_ITEMS_PER_CALL,
    generate_companion,
    translate_curriculum,
)
from agents.curriculum_research.validation import is_measurable
from agents.curriculum_research.curriculum_agent import CurriculumAgent
from agents.curriculum_research.llm import CurriculumModelError
from tests.curriculum_fakes import FakeLLM, bilingual_answers, in_process_tools
from tests.curriculum_fakes import translation_answer as fake_translation
from tests.curriculum_fixtures import valid_curriculum


# --------------------------------------------------------------------
# translate_curriculum: structure, numbers and ids preserved
# --------------------------------------------------------------------

def test_translate_curriculum_keeps_every_number_id_and_the_level_literal():
    content = valid_curriculum("en", hours=40)
    llm = FakeLLM({"TranslationBatch": fake_translation("ar")})

    translated = translate_curriculum(llm, content)

    assert translated["language"] == "ar"
    assert translated["total_hours"] == content["total_hours"] == 40
    assert translated["level"] == content["level"] == "Beginner"   # never translated: a controlled literal
    assert [m["id"] for m in translated["modules"]] == [m["id"] for m in content["modules"]]
    assert [m["hours"] for m in translated["modules"]] == [m["hours"] for m in content["modules"]]
    assert [o["id"] for o in translated["learning_outcomes"]] == [o["id"] for o in content["learning_outcomes"]]
    assert [m["outcome_ids"] for m in translated["modules"]] == [m["outcome_ids"] for m in content["modules"]]
    assert translated["tools_resources"] == content["tools_resources"]   # tool names stay as written


def test_translate_curriculum_translates_every_collected_text_field():
    content = valid_curriculum("en", hours=40)
    llm = FakeLLM({"TranslationBatch": fake_translation("ar")})

    translated = translate_curriculum(llm, content)

    assert translated["title"] != content["title"] and translated["title"].startswith("يطبق")
    assert translated["overview"] != content["overview"]
    assert translated["modules"][0]["title"] != content["modules"][0]["title"]
    assert translated["modules"][0]["description"] != content["modules"][0]["description"]
    assert translated["modules"][0]["topics"][0] != content["modules"][0]["topics"][0]
    assert translated["modules"][0]["activities"][0]["description"] != content["modules"][0]["activities"][0]["description"]
    assert translated["assessments"][0]["description"] != content["assessments"][0]["description"]
    assert translated["capstone"]["description"] != content["capstone"]["description"]
    assert translated["capstone"]["deliverables"] != content["capstone"]["deliverables"]


def test_arabic_to_english_is_the_companion_direction_too():
    content = valid_curriculum("ar", hours=30)
    llm = FakeLLM({"TranslationBatch": fake_translation("en")})

    translated = translate_curriculum(llm, content)

    assert translated["language"] == "en" == COMPANION_LANGUAGE["ar"]
    assert translated["total_hours"] == 30


def test_a_translation_count_that_never_matches_raises_after_bounded_attempts():
    content = valid_curriculum("en", hours=40)
    calls = {"n": 0}

    def always_two(prompt: str) -> dict:
        calls["n"] += 1
        return {"translations": ["يطبق البند الأول.", "يطبق البند الثاني."]}   # never matches any count

    llm = FakeLLM({"TranslationBatch": always_two})

    with pytest.raises(CurriculumModelError, match="bilingual companion was not created"):
        translate_curriculum(llm, content)
    # The first chunk, then its first item alone, then that item's one retry -- never more.
    assert calls["n"] == 3


def _numbered_translation(prompt: str) -> dict:
    """Translations that carry their source item's own text, so ordering can be checked."""
    import re

    items = re.findall(r"^\d+\. (.*)$", prompt.split("Translate each of these", 1)[1], flags=re.MULTILINE)
    return {"translations": [f"يطبق ترجمة البند التدريبي: {item}" for item in items]}


def test_translation_is_chunked_and_every_text_lands_back_in_its_own_place():
    """A real 3B model lost track of a whole curriculum in one call; now no call asks for more than a few items."""
    content = valid_curriculum("en", hours=40)
    llm = FakeLLM({"TranslationBatch": _numbered_translation})

    translated = translate_curriculum(llm, content)

    prompts = llm.prompts("TranslationBatch")
    assert len(prompts) > 1
    assert all(len(re.findall(r"^\d+\. ", p.split("Translate each of these", 1)[1], flags=re.MULTILINE))
               <= MAX_ITEMS_PER_CALL for p in prompts)
    assert translated["title"].endswith(content["title"])
    assert translated["learning_outcomes"][-1]["statement"].endswith(content["learning_outcomes"][-1]["statement"])
    assert translated["modules"][-1]["description"].endswith(content["modules"][-1]["description"])
    assert translated["capstone"]["deliverables"][-1].endswith(content["capstone"]["deliverables"][-1])


def test_a_miscounted_chunk_falls_back_to_one_item_per_call_in_order():
    content = valid_curriculum("en", hours=40)
    first = {"done": False}

    def miscount_first_chunk(prompt: str) -> dict:
        if not first["done"]:
            first["done"] = True
            return {"translations": ["يطبق شيئا ما فقط."]}   # 1 item for a 5-item chunk
        return _numbered_translation(prompt)

    llm = FakeLLM({"TranslationBatch": miscount_first_chunk})

    translated = translate_curriculum(llm, content)

    assert translated["title"].endswith(content["title"])
    assert translated["overview"].endswith(content["overview"])
    one_item_calls = [p for p in llm.prompts("TranslationBatch") if "these 1 items" in p]
    assert len(one_item_calls) == MAX_ITEMS_PER_CALL


def test_an_untranslated_item_or_a_vague_outcome_verb_is_retranslated_on_its_own():
    """The real model once left an objective in English and translated an outcome's verb as "فهم" (understand)."""
    content = valid_curriculum("en", hours=40)
    vague_outcome = content["learning_outcomes"][0]["statement"]

    def flawed_then_fixed(prompt: str) -> dict:
        body = prompt.split("Translate each of these", 1)[1]
        items = re.findall(r"^\d+\. (.*)$", body, flags=re.MULTILINE)
        retry = "Answer entirely in" in body
        out = []
        for item in items:
            if item == content["overview"] and not retry:
                out.append(item)   # left in English
            elif item == vague_outcome and not retry:
                out.append("فهم أساسيات الموضوع والمفاهيم المرتبطة به")   # vague verb
            else:
                out.append(f"يطبق ترجمة البند التدريبي: {item}")
        return {"translations": out}

    llm = FakeLLM({"TranslationBatch": flawed_then_fixed})

    translated = translate_curriculum(llm, content)

    assert translated["overview"].startswith("يطبق")
    assert is_measurable(translated["learning_outcomes"][0]["statement"], "ar")
    retries = [p for p in llm.prompts("TranslationBatch") if "Answer entirely in" in p]
    assert len(retries) == 2
    assert "measurable action verb" in next(p for p in retries if vague_outcome in p).split("Translate each", 1)[1]


def test_a_companion_that_fails_validation_is_kept_as_an_unapprovable_draft(tmp_path):
    """
    The same rule as every curriculum: saved, linked and visible, but a
    draft with its validation errors -- never pending review, so it
    cannot be approved until fixed. (It used to be silently discarded,
    so the admin never saw a second language whenever the source was a draft.)
    """
    toolbox = in_process_tools(tmp_path)
    content = valid_curriculum("en", hours=40)
    en_id = toolbox.save_curriculum_draft(curriculum=content, origin="request")["curriculum_id"]

    counter = iter(range(1, 10_000))

    def no_verb(prompt: str) -> dict:
        # No verb at all (nothing safe_repair.py could fix) -- and each one distinct, so no duplicate retries.
        items = re.findall(r"^\d+\. (.*)$", prompt.split("Translate each of these", 1)[1], flags=re.MULTILINE)
        return {"translations": [f"الموضوع رقم {next(counter)} والمفاهيم المرتبطة به بشكل عام" for _ in items]}

    errors = []
    companion = generate_companion(toolbox, FakeLLM({"TranslationBatch": no_verb}), content, en_id,
                                   on_error=errors.append)

    assert companion is not None and not errors
    assert companion["review_status"] == "draft"
    assert "outcome_not_measurable" in {e["code"] for e in companion["validation"]["errors"]}
    record = toolbox.store.get(companion["curriculum_id"])
    assert record["status"] == "draft" and record["content"]["language"] == "ar"
    assert toolbox.store.translation_of(en_id) == companion["curriculum_id"]
    assert toolbox.get_curriculum(companion["curriculum_id"])["documents"] == []   # a draft never gets a PDF
    with pytest.raises(Exception):
        toolbox.store.approve(companion["curriculum_id"], 1, "admin-test", None)
    assert toolbox.store.get(companion["curriculum_id"])["status"] == "draft"
    toolbox.close()


def test_a_draft_english_source_still_gets_its_arabic_companion_end_to_end(tmp_path):
    """The exact real-app scenario: the English draft fails validation, yet both languages are saved and linked."""
    toolbox = in_process_tools(tmp_path)
    answers = bilingual_answers("en")
    # No verb at all: neither the (fake) model repair nor safe_repair.py can fix it, so it stays a draft.
    answers["CurriculumOutline"]["learning_outcomes"][0]["statement"] = "The development tools in general"
    answers["RepairPatch"] = {"learning_outcomes": [{"id": "LO1", "statement": "The development tools in general"}]}
    agent = CurriculumAgent(tools=toolbox, llm=FakeLLM(answers))

    response = agent.process_request({"request_id": "r1", "user_input":
                                      "Create a beginner 40-hour curriculum on cloud computing for IT support staff",
                                      "context": {}})

    result = response["result"]
    assert toolbox.store.get(result["curriculum_id"])["status"] == "draft"
    companion_id = result.get("bilingual_companion_id")
    assert companion_id, result.get("bilingual_companion_error")
    assert toolbox.store.translation_of(result["curriculum_id"]) == companion_id
    assert toolbox.store.get(companion_id)["content"]["language"] == "ar"
    # Neither has a PDF before an admin approves it.
    assert toolbox.get_curriculum(companion_id)["documents"] == []
    assert toolbox.get_curriculum(result["curriculum_id"])["documents"] == []
    agent.close()


def test_a_translated_vague_verb_is_safely_fixed_before_the_companion_is_saved(tmp_path):
    toolbox = in_process_tools(tmp_path)
    content = valid_curriculum("en", hours=40)
    en_id = toolbox.save_curriculum_draft(curriculum=content, origin="request")["curriculum_id"]
    outcome_sources = {o["statement"] for o in content["learning_outcomes"]}
    counter = iter(range(1, 10_000))

    def understand_everywhere(prompt: str) -> dict:
        items = re.findall(r"^\d+\. (.*)$", prompt.split("Translate each of these", 1)[1], flags=re.MULTILINE)
        return {"translations": [(f"يفهم المفهوم رقم {next(counter)} في هذا الموضوع التدريبي" if item in outcome_sources
                                  else f"يطبق البند التدريبي رقم {next(counter)} في هذا الموضوع")
                                 for item in items]}

    companion = generate_companion(toolbox, FakeLLM({"TranslationBatch": understand_everywhere}), content, en_id)

    record = toolbox.store.get(companion["curriculum_id"])
    assert all(o["statement"].startswith("يشرح") for o in record["content"]["learning_outcomes"])
    assert companion["review_status"] == "pending_review"
    assert any("استُبدل الفعل" in a for a in record["content"]["assumptions"])   # the change is explained
    toolbox.close()


def test_two_sources_translated_identically_are_retranslated_on_their_own():
    content = valid_curriculum("en", hours=40)
    retried, numbers = [], iter(range(1, 10_000))

    def same_for_title_and_overview(prompt: str) -> dict:
        body = prompt.split("Translate each of these", 1)[1]
        items = re.findall(r"^\d+\. (.*)$", body, flags=re.MULTILINE)
        out = []
        for item in items:
            if item in (content["title"], content["overview"]) and "Answer entirely in" not in body:
                out.append("يطبق نصًا واحدًا مكررًا لعنصرين مختلفين")
            else:
                if "Answer entirely in" in body:
                    retried.append(item)
                out.append(f"يطبق ترجمة البند التدريبي رقم {next(numbers)}")
        return {"translations": out}

    translated = translate_curriculum(FakeLLM({"TranslationBatch": same_for_title_and_overview}), content)

    assert translated["title"] != translated["overview"]
    assert retried == [content["overview"]]   # only the second of the two was asked again


def test_a_model_failure_raises_curriculum_model_error():
    content = valid_curriculum("en")
    llm = FakeLLM({"TranslationBatch": RuntimeError("ollama is down")})

    with pytest.raises(CurriculumModelError):
        translate_curriculum(llm, content)


# --------------------------------------------------------------------
# generate_companion: end to end against the real toolbox
# --------------------------------------------------------------------

def test_generate_companion_saves_and_links_the_sibling_without_an_unapproved_pdf(tmp_path):
    toolbox = in_process_tools(tmp_path)
    content = valid_curriculum("en", hours=40)
    saved = toolbox.save_curriculum_draft(curriculum=content, origin="request")
    en_id = saved["curriculum_id"]
    llm = FakeLLM({"TranslationBatch": fake_translation("ar")})

    companion = generate_companion(toolbox, llm, content, en_id)

    assert companion is not None and companion["status"] == "ok"
    ar_id = companion["curriculum_id"]
    assert ar_id != en_id

    record = toolbox.get_curriculum(ar_id)["curriculum"]
    assert record["content"]["language"] == "ar" and record["content"]["total_hours"] == 40
    assert toolbox.store.translation_of(en_id) == ar_id
    assert toolbox.store.translation_of(ar_id) == en_id
    assert companion["review_status"] == "pending_review"
    assert toolbox.get_curriculum(ar_id)["documents"] == []   # its PDF comes only with its own approval
    toolbox.store.approve(ar_id, 1, "admin-test")
    assert toolbox.render_curriculum_pdf(ar_id)["document"]["language"] == "ar"
    toolbox.close()


def test_generate_companion_never_blocks_or_undoes_the_original_on_failure(tmp_path):
    toolbox = in_process_tools(tmp_path)
    content = valid_curriculum("en", hours=40)
    saved = toolbox.save_curriculum_draft(curriculum=content, origin="request")
    en_id = saved["curriculum_id"]
    llm = FakeLLM({"TranslationBatch": RuntimeError("ollama is down")})
    errors = []

    companion = generate_companion(toolbox, llm, content, en_id, on_error=errors.append)

    assert companion is None and errors
    assert toolbox.store.translation_of(en_id) is None   # nothing was linked
    original = toolbox.get_curriculum(en_id)["curriculum"]
    assert original["content"] == content   # the original is completely unaffected
    toolbox.close()


# --------------------------------------------------------------------
# curriculum_agent.py: fresh generation gets a companion, a revision does not
# --------------------------------------------------------------------

def test_a_fresh_curriculum_request_gets_a_bilingual_companion(tmp_path):
    toolbox = in_process_tools(tmp_path)
    llm = FakeLLM(bilingual_answers("en"))
    agent = CurriculumAgent(tools=toolbox, llm=llm)

    response = agent.process_request({"request_id": "r1", "user_input":
                                      "Create a beginner 40-hour curriculum on cloud computing for IT support staff",
                                      "context": {}})

    assert response["status"] == "success"
    companion_id = response["result"].get("bilingual_companion_id")
    assert companion_id, response["result"].get("bilingual_companion_error")
    companion_record = toolbox.get_curriculum(companion_id)["curriculum"]
    assert companion_record["content"]["language"] == "ar"
    assert toolbox.store.translation_of(response["result"]["curriculum_id"]) == companion_id
    agent.close()


def test_a_revision_does_not_generate_a_second_companion(tmp_path):
    toolbox = in_process_tools(tmp_path)
    llm = FakeLLM(bilingual_answers("en"))
    agent = CurriculumAgent(tools=toolbox, llm=llm)

    first = agent.process_request({"request_id": "r1", "user_input":
                                   "Create a beginner 40-hour curriculum on cloud computing for IT support staff",
                                   "context": {"conversation_id": "c1"}})
    curriculum_id = first["result"]["curriculum_id"]
    before = toolbox.store.translation_of(curriculum_id)

    revised = agent.process_request({"request_id": "r2", "user_input": "Add a module on logging and monitoring",
                                     "context": {"conversation_id": "c1"}})

    assert revised["status"] == "success"
    assert "bilingual_companion_id" not in revised["result"] and "bilingual_companion_error" not in revised["result"]
    assert toolbox.store.translation_of(curriculum_id) == before   # unchanged
    agent.close()




def test_an_echoed_item_number_is_removed_from_the_translation():
    """The real model once answered LO3 as "2. ..." -- the prompt's own numbering, read as the outcome's verb."""
    content = valid_curriculum("en", hours=40)
    numbers = iter(range(1, 10_000))

    def echo_numbers(prompt: str) -> dict:
        items = re.findall(r"^\d+\. (.*)$", prompt.split("Translate each of these", 1)[1], flags=re.MULTILINE)
        return {"translations": [f"{i + 1}. يطبق البند التدريبي رقم {next(numbers)}" for i in range(len(items))]}

    translated = translate_curriculum(FakeLLM({"TranslationBatch": echo_numbers}), content)

    assert all(o["statement"].startswith("يطبق") for o in translated["learning_outcomes"])
    assert translated["title"].startswith("يطبق")


def test_a_source_that_starts_with_a_number_keeps_it():
    content = valid_curriculum("en", hours=40)
    content["modules"][0]["topics"] = ["1. Cleaning data"]

    def keep(prompt: str) -> dict:
        items = re.findall(r"^\d+\. (.*)$", prompt.split("Translate each of these", 1)[1], flags=re.MULTILINE)
        return {"translations": [("1. تنظيف البيانات" if item == "1. Cleaning data" else f"يطبق بندًا تدريبيًا {i}")
                                 for i, item in enumerate(items)]}

    translated = translate_curriculum(FakeLLM({"TranslationBatch": keep}), content)

    assert translated["modules"][0]["topics"] == ["1. تنظيف البيانات"]


def test_an_empty_translation_is_retranslated_not_kept():
    content = valid_curriculum("en", hours=40)
    retried, numbers = [], iter(range(1, 10_000))

    def empty_overview(prompt: str) -> dict:
        body = prompt.split("Translate each of these", 1)[1]
        items = re.findall(r"^\d+\. (.*)$", body, flags=re.MULTILINE)
        if "Answer entirely in" in body:
            retried.extend(items)
        return {"translations": [("" if item == content["overview"] and "Answer entirely in" not in body
                                  else f"يطبق ترجمة البند رقم {next(numbers)}")
                                 for i, item in enumerate(items)]}

    translated = translate_curriculum(FakeLLM({"TranslationBatch": empty_overview}), content)

    assert translated["overview"].strip()
    assert content["overview"] in retried


# Real-model runs (qwen2.5:3b, 2026-11-11): 3 of 5 Arabic companions of a valid English curriculum were drafts.
# The exact texts the model wrote are used below.

def _items(prompt: str) -> list[str]:
    return re.findall(r"^\d+\. (.*)$", prompt.split("Translate each of these", 1)[1], flags=re.MULTILINE)


def test_the_can_lead_in_is_removed_when_an_action_verb_follows():
    content = valid_curriculum("en", hours=40)
    first = content["learning_outcomes"][0]["statement"]

    def literal_can(prompt: str) -> dict:
        return {"translations": ["يمكن للمتعلم أن يطبق مفاهيم التسويق الرقمي الرئيسية" if item == first
                                 else f"يطبق البند التدريبي: {item}" for item in _items(prompt)]}

    llm = FakeLLM({"TranslationBatch": literal_can})
    translated = translate_curriculum(llm, content)

    assert translated["learning_outcomes"][0]["statement"] == "يطبق مفاهيم التسويق الرقمي الرئيسية"
    assert not [p for p in llm.prompts("TranslationBatch") if "Answer entirely in" in p]   # fixed without a retry


def test_a_can_lead_in_without_a_measurable_verb_after_it_is_retranslated_not_kept():
    content = valid_curriculum("en", hours=40)
    first = content["learning_outcomes"][0]["statement"]

    def literal_then_fixed(prompt: str) -> dict:
        retry = "Answer entirely in" in prompt
        return {"translations": [("يحدد مسارات التسويق الرقمي ويقيمها" if retry else
                                  "يمكن للطلاب اكتشاف وفحص مسارات التسويق الرقمي") if item == first
                                 else f"يطبق البند التدريبي: {item}" for item in _items(prompt)]}

    llm = FakeLLM({"TranslationBatch": literal_then_fixed})
    translated = translate_curriculum(llm, content)

    assert translated["learning_outcomes"][0]["statement"] == "يحدد مسارات التسويق الرقمي ويقيمها"
    retry = next(p for p in llm.prompts("TranslationBatch") if "Answer entirely in" in p)
    assert "يمكن للطلاب اكتشاف" in retry and "does not start with a measurable action verb" in retry


def test_an_untranslated_title_is_retried_with_its_previous_answer_until_it_is_arabic():
    content = valid_curriculum("en", hours=40)
    content["modules"][1]["title"] = "Formulas: IF, XLOOKUP, SUMIFS"
    attempts = []

    def stubborn(prompt: str) -> dict:
        out = []
        for item in _items(prompt):
            if item == "Formulas: IF, XLOOKUP, SUMIFS":
                attempts.append(prompt)
                out.append(item if len(attempts) < 3 else "الصيغ: IF و XLOOKUP و SUMIFS")   # twice unchanged
            else:
                out.append(f"يطبق البند التدريبي: {item}")
        return {"translations": out}

    translated = translate_curriculum(FakeLLM({"TranslationBatch": stubborn}), content)

    assert translated["modules"][1]["title"] == "الصيغ: IF و XLOOKUP و SUMIFS"
    assert len(attempts) == 3                              # the first answer, then two retries
    assert "Formulas: IF, XLOOKUP, SUMIFS\": it is still mostly not in" in attempts[2]


def test_an_item_still_wrong_after_every_retry_keeps_the_companion_a_draft(tmp_path):
    toolbox = in_process_tools(tmp_path)
    content = valid_curriculum("en", hours=40)
    content["modules"][1]["title"] = "Formulas: IF, XLOOKUP, SUMIFS"
    en_id = toolbox.save_curriculum_draft(curriculum=content, origin="request")["curriculum_id"]
    counter = iter(range(1, 10_000))

    def never_translates_the_title(prompt: str) -> dict:
        return {"translations": [item if item == "Formulas: IF, XLOOKUP, SUMIFS"
                                 else f"يطبق البند التدريبي رقم {next(counter)}: {item}" for item in _items(prompt)]}

    llm = FakeLLM({"TranslationBatch": never_translates_the_title})
    companion = generate_companion(toolbox, llm, content, en_id)

    assert companion["review_status"] == "draft"                        # validation is not bypassed
    assert any(e["code"] == "language_mismatch" for e in companion["validation"]["errors"])
    title_calls = [p for p in llm.prompts("TranslationBatch") if "Formulas: IF, XLOOKUP, SUMIFS" in _items(p)]
    assert len(title_calls) == 1 + 3                                    # bounded: the first answer and 3 retries
    assert toolbox.store.translation_of(en_id) == companion["curriculum_id"]
    toolbox.close()


def test_both_languages_reach_pending_review_after_the_fixes(tmp_path):
    toolbox = in_process_tools(tmp_path)
    content = valid_curriculum("en", hours=40)
    content["modules"][1]["title"] = "Formulas: IF, XLOOKUP, SUMIFS"
    saved = toolbox.save_curriculum_draft(curriculum=content, origin="request")
    assert saved["review_status"] == "pending_review"
    first = content["learning_outcomes"][0]["statement"]
    counter = iter(range(1, 10_000))

    def real_model_like(prompt: str) -> dict:
        retry = "Answer entirely in" in prompt
        out = []
        for item in _items(prompt):
            if item == "Formulas: IF, XLOOKUP, SUMIFS":
                out.append("الصيغ: IF و XLOOKUP و SUMIFS" if retry else item)
            elif item == first:
                out.append("يستطيع الطالب استخدام برامج وأدوات التسويق الرقمي الأساسية")
            else:
                out.append(f"يطبق البند التدريبي رقم {next(counter)}: {item}")
        return {"translations": out}

    companion = generate_companion(toolbox, FakeLLM({"TranslationBatch": real_model_like}), content, saved["curriculum_id"])

    assert companion["review_status"] == "pending_review", companion["validation"]["errors"]
    record = toolbox.store.get(companion["curriculum_id"])
    assert record["content"]["learning_outcomes"][0]["statement"] == "استخدام برامج وأدوات التسويق الرقمي الأساسية"
    assert toolbox.store.get(saved["curriculum_id"])["status"] == "pending_review"   # the English one is untouched
    assert toolbox.store.translation_of(saved["curriculum_id"]) == companion["curriculum_id"]
    toolbox.close()


def test_an_outcome_label_the_model_added_is_removed():
    content = valid_curriculum("en", hours=40)
    first = content["learning_outcomes"][0]["statement"]

    def labelled(prompt: str) -> dict:
        return {"translations": [("LO5: يطبق تعابير SQL لاستخراج البيانات" if item == first
                                  else f"يطبق البند التدريبي: {item}") for item in _items(prompt)]}

    translated = translate_curriculum(FakeLLM({"TranslationBatch": labelled}), content)

    assert translated["learning_outcomes"][0]["statement"] == "يطبق تعابير SQL لاستخراج البيانات"


def test_an_identical_source_text_reuses_its_acceptable_translation():
    # Real model: "Introduction to Python" was the module title AND its first topic; the title came back as
    # "introduction إلى Python" on every retry while the topic was "مقدمة إلى Python".
    content = valid_curriculum("en", hours=40)
    content["modules"][0]["title"] = "Introduction to Python"
    content["modules"][0]["topics"] = ["Introduction to Python", *content["modules"][0]["topics"][1:]]
    seen = []

    def title_always_bad(prompt: str) -> dict:
        out = []
        for item in _items(prompt):
            if item == "Introduction to Python":
                seen.append(item)
                out.append("introduction إلى Python" if len(seen) == 1 else "مقدمة إلى Python")
            else:
                out.append(f"يطبق البند التدريبي: {item}")
        return {"translations": out}

    llm = FakeLLM({"TranslationBatch": title_always_bad})
    translated = translate_curriculum(llm, content)

    assert translated["modules"][0]["title"] == translated["modules"][0]["topics"][0] == "مقدمة إلى Python"
    assert not [p for p in llm.prompts("TranslationBatch") if "Answer entirely in" in p]   # reused, no retry needed


def test_a_retry_that_returns_the_wrong_count_does_not_lose_the_companion(tmp_path):
    # Real model: one retry came back with 4 items for 1 source text, and the whole companion was not created.
    toolbox = in_process_tools(tmp_path)
    content = valid_curriculum("en", hours=40)
    en_id = toolbox.save_curriculum_draft(curriculum=content, origin="request")["curriculum_id"]
    counter = iter(range(1, 10_000))

    def bad_overview_and_miscounted_retries(prompt: str) -> dict:
        items = _items(prompt)
        if "Answer entirely in" in prompt:
            return {"translations": ["أ", "ب", "ج", "د"]}                     # always the wrong count
        return {"translations": [item if item == content["overview"]
                                 else f"يطبق البند التدريبي رقم {next(counter)}: {item}" for item in items]}

    companion = generate_companion(toolbox, FakeLLM({"TranslationBatch": bad_overview_and_miscounted_retries}),
                                   content, en_id)

    assert companion is not None                                          # saved and linked, not lost
    assert toolbox.store.translation_of(en_id) == companion["curriculum_id"]
    assert companion["review_status"] == "draft"                          # the English overview still fails the check


def test_a_runaway_translation_is_retried_and_never_quoted_in_full():
    content = valid_curriculum("en", hours=40)
    first = content["learning_outcomes"][0]["statement"]
    runaway = "ينفذ " + "أ Divider: ي evaluates " * 400

    def loops_once(prompt: str) -> dict:
        retry = "Answer entirely in" in prompt
        return {"translations": [("يطبق المفاهيم الأساسية" if retry else runaway) if item == first
                                 else f"يطبق البند التدريبي: {item}" for item in _items(prompt)]}

    llm = FakeLLM({"TranslationBatch": loops_once})
    translated = translate_curriculum(llm, content)

    assert translated["learning_outcomes"][0]["statement"] == "يطبق المفاهيم الأساسية"
    retry = next(p for p in llm.prompts("TranslationBatch") if "Answer entirely in" in p)
    assert len(retry) < 3000 and "Divider" not in retry                    # the runaway text is never sent back


def test_the_retry_after_a_runaway_asks_for_one_short_translation():
    content = valid_curriculum("en", hours=40)
    first = content["learning_outcomes"][0]["statement"]
    runaway = "ينفذ " + "أ Divider: ي evaluates " * 400

    def loops_until_told(prompt: str) -> dict:
        told = "repeated the same words over and over" in prompt
        return {"translations": [("يطبق المفاهيم الأساسية" if told else runaway) if item == first
                                 else f"يطبق البند التدريبي: {item}" for item in _items(prompt)]}

    translated = translate_curriculum(FakeLLM({"TranslationBatch": loops_until_told}), content)

    assert translated["learning_outcomes"][0]["statement"] == "يطبق المفاهيم الأساسية"


@pytest.mark.parametrize("arabic, english, expected", [
    # Real qwen2.5:3b translations that kept the Arabic companion a draft after every retry.
    ("ينطبق استعلامات SQL لاستخراج البيانات", "Apply SQL queries to extract data.", "يطبق استعلامات SQL لاستخراج البيانات"),
    ("ين Analyze المعايير الأساسية للتسويق الرقمي", "Analyze basic digital marketing metrics.",
     "يحلل المعايير الأساسية للتسويق الرقمي"),
])
def test_an_outcome_verb_still_wrong_after_the_retries_gets_the_source_verbs_arabic_form(arabic, english, expected):
    content = valid_curriculum("en", hours=40)
    content["learning_outcomes"][0]["statement"] = english

    def always_wrong_verb(prompt: str) -> dict:
        return {"translations": [arabic if item == english else f"يطبق البند التدريبي: {item}" for item in _items(prompt)]}

    translated = translate_curriculum(FakeLLM({"TranslationBatch": always_wrong_verb}), content)

    assert translated["learning_outcomes"][0]["statement"] == expected
    assert is_measurable(expected, "ar")


def test_an_outcome_with_no_known_source_verb_is_left_for_validation_to_report():
    content = valid_curriculum("en", hours=40)
    content["learning_outcomes"][0]["statement"] = "Understand project management concepts."

    def no_verb(prompt: str) -> dict:
        return {"translations": ["مفاهيم إدارة المشاريع" if item == "Understand project management concepts."
                                 else f"يطبق البند التدريبي: {item}" for item in _items(prompt)]}

    translated = translate_curriculum(FakeLLM({"TranslationBatch": no_verb}), content)

    assert translated["learning_outcomes"][0]["statement"] == "مفاهيم إدارة المشاريع"   # nothing invented
