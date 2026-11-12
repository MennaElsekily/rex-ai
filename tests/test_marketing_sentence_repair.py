"""
agents/marketing/sentence_repair.py: after the model's one repair, a sentence the checks flag (an unsupported claim,
another writing system, English glued to Arabic) is removed -- never reworded, nothing added -- and everything is
checked again by the unchanged rules. Texts below are what the real qwen2.5:3b wrote (2026-11-12).
"""

from __future__ import annotations

from agents.marketing.facts import allowed_for
from agents.marketing.sentence_repair import (
    repair_channel_sentences,
    repair_strategy_sentences,
    without_flagged_sentences,
)
from agents.marketing.validation import check_channel, check_strategy, text_problems
from tests.curriculum_fixtures import valid_curriculum
from tests.marketing_fakes import ADMIN, TEXTS, approved_course, fake_llm, in_process_marketing_tools, make_campaign

COURSE = valid_curriculum()          # it lists prerequisites
ALLOWED = allowed_for(COURSE)
REAL_AD = ("Advance your machine learning skills with our Machine Learning Foundations course. Practise building simple "
           "models with Python. Learn to evaluate a classifier. No prerequisites, just a solid foundation in Python.")


def codes(text):
    return {p["code"] for p in text_problems(text, ALLOWED)}


def test_the_unsupported_prerequisite_sentence_is_removed_and_the_rest_kept_word_for_word():
    assert COURSE["prerequisites"] and "prerequisite_claim" in codes(REAL_AD)

    fixed = without_flagged_sentences(REAL_AD, ALLOWED)

    assert fixed == ("Advance your machine learning skills with our Machine Learning Foundations course. Practise building "
                     "simple models with Python. Learn to evaluate a classifier.")
    assert codes(fixed) == set()


def test_a_text_that_is_only_the_claim_is_left_unchanged_so_it_still_fails():
    text = "Start machine learning today, no prior experience needed."
    assert without_flagged_sentences(text, ALLOWED) == text and "prerequisite_claim" in codes(text)


def test_a_text_without_removable_problems_is_never_touched():
    text = TEXTS["en"]["ad_copy"][0]["primary_text"]
    assert without_flagged_sentences(text, ALLOWED) is text


def test_a_sentence_in_another_writing_system_is_removed_from_an_arabic_email():
    allowed = allowed_for(valid_curriculum("ar"))
    body = TEXTS["ar"]["email"][0]["body"]
    with_chinese = body + " هذا الدرس يشرح 逻辑 البيانات."
    assert "foreign_script" in {p["code"] for p in text_problems(with_chinese, allowed)}

    assert without_flagged_sentences(with_chinese, allowed).strip() == body.strip()


def test_the_channel_repair_only_removes_and_the_unchanged_checks_then_pass():
    variations = [dict(TEXTS["en"]["ad_copy"][0], primary_text=REAL_AD), dict(TEXTS["en"]["ad_copy"][1])]
    assert check_channel("ad_copy", variations, ALLOWED, 2)

    fixed = repair_channel_sentences("ad_copy", variations, ALLOWED)

    assert check_channel("ad_copy", fixed, ALLOWED, 2) == []
    assert fixed[1] == variations[1] and fixed[0]["headline"] == variations[0]["headline"]


def test_a_flagged_hashtag_is_dropped_and_the_others_kept():
    variation = dict(TEXTS["en"]["social_captions"][0], hashtags=["MachineLearning", "Guaranteed", "Python"])

    fixed = repair_channel_sentences("social_captions", [variation], ALLOWED)[0]

    assert fixed["hashtags"] == ["MachineLearning", "Python"]


def test_a_flagged_strategy_step_is_dropped_only_while_three_steps_remain():
    strategy = {"objective": "Promote the course.", "main_message": "Learn machine learning with Python.",
                "call_to_action": "Explore the course",
                "content_sequence": ["Post the course overview", "Share a module highlight", "Publish the best course ad",
                                     "Send the outline by e-mail"],
                "recommended_channels": ["social_captions"], "target_audience": "Analysts"}
    assert check_strategy(strategy, ALLOWED)

    fixed = repair_strategy_sentences(strategy, ALLOWED)

    assert fixed["content_sequence"] == ["Post the course overview", "Share a module highlight", "Send the outline by e-mail"]
    assert check_strategy(fixed, ALLOWED) == []
    three = dict(strategy, content_sequence=strategy["content_sequence"][1:])
    assert repair_strategy_sentences(three, ALLOWED)["content_sequence"] == three["content_sequence"]   # never below 3


def test_end_to_end_the_campaign_is_saved_without_the_claim_after_the_model_kept_it(tmp_path):
    toolbox = in_process_marketing_tools(tmp_path)
    cid = approved_course(toolbox)
    ads = {"variations": [dict(TEXTS["en"]["ad_copy"][0], primary_text=REAL_AD), TEXTS["en"]["ad_copy"][1]]}
    llm = fake_llm(AdCopySet=[ads, ads])                     # the model keeps the claim in its repair too

    response = make_campaign(toolbox, cid, llm=llm, channels=["ad_copy"], variations=2)

    assert response["status"] == "success", response.get("error_message")
    record = toolbox.store.get(response["result"]["campaign_id"])
    saved = record["content"]["channels"]["ad_copy"]["variations"][0]["primary_text"]
    assert "prerequisite" not in saved.casefold() and saved.startswith("Advance your machine learning skills")
    assert record["status"] == "pending_review"              # still waits for the administrator's approval
    assert llm.steps().count("AdCopySet") == 2               # the model's own repair was tried first
    assert ADMIN not in str(record.get("approved_by"))


def test_a_one_letter_arabic_prefix_glued_to_an_english_term_only_gets_a_space():
    from agents.marketing.sentence_repair import spaced_prefixes

    assert spaced_prefixes("تعلم بناء تطبيقات ويب وNode.js وExpress") == "تعلم بناء تطبيقات ويب و Node.js و Express"
    assert spaced_prefixes("دورة لPython") == "دورة ل Python"
    assert spaced_prefixes("الكتابة وتحليلMongoDB") == "الكتابة وتحليلMongoDB"     # not a prefix: left for the checks
    assert spaced_prefixes("تحليل Power BI") == "تحليل Power BI"
