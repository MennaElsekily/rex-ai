"""
Call-to-action suggestions that break the limits (found with the real model: 80 characters, 12 and 10 words, twice in a row).
Everything here uses a fake model; no Ollama. The limits themselves (3 to 60 characters, at most 8 words) are unchanged.
"""

import pytest

from agents.marketing import prompts, schemas
from agents.marketing.facts import course_facts
from tests.marketing_fakes import approved_course, channel_answer, fake_llm, in_process_marketing_tools, make_campaign

GOOD = {"variations": [{"text": "Explore the course"}, {"text": "Join the first module"}]}
TOO_LONG = "Start applying machine learning to real-world problems with our course today"          # 11 words, 76 characters
EIGHTY = "Gain practical skills in data preparation and model evaluation through hands-on work now"    # > 60 characters
NINE_WORDS_SHORT = "Join us to learn ML in a fun way"                                              # 9 words, under 60 characters
TWELVE_WORDS = "Start your journey into machine learning by joining the course and its practical modules"
WRONG_ONE = {"variations": [{"text": "Explore the course"}, {"text": TOO_LONG}]}                    # second one only
WRONG_BOTH = {"variations": [{"text": TWELVE_WORDS}, {"text": NINE_WORDS_SHORT}]}


@pytest.fixture
def toolbox(tmp_path):
    return in_process_marketing_tools(tmp_path)


def run(toolbox, llm, **request):
    cid = approved_course(toolbox)
    return cid, make_campaign(toolbox, cid, llm=llm, channels=request.pop("channels", ["ctas"]), **request)


def saved(toolbox):
    return toolbox.store.list()


# ---- the checks stay deterministic --------------------------------------------------------------------------------------------------

def test_the_limits_are_unchanged():
    assert schemas.LIMITS["ctas"]["text"] == (3, 60) and schemas.MAX_CTA_WORDS == 8


@pytest.mark.parametrize("text, code", [
    (EIGHTY, "length"),                                    # more than 60 characters
    (TWELVE_WORDS, "word_count"),                          # more than 8 words (and more than 60 characters)
    (NINE_WORDS_SHORT, "word_count"),                      # more than 8 words but few characters
])
def test_an_over_limit_call_to_action_is_rejected_with_its_exact_numbers(text, code):
    from agents.marketing.validation import check_channel
    from agents.marketing.facts import allowed_for

    problems = check_channel("ctas", [{"text": "Explore the course"}, {"text": text}], allowed_for({}, []), 2)

    assert code in {p["code"] for p in problems}
    messages = " ".join(p["message"] for p in problems)
    assert f'{len(text.strip())} characters' in messages or f"{len(text.split())} words" in messages
    assert f'You wrote: "{text}"' in messages                                           # the offending text is quoted
    assert all(p["where"].startswith("ctas[2]") for p in problems)                      # only the second variation is blamed


def test_limits_are_inclusive_at_the_boundary():
    from agents.marketing.validation import check_channel
    from agents.marketing.facts import allowed_for

    eight_words = "Join the first module of the course"                                  # 7 words
    eight = "Explore the course and join the first module"                              # 8 words, 44 characters
    sixty = ("Join the course " * 4)[:60].strip()
    assert len(eight.split()) == 8 and len(sixty) <= 60
    assert check_channel("ctas", [{"text": eight}, {"text": eight_words}], allowed_for({}, []), 2) == []
    nine = eight + " now"
    assert [p["code"] for p in check_channel("ctas", [{"text": nine}, {"text": eight_words}], allowed_for({}, []), 2)] == ["word_count"]


# ---- generation, repair and persistence ---------------------------------------------------------------------------------------------

def test_valid_suggestions_pass_without_a_repair_and_are_saved_unchanged(toolbox):
    llm = fake_llm(CtaSet=[GOOD])

    cid, response = run(toolbox, llm)

    assert response["status"] == "success" and llm.steps().count("CtaSet") == 1
    campaign = saved(toolbox)
    assert len(campaign) == 1 and campaign[0]["status"] == "pending_review"
    content = toolbox.store.get(campaign[0]["campaign_id"])["content"]
    assert [v["text"] for v in content["channels"]["ctas"]["variations"]] == ["Explore the course", "Join the first module"]


def test_text_over_sixty_characters_is_repaired_and_the_repaired_text_is_saved_untouched(toolbox):
    llm = fake_llm(CtaSet=[{"variations": [{"text": "Explore the course"}, {"text": EIGHTY}]}, GOOD])

    cid, response = run(toolbox, llm)

    assert response["status"] == "success" and llm.steps().count("CtaSet") == 2
    stored = toolbox.store.get(saved(toolbox)[0]["campaign_id"])["content"]["channels"]["ctas"]["variations"]
    assert [v["text"] for v in stored] == ["Explore the course", "Join the first module"]      # nothing was truncated or edited by code


def test_more_than_eight_words_is_repaired(toolbox):
    llm = fake_llm(CtaSet=[{"variations": [{"text": NINE_WORDS_SHORT}, {"text": "Explore the course"}]}, GOOD])

    cid, response = run(toolbox, llm)

    assert response["status"] == "success" and llm.steps().count("CtaSet") == 2


def test_two_invalid_variations_are_both_reported_and_both_repaired(toolbox):
    llm = fake_llm(CtaSet=[WRONG_BOTH, GOOD])

    cid, response = run(toolbox, llm)

    assert response["status"] == "success"
    repair = llm.prompts("CtaSet")[1]
    assert "previous answer had these problems" in repair
    assert "variation 1" in repair and "variation 2" in repair                          # the checker's problems name each variation
    assert f"{len(TWELVE_WORDS.split())} words" in repair and "9 words" in repair                                 # with the exact counts
    assert f'"{TWELVE_WORDS}"' in repair and f'"{NINE_WORDS_SHORT}"' in repair          # and the text that was written
    assert "Rewrite each one as a short action phrase" in repair


def test_the_repair_names_only_the_failed_suggestion_and_keeps_the_valid_one(toolbox):
    llm = fake_llm(CtaSet=[WRONG_ONE, GOOD])

    cid, response = run(toolbox, llm)

    assert response["status"] == "success"
    repair = llm.prompts("CtaSet")[1]
    failed, kept = repair.split("This is what you wrote (these failed the checks):")[1].split("These suggestions are fine")
    assert TOO_LONG in failed and "Explore the course" not in failed
    assert 'keep them exactly as they are' in repair and '1. "Explore the course"' in kept
    assert "variation 2" in repair and "variation 1:" not in repair.split("This is what you wrote")[0]


def test_a_repair_that_still_breaks_the_limits_saves_nothing(toolbox):
    llm = fake_llm(CtaSet=[WRONG_BOTH, WRONG_BOTH])

    cid, response = run(toolbox, llm)

    assert response["status"] == "error" and response["result"]["failed_content"] == "ctas"
    assert "did not pass the checks, even after one repair attempt" in response["error_message"]
    assert "Nothing was saved" in response["error_message"]
    assert "words; it must have at most 8" in response["error_message"]
    assert saved(toolbox) == []                                                         # not a draft, not pending review
    assert llm.steps().count("CtaSet") == 2                                             # one repair, never a third attempt


def test_a_failed_repair_leaves_an_existing_campaign_untouched(toolbox):
    cid = approved_course(toolbox)
    first = make_campaign(toolbox, cid, llm=fake_llm(CtaSet=[GOOD]), channels=["ctas"])
    campaign_id = first["result"]["campaign_id"]
    before = toolbox.store.get(campaign_id)

    again = make_campaign(toolbox, cid, llm=fake_llm(CtaSet=[WRONG_BOTH, WRONG_BOTH]), channels=["ctas"], campaign_id=campaign_id)

    assert again["status"] == "error"
    after = toolbox.store.get(campaign_id)
    assert after["version"] == before["version"] == 1 and after["content"] == before["content"]
    assert len(saved(toolbox)) == 1


def test_a_valid_campaign_is_saved_after_a_successful_repair_with_the_normal_lifecycle(toolbox):
    llm = fake_llm(CtaSet=[WRONG_BOTH, GOOD])

    cid, response = run(toolbox, llm, channels=["social_captions", "ctas"])

    assert response["status"] == "success"
    record = toolbox.store.get(response["result"]["campaign_id"])
    assert record["status"] == "pending_review" and record["approved_version"] is None        # approval is still the admin's step
    assert toolbox.store.get_approved(record["campaign_id"]) is None                          # and nothing is exportable yet


# ---- the prompts ----------------------------------------------------------------------------------------------------------------------

def test_the_first_cta_request_asks_for_well_under_the_limits_and_uses_a_compact_fact_block(toolbox):
    llm = fake_llm(CtaSet=[GOOD])
    cid, _ = run(toolbox, llm, channels=["social_captions", "ctas"])
    course = course_facts(toolbox.curricula.get(cid)["content"])

    cta = llm.prompts("CtaSet")[0]
    social = llm.prompts("SocialCaptionSet")[0]

    assert "aim for 3 to 6 words" in cta and "at most 8 words and at most 60 characters" in cta
    assert f"Course: {course['title']}" in cta and "Target audience (confirmed)" in cta
    module_title = course["modules"][0]["title"]
    assert module_title in social and module_title not in cta                              # the full facts stay for the other content
    assert "Modules:" not in cta and "Learning outcomes:" not in cta and "Objectives:" not in cta


def test_the_claims_checks_still_use_the_whole_course_for_calls_to_action(toolbox):
    claims = {"variations": [{"text": "Get a free certificate now"}, {"text": "Book before March"}]}

    cid, response = run(toolbox, fake_llm(CtaSet=[claims, claims]))

    assert response["result"]["kind"] == "validation_failed"
    assert {"price_claim", "certificate_claim", "date_claim"} <= {p["code"] for p in response["result"]["problems"]}
    assert saved(toolbox) == []


def test_the_repair_note_covers_every_suggestion_when_the_failures_cannot_be_located():
    previous = [{"text": "a b c d e f g h i"}, {"text": "Explore the course"}]
    assert prompts.cta_repair_note(previous, None).count("words,") == 2                      # unknown: all are quoted
    note = prompts.cta_repair_note(previous, {1})
    assert "These suggestions are fine" in note and '2. "Explore the course"' in note.split("These suggestions are fine")[1]
