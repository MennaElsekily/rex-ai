"""
Regression tests for the exact validation failures reported against the
Marketing Agent: a Facebook/Instagram primary text of 303 characters
(limit 40-300) and email subjects of 78 and 83 characters (limit 8-70)
failed the whole campaign instead of being repaired. agents/marketing/
length_repair.py now mechanically trims (or, for too-short text, extends
with an already-verified course fact) any field whose ONLY remaining
problem, after the model's own one-shot repair, is its length -- never a
claim, structure or count problem, which still fails the campaign exactly
as before.
"""

import pytest

from agents.marketing import schemas
from agents.marketing.facts import allowed_for
from agents.marketing.length_repair import repair_channel_lengths, repair_variation_lengths
from agents.marketing.validation import check_channel
from tests.curriculum_fixtures import valid_curriculum
from tests.marketing_fakes import TEXTS, approved_course, channel_answer, fake_llm, in_process_marketing_tools, make_campaign

EN = allowed_for(valid_curriculum())


def codes(problems):
    return {p["code"] for p in problems}


def seeded(length: int, seed: str = "practical hands-on course modules and exercises for real projects ") -> str:
    """A plain, claim-free sentence repeated to exactly `length` characters."""
    text = (seed * (length // len(seed) + 2))[:length]
    assert len(text) == length
    return text


# --------------------------------------------------------------------
# The exact reported failures
# --------------------------------------------------------------------

def test_a_facebook_primary_text_of_303_characters_is_repaired_to_fit():
    too_long = seeded(303)
    assert len(too_long) == 303 and not (40 <= 303 <= 300)

    variation = repair_variation_lengths("ad_copy", {"headline": "Start now", "primary_text": too_long, "cta": "Learn more"},
                                         EN, "en")

    assert 40 <= len(variation["primary_text"]) <= 300
    assert not codes(check_channel("ad_copy", [variation], EN, 1)) & {"length"}


@pytest.mark.parametrize("subject_length", [78, 83])
def test_an_email_subject_over_70_characters_is_repaired_to_fit(subject_length):
    too_long = seeded(subject_length)
    assert len(too_long) == subject_length and subject_length > 70

    variation = repair_variation_lengths(
        "email", {"subject": too_long, "preheader": "See what you will practise.",
                 "body": seeded(200), "cta": "View the course"}, EN, "en")

    assert 8 <= len(variation["subject"]) <= 70
    assert not codes(check_channel("email", [variation], EN, 1)) & {"length"}


# --------------------------------------------------------------------
# Exact minimum and maximum boundaries
# --------------------------------------------------------------------

@pytest.mark.parametrize("channel, field, low, high", [
    ("ad_copy", "primary_text", 40, 300),
    ("email", "subject", 8, 70),
])
def test_text_exactly_at_the_boundary_is_accepted(channel, field, low, high):
    base = {"ad_copy": {"headline": "Start now", "primary_text": "", "cta": "Learn more"},
           "email": {"subject": "", "preheader": "See what you will practise.", "body": seeded(200),
                     "cta": "View the course"}}[channel]

    for length in (low, high):
        variation = {**base, field: seeded(length)}
        problems = check_channel(channel, [variation], EN, 1)
        assert not codes(problems) & {"length"}, (length, problems)


@pytest.mark.parametrize("channel, field, low, high", [("ad_copy", "primary_text", 40, 300), ("email", "subject", 8, 70)])
def test_text_one_character_past_the_boundary_is_rejected(channel, field, low, high):
    base = {"ad_copy": {"headline": "Start now", "primary_text": "", "cta": "Learn more"},
           "email": {"subject": "", "preheader": "See what you will practise.", "body": seeded(200),
                     "cta": "View the course"}}[channel]

    for length in (low - 1, high + 1):
        variation = {**base, field: seeded(length)}
        assert "length" in codes(check_channel(channel, [variation], EN, 1))


# --------------------------------------------------------------------
# Multiple variations with different lengths
# --------------------------------------------------------------------

def test_only_the_variation_that_is_out_of_range_is_repaired():
    good = {"subject": "Your first machine learning course", "preheader": "Practical and beginner friendly.",
           "body": seeded(200), "cta": "View the course"}
    bad = {**good, "subject": seeded(90)}

    repaired = repair_channel_lengths("email", [good, bad], [{"code": "length", "where": "email[2].subject"}], EN, "en")

    assert repaired[0] == good   # untouched
    assert 8 <= len(repaired[1]["subject"]) <= 70 and repaired[1]["subject"] != good["subject"]


# --------------------------------------------------------------------
# Automatic repair end to end (generator.py), and failed repair
# --------------------------------------------------------------------

def test_the_whole_campaign_is_repaired_automatically_when_only_the_length_is_wrong(tmp_path):
    toolbox = in_process_marketing_tools(tmp_path)
    cid = approved_course(toolbox)
    bad_ad_copy = {"variations": [
        {"headline": "Start machine learning with Python", "primary_text": seeded(303), "cta": "Learn more"},
        {"headline": "Build your first classifier", "primary_text": TEXTS["en"]["ad_copy"][1]["primary_text"], "cta": "See the modules"},
    ]}
    # The model repeats the same too-long text on its own one-shot repair (as the admin observed in practice).
    llm = fake_llm(AdCopySet=[bad_ad_copy, bad_ad_copy])

    response = make_campaign(toolbox, cid, llm=llm, channels=["ad_copy"], variations=2)

    assert response["status"] == "success", response.get("error_message")
    saved = toolbox.store.get(response["result"]["campaign_id"])
    primary_texts = [v["primary_text"] for v in saved["content"]["channels"]["ad_copy"]["variations"]]
    assert all(40 <= len(t) <= 300 for t in primary_texts)
    assert llm.steps().count("AdCopySet") == 2   # no extra model call was needed to fix a pure length problem


def test_a_length_problem_mixed_with_a_claim_problem_never_saves_the_claim(tmp_path):
    """
    A field that is both too long AND makes an unsupported claim: trimming alone never hides the claim. Since
    2026-11-12 the claim's sentence is removed first (sentence_repair.py), then the length is fixed; the saved text
    contains no claim. A text that IS only the claim is still refused (next test).
    """
    toolbox = in_process_marketing_tools(tmp_path)
    cid = approved_course(toolbox)
    bad_ad_copy = {"variations": [
        {"headline": "Start machine learning with Python",
         "primary_text": "Guaranteed to get you hired! " + seeded(303), "cta": "Learn more"},
        {"headline": "Build your first classifier", "primary_text": TEXTS["en"]["ad_copy"][1]["primary_text"], "cta": "See the modules"},
    ]}
    llm = fake_llm(AdCopySet=[bad_ad_copy, bad_ad_copy])

    response = make_campaign(toolbox, cid, llm=llm, channels=["ad_copy"], variations=2)

    assert response["status"] == "success", response.get("error_message")
    saved = toolbox.store.get(response["result"]["campaign_id"])["content"]["channels"]["ad_copy"]["variations"][0]
    assert "guarantee" not in saved["primary_text"].casefold() and "hired" not in saved["primary_text"].casefold()
    assert 40 <= len(saved["primary_text"]) <= 300


def test_a_text_that_is_only_an_unsupported_claim_is_still_refused(tmp_path):
    toolbox = in_process_marketing_tools(tmp_path)
    cid = approved_course(toolbox)
    bad_ad_copy = {"variations": [
        {"headline": "Start machine learning with Python",
         "primary_text": "Guaranteed to get you hired as a machine learning engineer within weeks of finishing!",
         "cta": "Learn more"},
        {"headline": "Build your first classifier", "primary_text": TEXTS["en"]["ad_copy"][1]["primary_text"], "cta": "See the modules"},
    ]}
    llm = fake_llm(AdCopySet=[bad_ad_copy, bad_ad_copy])

    response = make_campaign(toolbox, cid, llm=llm, channels=["ad_copy"], variations=2)

    assert response["status"] == "error" and response["result"]["kind"] == "validation_failed"
    assert toolbox.store.list() == []   # nothing was saved
    assert "guarantee" in response["error_message"]


def test_a_too_short_field_is_extended_with_a_verified_fact_not_invented_text():
    allowed = allowed_for(valid_curriculum())   # the fixture curriculum states total_hours

    variation = repair_variation_lengths("whatsapp", {"message": "Hi there, join us!"}, allowed, "en")

    assert 20 <= len(variation["message"]) <= 450
    assert "Hi there, join us!" in variation["message"]   # nothing original was removed
    assert str(int(sorted(allowed.hours)[0])) in variation["message"]


def test_a_too_short_field_with_no_verified_fact_to_add_is_left_unrepaired_and_still_fails():
    bare = {"language": "en"}   # no hours, no numbers to safely add
    allowed = allowed_for(bare)

    variation = repair_variation_lengths("whatsapp", {"message": "Hi there, join us!"}, allowed, "en")

    assert variation["message"] == "Hi there, join us!"   # unchanged: nothing was invented to pad it
    assert "length" in codes(check_channel("whatsapp", [variation], allowed, 1))


# --------------------------------------------------------------------
# Factual accuracy: the repair never adds a new claim
# --------------------------------------------------------------------

def test_trimming_never_introduces_a_new_unsupported_claim():
    too_long = seeded(303)
    variation = repair_variation_lengths("ad_copy", {"headline": "Start now", "primary_text": too_long, "cta": "Learn more"},
                                         EN, "en")

    assert variation["primary_text"] in too_long   # the trim only ever removes characters from the original text


def test_a_trimmed_text_never_ends_on_a_joining_word():
    # Real model (2026-11-12): an Arabic e-mail subject trimmed to 70 characters ended with "... JavaScript و".
    from agents.marketing.length_repair import _trim_to_fit

    arabic = _trim_to_fit("تصميم تطبيقات الويب المتكيّم باستخدام HTML و CSS و JavaScript و Node.js و Express و MongoDB لبناء تطبيقات", 70)
    english = _trim_to_fit("Learn SQL and Power BI with hands-on labs and real projects for data analysts and managers today", 70)

    assert len(arabic) <= 70 and not arabic.endswith((" و", "و")) and arabic.split()[-1] != "و"
    assert len(english) <= 70 and english.split()[-1] not in ("and", "for", "with")
