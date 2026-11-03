"""
Grounding checks for Agent 5: a campaign may say only what the approved course (and the administrator's own words) support.
Covers invented facts and claims, channel limits, variations that are not different, and the Arabic checks.
"""

import copy

import pytest

from agents.marketing import schemas
from agents.marketing.facts import allowed_for
from agents.marketing.validation import check_channel, check_strategy, text_problems, validate_content
from tests.curriculum_fixtures import valid_curriculum
from tests.marketing_fakes import STRATEGY, TEXTS

EN, AR = allowed_for(valid_curriculum("en")), allowed_for(valid_curriculum("ar"))


def codes(problems):
    return {p["code"] for p in problems}


# ---- invented facts and claims -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, code", [
    ("Enrol for only $199 today", "price_claim"), ("The fees are low", "price_claim"), ("A free course for everyone", "price_claim"),
    ("Cheap and affordable", "price_claim"), ("Costs 500 EGP", "price_claim"),
    ("Get a 20% discount", "offer_claim"), ("Limited seats available", "offer_claim"), ("Don't miss the early bird offer", "offer_claim"),
    ("Last chance to join", "offer_claim"), ("Hurry, act now", "offer_claim"),
    ("We guarantee your success", "guarantee_claim"), ("Land a dream job after the course", "guarantee_claim"),
    ("Double your salary", "guarantee_claim"), ("Get hired by top firms", "guarantee_claim"),
    ("The best machine learning course", "superlative_claim"), ("Rated number one in the region", "superlative_claim"),
    ("A world-class programme", "superlative_claim"),
    ("Earn a certificate of completion", "certificate_claim"), ("An accredited diploma", "certificate_claim"),
    ("Visit https://example.com to join", "contact_detail"), ("Write to info@academy.org", "contact_detail"),
    ("Call +20 100 123 4567", "contact_detail"), ("See www.academy.com", "contact_detail"),
    ("Starts in March", "date_claim"), ("Registration deadline is near", "date_claim"), ("Join us in May", "date_claim"),
    ("No experience needed", "prerequisite_claim"), ("Anyone can join", "prerequisite_claim"),
])
def test_unsupported_claims_are_rejected_in_english(text, code):
    assert code in codes(text_problems(text, EN)), text


@pytest.mark.parametrize("text, code", [
    ("بسعر خاص جدًا", "price_claim"), ("دورة مجانية للجميع", "price_claim"), ("رسوم مخفضة", "price_claim"),
    ("خصم كبير للمسجلين", "offer_claim"), ("مقاعد محدودة", "offer_claim"), ("لا تفوّت الفرصة", "offer_claim"),
    ("نضمن لك النجاح ونوفر وظيفة مضمونة", "guarantee_claim"), ("رواتب عالية بعد الدورة", "guarantee_claim"),
    ("أفضل دورة في المنطقة", "superlative_claim"), ("لا مثيل لها", "superlative_claim"),
    ("احصل على شهادة معتمدة", "certificate_claim"), ("نبدأ في مارس", "date_claim"), ("الموعد النهائي قريب", "date_claim"),
    ("بدون خبرة سابقة", "prerequisite_claim"),
])
def test_unsupported_claims_are_rejected_in_arabic(text, code):
    assert code in codes(text_problems(text, AR)), text


@pytest.mark.parametrize("text", [
    "Learn to prepare a dataset for modelling in a beginner course.", "This course may suit you, and you may enjoy it.",
    "Best practices for cleaning data.", "Practise cleaning data and training a classifier with Python.",
    "A 40-hour beginner course with 2 modules.", "Two modules of 20 hours each cover data preparation and evaluation.",
    "Join the course", "Ask about the outline",
])
def test_ordinary_course_text_passes_in_english(text):
    assert text_problems(text, EN) == [], text


@pytest.mark.parametrize("text", [
    "التدرب والممارسة العملية على تجهيز البيانات", "أفضل الممارسات في تنظيف البيانات", "دورة من 40 ساعة للمبتدئين",
    "وحدتان تطبيقيتان تغطيان التصنيف", "ممارسة عملية مع بايثون وSQL"])
def test_ordinary_course_text_passes_in_arabic(text):
    assert text_problems(text, AR) == [], text                    # (in particular "ممارسة" is not the month "مارس")


def test_hours_and_numbers_must_come_from_the_course():
    assert "hours_mismatch" in codes(text_problems("A 12 hours course", EN))
    assert "hours_mismatch" in codes(text_problems("دورة من 12 ساعة", AR))
    assert "number_claim" in codes(text_problems("Over 5000 learners enrolled", EN))
    assert "number_claim" in codes(text_problems("انضم إليها 5000 متعلم", AR))
    assert "number_claim" in codes(text_problems("Since 2015 we teach data", EN))               # a year is a number the course never gives
    assert text_problems("The 40 hours are split across 2 modules of 20 hours", EN) == []        # the course's own figures
    assert text_problems("دورة من ٤٠ ساعة", AR) == []                                              # Arabic-Indic digits count as digits


def test_accreditation_claims_are_rejected_as_in_the_other_agents():
    assert "accreditation_claim" in codes(text_problems("This course is accredited internationally", EN))
    assert "accreditation_claim" in codes(text_problems("دورة معتمدة من الوزارة", AR))            # Agent 3's own pattern
    assert codes(text_problems("دورة معتمدة دوليًا", AR)) == {"certificate_claim"}                  # and this agent's wider one


def test_the_administrators_own_words_are_a_source_of_facts():
    notes = "Starts in March. Register at www.academy.example. Early bird for 3 days."
    allowed = allowed_for(valid_curriculum("en"), [notes])
    assert text_problems("Starts in March - early bird for 3 days", allowed) == []
    assert "date_claim" in codes(text_problems("Starts in June", allowed))                       # only what they actually supplied
    assert "number_claim" in codes(text_problems("Early bird for 4 days", allowed))


def test_a_term_the_curriculum_itself_uses_is_allowed():
    content = valid_curriculum("en")
    content["overview"] = "A practical course that ends with a certificate of completion."
    assert text_problems("Finish with a certificate", allowed_for(content)) == []
    assert "certificate_claim" in codes(text_problems("Finish with a certificate", EN))


# ---- language ---------------------------------------------------------------------------------------------------------------

def test_foreign_scripts_are_rejected():
    assert "foreign_script" in codes(text_problems("Learn data 最后 science", EN))
    assert "foreign_script" in codes(text_problems("تعلّم البيانات и ещё", AR))


def test_english_inside_arabic_text_is_limited_to_technical_terms():
    assert "foreign_word" in codes(text_problems("تعلم المزيد uso من الأدوات", AR))
    assert "foreign_word" in codes(text_problems("تعلم والevaluating للنماذج", AR))             # glued to Arabic letters
    assert text_problems("نستخدم Python و SQL و pandas للتحليل", AR) == []
    assert text_problems("تعلم scikit-learn بسهولة", AR) == []
    assert text_problems("تعلم LO1 بسهولة", AR) == []
    assert text_problems("نحلل البيانات ونتعلم", AR) == []


def test_an_english_word_the_course_uses_is_fine_in_arabic_and_english_text_is_unrestricted():
    content = valid_curriculum("ar")
    content["modules"][0]["topics"].append("dataframes")
    assert text_problems("نستخدم dataframes في الوحدة", allowed_for(content)) == []
    assert "foreign_word" in codes(text_problems("نستخدم dataframes في الوحدة", AR))
    assert text_problems("Use the dataframes in module one", EN) == []


def test_arabic_hashtags_may_be_latin_words():
    caption = {"text": "هل تريد بناء نماذج بسيطة وتقييم نموذج تصنيف؟ ابدأ خطوة بعد خطوة.", "hashtags": ["datascience", "تعلم_الآلة"]}
    other = {"text": "للخريجين الجدد: ابدأ رحلتك مع تعلم الآلة وتدرب على تجهيز البيانات بأسلوب عملي.", "hashtags": []}
    assert check_channel("social_captions", [caption, other], AR, 2) == []


def test_repeated_sentences_and_unfinished_text_are_rejected():
    assert "repeated_text" in codes(text_problems("Learn data skills now. Learn data skills now.", EN))
    assert "incomplete_text" in codes(text_problems("Learn data skills and,", EN))


# ---- channels, limits and variations --------------------------------------------------------------------------------------

def variations(channel, language="en", count=2):
    return copy.deepcopy(TEXTS[language][channel][:count])


@pytest.mark.parametrize("channel", schemas.CHANNELS)
@pytest.mark.parametrize("language, allowed", [("en", EN), ("ar", AR)])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_the_canned_answers_pass_for_every_channel_language_and_count(channel, language, allowed, count):
    assert check_channel(channel, variations(channel, language, count), allowed, count) == []


@pytest.mark.parametrize("channel", schemas.CHANNELS)
def test_the_number_of_variations_must_match_the_request(channel):
    problems = check_channel(channel, variations(channel, count=2), EN, 3)
    assert codes(problems) == {"variation_count"} and "exactly 3" in problems[0]["message"]
    assert codes(check_channel(channel, [], EN, 2)) == {"variation_count"} and codes(check_channel(channel, None, EN, 2)) == {"variation_count"}


@pytest.mark.parametrize("channel, field, too_short, too_long", [
    ("social_captions", "text", "Short one.", "x " * 200), ("ad_copy", "headline", "Hi", "word " * 30),
    ("ad_copy", "primary_text", "Too short.", "word " * 80), ("ad_copy", "cta", "Go" * 0 + "a", "Learn about this wonderful course today now"),
    ("linkedin_posts", "text", "Too short for a post.", "word " * 200), ("email", "subject", "Hi", "word " * 30),
    ("email", "preheader", "Hi", "word " * 40), ("email", "body", "Too short.", "word " * 300), ("email", "cta", "a", "word " * 20),
    ("whatsapp", "message", "Hi there", "word " * 120), ("headlines", "text", "Hi", "word " * 30), ("ctas", "text", "a", "Join" + "x" * 70),
])
def test_each_channels_length_limits_are_enforced(channel, field, too_short, too_long):
    for bad in (too_short, too_long):
        items = variations(channel)
        items[0][field] = bad
        assert "length" in codes(check_channel(channel, items, EN, 2)), (channel, field, bad[:20])


def test_empty_text_and_missing_fields_are_rejected():
    items = variations("email")
    items[1]["subject"] = "   "
    del items[0]["body"]
    assert "empty_text" in codes(check_channel("email", items, EN, 2))


def test_hashtag_rules():
    items = variations("social_captions")
    items[0]["hashtags"] = ["one", "two", "three", "four", "five", "six"]
    assert "hashtags" in codes(check_channel("social_captions", items, EN, 2))
    items[0]["hashtags"] = ["two words"]
    assert "hashtags" in codes(check_channel("social_captions", items, EN, 2))
    items[0]["hashtags"] = ["#Python", "Learning"]
    assert check_channel("social_captions", items, EN, 2) == []


@pytest.mark.parametrize("channel", schemas.CHANNELS)
def test_variations_that_are_the_same_or_nearly_the_same_are_rejected(channel):
    items = variations(channel)
    main = schemas.MAIN_FIELD[channel]
    items[1][main] = items[0][main]
    assert "variations_similar" in codes(check_channel(channel, items, EN, 2))
    items[1][main] = items[0][main].replace("a", "e", 1) if "a" in items[0][main] else items[0][main] + "."
    assert "variations_similar" in codes(check_channel(channel, items, EN, 2))


def test_claims_inside_a_variation_name_the_channel_and_the_variation():
    items = variations("whatsapp")
    items[1]["message"] = "Hi! Our best course starts in March, with a free workshop."
    problems = check_channel("whatsapp", items, EN, 2)

    assert {"superlative_claim", "date_claim", "price_claim"} <= codes(problems)
    assert all("WhatsApp promotional messages, variation 2" in p["message"] for p in problems)


# ---- the strategy --------------------------------------------------------------------------------------------------------------

def strategy(language="en"):
    return {**copy.deepcopy(STRATEGY[language]), "target_audience": "New graduates and analysts"}


@pytest.mark.parametrize("language, allowed", [("en", EN), ("ar", AR)])
def test_the_canned_strategies_pass(language, allowed):
    assert check_strategy(strategy(language), allowed) == []


def test_strategy_problems():
    bad = strategy()
    bad["objective"], bad["main_message"] = "Reach 10000 learners this year", "Guaranteed success"
    bad["recommended_channels"], bad["content_sequence"], bad["call_to_action"] = ["tiktok"], ["only one"], ""
    found = codes(check_strategy(bad, EN))
    assert {"number_claim", "guarantee_claim", "structure", "empty_text"} <= found
    assert codes(check_strategy(None, EN)) == {"structure"}
    assert "structure" in codes(check_strategy({**strategy(), "target_audience": ""}, EN))
    stepped = strategy()
    stepped["content_sequence"] = ["1. Introduce the course", "2. Share what learners practise", "3. Invite readers to join"]
    assert check_strategy(stepped, EN) == []                                                      # step numbers are not "numbers"


# ---- the whole stored campaign ---------------------------------------------------------------------------------------------------

@pytest.fixture
def stored(tmp_path):
    from tests.marketing_fakes import approved_course, in_process_marketing_tools, make_campaign

    toolbox = in_process_marketing_tools(tmp_path)
    cid = approved_course(toolbox)
    campaign_id = make_campaign(toolbox, cid)["result"]["campaign_id"]
    record = toolbox._approved_record(cid)
    return copy.deepcopy(toolbox.store.get(campaign_id)["content"]), record


def test_a_generated_campaign_is_valid(stored):
    content, record = stored
    result = validate_content(content, record)
    assert result == {"valid": True, "errors": [], "warnings": []}


@pytest.mark.parametrize("mutate, code", [
    (lambda c: c.update(campaign_type="newsletter"), "structure"), (lambda c: c.update(language="ar"), "language"),
    (lambda c: c.update(curriculum_version=7), "traceability"), (lambda c: c.update(curriculum_id="CUR-aaaaaaaaaaaa"), "traceability"),
    (lambda c: c.update(audience={"text": "", "source": "curriculum"}), "audience"), (lambda c: c.update(audience={"text": "x", "source": "model"}), "audience"),
    (lambda c: c.update(tone="sarcastic"), "structure"), (lambda c: c.update(variations=5), "structure"),
    (lambda c: c.update(channels={}), "structure"), (lambda c: c["channels"].update(tiktok={"variations": []}), "structure"),
    (lambda c: c["strategy"].update(objective="We guarantee a job"), "guarantee_claim"),
    (lambda c: c["channels"]["email"]["variations"][0].update(subject="Free course for you"), "price_claim"),
])
def test_a_stored_campaign_with_a_problem_is_not_valid(stored, mutate, code):
    content, record = stored
    mutate(content)
    result = validate_content(content, record)
    assert not result["valid"] and code in codes(result["errors"])


def test_not_an_object_is_not_valid(stored):
    assert not validate_content("nope", stored[1])["valid"] and not validate_content(None, stored[1])["valid"]


# ---- "promotional" in a channel name is not an offer (found with the real model) -----------------------------------------------------

@pytest.mark.parametrize("language, allowed", [("en", EN), ("ar", AR)])
@pytest.mark.parametrize("channel", schemas.CHANNELS)
def test_every_channel_name_is_accepted_as_text(channel, language, allowed):
    """The model is given these names and repeats them in its content sequence, so the checks must not reject them."""
    name = schemas.CHANNEL_LABELS[channel][language]
    assert text_problems(name, allowed) == [], name
    if language == "en":
        assert text_problems(f"3. {name}", allowed, enumerated=True) == [], name            # as a numbered step of the sequence


@pytest.mark.parametrize("text", [
    "Send the WhatsApp promotional messages in week two", "Publish WhatsApp promotional messages after the e-mail",
    "A promotional message that introduces the course", "Share promotional content about the modules", "The promotional campaign starts with a caption"])
def test_promotional_messages_and_content_are_not_offers(text):
    assert "offer_claim" not in codes(text_problems(text, EN)), text


def test_the_strategy_may_name_the_whatsapp_channel_in_a_step():
    strategy = {"objective": "Attract new graduates and analysts to the beginner course", "target_audience": "New graduates and analysts",
                "main_message": "Learn machine learning by practising data preparation and evaluating simple models.",
                "recommended_channels": ["email", "whatsapp"], "call_to_action": "Join the course",
                "content_sequence": ["Introduce the course by e-mail", "Share what learners will practise",
                                     "Follow up with WhatsApp promotional messages", "Invite readers to take the next step"]}
    assert check_strategy(strategy, EN) == []


@pytest.mark.parametrize("text", [
    "Use promo code SPRING at checkout", "Our promo ends soon", "Get the promotional offer today", "A promotional price for early joiners",
    "Enjoy the promotional discount", "Ask for the promotion offer", "Use this coupon", "Save with our promotional rates",
    "Take advantage of the promotional deal", "Apply the promocode", "50% off for everyone", "Get a discount"])
def test_actual_promotions_are_still_rejected(text):
    assert "offer_claim" in codes(text_problems(text, EN)), text


def test_arabic_offers_and_invalid_english_fragments_are_still_rejected():
    assert "offer_claim" in codes(text_problems("خصم كبير للمسجلين وعرض محدود", AR))
    assert "foreign_word" in codes(text_problems("تعلم كيف تعمل preprocess للبيانات", AR))                 # the real model's WhatsApp output
    assert "foreign_word" in codes(text_problems("الانضباط disciplين في العمل", AR))                          # the real model's strategy output
    assert "foreign_word" in codes(text_problems("ابدأ بـ whatsapp ثم أرسل رسالة", AR))


# ---- call to action: at most 8 words and at most 60 characters (found with the real model) ---------------------------------------------

def cta_variations(*texts):
    return [{"text": t} for t in texts]


def test_a_cta_of_eight_words_and_more_than_forty_characters_is_valid():
    real_model_style = "Explore the beginner course and start practising today"        # 8 words, 53 characters
    assert len(real_model_style.split()) == 8 and 40 < len(real_model_style) <= 60
    assert check_channel("ctas", cta_variations(real_model_style, "Join the first module"), EN, 2) == []


def test_a_cta_of_more_than_eight_words_is_rejected_even_when_it_is_short():
    nine_words = "Join us to learn it all in one go"                                    # 9 words, 33 characters
    assert len(nine_words.split()) == 9 and len(nine_words) <= 60
    problems = check_channel("ctas", cta_variations(nine_words, "Join the first module"), EN, 2)
    assert codes(problems) == {"word_count"} and "9 words" in problems[0]["message"] and "at most 8" in problems[0]["message"]


def test_a_cta_of_more_than_sixty_characters_is_rejected_even_with_few_words():
    long_words = "Internationalisation professionalisation transformation"               # 3 words, 55 characters: still fine
    assert check_channel("ctas", cta_variations(long_words, "Join the first module"), EN, 2) == []
    too_long = "Internationalisation professionalisation transformationalisation"        # 3 words, 64 characters
    assert len(too_long.split()) <= 8 and len(too_long) > 60
    problems = check_channel("ctas", cta_variations(too_long, "Join the first module"), EN, 2)
    assert codes(problems) == {"length"} and "must have 3 to 60" in problems[0]["message"]


def test_the_limits_are_eight_words_and_sixty_characters_everywhere_they_are_stated():
    from agents.marketing import prompts

    assert schemas.LIMITS["ctas"]["text"] == (3, 60) and schemas.MAX_CTA_WORDS == 8
    assert "8 words (and at most 60 characters)" in schemas.CallToAction.model_fields["text"].description
    assert "at most 8 words and at most 60 characters" in prompts.CHANNEL_TASKS["ctas"] and "40 characters" not in prompts.CHANNEL_TASKS["ctas"]


def test_the_other_buttons_keep_their_own_limits():
    assert schemas.LIMITS["ad_copy"]["cta"] == (3, 30) and schemas.LIMITS["email"]["cta"] == (3, 40)
    ad = [{"headline": "Start machine learning with Python", "primary_text": "A practical first course for new graduates and analysts: prepare data, then evaluate a model.",
           "cta": "x" * 31}, TEXTS["en"]["ad_copy"][1]]
    assert "length" in codes(check_channel("ad_copy", ad, EN, 2))


def test_cta_claims_are_still_checked():
    problems = check_channel("ctas", cta_variations("Get a free certificate now", "Book before March"), EN, 2)
    assert {"price_claim", "certificate_claim", "date_claim"} <= codes(problems)


def test_arabic_ctas_follow_the_same_limits():
    arabic = "ابدأ دورتك العملية الأولى في تعلم الآلة مع الوحدة الأولى"               # 10 words
    assert len(arabic.split()) > 8
    assert "word_count" in codes(check_channel("ctas", cta_variations(arabic, "اكتشف الدورة"), AR, 2))
    assert check_channel("ctas", cta_variations("ابدأ دورتك العملية الأولى اليوم", "اكتشف الدورة"), AR, 2) == []
