"""
REX Assistant's course-recommendation rules
(agents/rex_assistant/recommendation.py): profile labels, the per-course
answer layout and the deterministic checks on the model's answer.

Run through REX's own generate_answer() in recommendation mode.

Mocked: the chat model.
"""

from agents.rex_assistant.graph import generate_answer
from agents.rex_assistant.prompts import MODE_QUESTION, MODE_RECOMMENDATION
from tests.test_rex_answer_generation import FakeLLM, _chunk, _evidence


def write_recommendation(llm, evidence, language, profile, declined_fields=None):
    return generate_answer(llm, evidence, language, mode=MODE_RECOMMENDATION,
                           profile=profile, declined=declined_fields)


def test_learner_profile_labels_and_course_layout_are_in_the_prompt():
    llm = FakeLLM()

    write_recommendation(llm, _evidence(), "en",
                         {"career_goal": "data analyst", "training_preference": "online"},
                         declined_fields=["budget"])

    assert "LEARNER PROFILE" in llm.user
    assert "Career goal: data analyst" in llm.user
    assert "Training preference (online/offline): online" in llm.user
    assert "Budget: not provided (the learner skipped it)" in llm.user
    assert "Background: not provided" in llm.user
    assert "Why it fits you" in llm.user


# --------------------------------------------------------------------
# Preferences the evidence does not cover (live finding: the model said
# a course was "available online" and fit a 5000 budget; the catalogue
# states neither delivery mode nor fees).
# --------------------------------------------------------------------

MARKETING = _chunk(
    "Digital Marketing Strategy & Planning. Level: Foundation. Delivery: "
    "Workshops, case exercises, and campaign-plan review. Entry requirements: "
    "No previous marketing role required.", section="Programme 01")
PROFILE = {"career_goal": "marketing", "skill_level": "beginner",
           "training_preference": "online", "budget": "5000",
           "available_time": "2 hours per day"}


def test_unstated_delivery_and_fees_are_flagged_after_a_recommendation():
    llm = FakeLLM("Digital Marketing Strategy & Planning is available online. [1]")

    answer = write_recommendation(llm, _evidence(chunks=[MARKETING]), "en", PROFILE)["answer"]

    assert "Not stated in the catalogue" in answer
    assert "online or offline" in answer
    assert "budget (5000)" in answer


def test_the_note_is_in_arabic_for_arabic_replies():
    result = write_recommendation(FakeLLM("دورة مناسبة. [1]"), _evidence(chunks=[MARKETING]), "ar", PROFILE)

    assert "غير مذكور في الكتالوج" in result["answer"]
    assert "أونلاين" in result["answer"]


def test_no_note_when_the_evidence_states_delivery_and_fees():
    chunk = _chunk("Delivery: fully online. Fees: 4500 EGP.")
    result = write_recommendation(FakeLLM("Fits. [1]"), _evidence(chunks=[chunk]), "en", PROFILE)

    assert result["answer"] == "Fits. [1]"


def test_no_note_for_preferences_the_learner_did_not_give():
    profile = {"career_goal": "marketing", "training_preference": "no_preference"}
    result = write_recommendation(FakeLLM("Fits. [1]"), _evidence(chunks=[MARKETING]), "en", profile)

    assert result["answer"] == "Fits. [1]"


def test_direct_questions_get_no_preference_note():
    result = generate_answer(FakeLLM("Workshops. [1]"), _evidence(chunks=[MARKETING]), "en",
                             mode=MODE_QUESTION, profile=PROFILE)

    assert result["answer"] == "Workshops. [1]"


def test_recommendation_prompt_demands_stated_facts_per_course():
    llm = FakeLLM("Fits. [1]")
    write_recommendation(llm, _evidence(chunks=[MARKETING]), "en", PROFILE)

    assert 'not stated in the catalogue' in llm.user
    assert 'never "the learner"' in llm.user
    assert "never call it a match" in llm.user


# Live finding: the model repeated a course, renamed another
# ("Digital Marketing Analytics") and called an Intermediate course a fit
# for a beginner.
MODEL_REPLY = """Given your profile, I recommend:

**Digital Marketing Strategy & Planning** [1]
- Why it fits you: designed for beginners.
- Level: Foundation

**Digital Marketing Analytics** [2]
- Why it fits you: suitable for beginners.
- Level: Intermediate

**Marketing Analytics & Campaign Optimization** [2]
- Why it fits you: builds on strategy.
- Level: Intermediate

**Digital Marketing Strategy & Planning** [1]
- Why it fits you: designed for beginners.
- Level: Foundation

Start with Digital Marketing Strategy & Planning."""

ANALYTICS = _chunk("Marketing Analytics & Campaign Optimization. Level: Intermediate. "
                   "Fees: not listed online.", reference=2)


def _recommend(reply, chunks, profile=PROFILE, language_code="en"):
    return write_recommendation(FakeLLM(reply), _evidence(chunks=chunks), language_code, profile)["answer"]


def test_repeated_courses_are_listed_once():
    answer = _recommend(MODEL_REPLY, [MARKETING, ANALYTICS])

    assert answer.count("**Digital Marketing Strategy & Planning**") == 1
    assert answer.rstrip().endswith("Start with Digital Marketing Strategy & Planning.")


def test_courses_not_named_in_the_evidence_are_dropped():
    answer = _recommend(MODEL_REPLY, [MARKETING, ANALYTICS])

    assert "Digital Marketing Analytics" not in answer
    assert "**Marketing Analytics & Campaign Optimization**" in answer


def test_courses_above_the_learners_level_are_marked_as_a_step_up():
    answer = _recommend(MODEL_REPLY, [MARKETING, ANALYTICS])

    assert "- Level: Intermediate (above your beginner level: this may be a step up)" in answer
    assert "- Level: Foundation\n" in answer


def test_nothing_is_dropped_when_no_course_name_can_be_verified():
    reply = "**Some Course** [1]\n- Level: Foundation"

    assert "**Some Course**" in _recommend(reply, [MARKETING])


def test_answers_without_course_blocks_are_unchanged():
    chunk = _chunk("Delivery: fully online. Fees: 4500 EGP.")
    assert _recommend("No course fits your goal.", [chunk]) == "No course fits your goal."


# --------------------------------------------------------------------
# Live finding (2026-10-27): the model wrote "**Name** [MOB-101]" headers,
# which the checks did not recognise, so none of them ran. It also added
# an irrelevant course with a wrong ID, repeated "Which course to start
# with" in every block, and did not flag a fee above the budget.
# --------------------------------------------------------------------

FLUTTER = _chunk("Flutter Mobile App Development\nCourse ID: MOB-101\nLevel: Beginner\n"
                 "Delivery mode: Online\nFee (EGP): 7000", reference=1)
ADVANCED_FLUTTER = _chunk("Advanced Flutter with Firebase\nCourse ID: MOB-201\nLevel: Intermediate\n"
                          "Delivery mode: Hybrid\nFee (EGP): 9500", reference=2)
ML = _chunk("Machine Learning Practitioner\nCourse ID: AI-201\nLevel: Intermediate\nFee (EGP): 11500",
            reference=3)
STUDENT = {"background": "computer engineer", "career_goal": "mobile development",
           "skill_level": "beginner", "training_preference": "hybrid", "budget": "till 5000",
           "available_time": "2 days per week"}
LIVE_REPLY = """**Flutter Mobile App Development** [MOB-101]
- Why it fits you: beginner friendly.
- Level: Beginner
- Delivery: Online
- Fees: 7000 EGP
- Entry requirements: none.
- Which course to start with: Start with Flutter Mobile App Development.

**Advanced Flutter with Firebase** [MOB-201]
- Why it fits you: a step after MOB-101.
- Level: Intermediate
- Delivery: Hybrid
- Fees: 9500 EGP
- Which course to start with: Start with MOB-101 first.

**Machine Learning Practitioner** [AI-101]
- Why it fits you: This course is not relevant to your current profile.
- Level: Not stated in the catalogue
- Delivery: Not stated in the catalogue
- Fees: Not stated in the catalogue

I recommend starting with Flutter Mobile App Development (MOB-101)."""


def _live(reply=LIVE_REPLY, chunks=(FLUTTER, ADVANCED_FLUTTER, ML)):
    return _recommend(reply, list(chunks), profile=STUDENT)


def test_headers_with_course_codes_are_recognised_and_checked():
    answer = _live()

    assert "- Level: Intermediate (above your beginner level: this may be a step up)" in answer


def test_fees_above_the_budget_are_flagged():
    answer = _live()

    assert "- Fees: 7000 EGP (above your budget of till 5000)" in answer
    assert "- Fees: 9500 EGP (above your budget of till 5000)" in answer


def test_blocks_saying_the_course_does_not_fit_are_dropped():
    answer = _live()

    assert "Machine Learning Practitioner" not in answer
    assert "**Flutter Mobile App Development** [MOB-101]" in answer


def test_per_course_start_with_lines_are_removed_but_the_closing_line_stays():
    answer = _live()

    assert "Which course to start with" not in answer
    assert answer.rstrip().endswith("I recommend starting with Flutter Mobile App Development (MOB-101).")


def test_a_course_code_that_belongs_to_another_course_is_removed():
    reply = "**Machine Learning Practitioner** [AI-101] [3]\n- Level: Intermediate\n- Fees: 11500 EGP"

    answer = _live(reply, chunks=(ML,))

    assert "AI-101" not in answer
    assert "**Machine Learning Practitioner** [3]" in answer


def test_fees_within_the_budget_are_not_flagged():
    profile = {**STUDENT, "budget": "10000 EGP"}
    answer = _recommend(LIVE_REPLY, [FLUTTER, ADVANCED_FLUTTER, ML], profile=profile)

    assert "above your budget" not in answer


# Live finding: "**Name (MOB-101)** [n]" headers -- code in parentheses and a
# literal "[n]" -- made every name fail the evidence check, so nothing was
# dropped (including an irrelevant course with a wrong code).
PAREN_REPLY = """**Flutter Mobile App Development (MOB-101)** [n]
- Why it fits you: beginner friendly.
- Level: Beginner
- Fees: 7000 EGP

**Machine Learning Practitioner (AI-101)** [n]
- Why it fits you: While this course is not directly related to mobile development, it helps.
- Level: Not stated in the catalogue
- Delivery: Not stated in the catalogue
- Fees: Not stated in the catalogue

Start with Flutter Mobile App Development."""


def test_parenthesised_codes_and_placeholders_are_handled():
    answer = _live(PAREN_REPLY)

    assert "**Flutter Mobile App Development (MOB-101)**" in answer
    assert "[n]" not in answer
    assert "Machine Learning Practitioner" not in answer
    assert "- Fees: 7000 EGP (above your budget of till 5000)" in answer


def test_a_wrong_code_in_parentheses_is_removed():
    reply = "**Machine Learning Practitioner (AI-101)** [3]\n- Level: Intermediate\n- Fees: 11500 EGP"

    answer = _live(reply, chunks=(ML,))

    assert "AI-101" not in answer
    assert answer.startswith("**Machine Learning Practitioner** [3]")


def test_a_fee_line_that_already_compares_with_the_budget_is_not_flagged_twice():
    # Live finding with qwen2.5:7b: "(exceeds your budget of 5000 EGP) (above your budget of 5000)".
    reply = "**Flutter Mobile App Development** [MOB-101]\n- Level: Beginner\n- Fees: 7000 EGP (exceeds your budget of 5000 EGP)"

    answer = _live(reply, chunks=(FLUTTER,))

    assert answer.count("budget") == 1
