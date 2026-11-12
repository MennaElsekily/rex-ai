"""
REX Assistant's course recommendations (agents/rex_assistant/
grounded_recommendation.py): the model may only choose verified
catalogue records by id; every course name and fact comes from a record.

Run through REX's own generate_answer() in recommendation mode, with the
records the knowledge base's get_course_records tool returns.

These tests were ported from the earlier free-text checks: each live
finding below still holds, now because a course or fact is never taken
from the model's text. One expectation is reversed on purpose: an answer
naming no verifiable course used to be passed through unchanged ("nothing
is dropped"); that is how invented courses reached the learner, so now
nothing invented is ever shown.

Mocked: the chat model.
"""

from agents.rex_assistant.graph import generate_answer
from agents.rex_assistant.prompts import MODE_QUESTION, MODE_RECOMMENDATION
from tests.test_rex_answer_generation import FakeLLM, _chunk, _evidence, _record


def write_recommendation(llm, evidence, language, profile, declined_fields=None, records=()):
    evidence = {**evidence, "course_records": list(records), "course_records_status": "ok"}
    return generate_answer(llm, evidence, language, mode=MODE_RECOMMENDATION,
                           profile=profile, declined=declined_fields)


STRATEGY = _record("MKT-101", "Digital Marketing Fundamentals", level="Beginner",
                   entry_requirements="No previous marketing experience required.",
                   target_audience="Beginners, students, and small-business teams.")
ANALYTICS_RECORD = _record("MKT-301", "Marketing Analytics and Campaign Optimization", level="Advanced",
                           entry_requirements="Experience running campaigns and basic spreadsheet skills. MKT-201 or DA-101 recommended.")
MARKETING_RECORDS = (STRATEGY, ANALYTICS_RECORD)


def test_learner_profile_labels_are_in_the_prompt_and_the_layout_in_the_answer():
    llm = FakeLLM("MKT-101 :: It matches your goal.")

    result = write_recommendation(llm, _evidence(chunks=[MARKETING]), "en",
                                  {"career_goal": "data analyst", "training_preference": "online"},
                                  declined_fields=["budget"], records=MARKETING_RECORDS)

    assert "LEARNER PROFILE" in llm.user
    assert "Career goal: data analyst" in llm.user
    assert "Training preference (online/offline): online" in llm.user
    assert "Budget: not provided (the learner skipped it)" in llm.user
    assert "Background: not provided" in llm.user
    assert "- Why it fits you: It matches your goal." in result["answer"]


# --------------------------------------------------------------------
# Preferences the catalogue does not cover (live finding: the model said
# a course was "available online" and fit a 5000 budget; the catalogue
# states neither delivery mode nor fees).
# --------------------------------------------------------------------

MARKETING = _chunk(
    "Digital Marketing Fundamentals. Level: Beginner. Entry requirements: "
    "No previous marketing experience required.", section="Programme 01",
    course_id="MKT-101", course_name="Digital Marketing Fundamentals")
PROFILE = {"career_goal": "marketing", "skill_level": "beginner",
           "training_preference": "online", "budget": "5000",
           "available_time": "2 hours per day"}


def test_unstated_delivery_and_fees_are_flagged_after_a_recommendation():
    llm = FakeLLM("MKT-101 :: It suits your marketing goal. It is available online.")

    answer = write_recommendation(llm, _evidence(chunks=[MARKETING]), "en", PROFILE,
                                  records=MARKETING_RECORDS)["answer"]

    assert "- Delivery: online or offline is not stated in the catalogue (your preference for online training " \
           "cannot be verified)" in answer
    assert "- Fee: not stated in the catalogue (so it cannot be checked against your budget of 5000)" in answer
    assert "available online" not in answer   # the model's claim is not shown


def test_the_note_is_in_arabic_for_arabic_replies():
    result = write_recommendation(FakeLLM("MKT-101 :: دورة مناسبة لهدفك."), _evidence(chunks=[MARKETING]), "ar",
                                  PROFILE, records=MARKETING_RECORDS)

    assert "غير مذكور في الكتالوج" in result["answer"]
    assert "أونلاين" in result["answer"]
    assert "**Digital Marketing Fundamentals** (MKT-101)" in result["answer"]   # the exact catalogue name


def test_no_note_when_the_catalogue_states_delivery_and_fees():
    record = {**STRATEGY, "delivery_mode": "Online", "fee": 4500.0, "fee_currency": "EGP"}

    answer = write_recommendation(FakeLLM("MKT-101 :: It fits."), _evidence(chunks=[MARKETING]), "en", PROFILE,
                                  records=[record])["answer"]

    assert "- Delivery: Online (matches your preference for online training)" in answer
    assert "- Fee: 4500 EGP (within your budget of 5000)" in answer
    assert "cannot be" not in answer


def test_no_note_for_preferences_the_learner_did_not_give():
    profile = {"career_goal": "marketing", "training_preference": "no_preference"}

    answer = write_recommendation(FakeLLM("MKT-101 :: It fits."), _evidence(chunks=[MARKETING]), "en", profile,
                                  records=MARKETING_RECORDS)["answer"]

    assert "- Delivery: online or offline is not stated in the catalogue\n" in answer
    assert "- Fee: not stated in the catalogue\n" in answer
    assert "preference" not in answer and "budget" not in answer


def test_direct_questions_get_no_preference_note():
    result = generate_answer(FakeLLM("Workshops. [1]"), _evidence(chunks=[MARKETING]), "en",
                             mode=MODE_QUESTION, profile=PROFILE)

    assert result["answer"] == "Workshops. [1]"


def test_recommendation_prompt_allows_only_catalogue_ids_and_no_facts():
    llm = FakeLLM("MKT-101 :: Fits.")
    write_recommendation(llm, _evidence(chunks=[MARKETING]), "en", PROFILE, records=MARKETING_RECORDS)

    assert "Use only course ids from the list" in llm.user
    assert "do not mention fees, delivery, duration, prerequisites" in llm.user
    assert 'speaking to the learner as "you"' in llm.user
    assert "MKT-101 | Digital Marketing Fundamentals" in llm.user


# Live finding: the model repeated a course, renamed another
# ("Digital Marketing Analytics") and called an Intermediate course a fit
# for a beginner.
MODEL_REPLY = """MKT-101 :: It is designed for beginners.
Digital Marketing Analytics :: It is suitable for beginners.
MKT-301 :: It builds on strategy.
MKT-101 :: It is designed for beginners."""

ANALYTICS = _chunk("Marketing Analytics and Campaign Optimization. Level: Advanced. "
                   "Fees: not listed online.", reference=2,
                   course_id="MKT-301", course_name="Marketing Analytics and Campaign Optimization")


def _recommend(reply, chunks, profile=PROFILE, language_code="en", records=MARKETING_RECORDS):
    return write_recommendation(FakeLLM(reply), _evidence(chunks=chunks), language_code, profile,
                                records=records)["answer"]


def test_repeated_courses_are_listed_once():
    answer = _recommend(MODEL_REPLY, [MARKETING, ANALYTICS])

    assert answer.count("**Digital Marketing Fundamentals**") == 2   # its block and the closing line
    assert answer.count("(MKT-101)\n") == 1
    assert answer.rstrip().startswith("Based on your profile") and "Start with **Digital Marketing Fundamentals" in answer


def test_renamed_courses_are_never_shown():
    answer = _recommend(MODEL_REPLY, [MARKETING, ANALYTICS], profile={**PROFILE, "skill_level": None})

    assert "Digital Marketing Analytics" not in answer
    assert "**Marketing Analytics and Campaign Optimization** (MKT-301)" in answer


def test_a_course_above_a_beginners_level_is_left_out_when_one_at_their_level_fits():
    answer = _recommend(MODEL_REPLY, [MARKETING, ANALYTICS])

    assert "- Level: Beginner\n" in answer
    assert "Marketing Analytics and Campaign Optimization" not in answer


def test_courses_above_the_learners_level_are_marked_with_their_entry_requirements():
    answer = _recommend("MKT-301 :: It builds on strategy.", [ANALYTICS])

    assert "- Level: Advanced (above your beginner level: check the entry requirements below" in answer
    assert "- Entry requirements: Experience running campaigns and basic spreadsheet skills. MKT-201 or DA-101 recommended." in answer


def test_invented_courses_are_never_shown_even_when_none_can_be_verified():
    # Reversed on purpose (see the module docstring): this used to pass through.
    answer = _recommend("**Some Course** [1]\n- Level: Foundation", [_chunk("Some text.")])

    assert "Some Course" not in answer
    assert answer.startswith("No suitable course was found in the current academy catalogue")


def test_an_answer_naming_no_course_says_none_was_found():
    assert _recommend("NONE", [MARKETING]).startswith("No suitable course was found in the current academy catalogue")


# --------------------------------------------------------------------
# Live finding (2026-10-27): the model wrote "**Name** [MOB-101]" headers,
# added an irrelevant course with a wrong ID, repeated "Which course to
# start with" in every block, and did not flag a fee above the budget.
# --------------------------------------------------------------------

FLUTTER = _chunk("Flutter Mobile App Development\nCourse ID: MOB-101\nLevel: Beginner\n"
                 "Delivery mode: Online\nFee (EGP): 7000", reference=1,
                 course_id="MOB-101", course_name="Flutter Mobile App Development")
ADVANCED_FLUTTER = _chunk("Advanced Flutter with Firebase\nCourse ID: MOB-201\nLevel: Intermediate\n"
                          "Delivery mode: Hybrid\nFee (EGP): 9500", reference=2,
                          course_id="MOB-201", course_name="Advanced Flutter with Firebase")
ML = _chunk("Machine Learning Practitioner\nCourse ID: AI-201\nLevel: Intermediate\nFee (EGP): 11500",
            reference=3, course_id="AI-201", course_name="Machine Learning Practitioner")
MOBILE_RECORDS = (
    _record("MOB-101", "Flutter Mobile App Development", level="Beginner", delivery_mode="Online", fee=7000.0,
            fee_currency="EGP", prerequisites_text="None"),
    _record("MOB-201", "Advanced Flutter with Firebase", level="Intermediate",
            delivery_mode="Hybrid (online sessions plus on-campus labs in Cairo)", fee=9500.0, fee_currency="EGP",
            prerequisite_courses=["MOB-101"], entry_requirements="Completed MOB-101 or can build a multi-screen Flutter app"),
    _record("AI-101", "Introduction to Artificial Intelligence and Python", level="Beginner"),
    _record("AI-201", "Machine Learning Practitioner", level="Intermediate", fee=11500.0, fee_currency="EGP"),
)
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

**Machine Learning Practitioner** [AI-201]
- Why it fits you: This course is not relevant to your current profile.
- Level: Not stated in the catalogue
- Delivery: Not stated in the catalogue
- Fees: Not stated in the catalogue

I recommend starting with Flutter Mobile App Development (MOB-101)."""


def _live(reply=LIVE_REPLY, chunks=(FLUTTER, ADVANCED_FLUTTER, ML), profile=STUDENT):
    return _recommend(reply, list(chunks), profile=profile, records=MOBILE_RECORDS)


def test_headers_with_course_codes_are_recognised_and_checked():
    answer = _live()

    assert "**Flutter Mobile App Development** (MOB-101)\n- Why it fits you: beginner friendly." in answer
    assert "- Delivery: Online (does not match your preference for hybrid training)" in answer


def test_fees_above_the_budget_are_flagged():
    answer = _live(profile={**STUDENT, "skill_level": None})

    assert "- Fee: 7000 EGP (above your budget of till 5000)" in answer
    assert "- Fee: 9500 EGP (above your budget of till 5000)" in answer


def test_blocks_saying_the_course_does_not_fit_are_dropped():
    answer = _live()

    assert "Machine Learning Practitioner" not in answer and "AI-201" not in answer
    assert "**Flutter Mobile App Development** (MOB-101)" in answer


def test_one_closing_line_says_where_to_start():
    answer = _live()

    assert "Which course to start with" not in answer and "I recommend starting" not in answer
    assert answer.count("Start with") == 1
    assert answer.rstrip().endswith("Start with **Flutter Mobile App Development** (MOB-101): "
                                    "it is the closest match to your goal at your level.")


def test_a_course_code_that_belongs_to_another_course_is_not_used():
    # The name is one course and the code another: no guess is made.
    reply = "**Machine Learning Practitioner** [AI-101] [3]\n- Level: Intermediate\n- Fees: 11500 EGP"

    answer = _live(reply, chunks=(ML,))

    assert "AI-101" not in answer and "Introduction to Artificial Intelligence" not in answer
    assert "**Machine Learning Practitioner** (AI-201)" in answer   # the search found its passage


def test_fees_within_the_budget_are_not_flagged():
    answer = _live(profile={**STUDENT, "budget": "10000 EGP"})

    assert "above your budget" not in answer
    assert "- Fee: 7000 EGP (within your budget of 10000 EGP)" in answer


# Live finding: "**Name (MOB-101)** [n]" headers -- code in parentheses and a
# literal "[n]" -- made every name fail the evidence check, so nothing was
# dropped (including an irrelevant course with a wrong code).
PAREN_REPLY = """**Flutter Mobile App Development (MOB-101)** [n]
- Why it fits you: beginner friendly.
- Level: Beginner
- Fees: 7000 EGP

**Machine Learning Practitioner (AI-201)** [n]
- Why it fits you: While this course is not directly related to mobile development, it helps.
- Level: Not stated in the catalogue
- Delivery: Not stated in the catalogue
- Fees: Not stated in the catalogue

Start with Flutter Mobile App Development."""


def test_parenthesised_codes_and_placeholders_are_handled():
    answer = _live(PAREN_REPLY)

    assert "**Flutter Mobile App Development** (MOB-101)" in answer
    assert "[n]" not in answer
    assert "Machine Learning Practitioner" not in answer
    assert "- Fee: 7000 EGP (above your budget of till 5000)" in answer


def test_a_wrong_code_in_parentheses_is_not_used():
    reply = "**Machine Learning Practitioner (AI-101)** [3]\n- Level: Intermediate\n- Fees: 11500 EGP"

    answer = _live(reply, chunks=(ML,))

    assert "AI-101" not in answer
    assert "**Machine Learning Practitioner** (AI-201)" in answer


def test_a_budget_comparison_by_the_model_is_not_repeated():
    # Live finding with qwen2.5:7b: "(exceeds your budget of 5000 EGP) (above your budget of 5000)".
    reply = "MOB-101 :: A good first course. It exceeds your budget of 5000 EGP."

    answer = _live(reply, chunks=(FLUTTER,))

    assert answer.count("budget") == 1
    assert "- Why it fits you: A good first course." in answer
