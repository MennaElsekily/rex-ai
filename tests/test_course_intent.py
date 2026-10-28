"""
agents/course_intent.py: recommendation vs course question vs general,
using the acceptance examples from the two-mode REX requirements.
"""

import pytest

from agents.course_intent import (
    COURSE_QUESTION,
    GENERAL,
    RECOMMENDATION,
    classify_course_intent,
)


@pytest.mark.parametrize("text", [
    "Recommend a course for me.",
    "I want to become a data analyst. What should I study?",
    "Can you recommend a course based on my background?",
    "Which courses are best for me?",
    "Which course should I take?",
    "رشحلي كورسات مناسبة ليا.",
    "عايز أبدأ في مجال الذكاء الاصطناعي، أتعلم إيه؟",
    "أنا عايز آخد كورس في البرمجة، تنصحني بإيه؟",
    "Ana 3ayez akhod course fel programming.",
    "Ana 3ayez at3alem programming, tensa7ny b course eh?",
    # Learning goals ask for course guidance.
    "I want to learn AI",
    "I would like to study data analysis",
    "عايز أتعلم الذكاء الاصطناعي",
    "أريد أن أتعلم البرمجة",
])
def test_recommendation_requests(text):
    assert classify_course_intent(text) == RECOMMENDATION


@pytest.mark.parametrize("text", [
    "What courses are available?",
    "Which courses are available?",
    "What does the Python course cover?",
    "How long does the data analysis course take?",
    "How long does that course take?",
    "What are the prerequisites?",
    "What are the prerequisites for this course?",
    "Is this course available online?",
    "What skills will I learn?",
    "Who is this course suitable for?",
    "What is the difference between these two courses?",
    "Okay, but what does the Python course teach?",
    "Who are the instructors?",
    "ما هي الدورات المتاحة؟",
    "مدة الكورس كام؟",
    "الكورس بيغطي إيه؟",
    "الكورس ده بكام؟",
    "ما هي متطلبات الالتحاق بالبرنامج؟",
])
def test_course_questions(text):
    assert classify_course_intent(text) == COURSE_QUESTION


@pytest.mark.parametrize("text", [
    "Hello",
    "Hello there",
    "أهلاً",
    "Thanks!",
    "yes",
    "Yes.",
    "How much is a pizza",
    "Find education tenders in Egypt",
    # A course word alone is not a request.
    "Make a good decision about the course",
    # Curriculum draft edits must not look like course questions.
    "make the course 40 hours",
    "عايز الدورة تكون 40 ساعة",
    "Add a capstone project",
    "",
])
def test_general_messages(text):
    assert classify_course_intent(text) == GENERAL


@pytest.mark.parametrize("text", [
    "How much does it cost?",
    "Is it online?",
    "How long does it take?",
    "بكام؟",
])
def test_follow_ups_need_a_course_under_discussion(text):
    assert classify_course_intent(text) == (
        GENERAL if "long does it take" not in text else COURSE_QUESTION
    )
    assert classify_course_intent(text, course_context=True) == COURSE_QUESTION


def test_short_replies_stay_general_even_with_course_context():
    # "Yes" during a questionnaire is interpreted by REX's pending state,
    # not as a course question.
    assert classify_course_intent("Yes.", course_context=True) == GENERAL


@pytest.mark.parametrize("text, expected", [
    # The example buttons in rex_ai_app.py, both languages.
    ("Help me choose a course", RECOMMENDATION),
    ("ساعدني أختار دورة", RECOMMENDATION),
    ("ساعدني في اختيار كورس", RECOMMENDATION),
    ("What does the data analysis course cover?", COURSE_QUESTION),
    ("إيه اللي بيغطيه كورس تحليل البيانات؟", COURSE_QUESTION),
])
def test_ui_example_prompts_are_routed(text, expected):
    assert classify_course_intent(text) == expected


@pytest.mark.parametrize("text", ["اختار الوقت المناسب للاجتماع", "Choose a date"])
def test_choosing_something_else_is_not_a_recommendation(text):
    assert classify_course_intent(text) != RECOMMENDATION


@pytest.mark.parametrize("text, expected", [
    ("Which courses do you offer?", True),
    ("What courses are available?", True),
    ("List all courses", True),
    ("ايه الكورسات المتاحة؟", True),
    ("What does the Flutter course cover?", False),
    ("How much is the SQL course?", False),
])
def test_catalogue_listing_questions(text, expected):
    from agents.course_intent import is_catalogue_listing

    assert is_catalogue_listing(text) is expected
