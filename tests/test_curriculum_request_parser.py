"""
Tests for agents/curriculum_research/request_parser.py: hours, audience,
topic, creation intent, accreditation detection and refinement parsing,
in English and Arabic. Pure functions -- no store, no orchestrator.
"""

import pytest

from agents.curriculum_research.request_parser import (
    CREATION_PHRASES,
    detect_refinements,
    extract_topic,
    has_creation_intent,
    is_accreditation_request,
    looks_like_edit_attempt,
    parse_request,
)


def _kinds(text):
    return [(item.kind, item.value) for item in detect_refinements(text)]


# --------------------------------------------------------------------
# Training hours
# --------------------------------------------------------------------

@pytest.mark.parametrize("text, hours", [
    ("Create a curriculum about Python for 40 hours", 40),
    ("Create a 40-hour curriculum about Python", 40),
    ("Create a curriculum about Python, 40hrs", 40),
    ("Create a curriculum about Python lasting 25 hours", 25),
    ("Create a curriculum about AI that lasts 20 hours", 20),
    ("Create a curriculum about Python for 7.5 hours", 8),
    ("Create a curriculum about Python for 140 hours", 140),
    ("أنشئ منهج عن البرمجة لمدة 30 ساعة", 30),
    ("أنشئ منهج عن البرمجة ٤٠ ساعة", 40),
    ("أنشئ منهج عن البرمجة 12 ساعات", 12),
])
def test_hours_are_extracted(text, hours):
    assert parse_request(text).training_hours == hours


def test_no_hours_when_none_given():
    assert parse_request("Create a curriculum about Python").training_hours is None


@pytest.mark.parametrize("text, topic", [
    ("Create a curriculum about Python for 40 hours", "Python"),
    ("Create a 40-hour curriculum about data analysis", "data analysis"),
    ("Create a curriculum about AI that lasts 20 hours", "AI"),
    ("أنشئ منهج عن تحليل البيانات لمدة 40 ساعة", "تحليل البيانات"),
])
def test_hours_phrase_never_leaks_into_the_topic(text, topic):
    assert parse_request(text).topic == topic


# --------------------------------------------------------------------
# Target audience
# --------------------------------------------------------------------

@pytest.mark.parametrize("text, audience", [
    ("Create a curriculum about Python for beginners", "Beginners"),
    ("Design a Python course for high school students", "High school students"),
    ("Create a curriculum about SQL designed for managers", "Managers"),
    ("Create a curriculum about SQL, target audience is nurses", "Nurses"),
    ("Create a curriculum about SQL aimed at new hires", "New hires"),
    ("أنشئ منهج عن البرمجة للمبتدئين", "المبتدئين"),
    ("أنشئ منهج عن البرمجة لطلاب الجامعة", "طلاب الجامعة"),
    ("أنشئ منهج عن البرمجة للمعلمين", "المعلمين"),
])
def test_obvious_audience_is_extracted(text, audience):
    assert parse_request(text).target_audience == audience


def test_a_subject_is_not_mistaken_for_an_audience():
    parsed = parse_request("Create a curriculum for data analysis")

    assert parsed.target_audience is None
    assert parsed.topic == "data analysis"


@pytest.mark.parametrize("text, topic", [
    ("Create a curriculum about Python for beginners", "Python"),
    ("Design a Python course for high school students", "Python"),
    ("Create a curriculum on project management for managers", "project management"),
    ("Create a 30-hour curriculum about SQL for nurses", "SQL"),
    ("أنشئ منهج عن تحليل البيانات للمبتدئين", "تحليل البيانات"),
    ("أنشئ منهج تحليل البيانات للمعلمين", "تحليل البيانات"),
])
def test_audience_phrase_never_leaks_into_the_topic(text, topic):
    assert parse_request(text).topic == topic


# --------------------------------------------------------------------
# Topic extraction
# --------------------------------------------------------------------

@pytest.mark.parametrize("text, topic", [
    ("Create a curriculum about Python programming", "Python programming"),
    ("create curriculum for cybersecurity", "cybersecurity"),
    ("Generate a curriculum on Power BI.", "Power BI"),
    ("Please create a course about SQL", "SQL"),
    ("Design a Python course", "Python"),
    ("Create a data analysis curriculum", "data analysis"),
    ("Create an accredited curriculum about SQL", "SQL"),
    ("Create a training course about Excel", "Excel"),
    ("Create a curriculum about SQL and a marketing campaign", "SQL"),
    ("صمم دورة عن الأمن السيبراني", "الأمن السيبراني"),
    ("إنشاء دورة عن البرمجة", "البرمجة"),
    ("أنشئ منهج عن تحليل البيانات", "تحليل البيانات"),
    ("أنشئ منهج لتحليل البيانات", "تحليل البيانات"),
])
def test_topic_is_extracted(text, topic):
    assert parse_request(text).topic == topic


@pytest.mark.parametrize("text", [
    "Create a curriculum",
    "Create a curriculum and marketing campaign",
    "Create a curriculum and a marketing campaign",
    "أنشئ منهج تدريبي ثم حملة تسويقية",
    "Find a curriculum about data analysis",
])
def test_no_topic_is_invented(text):
    assert parse_request(text).topic is None


def test_extract_topic_rejects_overlong_text():
    assert extract_topic("Create a curriculum about " + "x " * 100) is None


# --------------------------------------------------------------------
# Creation intent
# --------------------------------------------------------------------

@pytest.mark.parametrize("phrase", CREATION_PHRASES)
def test_every_documented_creation_phrase_is_recognised(phrase):
    assert has_creation_intent(f"{phrase} about data analysis")


@pytest.mark.parametrize("text", [
    "Find a curriculum about data analysis",
    "I need a training curriculum for artificial intelligence",
    "Show me a syllabus for project management",
    "Make sure the curriculum is ready",
    "أريد منهج تدريبي لتحليل البيانات",
    "ابحث عن مناهج تدريبية",
    "make it 40 hours",
])
def test_search_and_refinement_requests_are_not_creation(text):
    assert not has_creation_intent(text)


# --------------------------------------------------------------------
# Accreditation / gap analysis
# --------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Do an accreditation gap analysis for my AI curriculum",
    "Is this curriculum accredited?",
    "Run a gap analysis on this draft",
    "Align the curriculum with the standards",
    "Compare it against standards",
    "تحليل الفجوات للمنهج",
    "أريد مواءمة المنهج مع معايير الاعتماد",
    "ما هي متطلبات الاعتماد الأكاديمي",
])
def test_accreditation_requests_are_detected(text):
    assert is_accreditation_request(text)


@pytest.mark.parametrize("text", [
    "Find a curriculum about compliance training",
    "Create a curriculum about data analysis",
    "الاعتماد على الذات",
    "أنشئ منهج عن تحليل البيانات",
])
def test_ordinary_requests_are_not_accreditation(text):
    assert not is_accreditation_request(text)


# --------------------------------------------------------------------
# Refinements
# --------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("make it 40 hours", [("set_hours", 40)]),
    ("Change the duration to 20 hours", [("set_hours", 20)]),
    ("40 hours", [("set_hours", 40)]),
    ("increase it to 50 hrs", [("set_hours", 50)]),
    ("اجعلها 40 ساعة", [("set_hours", 40)]),
    ("غيّر المدة إلى 20 ساعة", [("set_hours", 20)]),
    ("Add a capstone to that curriculum", [("add_capstone", None)]),
    ("add a capstone project", [("add_capstone", None)]),
    ("أضف مشروعًا ختاميًا", [("add_capstone", None)]),
    ("أضف مشروع التخرج", [("add_capstone", None)]),
    ("add a module on SQL", [("add_module", "SQL")]),
    ("Add a new unit about Machine Learning to the course",
     [("add_module", "Machine Learning")]),
    ("أضف وحدة عن SQL", [("add_module", "SQL")]),
    ("change the audience to teachers", [("set_audience", "Teachers")]),
    ("make it for managers instead", [("set_audience", "Managers")]),
    ("غيّر الفئة المستهدفة إلى المعلمين", [("set_audience", "المعلمين")]),
    ("add a module on SQL and make it 50 hours",
     [("add_module", "SQL"), ("set_hours", 50)]),
    ("add a capstone module", [("add_capstone", None)]),
])
def test_refinements_are_detected(text, expected):
    assert _kinds(text) == expected


@pytest.mark.parametrize("text", [
    "yes",
    "Hello there",
    "What are the modules?",
    "How much does a 3 hours class cost",
    "Find a curriculum about data analysis",
    "Recommend a course for me",
    "make it more fun",
])
def test_unrelated_messages_are_not_refinements(text):
    assert detect_refinements(text) == []


@pytest.mark.parametrize("text", [
    "remove the capstone",
    "delete the last module",
    "احذف المشروع الختامي",
])
def test_removals_are_reported_as_unsupported(text):
    assert _kinds(text) == [("unsupported", "remove")]


def test_edit_attempts_are_recognised_even_when_unsupported():
    assert looks_like_edit_attempt("make it more fun")
    assert looks_like_edit_attempt("Change the tone")
    assert not looks_like_edit_attempt("What are the modules?")


# --------------------------------------------------------------------
# Review regressions: messages that must NOT be treated as edits, and
# edits that must be recognised in their natural phrasings.
# --------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    # Searches that merely mention a capstone / hours.
    "Show me a curriculum with a capstone",
    "I want a curriculum with a capstone project",
    "Find a curriculum about SQL and add a capstone",
    "Show me a curriculum without a capstone",
    # Answers that belong to other agents (REX asks about available time).
    "5 hours per week",
    "About 10 hours",
    "I have 10 hours",
    "I can study 20 hours a week",
    "٥ ساعات في الأسبوع",
    "نعم",
    "no",
    "I'm a beginner",
])
def test_messages_that_only_mention_edits_are_not_refinements(text):
    assert detect_refinements(text) == []


@pytest.mark.parametrize("text, expected", [
    ("من فضلك اجعلها 40 ساعة", [("set_hours", 40)]),
    ("I'd like to add a module on SQL", [("add_module", "SQL")]),
    ("now change the audience to teachers", [("set_audience", "Teachers")]),
    ("Please add a capstone", [("add_capstone", None)]),
    ("can you make it 30 hours", [("set_hours", 30)]),
    ("add a module on how to design a course",
     [("add_module", "how to design a course")]),
    ("add a module on accreditation", [("add_module", "accreditation")]),
    ("ضيف وحدة عن SQL", [("add_module", "SQL")]),
])
def test_natural_phrasings_of_edits_are_recognised(text, expected):
    assert _kinds(text) == expected


def test_module_titles_are_whitespace_clean():
    assert _kinds("add a module on   SQL    basics  ") == [
        ("add_module", "SQL basics")
    ]


@pytest.mark.parametrize("text", [
    "5 hours per week", "10 hours a day", "3 hours each session",
    "٥ ساعات في الأسبوع", "ساعتين 3 ساعات يوميا",
])
def test_hours_per_week_or_day_are_not_a_total_duration(text):
    from agents.curriculum_research.request_parser import find_hours
    from agents.text_utils import prepare

    hours, span = find_hours(*prepare(text))

    assert hours is None
    assert span is not None  # the phrase is still removed from the text


def test_a_weekly_rate_is_dropped_from_the_topic_and_flagged():
    parsed = parse_request("Create a curriculum about Python, 5 hours per week")

    assert parsed.topic == "Python"
    assert parsed.training_hours is None
    assert parsed.ignored_hours_rate is True


def test_a_trailing_instruction_is_not_part_of_the_topic():
    parsed = parse_request(
        "Create a curriculum about Python for 40 hours and make it 50 hours"
    )

    assert parsed.topic == "Python"
    assert parsed.training_hours == 40


def test_refinement_priority_over_creation_wording():
    from agents.curriculum_research.request_parser import (
        refinement_takes_priority,
    )

    assert refinement_takes_priority("add a module on how to design a course")
    assert refinement_takes_priority("من فضلك اضف وحدة عن SQL")
    # "make" is ambiguous, so creation wins unless it is clearly an edit.
    assert not refinement_takes_priority("make a curriculum about SQL")
    assert not refinement_takes_priority("Create a curriculum about SQL")


@pytest.mark.parametrize("text", [
    "translate this curriculum to Arabic",
    "Export this curriculum as PDF",
    "please download the curriculum",
    "Compare these two curricula",
    "Summarize this curriculum",
    "ترجم هذا المنهج إلى الإنجليزية",
    "صدرها بصيغة PDF",
    "قارن بين المنهجين",
])
def test_unsupported_actions_are_recognised(text):
    from agents.curriculum_research.request_parser import is_unsupported_action

    assert is_unsupported_action(text)


@pytest.mark.parametrize("text", [
    "Find a curriculum about translation",
    "Create a curriculum about export management",
    "Find a curriculum about data analysis",
    "حملة تسويقية",
    "أنشئ منهج عن الترجمة",
])
def test_topics_that_sound_like_actions_are_not_unsupported(text):
    from agents.curriculum_research.request_parser import is_unsupported_action

    assert not is_unsupported_action(text)
