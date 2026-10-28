"""
Rule-based understanding (agents/curriculum_research/understanding.py)
and revisions (revisions.py) of the rebuilt Agent 3, English and Arabic.
Pure functions: no model, no tools.
"""

import pytest

from agents.curriculum_research import understanding as rules
from agents.curriculum_research.revisions import (
    RevisionError,
    add_module,
    apply_change,
    detect_changes,
)
from tests.curriculum_fixtures import valid_curriculum


@pytest.mark.parametrize("text", [
    "Suggest new courses for the AI track",
    "recommend new courses we should add to the catalogue",
    "Review the catalogue and identify missing learning paths",
    "What new courses should we offer next year?",
    "Do a gap analysis of our course catalogue",
    "اقترح دورات جديدة للأكاديمية",
    "ما هي المسارات الناقصة في الكتالوج؟",
    "حلل الكتالوج واقترح برامج جديدة",
])
def test_catalogue_planning_requests_are_suggestions(text):
    assert rules.is_suggest_request(text)
    assert not rules.is_alignment_request(text)


@pytest.mark.parametrize("text", [
    "Create a curriculum about Python", "Compare the curriculum with the NICE framework",
    "Approve the curriculum", "recommend a course for me", "اعرض المنهج",
])
def test_other_requests_are_not_suggestions(text):
    assert not rules.is_suggest_request(text)


@pytest.mark.parametrize("text, expected", [
    ("Align this curriculum with the Saudi NQF", True),
    ("Compare it against the ESCO framework", True),
    ("Check accreditation readiness", True),
    ("قارن المنهج مع معايير الإطار الوطني", True),
    ("Create a curriculum about project management", False),
    ("Suggest new courses based on catalogue gaps", False),
])
def test_alignment_requests(text, expected):
    assert rules.is_alignment_request(text) is expected


@pytest.mark.parametrize("text", ["Approve the curriculum", "please approve it", "publish this draft",
                                  "اعتمد المنهج", "وافق على المسودة", "اعتماد النسخة"])
def test_approval_attempts_are_recognised(text):
    assert rules.is_approve_attempt(text)


@pytest.mark.parametrize("text", ["Create a curriculum about approval workflows in SAP",
                                  "add a module on code review"])
def test_mentioning_approval_is_not_an_approval_attempt(text):
    assert not rules.is_approve_attempt(text)


@pytest.mark.parametrize("text, expected", [
    ("generate suggestion 2", 2), ("Use the second suggestion", 2), ("2nd one", None),
    ("أنشئ الاقتراح 3", 3), ("اختر الاقتراح الأول", 1), ("suggestion 2 looks interesting but what about its "
                                                          "duration and audience exactly", None),
])
def test_suggestion_choice(text, expected):
    assert rules.suggestion_choice(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("a beginner course", "Beginner"), ("intermediate learners", "Intermediate"), ("Advanced SQL", "Advanced"),
    ("منهج متقدم", "Advanced"), ("للمبتدئين", "Beginner"), ("Python Fundamentals", None),
    ("from beginner to advanced", None),  # two levels: not a single stated level
])
def test_level(text, expected):
    assert rules.find_level(text) == expected


def test_ids_in_text():
    assert rules.find_course_id("Create a curriculum based on da-201 please") == "DA-201"
    assert rules.find_course_id("no course here") is None
    assert rules.find_curriculum_id("show CUR-0123456789AB") == "CUR-0123456789ab"


@pytest.mark.parametrize("text, reply, expected", [
    ("Create a curriculum about SQL", "en", "en"),
    ("Create a curriculum about SQL in Arabic", "en", "ar"),
    ("أنشئ منهج عن بايثون باللغة الإنجليزية", "ar", "en"),
    ("أنشئ منهج عن بايثون", "ar", "ar"),
    ("translate it to Arabic", "en", "ar"),
    ("ترجمه إلى الإنجليزية", "ar", "en"),
])
def test_output_language(text, reply, expected):
    assert rules.output_language(text, reply) == expected


def test_request_slots_reuse_the_tested_parser():
    slots = rules.request_slots("Create an intermediate curriculum about SQL for analysts, 40 hours, no capstone")
    assert slots["topic"] == "SQL"
    assert (slots["total_hours"], slots["level"], slots["capstone"]) == (40, "Intermediate", False)
    assert slots["target_audience"] == "Analysts"

    arabic = rules.request_slots("أنشئ منهج عن تحليل البيانات لمدة 30 ساعة للمبتدئين")
    assert (arabic["topic"], arabic["total_hours"], arabic["level"]) == ("تحليل البيانات", 30, "Beginner")
    assert arabic["target_audience"] is None  # "for beginners" is the level, not a separate audience

    rate = rules.request_slots("Create a curriculum about Excel, 5 hours per week")
    assert rate["total_hours"] is None and rate["hours_rate_ignored"]


@pytest.mark.parametrize("text, expected", [
    ("40", {"total_hours": 40}), ("40 hours, intermediate", {"total_hours": 40, "level": "Intermediate"}),
    ("٣٠ ساعة", {"total_hours": 30}), ("Kubernetes", {"topic": "Kubernetes"}), ("use defaults", {}),
])
def test_answer_slots(text, expected):
    slots = rules.answer_slots(text)
    for key, value in expected.items():
        assert slots[key] == value
    if text == "use defaults":
        assert rules.accepts_defaults(text) and not slots["topic"]


def test_send_requests_are_recognised():
    assert rules.is_send_request("email it to the dean") and rules.is_send_request("أرسل المنهج")
    assert not rules.is_send_request("Create a curriculum about email marketing")


# --------------------------------------------------------------------
# Revisions
# --------------------------------------------------------------------

@pytest.mark.parametrize("text, kinds", [
    ("make it 30 hours", ["set_hours"]),
    ("40 hours", ["set_hours"]),
    ("change the audience to university students", ["set_audience"]),
    ("add a module on Git", ["add_module"]),
    ("add a capstone project", ["no_change"]),  # the curriculum already has one
    ("remove module 2", ["remove_module"]),
    ("remove the capstone", ["remove_capstone"]),
    ("make it advanced", ["set_level"]),
    ("translate it to Arabic", ["set_language"]),
    ("make the outcomes more practical", ["instruction"]),
    ("احذف الوحدة 1", ["remove_module"]),
    ("اجعلها 20 ساعة", ["set_hours"]),
    ("what is the weather today", []),
    ("Create a curriculum about SQL", []),
])
def test_detect_changes(text, kinds):
    assert [c["kind"] for c in detect_changes(text, valid_curriculum())] == kinds


def test_add_capstone_is_detected_when_there_is_none():
    curriculum = valid_curriculum() | {"capstone": None, "capstone_omitted_reason": "none"}
    assert [c["kind"] for c in detect_changes("add a capstone project", curriculum)] == ["add_capstone"]


def test_hours_change_rescales_modules_proportionally_and_keeps_the_original():
    original = valid_curriculum(hours=40)
    revised, summary = apply_change(original, {"kind": "set_hours", "value": 30})

    assert [m["hours"] for m in revised["modules"]] == [15, 15] and revised["total_hours"] == 30
    assert [m["hours"] for m in original["modules"]] == [20, 20]  # not modified in place
    assert "40 to 30" in summary


def test_removing_a_module_renumbers_and_keeps_the_total():
    curriculum = add_module(valid_curriculum(hours=40), {"title": "Deployment", "topics": [], "description": "d",
                                                         "activities": [], "outcome_ids": ["LO2"]})
    assert [m["id"] for m in curriculum["modules"]] == ["M1", "M2", "M3"] and curriculum["total_hours"] == 40

    revised, _ = apply_change(curriculum, {"kind": "remove_module", "module_id": "M2"})
    assert [m["id"] for m in revised["modules"]] == ["M1", "M2"]
    assert [m["title"] for m in revised["modules"]] == ["Data preparation", "Deployment"]
    assert sum(m["hours"] for m in revised["modules"]) == 40


@pytest.mark.parametrize("change, message", [
    ({"kind": "set_hours", "value": 1}, "cannot cover 2 modules"),
    ({"kind": "set_hours", "value": 5000}, "between 1 and 1000"),
    ({"kind": "remove_module", "module_id": "M9"}, "no module M9"),
])
def test_impossible_changes_are_refused(change, message):
    with pytest.raises(RevisionError, match=message):
        apply_change(valid_curriculum(), change)
