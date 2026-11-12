"""
agents/curriculum_research/editing.py: the pure text <-> structured-
content conversion behind the admin's curriculum editor. No database,
no Streamlit -- these are plain function tests.
"""

from agents.curriculum_research.editing import (
    apply_edits,
    capstone_fields,
    format_assessments,
    format_lines,
    format_modules,
    format_outcomes,
    parse_assessments,
    parse_capstone,
    parse_lines,
    parse_modules,
    parse_outcomes,
)


def test_plain_lists_round_trip():
    items = ["Basic computer skills", "Comfort with spreadsheets"]
    assert parse_lines(format_lines(items)) == items


def test_blank_lines_are_ignored():
    assert parse_lines("One\n\n  \nTwo\n") == ["One", "Two"]


# ------------------------------------------------------------ outcomes

def test_outcomes_round_trip_with_their_ids():
    outcomes = [{"id": "LO1", "statement": "Compare cloud service models"},
               {"id": "LO2", "statement": "Deploy a virtual server"}]
    assert parse_outcomes(format_outcomes(outcomes), []) == outcomes


def test_an_outcome_without_an_id_gets_the_next_unused_one():
    text = "LO1: Compare cloud service models\nApply basic cloud security controls"
    parsed = parse_outcomes(text, [])
    assert parsed == [{"id": "LO1", "statement": "Compare cloud service models"},
                      {"id": "LO2", "statement": "Apply basic cloud security controls"}]


def test_a_removed_outcome_line_is_simply_gone():
    outcomes = [{"id": "LO1", "statement": "A"}, {"id": "LO2", "statement": "B"}]
    edited_text = format_outcomes(outcomes).splitlines()[0]   # only the first line kept, as if the admin deleted the rest
    assert parse_outcomes(edited_text, []) == [{"id": "LO1", "statement": "A"}]


# -------------------------------------------------------------- modules

def test_modules_round_trip():
    modules = [{"id": "M1", "title": "Cloud foundations", "hours": 10, "topics": ["Service models", "Networking"],
               "description": "Guided explanation.", "activities": [{"type": "lab", "description": "Hands-on lab"}],
               "outcome_ids": ["LO1"]}]
    assert parse_modules(format_modules(modules)) == modules


def test_two_modules_round_trip_and_stay_in_order():
    modules = [{"id": "M1", "title": "A", "hours": 10, "topics": ["x"], "description": "d1",
               "activities": [], "outcome_ids": ["LO1"]},
              {"id": "M2", "title": "B", "hours": 15, "topics": ["y"], "description": "d2",
               "activities": [], "outcome_ids": ["LO2"]}]
    parsed = parse_modules(format_modules(modules))
    assert [m["id"] for m in parsed] == ["M1", "M2"]
    assert parsed == modules


def test_a_new_module_with_no_id_is_assigned_one():
    text = "title: New module\nhours: 5\ntopics: a, b\ndescription: d\nactivities: \noutcome_ids: LO1"
    [module] = parse_modules(text)
    assert module["id"] == "M1"


def test_adding_a_module_after_existing_ones_gets_the_next_number():
    text = ("id: M1\ntitle: First\nhours: 10\ntopics: \ndescription: \nactivities: \noutcome_ids: \n"
           + "-" * 20 +
           "\ntitle: Second (new)\nhours: 5\ntopics: \ndescription: \nactivities: \noutcome_ids: ")
    modules = parse_modules(text)
    assert [m["id"] for m in modules] == ["M1", "M2"]


def test_removing_a_module_block_removes_it():
    modules = [{"id": "M1", "title": "A", "hours": 10, "topics": [], "description": "", "activities": [],
               "outcome_ids": []},
              {"id": "M2", "title": "B", "hours": 15, "topics": [], "description": "", "activities": [],
               "outcome_ids": []}]
    full_text = format_modules(modules)
    only_first_block = full_text.split("-" * 20)[0]

    assert [m["id"] for m in parse_modules(only_first_block)] == ["M1"]


def test_multiple_activities_round_trip():
    modules = [{"id": "M1", "title": "A", "hours": 10, "topics": [], "description": "",
               "activities": [{"type": "lab", "description": "Do a lab"}, {"type": "quiz", "description": "Take a quiz"}],
               "outcome_ids": []}]
    assert parse_modules(format_modules(modules))[0]["activities"] == modules[0]["activities"]


def test_a_blank_modules_text_gives_no_modules():
    assert parse_modules("") == []
    assert parse_modules("   \n\n  ") == []


# --------------------------------------------------------- assessments

def test_assessments_round_trip():
    assessments = [{"id": "A1", "type": "quiz", "description": "A short quiz", "outcome_ids": ["LO1", "LO2"]}]
    assert parse_assessments(format_assessments(assessments)) == assessments


def test_a_new_assessment_is_assigned_an_id():
    text = "type: project\ndescription: Final project\noutcome_ids: LO1"
    [assessment] = parse_assessments(text)
    assert assessment["id"] == "A1" and assessment["type"] == "project"


# ------------------------------------------------------------- capstone

def test_capstone_fields_and_parse_round_trip():
    capstone = {"title": "Capstone", "description": "Deploy an app", "outcome_ids": ["LO1"],
               "deliverables": ["Report", "Demo"]}
    fields = capstone_fields(capstone)
    rebuilt = parse_capstone(True, fields["title"], fields["description"], fields["outcome_ids"], fields["deliverables"])
    assert rebuilt == capstone


def test_capstone_can_be_removed_by_unchecking_it():
    assert parse_capstone(False, "title", "description", "LO1", "Report") is None


def test_capstone_fields_of_none_are_all_blank():
    fields = capstone_fields(None)
    assert fields == {"title": "", "description": "", "outcome_ids": "", "deliverables": ""}


# --------------------------------------------------------------- apply

def test_apply_edits_keeps_untouched_fields_and_recomputes_hours():
    content = {"title": "Old", "language": "en", "total_hours": 40, "references": [{"title": "Source"}],
              "modules": [{"id": "M1", "title": "A", "hours": 10, "topics": [], "description": "", "activities": [],
                          "outcome_ids": []}]}
    edits = {"title": "New title", "modules": [
        {"id": "M1", "title": "A", "hours": 20, "topics": [], "description": "", "activities": [], "outcome_ids": []}]}

    updated = apply_edits(content, edits)

    assert updated["title"] == "New title" and updated["total_hours"] == 20
    assert updated["language"] == "en" and updated["references"] == [{"title": "Source"}]   # untouched
    assert content["title"] == "Old"   # the original dict is never mutated
