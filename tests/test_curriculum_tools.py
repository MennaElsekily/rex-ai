"""
Agent 3's tools (agents/curriculum_research/curriculum_tools.py) and the
modules behind them: catalogue access over a temporary index, overlap,
gap report, allowlisted research, frameworks, validation, storage and
review rules. Temp index, database and folders; Tavily mocked.
"""

import json

import pytest

from agents.curriculum_research import validation
from agents.curriculum_research.academy_courses import AcademyCatalogue
from agents.curriculum_research.curriculum_store import CurriculumStore, CurriculumStoreError
from agents.curriculum_research.curriculum_tools import CurriculumToolbox
from agents.curriculum_research.research import allowed_domains, domain_allowed
from agents.rfp_research.api_clients.tavily import WebPage, WebSearchResult
from tests.curriculum_fixtures import DATA_STATUS, FRAMEWORK, build_test_index, valid_curriculum


@pytest.fixture
def catalogue(tmp_path):
    def fake_evidence(question):
        # Stands in for the existing EvidenceRetriever (which translates Arabic).
        hit = "SQL for Analysts" if "sql" in question.casefold() or "استعلام" in question else None
        return {"status": "ok", "chunks": [{"section_path": hit, "text": hit}] if hit else []}

    return AcademyCatalogue(vector_store=build_test_index(tmp_path / "index"), evidence_search=fake_evidence,
                            translate=lambda text: "machine learning basics")


@pytest.fixture
def frameworks_dir(tmp_path):
    directory = tmp_path / "frameworks"
    directory.mkdir()
    (directory / "test-framework.json").write_text(json.dumps(FRAMEWORK), encoding="utf-8")
    return directory


@pytest.fixture
def tools(tmp_path, catalogue, frameworks_dir):
    return CurriculumToolbox(store=CurriculumStore(tmp_path / "curricula.sqlite3"), catalogue=catalogue,
                             frameworks_dir=frameworks_dir, documents_dir=tmp_path / "docs", research_domains=[])


# --------------------------------------------------------------------
# Academy catalogue (existing index, read-only)
# --------------------------------------------------------------------

def test_courses_come_from_the_index_with_their_metadata(tools):
    result = tools.list_academy_courses()

    assert result["status"] == "ok" and result["data_status"] == [DATA_STATUS]
    by_id = {c["course_id"]: c for c in result["courses"]}
    assert set(by_id) == {"DA-101", "DA-201", "CL-101"}  # the FAQ chunk is not a course
    sql = by_id["DA-201"]
    assert (sql["name"], sql["level"], sql["total_hours"], sql["department"]) == (
        "SQL for Analysts", "Intermediate", 40, "Data Analytics")
    assert sql["learning_outcomes"] == ["Write SQL queries with joins", "Build dashboards from queries"]
    assert sql["prerequisite_courses"] == ["DA-101"] and sql["missing"] == []
    assert not any(k.startswith("_") for k in sql)


@pytest.mark.parametrize("filters, ids", [
    ({"department": "data analytics"}, ["DA-101", "DA-201"]),
    ({"level": "beginner"}, ["CL-101", "DA-101"]),
    ({"audience_contains": "cloud"}, ["CL-101"]),
])
def test_courses_are_filtered_by_metadata(tools, filters, ids):
    assert sorted(c["course_id"] for c in tools.list_academy_courses(**filters)["courses"]) == ids


@pytest.mark.parametrize("reference, how", [("da-201", "id"), ("SQL for analysts", "name"),
                                            ("استعلامات قواعد البيانات", "search")])
def test_a_course_is_found_by_id_name_or_search(tools, reference, how):
    result = tools.get_course_details(reference)

    assert result["status"] == "ok" and result["course"]["course_id"] == "DA-201"
    assert result["matched_by"] == how and "Learning Outcomes" in result["course"]["sections"]


def test_an_unknown_course_is_not_found_not_invented(tools):
    assert tools.get_course_details("Underwater basket weaving")["status"] == "not_found"
    assert tools.get_course_details("")["status"] == "invalid_request"


def test_overlap_flags_a_near_duplicate_and_not_a_new_topic(tools):
    duplicate = tools.check_course_overlap("SQL for Analysts", ["joins", "aggregations"])
    new = tools.check_course_overlap("Cybersecurity awareness", ["phishing", "passwords"])

    assert duplicate["duplicate"] is True and duplicate["overlaps"][0]["course_id"] == "DA-201"
    assert new["duplicate"] is False


def test_the_gap_report_only_states_what_the_catalogue_shows(tools):
    report = tools.catalogue_gap_report()

    gaps = {(g["type"], g["department"], g.get("level")) for g in report["gaps"]}
    assert ("missing_level", "Data Analytics", "Advanced") in gaps
    assert ("single_course_department", "Cloud", None) in gaps
    assert ("missing_level", "Data Analytics", "Beginner") not in gaps
    assert all(g["basis"] == "catalogue data" and g["evidence_course_ids"] for g in report["gaps"])
    assert report["data_status"] == [DATA_STATUS] and report["course_count"] == 3


# --------------------------------------------------------------------
# Research: approved domains only
# --------------------------------------------------------------------

def test_research_is_off_without_approved_domains(tools):
    result = tools.search_curriculum_sources("machine learning curriculum")

    assert result["status"] == "not_configured" and result["sources"] == []


def test_research_keeps_only_approved_domains_and_their_metadata(tmp_path, catalogue):
    calls = []

    def fake_search(query, **kwargs):
        calls.append(kwargs)
        pages = [WebPage(title="CS2023", url="https://csed.acm.org/report", content="AI knowledge area",
                         source_domain="csed.acm.org", published_date="2024-01-01"),
                 WebPage(title="Blog", url="https://random-blog.example/ml", content="Opinions",
                         source_domain="random-blog.example")]
        return WebSearchResult(status="success", message="ok", query=query, pages=pages)

    tools = CurriculumToolbox(store=CurriculumStore(tmp_path / "c.sqlite3"), catalogue=catalogue,
                              research_domains=["acm.org"], search=fake_search)
    result = tools.search_curriculum_sources("AI curriculum guidelines")

    assert calls[0]["include_domains"] == ["acm.org"]
    [source] = result["sources"]
    assert (source["url"], source["publisher"], source["published"]) == (
        "https://csed.acm.org/report", "csed.acm.org", "2024-01-01")
    assert source["source_id"].startswith("SRC-") and source["retrieved_at"]


def test_the_allowlist_setting_is_parsed_strictly():
    assert allowed_domains(" acm.org, www.NIST.gov , not a domain, acm.org") == ["acm.org", "nist.gov"]
    assert domain_allowed("https://csed.acm.org/x", ["acm.org"])
    assert not domain_allowed("https://acm.org.evil.example/x", ["acm.org"])


# --------------------------------------------------------------------
# Frameworks
# --------------------------------------------------------------------

def test_frameworks_are_loaded_by_id_with_provenance(tools):
    assert [f["framework_id"] for f in tools.list_frameworks()["frameworks"]] == ["test-framework"]
    loaded = tools.get_framework_requirements("test-framework")
    assert loaded["status"] == "ok" and len(loaded["framework"]["requirements"]) == 3


@pytest.mark.parametrize("framework_id, status", [("sfia", "not_found"), ("../../secrets", "invalid_request"),
                                                  ("A B", "invalid_request")])
def test_unknown_or_unsafe_framework_ids(tools, framework_id, status):
    assert tools.get_framework_requirements(framework_id)["status"] == status


def test_a_framework_file_without_provenance_is_refused(tools, frameworks_dir):
    (frameworks_dir / "no-source.json").write_text(json.dumps({**FRAMEWORK, "framework_id": "no-source",
                                                               "source_url": ""}), encoding="utf-8")

    assert tools.get_framework_requirements("no-source")["status"] == "error"
    assert "no-source" not in [f["framework_id"] for f in tools.list_frameworks()["frameworks"]]


def test_pre_mapping_links_requirements_to_curriculum_elements(tools):
    result = tools.compare_with_framework(valid_curriculum(), "test-framework")

    by_requirement = {m["requirement_id"]: m for m in result["pre_mapping"]}
    assert "LO1" in [c["element_id"] for c in by_requirement["R1"]["candidates"]]
    assert result["framework"]["source_url"] == FRAMEWORK["source_url"]


# --------------------------------------------------------------------
# Validation rules
# --------------------------------------------------------------------

@pytest.mark.parametrize("language", ["en", "ar"])
def test_a_complete_curriculum_is_valid(language):
    result = validation.validate_curriculum(valid_curriculum(language), 40)

    assert result["valid"], result["errors"]


def _codes(curriculum, hours=40, framework=None):
    return {e["code"] for e in validation.validate_curriculum(curriculum, hours, framework)["errors"]}


def test_hours_must_add_up_and_be_positive():
    curriculum = valid_curriculum()
    curriculum["modules"][0]["hours"] = 0

    assert {"module_hours", "hours_total"} <= _codes(curriculum)
    assert "hours_total" in _codes(valid_curriculum(hours=40), hours=45)


@pytest.mark.parametrize("statement, language, ok", [
    ("Learners will be able to evaluate a model", "en", True),
    ("Understand machine learning", "en", False),
    ("Know the main algorithms", "en", False),
    ("يحلل البيانات باستخدام بايثون", "ar", True),
    ("أن يطبق خطوات تجهيز البيانات", "ar", True),
    ("يفهم مفاهيم تعلم الآلة", "ar", False),
])
def test_outcomes_must_be_measurable(statement, language, ok):
    assert validation.is_measurable(statement, language) is ok


def test_outcomes_modules_and_assessments_must_link_up():
    curriculum = valid_curriculum()
    curriculum["assessments"][0]["outcome_ids"] = ["LO1"]
    curriculum["capstone"]["outcome_ids"] = ["LO1"]
    curriculum["modules"][1]["outcome_ids"] = ["LO9"]

    assert {"outcome_unlinked", "dangling_outcome_ref"} <= _codes(curriculum)


def test_missing_sections_and_capstone_are_reported():
    curriculum = valid_curriculum()
    curriculum["capstone"] = None
    curriculum["assessments"] = []

    assert "missing_section" in _codes(curriculum)
    curriculum = valid_curriculum()
    curriculum["capstone"], curriculum["capstone_omitted_reason"] = None, "Short awareness session"
    assert "missing_section" not in _codes(curriculum)


def test_sources_must_be_real_retrieved_references():
    curriculum = valid_curriculum()
    curriculum["competencies"][0]["source_ids"] = ["SRC-invented"]

    assert "unknown_source" in _codes(curriculum)


@pytest.mark.parametrize("text", ["This course is accredited by the Ministry.", "The course is officially approved.",
                                  "المنهج معتمد من الهيئة الوطنية."])
def test_accreditation_claims_are_rejected(text):
    curriculum = valid_curriculum("ar" if "المنهج" in text else "en")
    curriculum["overview"] += " " + text

    assert "accreditation_claim" in _codes(curriculum)


def test_a_disclaimer_is_not_a_claim():
    curriculum = valid_curriculum()
    curriculum["overview"] += " It is not accredited; alignment is not accreditation."

    assert "accreditation_claim" not in _codes(curriculum)


def test_the_language_must_match():
    curriculum = valid_curriculum("en")
    curriculum["language"] = "ar"

    assert "language_mismatch" in _codes(curriculum)


def test_assessments_and_capstone_count_for_the_language():
    """Regression (§17): an Arabic curriculum with English assessments and capstone."""
    curriculum = valid_curriculum("ar")
    english = valid_curriculum("en")
    assert "language_mismatch" not in _codes(curriculum)
    curriculum["assessments"] = english["assessments"] * 4
    curriculum["capstone"] = english["capstone"]
    for module, other in zip(curriculum["modules"], english["modules"]):
        module["activities"] = other["activities"] * 3

    assert "language_mismatch" in _codes(curriculum)


def test_alignment_needs_real_requirements_and_evidence():
    curriculum = valid_curriculum()
    curriculum["alignment"] = {"framework_id": "test-framework", "items": [
        {"requirement_id": "R1", "status": "covered", "element_ids": ["LO1"]},
        {"requirement_id": "R2", "status": "covered", "element_ids": []},          # no evidence
        {"requirement_id": "R9", "status": "not_covered", "element_ids": []},      # not in the framework
    ]}

    errors = validation.validate_curriculum(curriculum, 40, FRAMEWORK)["errors"]
    assert sorted(e["path"] for e in errors if e["code"] == "alignment_evidence") == [
        "alignment.items[1]", "alignment.items[2]"]


def test_rebalance_keeps_the_total_and_proportions():
    modules = [{"hours": 10}, {"hours": 20}, {"hours": 30}]

    assert validation.rebalance_hours(modules, 30) == [5, 10, 15]
    assert sum(validation.rebalance_hours(modules, 7)) == 7
    with pytest.raises(ValueError):
        validation.rebalance_hours(modules, 2)


# --------------------------------------------------------------------
# Storage and review
# --------------------------------------------------------------------

def test_saving_validates_server_side_and_never_approves(tools):
    good = tools.save_curriculum_draft(valid_curriculum(), topic="machine learning")
    bad_curriculum = valid_curriculum()
    bad_curriculum["modules"][0]["hours"] = 1
    bad = tools.save_curriculum_draft(bad_curriculum)

    assert (good["status"], good["version"], good["review_status"]) == ("ok", 1, "pending_review")
    assert good["validation"]["valid"]
    assert tools.get_curriculum(good["curriculum_id"])["curriculum"]["status"] == "pending_review"
    assert (bad["status"], bad["review_status"]) == ("ok", "draft") and bad["validation"]["valid"] is False
    assert tools.get_curriculum(bad["curriculum_id"])["curriculum"]["status"] == "draft"


def test_revisions_add_versions_and_keep_history(tools):
    first = tools.save_curriculum_draft(valid_curriculum(hours=40))
    second = tools.save_curriculum_draft(valid_curriculum(hours=60), curriculum_id=first["curriculum_id"],
                                         change_summary="Duration 40 -> 60 hours")

    record = tools.get_curriculum(first["curriculum_id"])
    assert second["version"] == 2 and record["curriculum"]["content"]["total_hours"] == 60
    assert [v["version"] for v in record["versions"]] == [1, 2]
    assert tools.get_curriculum(first["curriculum_id"], 1)["curriculum"]["content"]["total_hours"] == 40


def test_review_rules_in_the_store(tmp_path):
    store = CurriculumStore(tmp_path / "c.sqlite3")
    saved = store.save_version(valid_curriculum(), {"valid": True}, [])
    cid = saved["curriculum_id"]

    with pytest.raises(CurriculumStoreError):
        store.approve(cid, 1, "system:curriculum_agent")       # the agent's own identity
    with pytest.raises(CurriculumStoreError):
        store.request_changes(cid, 1, "admin-session-1", "")   # notes required
    store.request_changes(cid, 1, "admin-session-1", "Add more practice")
    with pytest.raises(CurriculumStoreError):
        store.approve(cid, 1, "admin-session-1")               # not pending any more
    store.save_version(valid_curriculum(), {"valid": True}, [], curriculum_id=cid, change_summary="More practice")
    approved = store.approve(cid, 2, "admin-session-1")

    assert approved["status"] == "approved" and approved["approved_version"] == 2
    assert [(a["event"], a["new_status"]) for a in store.audit_trail(cid)] == [
        ("version_saved", "pending_review"), ("review", "changes_requested"),
        ("version_saved", "pending_review"), ("review", "approved")]


@pytest.mark.parametrize("bad", ["CUR-1", "../etc/passwd", "CUR-zzzzzzzzzzzz", None])
def test_ids_are_validated(tools, bad):
    assert tools.get_curriculum(bad)["status"] == "invalid_request"


@pytest.mark.parametrize("curriculum", ["text", {"title": "x", "total_hours": 0}, {"title": "x", "total_hours": "40"}])
def test_invalid_curricula_are_refused_before_saving(tools, curriculum):
    assert tools.save_curriculum_draft(curriculum)["status"] == "invalid_request"
    assert tools.list_curricula()["curricula"] == []
