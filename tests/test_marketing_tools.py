"""
Agent 5's toolbox: approved courses only (read through Agent 3's public store methods), validated saves that can never
approve, admin-only approval, and downloads of the approved version only. Includes the parity test that keeps this
agent's own approval rule identical to Agent 4's.
"""

import copy
import json

import pytest

from agents.curriculum_research.curriculum_store import CurriculumStore
from agents.document_generation.document_store import DocumentStore
from agents.document_generation.document_tools import DocumentToolbox
from agents.marketing.marketing_store import MarketingStore
from agents.marketing.marketing_tools import MarketingToolbox
from tests.curriculum_fixtures import valid_curriculum
from tests.marketing_fakes import ADMIN, approved_course, in_process_marketing_tools, make_campaign, save_curriculum


@pytest.fixture
def toolbox(tmp_path):
    return in_process_marketing_tools(tmp_path)


# ---- approved courses only ------------------------------------------------------------------------------------------

def test_only_approved_courses_are_listed_and_offered(toolbox):
    approved = approved_course(toolbox)
    pending = save_curriculum(toolbox)["curriculum_id"]
    draft = save_curriculum(toolbox, valid=False)["curriculum_id"]

    listed = toolbox.list_approved_courses()

    assert listed["status"] == "ok" and listed["total"] == 1 and listed["has_more"] is False
    assert [c["curriculum_id"] for c in listed["courses"]] == [approved]
    assert pending not in str(listed) and draft not in str(listed)
    assert {"title", "language", "level", "total_hours", "approved_version", "target_audience"} <= set(listed["courses"][0])


def test_the_list_is_paged(toolbox):
    ids = [approved_course(toolbox) for _ in range(3)]
    first, second = toolbox.list_approved_courses(limit=2), toolbox.list_approved_courses(limit=2, offset=2)
    assert len(first["courses"]) == 2 and first["has_more"] and len(second["courses"]) == 1 and not second["has_more"]
    assert {c["curriculum_id"] for c in first["courses"] + second["courses"]} == set(ids) and first["total"] == 3


def test_an_approved_course_is_returned_with_its_content(toolbox):
    cid = approved_course(toolbox)
    found = toolbox.get_approved_course(cid)
    assert found["status"] == "ok" and found["course"]["version"] == 1 and found["newer_unapproved_version"] is False
    assert found["course"]["content"]["title"] == valid_curriculum()["title"]


@pytest.mark.parametrize("make_status", ["pending", "draft", "changes"])
def test_unapproved_courses_are_refused_with_the_reason(toolbox, make_status):
    saved = save_curriculum(toolbox, valid=make_status != "draft")
    if make_status == "changes":
        toolbox.curricula.request_changes(saved["curriculum_id"], 1, ADMIN, "fix it")

    found = toolbox.get_approved_course(saved["curriculum_id"])

    assert found["status"] == "not_approved" and "approve" in found["message"].lower()
    assert "course" not in found                                                    # nothing of the draft is handed over


def test_an_unknown_course_is_not_found_and_a_bad_id_is_invalid(toolbox):
    assert toolbox.get_approved_course("CUR-aaaaaaaaaaaa")["status"] == "not_found"
    assert toolbox.get_approved_course("nonsense")["status"] == "invalid_request"


def test_a_newer_unapproved_revision_never_replaces_the_approved_version(toolbox):
    cid = approved_course(toolbox)
    revised = valid_curriculum()
    revised["title"] = "Revised title that is not approved"
    save_curriculum(toolbox, revised, curriculum_id=cid)

    found = toolbox.get_approved_course(cid)

    assert found["status"] == "ok" and found["course"]["version"] == 1 and found["newer_unapproved_version"] is True
    assert found["course"]["content"]["title"] == valid_curriculum()["title"]
    assert toolbox.get_approved_course(cid, 2)["status"] == "not_approved"           # the revision itself is refused


# ---- parity with Agent 4's approval rule ------------------------------------------------------------------------------

SCENARIOS = ["approved", "pending", "draft", "changes_requested", "approved_then_revised", "approved_then_revised_and_changes",
             "two_approved_versions"]


def build_scenario(curricula: CurriculumStore, name: str) -> tuple[str, list]:
    """A curriculum in one of the states it can be in; returns (id, versions worth asking for)."""
    saved = curricula.save_version(valid_curriculum(), {"valid": name != "draft", "errors": []}, [])
    cid = saved["curriculum_id"]
    if name in ("approved", "approved_then_revised", "approved_then_revised_and_changes", "two_approved_versions"):
        curricula.approve(cid, 1, ADMIN)
    if name == "changes_requested":
        curricula.request_changes(cid, 1, ADMIN, "fix")
    if name in ("approved_then_revised", "approved_then_revised_and_changes", "two_approved_versions"):
        curricula.save_version(valid_curriculum(), {"valid": True, "errors": []}, [], curriculum_id=cid)
    if name == "approved_then_revised_and_changes":
        curricula.request_changes(cid, 2, ADMIN, "fix the revision")
    if name == "two_approved_versions":
        curricula.approve(cid, 2, ADMIN)
    return cid, [None, 1, 2, 9]


def outcome(call):
    try:
        record = call()
        return ("ok", record["version"], record["current_version"], record["version_status"])
    except LookupError:
        return ("not_found",)
    except PermissionError:
        return ("not_approved",)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_approval_rule_is_the_same_as_agent_4s(tmp_path, scenario):
    """Agent 5 has its own copy of the "approved only" rule; for every status a curriculum can be in, it must agree with Agent 4's."""
    curricula = CurriculumStore(tmp_path / "curricula.sqlite3")
    mine = MarketingToolbox(store=MarketingStore(tmp_path / "m.sqlite3"), curricula=curricula)
    theirs = DocumentToolbox(store=DocumentStore(tmp_path / "d.sqlite3"), curricula=curricula, documents_dir=tmp_path / "pdfs")
    cid, versions = build_scenario(curricula, scenario)

    for version in versions:
        assert outcome(lambda: mine._approved_record(cid, version)) == outcome(lambda: theirs._approved_record(cid, version)), \
            (scenario, version)
    assert outcome(lambda: mine._approved_record("CUR-aaaaaaaaaaaa")) == outcome(lambda: theirs._approved_record("CUR-aaaaaaaaaaaa"))
    # the listing agrees too
    assert [c["curriculum_id"] for c in mine.list_approved_courses()["courses"]] == \
           [c["curriculum_id"] for c in theirs.list_approved_curricula()["curricula"]]


# ---- saving ---------------------------------------------------------------------------------------------------------------

@pytest.fixture
def campaign(toolbox):
    cid = approved_course(toolbox)
    result = make_campaign(toolbox, cid)["result"]
    return cid, result["campaign_id"], copy.deepcopy(toolbox.store.get(result["campaign_id"])["content"])


def test_a_valid_campaign_is_saved_as_pending_review_never_approved(toolbox, campaign):
    cid, campaign_id, content = campaign
    saved = toolbox.save_campaign_draft(content, campaign_id, "again")

    assert saved["status"] == "ok" and saved["review_status"] == "pending_review" and saved["version"] == 2
    assert saved["validation"]["valid"] and toolbox.store.get_approved(campaign_id) is None


def test_an_invalid_campaign_is_kept_as_a_draft_with_its_problems(toolbox, campaign):
    cid, campaign_id, content = campaign
    content["channels"]["whatsapp"]["variations"][0]["message"] = "Join today and get a discount of 50 dollars, guaranteed!"

    saved = toolbox.save_campaign_draft(content, campaign_id)

    assert saved["status"] == "ok" and saved["review_status"] == "draft" and not saved["validation"]["valid"]
    codes = {e["code"] for e in saved["validation"]["errors"]}
    assert {"offer_claim", "price_claim", "guarantee_claim", "number_claim"} <= codes


def test_a_campaign_for_an_unapproved_course_is_refused(toolbox, campaign):
    cid, campaign_id, content = campaign
    other = save_curriculum(toolbox)["curriculum_id"]                                 # pending, not approved
    content["curriculum_id"] = other

    assert toolbox.save_campaign_draft(content)["status"] == "not_approved"
    assert toolbox.save_campaign_draft({**content, "curriculum_id": "CUR-aaaaaaaaaaaa"})["status"] == "not_found"


def test_a_campaign_cannot_name_a_different_language_or_course_version(toolbox, campaign):
    cid, campaign_id, content = campaign
    assert toolbox.save_campaign_draft({**content, "language": "ar"}, campaign_id)["status"] == "invalid_request"   # a new version keeps its language
    wrong_language = toolbox.save_campaign_draft({**content, "language": "ar"})                                       # a new campaign: kept as a draft
    assert wrong_language["review_status"] == "draft" and "language" in {e["code"] for e in wrong_language["validation"]["errors"]}
    revised = valid_curriculum()
    save_curriculum(toolbox, revised, curriculum_id=cid)                              # version 2, not approved
    assert toolbox.save_campaign_draft({**content, "curriculum_version": 2})["status"] == "not_approved"


def test_oversized_or_malformed_content_is_refused(toolbox):
    assert toolbox.save_campaign_draft("nope")["status"] == "invalid_request"
    assert toolbox.save_campaign_draft({"x": "y" * 300_000})["status"] == "invalid_request"


def test_the_save_tools_cannot_be_used_to_approve(toolbox, campaign):
    cid, campaign_id, content = campaign
    assert toolbox.save_admin_edit(content, campaign_id, "system:marketing_agent")["status"] == "invalid_request"
    assert toolbox.save_admin_edit(content, campaign_id, "")["status"] == "invalid_request"
    saved = toolbox.save_admin_edit(content, campaign_id, ADMIN, "edited")
    assert saved["review_status"] == "pending_review" and toolbox.store.get(campaign_id)["version_created_by"] == ADMIN


def test_get_and_list_campaigns(toolbox, campaign):
    cid, campaign_id, _ = campaign
    got = toolbox.get_campaign(campaign_id)
    assert got["status"] == "ok" and got["campaign"]["version_status"] == "pending_review" and len(got["versions"]) == 1
    assert toolbox.get_campaign("MKT-aaaaaaaaaaaa")["status"] == "not_found"
    assert [c["campaign_id"] for c in toolbox.list_campaigns("pending_review")["campaigns"]] == [campaign_id]
    assert toolbox.list_campaigns("approved")["campaigns"] == [] and toolbox.list_campaigns("nonsense")["status"] == "invalid_request"


# ---- approval and downloads -----------------------------------------------------------------------------------------------

def test_approval_needs_an_admin_and_a_course_that_is_still_approved(toolbox, campaign):
    cid, campaign_id, _ = campaign
    with pytest.raises(ValueError):
        toolbox.approve_campaign(campaign_id, 1, "system:marketing_agent")
    assert toolbox.store.get_approved(campaign_id) is None

    toolbox.curricula.save_version(valid_curriculum(), {"valid": True, "errors": []}, [], curriculum_id=cid)   # revision, v2
    toolbox.curricula.request_changes(cid, 2, ADMIN, "revise")
    assert toolbox.approve_campaign(campaign_id, 1, ADMIN)["version_status"] == "approved"      # v1 of the course stays approved


def test_a_campaign_whose_course_is_no_longer_approved_cannot_be_approved(tmp_path):
    toolbox = in_process_marketing_tools(tmp_path)
    cid = approved_course(toolbox)
    campaign_id = make_campaign(toolbox, cid)["result"]["campaign_id"]
    # the course version the campaign names is replaced by a store that does not know it as approved
    toolbox.curricula._connection.execute("UPDATE curriculum_versions SET status = 'changes_requested' WHERE curriculum_id = ?", (cid,))
    toolbox.curricula._connection.commit()

    with pytest.raises(ValueError, match="approved"):
        toolbox.approve_campaign(campaign_id, 1, ADMIN)


def test_only_the_approved_version_can_be_downloaded(toolbox, campaign):
    cid, campaign_id, content = campaign
    with pytest.raises(ValueError, match="no approved version"):
        toolbox.approved_export(campaign_id, "markdown")                              # pending: not ready
    toolbox.approve_campaign(campaign_id, 1, ADMIN, "ok")
    toolbox.save_admin_edit({**content, "tone": "friendly"}, campaign_id, ADMIN)      # v2 is pending review again

    markdown = toolbox.approved_export(campaign_id, "markdown")
    as_json = toolbox.approved_export(campaign_id, "json")

    assert markdown["version"] == 1 and markdown["filename"] == f"{campaign_id}_v1_en.md" and markdown["mime"] == "text/markdown"
    body = markdown["data"].decode("utf-8")
    assert "Professional" in body and "Friendly" not in body                           # the approved v1, not the pending v2
    parsed = json.loads(as_json["data"].decode("utf-8"))
    assert parsed["status"] == "approved" and parsed["version"] == 1 and parsed["content"]["tone"] == "professional"
    assert as_json["filename"].endswith("_v1_en.json")
    with pytest.raises(ValueError):
        toolbox.approved_export(campaign_id, "pdf")


def test_arabic_campaigns_download_as_utf8_text(tmp_path):
    toolbox = in_process_marketing_tools(tmp_path)
    cid = approved_course(toolbox, valid_curriculum("ar"))
    campaign_id = make_campaign(toolbox, cid, "ar")["result"]["campaign_id"]
    toolbox.approve_campaign(campaign_id, 1, ADMIN)

    markdown = toolbox.approved_export(campaign_id)["data"].decode("utf-8")
    as_json = toolbox.approved_export(campaign_id, "json")["data"].decode("utf-8")

    assert "حملة تسويقية" in markdown and "الجمهور المستهدف" in markdown and "أساسيات تعلم الآلة" in markdown
    assert "أساسيات تعلم الآلة" in as_json and "\\u" not in as_json                    # kept as readable Arabic text
