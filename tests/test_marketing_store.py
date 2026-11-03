"""
Agent 5's own database: versions, review status and audit trail. Nothing here can make content "approved" except the
admin-only approve(), draft and pending-review content is never what get_approved() returns, and an edit or a
regeneration is always a new version.
"""

import pytest

from agents.marketing.marketing_store import (
    APPROVED,
    DRAFT,
    PENDING_REVIEW,
    MarketingStore,
    MarketingStoreError,
    check_campaign_id,
)
from tests.marketing_fakes import ADMIN

CONTENT = {"campaign_type": "marketing_campaign", "title": "Course", "curriculum_id": "CUR-0123456789ab", "curriculum_version": 1,
           "language": "en", "marker": "v1"}
VALID, INVALID = {"valid": True, "errors": []}, {"valid": False, "errors": [{"code": "x", "message": "bad"}]}


@pytest.fixture
def store(tmp_path):
    return MarketingStore(tmp_path / "marketing.sqlite3")


def test_a_valid_save_is_pending_review_and_an_invalid_one_is_a_draft(store):
    good, bad = store.save_version(CONTENT, VALID), store.save_version(CONTENT, INVALID)

    assert good["status"] == PENDING_REVIEW and bad["status"] == DRAFT
    assert good["campaign_id"].startswith("MKT-") and good["version"] == 1
    assert APPROVED not in (good["status"], bad["status"])           # saving can never approve


def test_every_edit_or_regeneration_is_a_new_immutable_version(store):
    first = store.save_version(CONTENT, VALID)
    second = store.save_version({**CONTENT, "marker": "v2"}, VALID, campaign_id=first["campaign_id"], change_summary="edited",
                                created_by=ADMIN)

    assert second["version"] == 2 and store.get(first["campaign_id"], 1)["content"]["marker"] == "v1"
    assert store.get(first["campaign_id"])["content"]["marker"] == "v2"           # the current version
    assert [v["version"] for v in store.versions(first["campaign_id"])] == [1, 2]


def test_a_new_version_must_be_for_the_same_course_and_language(store):
    first = store.save_version(CONTENT, VALID)
    with pytest.raises(MarketingStoreError):
        store.save_version({**CONTENT, "curriculum_id": "CUR-aaaaaaaaaaaa"}, VALID, campaign_id=first["campaign_id"])
    with pytest.raises(MarketingStoreError):
        store.save_version({**CONTENT, "language": "ar"}, VALID, campaign_id=first["campaign_id"])
    with pytest.raises(MarketingStoreError):
        store.save_version(CONTENT, VALID, campaign_id="MKT-aaaaaaaaaaaa")


def test_only_an_admin_can_approve_a_pending_current_version(store):
    saved = store.save_version(CONTENT, VALID)
    cid = saved["campaign_id"]
    for reviewer in (None, "", "   ", "system:marketing_agent", "system:anything"):
        with pytest.raises(MarketingStoreError, match="administrator"):
            store.approve(cid, 1, reviewer)
    assert store.get(cid)["version_status"] == PENDING_REVIEW and store.get_approved(cid) is None

    approved = store.approve(cid, 1, ADMIN, "looks good")

    assert approved["version_status"] == APPROVED and approved["status"] == APPROVED and approved["approved_version"] == 1
    assert store.get_approved(cid)["version"] == 1


def test_a_draft_cannot_be_approved(store):
    saved = store.save_version(CONTENT, INVALID)
    with pytest.raises(MarketingStoreError, match="pending review"):
        store.approve(saved["campaign_id"], 1, ADMIN)
    assert store.get_approved(saved["campaign_id"]) is None


def test_an_approved_version_cannot_be_approved_again_and_an_old_version_cannot_be_approved(store):
    cid = store.save_version(CONTENT, VALID)["campaign_id"]
    store.approve(cid, 1, ADMIN)
    with pytest.raises(MarketingStoreError):
        store.approve(cid, 1, ADMIN)                                  # already approved
    store.save_version({**CONTENT, "marker": "v2"}, VALID, campaign_id=cid)
    with pytest.raises(MarketingStoreError, match="current version"):
        store.approve(cid, 1, ADMIN)                                  # v1 is no longer the current version
    with pytest.raises(MarketingStoreError):
        store.approve(cid, 5, ADMIN)                                  # no such version


def test_a_newer_version_under_review_never_replaces_the_approved_one(store):
    cid = store.save_version(CONTENT, VALID)["campaign_id"]
    store.approve(cid, 1, ADMIN)
    store.save_version({**CONTENT, "marker": "v2"}, VALID, campaign_id=cid)

    head = store.get(cid)
    assert head["version"] == 2 and head["version_status"] == PENDING_REVIEW        # the latest is not approved
    assert head["approved_version"] == 1
    assert store.get_approved(cid)["content"]["marker"] == "v1"                     # the only content that is "ready"

    store.save_version({**CONTENT, "marker": "v3"}, INVALID, campaign_id=cid)       # a draft changes nothing either
    assert store.get_approved(cid)["version"] == 1


def test_pending_and_draft_campaigns_have_no_approved_content(store):
    pending, draft = store.save_version(CONTENT, VALID)["campaign_id"], store.save_version(CONTENT, INVALID)["campaign_id"]
    assert store.get_approved(pending) is None and store.get_approved(draft) is None
    assert store.get(pending)["approved_version"] is None and store.get(pending)["approved_at"] is None


def test_the_audit_trail_records_every_save_and_the_approval(store):
    cid = store.save_version(CONTENT, VALID)["campaign_id"]
    store.save_version({**CONTENT, "marker": "v2"}, VALID, campaign_id=cid, change_summary="edited", created_by=ADMIN)
    store.approve(cid, 2, ADMIN, "ok")

    trail = store.audit_trail(cid)

    assert [(r["event"], r["version"], r["new_status"], r["actor"]) for r in trail] == [
        ("version_saved", 1, PENDING_REVIEW, "system:marketing_agent"), ("version_saved", 2, PENDING_REVIEW, ADMIN),
        ("review", 2, APPROVED, ADMIN)]
    assert trail[-1]["notes"] == "ok" and trail[1]["notes"] == "edited"
    assert store.audit_trail() == trail                                              # (one campaign here)


def test_counts_and_lists(store):
    a, b = store.save_version(CONTENT, VALID)["campaign_id"], store.save_version(CONTENT, INVALID)["campaign_id"]
    store.approve(a, 1, ADMIN)

    assert store.counts() == {"draft": 1, "pending_review": 0, "approved": 1}
    assert [c["campaign_id"] for c in store.list(DRAFT)] == [b] and [c["campaign_id"] for c in store.list(APPROVED)] == [a]
    assert len(store.list(curriculum_id="CUR-0123456789ab")) == 2 and store.list(curriculum_id="CUR-aaaaaaaaaaaa") == []
    with pytest.raises(MarketingStoreError):
        store.list("nonsense")


@pytest.mark.parametrize("bad", [None, "", "MKT-short", "GDOC-0123456789ab", "MKT-0123456789AB", 12])
def test_campaign_ids_are_checked(bad):
    with pytest.raises(MarketingStoreError):
        check_campaign_id(bad)


def test_the_database_is_its_own_file_and_survives_reopening(tmp_path):
    path = tmp_path / "own.sqlite3"
    cid = MarketingStore(path).save_version(CONTENT, VALID)["campaign_id"]
    assert MarketingStore(path).get(cid)["content"]["marker"] == "v1" and path.exists()
