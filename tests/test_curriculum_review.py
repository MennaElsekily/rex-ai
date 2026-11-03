"""
Curriculum review (Agent 3's store): approval and requested changes only from the current version while it is pending review,
decided inside one transaction so that concurrent reviewers get one consistent result, with version history and audit kept.
"""

import threading

import pytest

from agents.admin_service import AdminAuthError, AdminService
from agents.curriculum_research.curriculum_store import CurriculumStore, CurriculumStoreError
from tests.curriculum_fixtures import valid_curriculum

ADMIN = "admin-session-test0001"
VALID, INVALID = {"valid": True, "errors": []}, {"valid": False, "errors": [{"code": "x"}]}


@pytest.fixture
def store(tmp_path):
    return CurriculumStore(tmp_path / "cur.sqlite3")


def make(store, valid=True, curriculum_id=None):
    return store.save_version(valid_curriculum(), VALID if valid else INVALID, [], curriculum_id=curriculum_id)["curriculum_id"]


def events(store, cid):
    return [(r["event"], r["version"], r["new_status"]) for r in store.audit_trail(cid)]


# ---- the transitions -----------------------------------------------------------------------------------------------------------------

def test_a_pending_version_is_approved_with_one_audit_row_and_consistent_records(store):
    cid = make(store)
    approved = store.approve(cid, 1, ADMIN, "reviewed")

    assert approved["status"] == "approved" and approved["approved_version"] == 1
    assert store.get(cid)["status"] == "approved" and store.get(cid, 1)["version_status"] == "approved"           # head and version agree
    assert [v["status"] for v in store.versions(cid)] == ["approved"]
    assert events(store, cid) == [("version_saved", 1, "pending_review"), ("review", 1, "approved")]
    assert store.audit_trail(cid)[-1]["actor"] == ADMIN and store.audit_trail(cid)[-1]["notes"] == "reviewed"
    assert store.count_approved() == 1 and [c["curriculum_id"] for c in store.list_approved()] == [cid]


def test_approving_twice_is_refused_and_adds_no_second_audit_row(store):
    cid = make(store)
    store.approve(cid, 1, ADMIN)
    with pytest.raises(CurriculumStoreError, match="This version is approved"):
        store.approve(cid, 1, ADMIN)
    assert events(store, cid).count(("review", 1, "approved")) == 1


@pytest.mark.parametrize("setup", ["draft", "changes_requested"])
def test_a_version_that_is_not_pending_review_cannot_be_approved(store, setup):
    cid = make(store, valid=setup != "draft")
    if setup == "changes_requested":
        store.request_changes(cid, 1, ADMIN, "fix it")
    with pytest.raises(CurriculumStoreError, match="only a version pending review"):
        store.approve(cid, 1, ADMIN)
    assert store.get(cid)["approved_version"] is None and store.count_approved() == 0


def test_only_the_current_version_can_be_reviewed(store):
    cid = make(store)
    make(store, curriculum_id=cid)                                                                          # version 2
    with pytest.raises(CurriculumStoreError, match="Only the current version"):
        store.approve(cid, 1, ADMIN)
    assert store.approve(cid, 2, ADMIN)["approved_version"] == 2


def test_a_revision_never_replaces_the_approved_version_until_it_is_approved(store):
    cid = make(store)
    store.approve(cid, 1, ADMIN)
    make(store, curriculum_id=cid)

    head = store.get(cid)
    assert head["status"] == "pending_review" and head["approved_version"] == 1                               # the approved one stays usable
    assert store.get(cid, 1)["version_status"] == "approved" and store.get(cid, 2)["version_status"] == "pending_review"
    store.request_changes(cid, 2, ADMIN, "not yet")
    assert store.get(cid)["approved_version"] == 1 and store.get(cid, 1)["version_status"] == "approved"


@pytest.mark.parametrize("version, cid_suffix, message", [(9, "", "Unknown curriculum"), (1, "x", "A curriculum id looks like")])
def test_unknown_versions_and_malformed_ids_are_refused(store, version, cid_suffix, message):
    cid = make(store)
    with pytest.raises(CurriculumStoreError, match=message):
        store.approve(cid + cid_suffix, version, ADMIN)
    with pytest.raises(CurriculumStoreError, match="Unknown curriculum"):
        store.approve("CUR-aaaaaaaaaaaa", 1, ADMIN)


@pytest.mark.parametrize("reviewer", [None, "", "   ", "system:curriculum_agent", 5])
def test_only_an_administrator_identity_can_review(store, reviewer):
    cid = make(store)
    with pytest.raises(CurriculumStoreError, match="administrator"):
        store.approve(cid, 1, reviewer)
    with pytest.raises(CurriculumStoreError, match="administrator"):
        store.request_changes(cid, 1, reviewer, "notes")
    assert store.get(cid)["status"] == "pending_review"


def test_requesting_changes_needs_notes_and_then_a_new_version_can_be_approved(store):
    cid = make(store)
    with pytest.raises(CurriculumStoreError, match="Say what should change"):
        store.request_changes(cid, 1, ADMIN, "  ")
    store.request_changes(cid, 1, ADMIN, "add a capstone")
    assert store.get(cid)["status"] == "changes_requested"
    with pytest.raises(CurriculumStoreError):
        store.approve(cid, 1, ADMIN)
    make(store, curriculum_id=cid)
    store.approve(cid, 2, ADMIN)
    assert [v["status"] for v in store.versions(cid)] == ["changes_requested", "approved"]
    assert events(store, cid) == [("version_saved", 1, "pending_review"), ("review", 1, "changes_requested"),
                                  ("version_saved", 2, "pending_review"), ("review", 2, "approved")]


# ---- concurrent reviewers ------------------------------------------------------------------------------------------------------------------

def race(actions):
    barrier, outcomes = threading.Barrier(len(actions)), []

    def run(action):
        barrier.wait(timeout=30)
        try:
            outcomes.append(("ok", action()))
        except Exception as exc:
            outcomes.append(("error", str(exc)))

    threads = [threading.Thread(target=run, args=(a,)) for a in actions]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    return outcomes


def test_two_approvals_at_once_give_one_approval_and_one_refusal(tmp_path):
    path = tmp_path / "cur.sqlite3"
    cid = make(CurriculumStore(path))
    stores = [CurriculumStore(path), CurriculumStore(path)]                                                  # two connections, as two processes

    outcomes = race([lambda s=s: s.approve(cid, 1, ADMIN)["approved_version"] for s in stores])

    assert sorted(o[0] for o in outcomes) == ["error", "ok"], outcomes
    assert "pending review" in next(o[1] for o in outcomes if o[0] == "error")
    check = CurriculumStore(path)
    assert check.get(cid)["approved_version"] == 1 and check.get(cid)["status"] == "approved"
    assert events(check, cid) == [("version_saved", 1, "pending_review"), ("review", 1, "approved")]


@pytest.mark.parametrize("repeat", range(5))
def test_approving_and_requesting_changes_at_once_leave_one_consistent_result(tmp_path, repeat):
    path = tmp_path / "cur.sqlite3"
    cid = make(CurriculumStore(path))
    a, b = CurriculumStore(path), CurriculumStore(path)

    outcomes = race([lambda: a.approve(cid, 1, ADMIN), lambda: b.request_changes(cid, 1, ADMIN, "needs work")])

    assert sorted(o[0] for o in outcomes) == ["error", "ok"], outcomes
    check = CurriculumStore(path)
    head, version = check.get(cid), check.get(cid, 1)
    assert head["status"] == version["version_status"] and head["status"] in ("approved", "changes_requested")
    assert (head["approved_version"] == 1) == (head["status"] == "approved")                                # never "changes requested" AND approved
    reviews = [e for e in events(check, cid) if e[0] == "review"]
    assert len(reviews) == 1 and reviews[0][2] == head["status"]


def test_a_review_that_fails_midway_changes_nothing(store, monkeypatch):
    cid = make(store)
    original = store._audit
    monkeypatch.setattr(store, "_audit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    with pytest.raises(RuntimeError):
        store.request_changes(cid, 1, ADMIN, "notes")
    monkeypatch.setattr(store, "_audit", original)

    assert store.get(cid)["status"] == "pending_review" and store.get(cid, 1)["version_status"] == "pending_review"
    assert events(store, cid) == [("version_saved", 1, "pending_review")]
    assert store.approve(cid, 1, ADMIN)["status"] == "approved"


# ---- the admin session still guards every review ------------------------------------------------------------------------------------------

def test_the_admin_service_still_requires_a_session_to_review(tmp_path):
    store = CurriculumStore(tmp_path / "cur.sqlite3")
    cid = make(store)
    admin = AdminService(orchestrator_factory=lambda: None, passcode="throwaway", curriculum_store=store,
                         curriculum_documents_dir=str(tmp_path / "pdfs"))
    for call in (lambda t: admin.approve_curriculum(t, cid, 1), lambda t: admin.request_curriculum_changes(t, cid, 1, "x")):
        for token in (None, "", "forged-token"):
            with pytest.raises(AdminAuthError):
                call(token)
    assert store.get(cid)["status"] == "pending_review"
    token = admin.login("throwaway")
    approved = admin.approve_curriculum(token, cid, 1, "ok")
    assert approved["status"] == "approved" and store.audit_trail(cid)[-1]["actor"].startswith("admin-session-")
