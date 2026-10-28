"""
Curriculum review in AdminService (agents/admin_service.py): only a valid
admin session can list, read, approve, request changes or download PDFs;
approval rules (current version, pending review, notes for changes) come
from the store; every decision is audited under the admin session. The
agent itself has no way to approve. Temporary database and PDF folder.
"""

import pytest

from agents.admin_service import AdminAuthError, AdminService
from agents.curriculum_research.curriculum_store import CurriculumStore, CurriculumStoreError
from agents.curriculum_research.curriculum_tools import CurriculumToolbox
from tests.curriculum_fixtures import valid_curriculum

PASSCODE = "pass-123"


@pytest.fixture
def store(tmp_path):
    return CurriculumStore(tmp_path / "curricula.sqlite3")


@pytest.fixture
def admin(store, tmp_path):
    return AdminService(orchestrator_factory=lambda: None, passcode=PASSCODE, curriculum_store=store,
                        curriculum_documents_dir=str(tmp_path / "docs"))


@pytest.fixture
def token(admin):
    return admin.login(PASSCODE)


def agent_save(store, content=None, curriculum_id=None, tmp_path=None):
    """Save as the agent does (through the tool, which validates)."""
    tools = CurriculumToolbox(store=store, catalogue=object(), documents_dir=tmp_path)
    return tools.save_curriculum_draft(content or valid_curriculum(), curriculum_id=curriculum_id, topic="ml")


@pytest.mark.parametrize("method, args", [
    ("list_curricula", ()), ("curriculum_counts", ()), ("get_curriculum", ("CUR-0123456789ab",)),
    ("approve_curriculum", ("CUR-0123456789ab", 1)),
    ("request_curriculum_changes", ("CUR-0123456789ab", 1, "notes")),
    ("curriculum_pdf", ("CUR-0123456789ab",)), ("curriculum_audit_trail", ()),
])
def test_every_curriculum_action_needs_an_admin_session(admin, store, tmp_path, method, args):
    saved = agent_save(store, tmp_path=tmp_path / "d")
    args = tuple(saved["curriculum_id"] if a == "CUR-0123456789ab" else a for a in args)
    for bad in (None, "", "forged-token"):
        with pytest.raises(AdminAuthError):
            getattr(admin, method)(bad, *args)
    assert store.get(saved["curriculum_id"])["status"] == "pending_review"


def test_list_get_and_counts(admin, token, store, tmp_path):
    first = agent_save(store, tmp_path=tmp_path / "d")
    invalid = valid_curriculum()
    invalid["total_hours"] = 41  # modules add up to 40 -> saved as a draft
    agent_save(store, invalid, tmp_path=tmp_path / "d")

    assert {row["status"] for row in admin.list_curricula(token)} == {"pending_review", "draft"}
    assert [r["curriculum_id"] for r in admin.list_curricula(token, status="pending_review")] == [first["curriculum_id"]]
    assert admin.curriculum_counts(token) == {"draft": 1, "pending_review": 1, "changes_requested": 0, "approved": 0}
    detail = admin.get_curriculum(token, first["curriculum_id"])
    assert detail["curriculum"]["content"]["title"] == "Machine Learning Foundations"
    assert [v["version"] for v in detail["versions"]] == [1] and detail["audit"]
    assert admin.get_curriculum(token, "CUR-ffffffffffff") is None
    with pytest.raises(ValueError):
        admin.list_curricula(token, status="published")


def test_approval_is_audited_under_the_admin_session(admin, token, store, tmp_path):
    saved = agent_save(store, tmp_path=tmp_path / "d")

    admin.approve_curriculum(token, saved["curriculum_id"], 1, notes="Looks good")

    record = store.get(saved["curriculum_id"])
    assert record["status"] == "approved" and record["approved_version"] == 1
    last = admin.curriculum_audit_trail(token, saved["curriculum_id"])[-1]
    assert last["new_status"] == "approved" and last["actor"].startswith("admin-session-")
    assert last["notes"] == "Looks good"


def test_review_rules_are_enforced(admin, token, store, tmp_path):
    saved = agent_save(store, tmp_path=tmp_path / "d")
    curriculum_id = saved["curriculum_id"]

    with pytest.raises(CurriculumStoreError, match="Say what should change"):
        admin.request_curriculum_changes(token, curriculum_id, 1, "")
    admin.request_curriculum_changes(token, curriculum_id, 1, "Add a security module.")
    assert store.get(curriculum_id)["status"] == "changes_requested"
    with pytest.raises(CurriculumStoreError, match="pending review"):
        admin.approve_curriculum(token, curriculum_id, 1)  # changes were requested

    agent_save(store, curriculum_id=curriculum_id, tmp_path=tmp_path / "d")  # the agent's revision: v2
    with pytest.raises(CurriculumStoreError, match="current version"):
        admin.approve_curriculum(token, curriculum_id, 1)
    admin.approve_curriculum(token, curriculum_id, 2)
    assert store.get(curriculum_id)["approved_version"] == 2


def test_a_draft_cannot_be_approved(admin, token, store, tmp_path):
    invalid = valid_curriculum()
    invalid["learning_outcomes"][0]["statement"] = "Understand data"
    saved = agent_save(store, invalid, tmp_path=tmp_path / "d")
    assert saved["review_status"] == "draft"

    with pytest.raises(CurriculumStoreError):
        admin.approve_curriculum(token, saved["curriculum_id"], 1)


def test_the_pdf_is_rendered_once_and_then_reused(admin, token, store, tmp_path):
    saved = agent_save(store, valid_curriculum("ar"), tmp_path=tmp_path / "d")

    first = admin.curriculum_pdf(token, saved["curriculum_id"])
    second = admin.curriculum_pdf(token, saved["curriculum_id"], 1)

    assert first["filename"] == f"{saved['curriculum_id']}_v1_ar.pdf" and first["data"].startswith(b"%PDF")
    assert second == first
    assert len(store.documents(saved["curriculum_id"])) == 1
    assert admin.curriculum_pdf(token, saved["curriculum_id"], 9) is None


def test_the_admin_side_never_opens_the_academy_catalogue(admin, token):
    tools = admin._curricula()
    with pytest.raises(RuntimeError, match="catalogue is not available"):
        tools.catalogue.courses()
