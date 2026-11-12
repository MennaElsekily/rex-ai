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


def test_the_pdf_is_rendered_once_on_approval_and_then_reused(admin, token, store, tmp_path):
    saved = agent_save(store, valid_curriculum("ar"), tmp_path=tmp_path / "d")
    admin.approve_curriculum(token, saved["curriculum_id"], 1)

    first = admin.curriculum_pdf(token, saved["curriculum_id"])
    second = admin.curriculum_pdf(token, saved["curriculum_id"], 1)

    assert first["filename"] == f"{saved['curriculum_id']}_v1_ar.pdf" and first["data"].startswith(b"%PDF")
    assert second == first
    assert len(store.documents(saved["curriculum_id"])) == 1   # rendered once, at approval
    assert admin.curriculum_pdf(token, saved["curriculum_id"], 9) is None


def test_no_pdf_is_served_or_rendered_before_approval(admin, token, store, tmp_path):
    pending = agent_save(store, tmp_path=tmp_path / "d")["curriculum_id"]
    invalid = valid_curriculum()
    invalid["learning_outcomes"][0]["statement"] = "Understand data"
    draft = agent_save(store, invalid, tmp_path=tmp_path / "d")["curriculum_id"]

    for cid in (pending, draft):
        with pytest.raises(ValueError, match="approved"):
            admin.curriculum_pdf(token, cid)
        with pytest.raises(ValueError, match="approved"):
            admin.curriculum_pdf(token, cid, 1)
        with pytest.raises(ValueError, match="approved version of every language"):
            admin.curricula_pdf_zip(token, cid)
        assert store.documents(cid) == []


def test_the_admin_side_never_opens_the_academy_catalogue(admin, token):
    tools = admin._curricula()
    with pytest.raises(RuntimeError, match="catalogue is not available"):
        tools.catalogue.courses()


# --------------------------------------------------------------------
# The bilingual companion: lookup, side-by-side review, ZIP download
# --------------------------------------------------------------------

def test_a_curriculum_without_a_companion_reports_none(admin, token, store, tmp_path):
    saved = agent_save(store, tmp_path=tmp_path / "d")
    assert admin.curriculum_translation(token, saved["curriculum_id"]) is None


def test_a_linked_pair_reports_each_other(admin, token, store, tmp_path):
    en = agent_save(store, valid_curriculum("en"), tmp_path=tmp_path / "d")
    ar = agent_save(store, valid_curriculum("ar"), tmp_path=tmp_path / "d")
    store.link_translation(en["curriculum_id"], ar["curriculum_id"])

    assert admin.curriculum_translation(token, en["curriculum_id"]) == ar["curriculum_id"]
    assert admin.curriculum_translation(token, ar["curriculum_id"]) == en["curriculum_id"]


def test_the_zip_contains_both_approved_language_pdfs(admin, token, store, tmp_path):
    en = agent_save(store, valid_curriculum("en"), tmp_path=tmp_path / "d")
    ar = agent_save(store, valid_curriculum("ar"), tmp_path=tmp_path / "d")
    store.link_translation(en["curriculum_id"], ar["curriculum_id"])
    admin.approve_curriculum(token, en["curriculum_id"], 1)
    with pytest.raises(ValueError, match="approved version of every language"):
        admin.curricula_pdf_zip(token, en["curriculum_id"])   # Arabic is not approved yet
    admin.approve_curriculum(token, ar["curriculum_id"], 1)

    bundle = admin.curricula_pdf_zip(token, en["curriculum_id"])

    import zipfile
    from io import BytesIO
    with zipfile.ZipFile(BytesIO(bundle["data"])) as archive:
        names = archive.namelist()
        assert f"{en['curriculum_id']}_v1_en.pdf" in names
        assert f"{ar['curriculum_id']}_v1_ar.pdf" in names
        for name in names:
            assert archive.read(name).startswith(b"%PDF")


def test_the_zip_has_only_one_pdf_without_a_companion(admin, token, store, tmp_path):
    saved = agent_save(store, tmp_path=tmp_path / "d")
    admin.approve_curriculum(token, saved["curriculum_id"], 1)

    bundle = admin.curricula_pdf_zip(token, saved["curriculum_id"])

    import zipfile
    from io import BytesIO
    with zipfile.ZipFile(BytesIO(bundle["data"])) as archive:
        assert len(archive.namelist()) == 1


@pytest.mark.parametrize("method, args", [("curriculum_translation", ("CUR-0123456789ab",)),
                                          ("curricula_pdf_zip", ("CUR-0123456789ab",))])
def test_the_bilingual_actions_also_require_an_admin_session(admin, store, tmp_path, method, args):
    saved = agent_save(store, tmp_path=tmp_path / "d")
    args = tuple(saved["curriculum_id"] if a == "CUR-0123456789ab" else a for a in args)
    with pytest.raises(AdminAuthError):
        getattr(admin, method)("forged-token", *args)


# --------------------------------------------------------------------
# Editing (agents/curriculum_research/editing.py + admin_service.edit_curriculum)
# --------------------------------------------------------------------

def test_editing_saves_a_new_version_and_keeps_the_original(admin, token, store, tmp_path):
    saved = agent_save(store, tmp_path=tmp_path / "d")
    cid = saved["curriculum_id"]
    original = store.get(cid)["content"]

    result = admin.edit_curriculum(token, cid, {"title": "A revised title", "overview": "A revised overview."})

    assert result["version"] == 2
    current = store.get(cid)
    assert current["content"]["title"] == "A revised title" and current["content"]["overview"] == "A revised overview."
    original_version = store.get(cid, 1)
    assert original_version["content"] == original   # version 1 is completely untouched


def test_editing_does_not_approve_the_new_version(admin, token, store, tmp_path):
    saved = agent_save(store, tmp_path=tmp_path / "d")
    cid = saved["curriculum_id"]

    result = admin.edit_curriculum(token, cid, {"title": "Edited"})

    assert result["status"] == "pending_review"
    assert store.get(cid)["status"] == "pending_review"


def test_editing_an_approved_version_invalidates_its_approval_without_losing_it(admin, token, store, tmp_path):
    saved = agent_save(store, tmp_path=tmp_path / "d")
    cid = saved["curriculum_id"]
    admin.approve_curriculum(token, cid, 1)
    approved_content = store.get(cid, 1)["content"]

    admin.edit_curriculum(token, cid, {"title": "Edited after approval"})

    head = store.get(cid)
    assert head["version"] == 2 and head["status"] == "pending_review"   # the new version needs approval again
    approved_version = store.get(cid)["approved_version"] if "approved_version" in store.get(cid) else None
    row = next(r for r in store.list() if r["curriculum_id"] == cid)
    assert row["approved_version"] == 1   # the admin's earlier approval is not silently carried over to the edit
    assert store.get(cid, 1)["content"] == approved_content   # what was approved is exactly preserved
    assert store.get(cid, 1)["content"]["title"] != "Edited after approval"


def test_the_edited_version_can_be_approved_like_any_other(admin, token, store, tmp_path):
    saved = agent_save(store, tmp_path=tmp_path / "d")
    cid = saved["curriculum_id"]
    admin.approve_curriculum(token, cid, 1)
    admin.edit_curriculum(token, cid, {"title": "Edited after approval"})

    admin.approve_curriculum(token, cid, 2)

    row = next(r for r in store.list() if r["curriculum_id"] == cid)
    assert row["approved_version"] == 2
    assert store.get(cid, 2)["content"]["title"] == "Edited after approval"


def test_adding_and_removing_modules(admin, token, store, tmp_path):
    saved = agent_save(store, tmp_path=tmp_path / "d")
    cid = saved["curriculum_id"]
    original_modules = store.get(cid)["content"]["modules"]
    assert len(original_modules) == 2   # the fixture's own two modules

    new_modules = [original_modules[0], {"id": "M3", "title": "A brand new module", "hours": 10,
                                        "topics": ["New topic"], "description": "New.", "activities": [],
                                        "outcome_ids": ["LO1"]}]
    admin.edit_curriculum(token, cid, {"modules": new_modules})

    updated = store.get(cid)["content"]
    assert [m["id"] for m in updated["modules"]] == ["M1", "M3"]   # M2 removed, M3 added
    assert updated["total_hours"] == sum(m["hours"] for m in new_modules)   # recomputed


def test_editing_an_unknown_curriculum_is_refused(admin, token):
    with pytest.raises(ValueError, match="Unknown curriculum"):
        admin.edit_curriculum(token, "CUR-0123456789ab", {"title": "x"})


def test_an_edit_that_fails_validation_is_saved_as_a_draft(admin, token, store, tmp_path):
    saved = agent_save(store, tmp_path=tmp_path / "d")
    cid = saved["curriculum_id"]

    result = admin.edit_curriculum(token, cid, {"learning_outcomes": [{"id": "LO1", "statement": "Understand data"}]})

    assert result["status"] == "draft"
    assert store.get(cid)["status"] == "draft"


def test_editing_one_bilingual_companion_never_touches_the_other(admin, token, store, tmp_path):
    en = agent_save(store, valid_curriculum("en"), tmp_path=tmp_path / "d")
    ar = agent_save(store, valid_curriculum("ar"), tmp_path=tmp_path / "d")
    store.link_translation(en["curriculum_id"], ar["curriculum_id"])
    ar_before = store.get(ar["curriculum_id"])["content"]

    admin.edit_curriculum(token, en["curriculum_id"], {"title": "Edited English title"})

    assert store.get(ar["curriculum_id"])["content"] == ar_before   # completely unaffected
    assert store.get(en["curriculum_id"])["content"]["title"] == "Edited English title"
    assert store.translation_of(en["curriculum_id"]) == ar["curriculum_id"]   # the link survives the edit


def test_editing_requires_an_admin_session(admin, store, tmp_path):
    saved = agent_save(store, tmp_path=tmp_path / "d")
    with pytest.raises(AdminAuthError):
        admin.edit_curriculum("forged-token", saved["curriculum_id"], {"title": "x"})


# --------------------------------------------------------------------
# PDF and ZIP consistency after an edit (brief part 4)
# --------------------------------------------------------------------

def test_the_pdf_reflects_the_latest_approved_edit_not_the_original(admin, token, store, tmp_path):
    from pypdf import PdfReader
    import io

    saved = agent_save(store, tmp_path=tmp_path / "d")
    cid = saved["curriculum_id"]
    admin.approve_curriculum(token, cid, 1)   # version 1's PDF, rendered on approval

    admin.edit_curriculum(token, cid, {"title": "A Completely Distinct New Title"})
    admin.approve_curriculum(token, cid, 2)
    latest_pdf = admin.curriculum_pdf(token, cid)   # defaults to the latest APPROVED version

    text = PdfReader(io.BytesIO(latest_pdf["data"])).pages[0].extract_text()
    assert "A Completely Distinct New Title" in text
    assert latest_pdf["filename"] == f"{cid}_v2_en.pdf"


def test_the_old_versions_pdf_is_never_overwritten_by_a_later_edit(admin, token, store, tmp_path):
    saved = agent_save(store, tmp_path=tmp_path / "d")
    cid = saved["curriculum_id"]
    admin.approve_curriculum(token, cid, 1)
    original_pdf = admin.curriculum_pdf(token, cid, 1)
    original_record = store.documents(cid)[0]

    admin.edit_curriculum(token, cid, {"title": "A new title"})
    admin.approve_curriculum(token, cid, 2)   # renders the new version's own PDF

    still_the_original = admin.curriculum_pdf(token, cid, 1)
    assert still_the_original["data"] == original_pdf["data"]
    assert still_the_original["filename"] == f"{cid}_v1_en.pdf"   # never confused with the new version's file
    assert store.documents(cid)[0] == original_record   # the v1 record still points at its own file
    assert len({d["filename"] for d in store.documents(cid)}) == 2


def test_the_zip_bundles_the_latest_version_of_each_language_after_an_edit(admin, token, store, tmp_path):
    import zipfile
    from io import BytesIO

    en = agent_save(store, valid_curriculum("en"), tmp_path=tmp_path / "d")
    ar = agent_save(store, valid_curriculum("ar"), tmp_path=tmp_path / "d")
    store.link_translation(en["curriculum_id"], ar["curriculum_id"])
    admin.approve_curriculum(token, en["curriculum_id"], 1)
    admin.approve_curriculum(token, ar["curriculum_id"], 1)
    admin.edit_curriculum(token, en["curriculum_id"], {"title": "Updated English Title"})

    # v2 is pending: the ZIP still carries the last APPROVED English version, never an unapproved one.
    with zipfile.ZipFile(BytesIO(admin.curricula_pdf_zip(token, en["curriculum_id"])["data"])) as archive:
        assert sorted(archive.namelist()) == sorted([f"{en['curriculum_id']}_v1_en.pdf",
                                                     f"{ar['curriculum_id']}_v1_ar.pdf"])
    admin.approve_curriculum(token, en["curriculum_id"], 2)

    bundle = admin.curricula_pdf_zip(token, en["curriculum_id"])

    with zipfile.ZipFile(BytesIO(bundle["data"])) as archive:
        names = archive.namelist()
        assert f"{en['curriculum_id']}_v2_en.pdf" in names   # the latest approved English version
        assert f"{en['curriculum_id']}_v1_en.pdf" not in names   # never the stale one
        assert f"{ar['curriculum_id']}_v1_ar.pdf" in names   # the Arabic companion is unaffected and still current


# --------------------------------------------------------------------
# The PDF downloaded after approval is the approved revision's own PDF
# --------------------------------------------------------------------

def _cover_text(pdf: dict) -> str:
    from io import BytesIO

    from pypdf import PdfReader

    return PdfReader(BytesIO(pdf["data"])).pages[0].extract_text()


def test_the_pdf_downloaded_after_approval_says_approved(admin, token, store, tmp_path):
    """The final PDF is generated only on approval, so its cover can only ever say "approved"."""
    saved = agent_save(store, tmp_path=tmp_path / "d")
    cid = saved["curriculum_id"]
    assert store.documents(cid) == []

    admin.approve_curriculum(token, cid, 1)

    after = admin.curriculum_pdf(token, cid, 1)
    assert "Approved by the academy" in _cover_text(after)
    assert "Pending review" not in _cover_text(after)
    assert after["filename"] == f"{cid}_v1_en.pdf"


def test_the_arabic_companion_gets_its_own_approved_pdf_and_the_zip_carries_both(admin, token, store, tmp_path):
    import zipfile
    from io import BytesIO

    from pypdf import PdfReader

    en = agent_save(store, valid_curriculum("en"), tmp_path=tmp_path / "d")["curriculum_id"]
    ar = agent_save(store, valid_curriculum("ar"), tmp_path=tmp_path / "d")["curriculum_id"]
    store.link_translation(en, ar)

    admin.approve_curriculum(token, ar, 1)   # each language is approved on its own

    assert store.get(ar)["status"] == "approved" and store.get(en)["status"] == "pending_review"
    assert [d["language"] for d in store.documents(ar)] == ["ar"] and store.documents(en) == []
    with pytest.raises(ValueError):
        admin.curricula_pdf_zip(token, en)   # never a ZIP holding an unapproved language

    admin.approve_curriculum(token, en, 1)
    with zipfile.ZipFile(BytesIO(admin.curricula_pdf_zip(token, en)["data"])) as archive:
        assert sorted(archive.namelist()) == sorted([f"{en}_v1_en.pdf", f"{ar}_v1_ar.pdf"])
        en_cover = PdfReader(BytesIO(archive.read(f"{en}_v1_en.pdf"))).pages[0].extract_text()
    assert "Approved by the academy" in en_cover


def test_an_edit_after_approval_needs_a_new_approval_and_its_pdf_says_so(admin, token, store, tmp_path):
    saved = agent_save(store, tmp_path=tmp_path / "d")
    cid = saved["curriculum_id"]
    admin.approve_curriculum(token, cid, 1)

    admin.edit_curriculum(token, cid, {"title": "An Edited Title"})

    record = store.get(cid)
    assert record["version"] == 2 and record["status"] == "pending_review" and record["approved_version"] == 1
    with pytest.raises(ValueError, match="not approved"):
        admin.curriculum_pdf(token, cid, 2)   # no PDF for the unapproved edit
    assert admin.curriculum_pdf(token, cid)["version"] == 1   # the default is the last approved version
    assert "Approved by the academy" in _cover_text(admin.curriculum_pdf(token, cid, 1))   # v1 stays approved
    with pytest.raises(CurriculumStoreError):
        admin.approve_curriculum(token, cid, 1)   # approval can never land on the outdated version
