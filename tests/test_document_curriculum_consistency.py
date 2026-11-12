"""
Document Generation always reads the curriculum's APPROVED revision
(agents/document_generation/document_tools.py's _approved_record, via
get_approved_curriculum), never a later edit that has not been approved
yet -- even though editing a curriculum (agents/admin_service.py's
edit_curriculum) saves it as a new version immediately. Covers brief
part 5: Agent 4 must not read stale content or generate from an
unapproved revision.
"""

import pytest
from pypdf import PdfReader
import io

from agents.admin_service import AdminService
from agents.document_generation.document_tools import DocumentToolbox
from tests.curriculum_fixtures import valid_curriculum
from tests.document_fakes import approved_curriculum, in_process_document_tools, save_curriculum

PASSCODE = "pass-123"


@pytest.fixture
def toolbox(tmp_path):
    return in_process_document_tools(tmp_path)


@pytest.fixture
def admin(toolbox, tmp_path):
    return AdminService(orchestrator_factory=lambda: None, passcode=PASSCODE, curriculum_store=toolbox.curricula,
                        curriculum_documents_dir=str(tmp_path / "cpdfs"))


@pytest.fixture
def token(admin):
    return admin.login(PASSCODE)


def test_document_generation_uses_the_approved_content_not_a_later_unapproved_edit(toolbox, admin, token):
    cid = approved_curriculum(toolbox)   # version 1, approved
    admin.edit_curriculum(token, cid, {"title": "An unapproved rewrite nobody signed off on"})

    found = toolbox.get_approved_curriculum(cid)

    assert found["status"] == "ok"
    assert found["curriculum"]["content"]["title"] != "An unapproved rewrite nobody signed off on"
    assert found["curriculum"]["version"] == 1   # still serving the approved version
    assert found["newer_unapproved_version"] is True   # but it does say a newer draft exists


def test_approving_the_edit_makes_document_generation_use_it(toolbox, admin, token):
    cid = approved_curriculum(toolbox)
    admin.edit_curriculum(token, cid, {"title": "A revised and now-approved title"})
    admin.approve_curriculum(token, cid, 2)

    found = toolbox.get_approved_curriculum(cid)

    assert found["curriculum"]["content"]["title"] == "A revised and now-approved title"
    assert found["curriculum"]["version"] == 2 and found["newer_unapproved_version"] is False


def test_an_unapproved_edit_of_a_never_approved_curriculum_is_still_refused(toolbox, admin, token):
    saved = save_curriculum(toolbox)   # pending_review, never approved
    cid = saved["curriculum_id"]
    admin.edit_curriculum(token, cid, {"title": "Edited while still only pending review"})

    found = toolbox.get_approved_curriculum(cid)

    assert found["status"] == "not_approved"


def test_the_approved_pdf_still_matches_the_approved_content_after_an_edit(toolbox, admin, token):
    cid = approved_curriculum(toolbox)
    approved_record = toolbox.curricula.get(cid, 1)
    admin.edit_curriculum(token, cid, {"title": "A title that must never appear in the approved PDF"})

    pdf_data = toolbox.approved_curriculum_pdf(cid)["data"]

    text = PdfReader(io.BytesIO(pdf_data)).pages[0].extract_text()
    assert "A title that must never appear in the approved PDF" not in text
    assert approved_record["content"]["title"] in text
