"""
The academy's application profile (agents/rfp_research/application_profile.py)
and its admin actions. Temp database and temp documents folder only.
"""

import pytest

from agents.admin_service import AdminAuthError, AdminService
from agents.rfp_research import application_profile
from agents.rfp_research.application_profile import MAX_DOCUMENT_BYTES, ApplicationProfile

ADMIN = "admin-session-1"
PDF = b"%PDF-1.7\n% academy profile\n"
DOCX = b"PK\x03\x04" + b"\x00" * 40
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


@pytest.fixture
def profile(tmp_path):
    p = ApplicationProfile(db_path=tmp_path / "tenders.sqlite3", documents_dir=tmp_path / "docs")
    yield p
    p.close()


# --------------------------------------------------------------------
# Fields
# --------------------------------------------------------------------

def test_fields_start_empty_and_nothing_is_invented(profile):
    assert profile.fields() == {}


def test_fields_are_saved_cleaned_and_removed_when_emptied(profile):
    profile.update_fields({"legal_name": "  Andalusia Academy LLC \x07", "contact_email": "info@academy.example"},
                          ADMIN)
    profile.update_fields({"legal_name": ""}, ADMIN)

    assert profile.fields() == {"contact_email": "info@academy.example"}


@pytest.mark.parametrize("values", [{"bank_password": "x"}, {"contact_email": "not an email"}])
def test_unknown_fields_and_bad_emails_are_refused(profile, values):
    with pytest.raises(ValueError):
        profile.update_fields(values, ADMIN)
    assert profile.fields() == {}


def test_an_admin_identity_is_required(profile):
    with pytest.raises(ValueError):
        profile.update_fields({"legal_name": "X"}, "")


# --------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------

def test_a_document_is_stored_under_a_random_name(profile):
    doc = profile.add_document("company_profile", r"C:\Users\me\..\Company Profile.pdf", PDF, ADMIN)

    assert doc["original_name"] == "Company Profile.pdf"  # display only
    assert "path" not in doc and "stored_name" not in doc
    path = profile.document_path(doc["document_id"])
    assert path.parent == profile.documents_dir and path.name == f"{doc['document_id']}.pdf"
    assert path.read_bytes() == PDF
    assert profile.document_types_present() == {"company_profile"}


@pytest.mark.parametrize("name, data", [("cv.docx", DOCX), ("stamp.png", PNG), ("scan.JPG", b"\xff\xd8\xff\xe0abc")])
def test_accepted_file_types(profile, name, data):
    assert profile.add_document("other", name, data, ADMIN)["size_bytes"] == len(data)


@pytest.mark.parametrize("doc_type, name, data, message", [
    ("company_profile", "run.exe", b"MZ...", "Only PDF"),
    ("company_profile", "profile.pdf", b"", "empty"),
    ("company_profile", "renamed.pdf", b"MZ not a pdf", "not a valid .pdf"),
    ("company_profile", "huge.pdf", b"%PDF" + b"0" * MAX_DOCUMENT_BYTES, "10 MB"),
    ("passport_scan", "p.pdf", PDF, "Unknown document type"),
], ids=["executable", "empty", "renamed", "over-10-MB", "unknown-type"])
def test_unsafe_or_wrong_files_are_refused(profile, doc_type, name, data, message):
    with pytest.raises(ValueError, match=message):
        profile.add_document(doc_type, name, data, ADMIN)
    assert profile.documents() == []
    assert not profile.documents_dir.exists() or list(profile.documents_dir.iterdir()) == []


def test_removing_a_document_deletes_its_file(profile):
    doc = profile.add_document("company_profile", "p.pdf", PDF, ADMIN)
    path = profile.document_path(doc["document_id"])

    assert profile.remove_document(doc["document_id"]) is True
    assert not path.exists() and profile.documents() == []
    assert profile.remove_document(doc["document_id"]) is False


def test_the_default_documents_folder_is_temporary_in_tests(tmp_path):
    p = ApplicationProfile()

    assert p.documents_dir == application_profile.DEFAULT_DOCUMENTS_DIR
    assert tmp_path in p.documents_dir.parents  # conftest.py
    p.close()


# --------------------------------------------------------------------
# Admin actions
# --------------------------------------------------------------------

@pytest.fixture
def admin(profile):
    return AdminService(orchestrator_factory=lambda: None, passcode="correct horse",
                        application_profile=profile)


def test_the_admin_manages_the_profile(admin):
    token = admin.login("correct horse")

    admin.update_application_profile(token, {"legal_name": "Andalusia Academy"})
    doc = admin.upload_application_document(token, "company_profile", "profile.pdf", PDF)
    view = admin.application_profile(token)

    assert view["fields"] == {"legal_name": "Andalusia Academy"}
    assert [d["document_id"] for d in view["documents"]] == [doc["document_id"]]
    assert view["documents"][0]["uploaded_by"].startswith("admin-session-")
    assert "legal_name" in view["field_labels"] and "company_profile" in view["document_types"]
    assert admin.remove_application_document(token, doc["document_id"]) is True


@pytest.mark.parametrize("action, args", [
    ("application_profile", ()),
    ("update_application_profile", ({"legal_name": "X"},)),
    ("upload_application_document", ("company_profile", "p.pdf", PDF)),
    ("remove_application_document", ("abc",)),
])
def test_profile_actions_require_an_admin_session(admin, action, args):
    with pytest.raises(AdminAuthError):
        getattr(admin, action)("forged-token", *args)
