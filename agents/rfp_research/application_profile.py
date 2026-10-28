"""
The academy's application profile: the real details and documents an
application needs, entered and uploaded by the admin.

Nothing here is ever invented or generated: an empty field stays empty,
and the application workflow (applications.py) lists what is missing
instead of filling it in.

Storage:
- the fields and the document list live in the tender database
  (rfp_opportunities.sqlite3), tables application_profile and
  application_documents;
- uploaded files are saved under agents/rfp_research/application_documents/
  (gitignored) with a random file name, so a file name typed by a user
  never becomes a path.

Protection: only PDF, DOCX, PNG and JPG files up to 10 MB whose content
matches their type are accepted. The profile and documents are not
exposed through MCP and never sent to an LLM. Admin access goes through
agents/admin_service.py.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agents.rfp_research import opportunity_store

DEFAULT_DOCUMENTS_DIR = Path(__file__).resolve().parent / "application_documents"

# Field -> label. All optional; the workflow says which an application needs.
PROFILE_FIELDS = {
    "legal_name": "Legal name",
    "registration_number": "Commercial registration number",
    "tax_number": "Tax registration number",
    "address": "Address",
    "country": "Country of registration",
    "contact_person": "Contact person",
    "contact_title": "Contact person's title",
    "contact_email": "Contact email",
    "contact_phone": "Contact phone",
    "website": "Website",
}
MAX_FIELD_CHARS = 300

DOCUMENT_TYPES = {
    "company_profile": "Company profile",
    "registration_certificate": "Commercial registration certificate",
    "tax_card": "Tax card",
    "experience_references": "Experience / reference letters",
    "cvs": "CVs of key staff",
    "other": "Other",
}
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
# Extension -> the bytes such a file starts with.
_SIGNATURES = {
    ".pdf": (b"%PDF",),
    ".docx": (b"PK\x03\x04",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
}

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any, limit: int = MAX_FIELD_CHARS) -> str:
    return _CONTROL.sub(" ", str(value or "")).strip()[:limit]


def _display_name(file_name: str) -> str:
    """The original file name, for display only (never used as a path)."""
    name = re.split(r"[\\/]", str(file_name or ""))[-1]
    return _clean(name, 120) or "document"


class ApplicationProfile:
    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        documents_dir: Optional[str | Path] = None,
    ) -> None:
        # Defaults are read when the profile opens, so tests can point
        # them at temporary locations (tests/conftest.py).
        self.db_path = Path(db_path or opportunity_store.DEFAULT_DB_PATH)
        self.documents_dir = Path(documents_dir or DEFAULT_DOCUMENTS_DIR)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS application_profile (
                    field TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS application_documents (
                    document_id TEXT PRIMARY KEY,
                    doc_type TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    uploaded_by TEXT NOT NULL
                );
                """
            )
            self._connection.commit()

    # ------------------------------------------------------------ fields

    def fields(self) -> dict[str, str]:
        """The saved fields (empty fields are absent)."""
        with self._lock:
            rows = self._connection.execute("SELECT field, value FROM application_profile").fetchall()
        return {field: value for field, value in rows if field in PROFILE_FIELDS}

    def update_fields(self, values: dict[str, Any], updated_by: str) -> dict[str, str]:
        """
        Save the given fields; an empty value removes that field. Unknown
        fields and an invalid email are refused (nothing is saved then).
        """
        if not isinstance(updated_by, str) or not updated_by.strip():
            raise ValueError("An admin identity is required")
        unknown = set(values) - set(PROFILE_FIELDS)
        if unknown:
            raise ValueError(f"Unknown profile fields: {', '.join(sorted(unknown))}")
        cleaned = {field: _clean(value) for field, value in values.items()}
        email = cleaned.get("contact_email")
        if email and not _EMAIL.match(email):
            raise ValueError("The contact email is not a valid email address.")

        now = _now()
        with self._lock:
            for field, value in cleaned.items():
                if value:
                    self._connection.execute(
                        "INSERT INTO application_profile (field, value, updated_at, updated_by) "
                        "VALUES (?, ?, ?, ?) ON CONFLICT(field) DO UPDATE SET value = excluded.value, "
                        "updated_at = excluded.updated_at, updated_by = excluded.updated_by",
                        (field, value, now, updated_by.strip()),
                    )
                else:
                    self._connection.execute("DELETE FROM application_profile WHERE field = ?", (field,))
            self._connection.commit()
        return self.fields()

    # --------------------------------------------------------- documents

    def add_document(self, doc_type: str, file_name: str, data: bytes, uploaded_by: str) -> dict[str, Any]:
        """
        Store one uploaded document. Refuses unknown types, other file
        types, files over 10 MB, empty files and files whose content does
        not match their extension (e.g. a renamed file).
        """
        if doc_type not in DOCUMENT_TYPES:
            raise ValueError(f"Unknown document type: {doc_type}")
        if not isinstance(uploaded_by, str) or not uploaded_by.strip():
            raise ValueError("An admin identity is required")
        extension = Path(_display_name(file_name)).suffix.lower()
        if extension not in _SIGNATURES:
            raise ValueError("Only PDF, DOCX, PNG and JPG files are accepted.")
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise ValueError("The file is empty.")
        if len(data) > MAX_DOCUMENT_BYTES:
            raise ValueError("The file is larger than 10 MB.")
        if not any(bytes(data).startswith(signature) for signature in _SIGNATURES[extension]):
            raise ValueError(f"The file's content is not a valid {extension} file.")

        document_id = uuid.uuid4().hex
        stored_name = f"{document_id}{extension}"
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        (self.documents_dir / stored_name).write_bytes(bytes(data))
        with self._lock:
            self._connection.execute(
                "INSERT INTO application_documents (document_id, doc_type, original_name, stored_name, "
                "size_bytes, sha256, uploaded_at, uploaded_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (document_id, doc_type, _display_name(file_name), stored_name, len(data),
                 hashlib.sha256(bytes(data)).hexdigest(), _now(), uploaded_by.strip()),
            )
            self._connection.commit()
        return self.document(document_id)

    def documents(self) -> list[dict[str, Any]]:
        """Every stored document (without its file path), oldest first."""
        with self._lock:
            cursor = self._connection.execute(
                "SELECT document_id, doc_type, original_name, size_bytes, sha256, uploaded_at, uploaded_by "
                "FROM application_documents ORDER BY uploaded_at")
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def document(self, document_id: str) -> Optional[dict[str, Any]]:
        return next((d for d in self.documents() if d["document_id"] == document_id), None)

    def document_types_present(self) -> set[str]:
        return {d["doc_type"] for d in self.documents()}

    def document_path(self, document_id: str) -> Optional[Path]:
        """The stored file, for attaching it to an application (internal use)."""
        with self._lock:
            row = self._connection.execute(
                "SELECT stored_name FROM application_documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self.documents_dir / row[0] if row else None

    def remove_document(self, document_id: str) -> bool:
        """Delete a document and its file. Returns False if it did not exist."""
        path = self.document_path(document_id)
        if path is None:
            return False
        with self._lock:
            self._connection.execute("DELETE FROM application_documents WHERE document_id = ?", (document_id,))
            self._connection.commit()
        path.unlink(missing_ok=True)
        return True

    def close(self) -> None:
        with self._lock:
            self._connection.close()
