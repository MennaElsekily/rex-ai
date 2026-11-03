"""
Generated documents, their versions and review history (SQLite).

This is Agent 4's own database (documents.sqlite3); curricula stay in the
Curriculum Agent's store, which this module never writes to. A document
record names the curriculum id and version it was built from.

Lifecycle (per document; approval is per version):

    draft  ->  pending_review  ->  approved

- Saving a version (the agent, or an administrator's edit) gives
  "pending_review" when it passed validation and "draft" when it did not.
  It can never give "approved".
- Only approve() changes a version to approved. It needs an administrator
  identity (AdminService passes one after its session check), is not an
  agent tool, and records the version's *approved PDF* in the same
  transaction: a version is never approved without its approved file.
- Versions are never changed once written: an edit or a regeneration is a
  new version. An approved version stays the approved one
  (approved_version) until a newer version is approved, and its PDFs are
  never replaced. The head row's status describes the latest version, so a
  document with a newer version under review still has approved_version
  set: consumers must read get_approved(), not the head status.
- Each version has a review PDF (made at generation) and, once approved,
  an approved PDF whose cover says so.

Schema changes are additive only: columns missing from an older database
are added when the store opens (see _MIGRATIONS).
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agents.sqlite_utils import connect, write_transaction

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "documents.sqlite3"

DRAFT = "draft"
PENDING_REVIEW = "pending_review"
APPROVED = "approved"
STATUSES = (DRAFT, PENDING_REVIEW, APPROVED)
AGENT_ACTOR = "system:document_agent"

DOCUMENT_ID = re.compile(r"^GDOC-[0-9a-f]{12}$")

# Columns added after the first release of the schema: (table, column, type).
_MIGRATIONS = (
    ("document_versions", "approved_filename", "TEXT"),
    ("document_versions", "approved_sha256", "TEXT"),
    ("document_versions", "approved_pages", "INTEGER"),
    ("document_versions", "approved_size_bytes", "INTEGER"),
)


class DocumentStoreError(ValueError):
    """A request the store refuses (unknown id, status not allowed...)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def check_document_id(document_id: Any) -> str:
    if not isinstance(document_id, str) or not DOCUMENT_ID.match(document_id):
        raise DocumentStoreError("A document id looks like GDOC- followed by 12 hexadecimal characters.")
    return document_id


def _approval_problem(version: int, current_version: int, version_status: str, filename: Optional[str],
                      reviewer: Any) -> Optional[str]:
    """Why this version cannot be approved now (None if it can)."""
    if not isinstance(reviewer, str) or not reviewer.strip() or reviewer.startswith("system:"):
        return "An administrator identity is required."
    if version != current_version:
        return "Only the current version can be approved."
    if version_status != PENDING_REVIEW:
        return f"This version is {version_status}: only a version pending review can be approved."
    if not filename:
        return "Render the PDF before approving, so the review file exists."
    return None


class DocumentStore:
    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        # The default is read when the store opens, so tests can redirect it.
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = connect(self.db_path)
        self._lock = threading.Lock()
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    curriculum_id TEXT NOT NULL,
                    curriculum_version INTEGER NOT NULL,
                    document_type TEXT NOT NULL,
                    language TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_version INTEGER NOT NULL,
                    approved_version INTEGER,
                    approved_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS document_versions (
                    document_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    curriculum_version INTEGER NOT NULL,
                    content_json TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    change_summary TEXT,
                    filename TEXT,
                    sha256 TEXT,
                    pages INTEGER,
                    size_bytes INTEGER,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    PRIMARY KEY (document_id, version)
                );
                CREATE TABLE IF NOT EXISTS document_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    version INTEGER,
                    event TEXT NOT NULL,
                    previous_status TEXT,
                    new_status TEXT,
                    actor TEXT NOT NULL,
                    notes TEXT,
                    at TEXT NOT NULL
                );
                """
            )
            for table, column, kind in _MIGRATIONS:
                existing = {row[1] for row in self._connection.execute(f"PRAGMA table_info({table})")}
                if column not in existing:
                    self._connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
            self._connection.commit()

    # ------------------------------------------------------------ saving

    def save_version(
        self,
        content: dict[str, Any],
        validation: dict[str, Any],
        *,
        document_id: Optional[str] = None,
        change_summary: Optional[str] = None,
        created_by: str = AGENT_ACTOR,
    ) -> dict[str, Any]:
        """A new document (no document_id) or a new version of one; never approved."""
        status = PENDING_REVIEW if validation.get("valid") else DRAFT
        now = _now()
        with write_transaction(self._connection, self._lock, DocumentStoreError):
            if document_id is None:
                document_id = f"GDOC-{uuid.uuid4().hex[:12]}"
                version, previous_status = 1, None
                self._connection.execute(
                    "INSERT INTO documents (document_id, curriculum_id, curriculum_version, document_type, language, "
                    "title, status, current_version, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                    (document_id, content["curriculum_id"], content["curriculum_version"], content["document_type"],
                     content["language"], content["title"], status, now, now))
            else:
                check_document_id(document_id)
                row = self._connection.execute(
                    "SELECT current_version, status, curriculum_id, document_type, language FROM documents "
                    "WHERE document_id = ?", (document_id,)).fetchone()
                if row is None:
                    raise DocumentStoreError(f"Unknown document {document_id}.")
                if (row[2], row[3], row[4]) != (content["curriculum_id"], content["document_type"], content["language"]):
                    raise DocumentStoreError("A new version must be for the same curriculum, document type and language.")
                version, previous_status = row[0] + 1, row[1]
                self._connection.execute(
                    "UPDATE documents SET curriculum_version = ?, title = ?, status = ?, current_version = ?, "
                    "updated_at = ? WHERE document_id = ?",
                    (content["curriculum_version"], content["title"], status, version, now, document_id))
            self._connection.execute(
                "INSERT INTO document_versions (document_id, version, status, curriculum_version, content_json, "
                "validation_json, change_summary, created_at, created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (document_id, version, status, content["curriculum_version"], _json(content), _json(validation),
                 change_summary, now, created_by))
            self._audit(document_id, version, "version_saved", previous_status, status, created_by, change_summary)
        return {"document_id": document_id, "version": version, "status": status}

    # ----------------------------------------------------------- reading

    def get(self, document_id: str, version: Optional[int] = None) -> Optional[dict[str, Any]]:
        """A document with one version's content (the current one by default)."""
        check_document_id(document_id)
        with self._lock:
            cursor = self._connection.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,))
            head = cursor.fetchone()
            if head is None:
                return None
            head = dict(zip([d[0] for d in cursor.description], head))
            cursor = self._connection.execute(
                "SELECT * FROM document_versions WHERE document_id = ? AND version = ?",
                (document_id, version or head["current_version"]))
            row = cursor.fetchone()
            if row is None:
                return None
            row = dict(zip([d[0] for d in cursor.description], row))
        return {**head, "version": row["version"], "version_status": row["status"],
                "content": json.loads(row["content_json"]), "validation": json.loads(row["validation_json"]),
                "change_summary": row["change_summary"], "filename": row["filename"], "sha256": row["sha256"],
                "pages": row["pages"], "size_bytes": row["size_bytes"],
                "approved_filename": row["approved_filename"], "approved_sha256": row["approved_sha256"],
                "approved_pages": row["approved_pages"], "version_created_at": row["created_at"],
                "version_created_by": row["created_by"]}

    def get_approved(self, document_id: str) -> Optional[dict[str, Any]]:
        """
        The approved version of a document, or None if none was approved. This
        (not the head row's status, nor current_version) is what anything
        consuming a document must read.
        """
        head = self.get(document_id)
        if head is None or head["approved_version"] is None:
            return None
        record = self.get(document_id, head["approved_version"])
        return record if record and record["version_status"] == APPROVED else None

    def list(self, status: Optional[str] = None, curriculum_id: Optional[str] = None,
             limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        if status is not None and status not in STATUSES:
            raise DocumentStoreError(f"Unknown status {status}.")
        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if curriculum_id:
            clauses.append("curriculum_id = ?")
            params.append(curriculum_id)
        query = "SELECT * FROM documents" + (" WHERE " + " AND ".join(clauses) if clauses else "") + \
                " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        with self._lock:
            cursor = self._connection.execute(query, [*params, max(1, int(limit)), max(0, int(offset))])
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def counts(self) -> dict[str, int]:
        """Documents per head status (all documents, no row limit)."""
        with self._lock:
            found = dict(self._connection.execute("SELECT status, COUNT(*) FROM documents GROUP BY status").fetchall())
        return {status: found.get(status, 0) for status in STATUSES}

    def versions(self, document_id: str) -> list[dict[str, Any]]:
        check_document_id(document_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT version, status, curriculum_version, change_summary, created_at, created_by, filename, "
                "validation_json, approved_filename FROM document_versions WHERE document_id = ? ORDER BY version",
                (document_id,)).fetchall()
        return [{"version": v, "status": s, "curriculum_version": cv, "change_summary": c, "created_at": t,
                 "created_by": b, "has_pdf": bool(f), "valid": json.loads(j).get("valid"),
                 "has_approved_pdf": bool(a)}
                for v, s, cv, c, t, b, f, j, a in rows]

    def audit_trail(self, document_id: Optional[str] = None) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM document_audit", []
        if document_id:
            query += " WHERE document_id = ?"
            params.append(check_document_id(document_id))
        with self._lock:
            cursor = self._connection.execute(query + " ORDER BY audit_id", params)
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # ------------------------------------------------------------ review

    def _audit(self, document_id, version, event, previous, new, actor, notes) -> None:
        """One audit row; the caller holds the lock and commits."""
        self._connection.execute(
            "INSERT INTO document_audit (document_id, version, event, previous_status, new_status, actor, notes, at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (document_id, version, event, previous, new, actor, notes, _now()))

    def check_approvable(self, document_id: str, version: int, reviewer: Any) -> dict[str, Any]:
        """The version's record if it can be approved by this reviewer now; otherwise DocumentStoreError."""
        record = self.get(document_id, version)
        if record is None:
            raise DocumentStoreError(f"Unknown document {document_id} version {version}.")
        problem = _approval_problem(version, record["current_version"], record["version_status"],
                                    record["filename"], reviewer)
        if problem:
            raise DocumentStoreError(problem)
        return record

    def approve(self, document_id: str, version: int, reviewer: str, notes: Optional[str] = None, *,
                approved_pdf: tuple[str, bytes, int], approved_at: Optional[str] = None) -> dict[str, Any]:
        """
        Admin only (AdminService checks the session first). approved_pdf is
        (file name, bytes, pages) of the version's approved PDF, already
        written by the caller: the status change and the record of that file
        happen in one transaction, so a version is never approved without it.
        The checks run again inside the transaction.
        """
        check_document_id(document_id)
        filename, data, pages = approved_pdf
        approved_at = approved_at or _now()
        with write_transaction(self._connection, self._lock, DocumentStoreError):
            row = self._connection.execute(
                "SELECT d.current_version, v.status, v.filename, v.approved_filename FROM documents d "
                "JOIN document_versions v ON v.document_id = d.document_id AND v.version = ? "
                "WHERE d.document_id = ?", (version, document_id)).fetchone()
            if row is None:
                raise DocumentStoreError(f"Unknown document {document_id} version {version}.")
            problem = _approval_problem(version, row[0], row[1], row[2], reviewer)
            if problem:
                raise DocumentStoreError(problem)
            if row[3]:
                raise DocumentStoreError("This version already has an approved PDF.")
            updated = self._connection.execute(
                "UPDATE document_versions SET status = 'approved', approved_filename = ?, approved_sha256 = ?, "
                "approved_pages = ?, approved_size_bytes = ? WHERE document_id = ? AND version = ? "
                "AND status = 'pending_review'",
                (filename, hashlib.sha256(data).hexdigest(), pages, len(data), document_id, version)).rowcount
            if updated != 1:
                raise DocumentStoreError("This version changed while it was being approved; try again.")
            self._connection.execute(
                "UPDATE documents SET status = 'approved', approved_version = ?, approved_at = ?, updated_at = ? "
                "WHERE document_id = ?", (version, approved_at, _now(), document_id))
            self._audit(document_id, version, "review", PENDING_REVIEW, APPROVED, reviewer.strip(), notes)
        return self.get(document_id, version)

    # --------------------------------------------------------------- PDFs

    def attach_pdf(self, document_id: str, version: int, filename: str, data: bytes, pages: int) -> dict[str, Any]:
        """Record a version's review PDF once. A version that has one keeps it: files are never replaced."""
        check_document_id(document_id)
        with write_transaction(self._connection, self._lock, DocumentStoreError):
            row = self._connection.execute(
                "SELECT filename FROM document_versions WHERE document_id = ? AND version = ?",
                (document_id, version)).fetchone()
            if row is None:
                raise DocumentStoreError(f"Unknown document {document_id} version {version}.")
            if row[0]:
                raise DocumentStoreError("This version already has a PDF.")
            self._connection.execute(
                "UPDATE document_versions SET filename = ?, sha256 = ?, pages = ?, size_bytes = ? "
                "WHERE document_id = ? AND version = ?",
                (filename, hashlib.sha256(data).hexdigest(), pages, len(data), document_id, version))
            self._audit(document_id, version, "pdf_rendered", None, None, AGENT_ACTOR, filename)
        return self.get(document_id, version)

    def close(self) -> None:
        with self._lock:
            self._connection.close()
