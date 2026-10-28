"""
Curriculum drafts, versions, documents and review history (SQLite).

Lifecycle (per curriculum; approval is per version):

    draft  ->  pending_review  ->  approved
                     |   ^
                     v   |  (a new version)
              changes_requested

- The agent saves versions: "pending_review" when the version passed
  validation, "draft" when it did not. It can never save "approved".
- Only approve() / request_changes() change the review status; they need
  a reviewer identity and are called by AdminService after its session
  check. Every status change is written to curriculum_audit.
- A revision is a new version; old versions are kept. An approved
  version stays approved (approved_version) until a newer one is approved.

Mapping to the tender convention (pending/approved/rejected):
pending_review ~ pending, changes_requested ~ rejected-with-feedback.

Schema changes are additive only, like the other stores.
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

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "curricula.sqlite3"

DRAFT = "draft"
PENDING_REVIEW = "pending_review"
CHANGES_REQUESTED = "changes_requested"
APPROVED = "approved"
STATUSES = (DRAFT, PENDING_REVIEW, CHANGES_REQUESTED, APPROVED)
AGENT_ACTOR = "system:curriculum_agent"

CURRICULUM_ID = re.compile(r"^CUR-[0-9a-f]{12}$")
DOCUMENT_ID = re.compile(r"^DOC-[0-9a-f]{12}$")


class CurriculumStoreError(ValueError):
    """A request the store refuses (unknown id, status not allowed...)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def check_curriculum_id(curriculum_id: Any) -> str:
    if not isinstance(curriculum_id, str) or not CURRICULUM_ID.match(curriculum_id):
        raise CurriculumStoreError("A curriculum id looks like CUR- followed by 12 hexadecimal characters.")
    return curriculum_id


class CurriculumStore:
    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        # The default is read when the store opens, so tests can redirect it.
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS curricula (
                    curriculum_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    topic TEXT,
                    language TEXT NOT NULL,
                    level TEXT,
                    target_audience TEXT,
                    total_hours INTEGER,
                    framework_id TEXT,
                    status TEXT NOT NULL,
                    current_version INTEGER NOT NULL,
                    approved_version INTEGER,
                    origin TEXT,
                    source_request TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS curriculum_versions (
                    curriculum_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    alignment_json TEXT,
                    sources_json TEXT NOT NULL,
                    change_summary TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    PRIMARY KEY (curriculum_id, version)
                );
                CREATE TABLE IF NOT EXISTS curriculum_documents (
                    document_id TEXT PRIMARY KEY,
                    curriculum_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    language TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    pages INTEGER,
                    size_bytes INTEGER,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS curriculum_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    curriculum_id TEXT NOT NULL,
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
            self._connection.commit()

    # ------------------------------------------------------------ saving

    def save_version(
        self,
        content: dict[str, Any],
        validation: dict[str, Any],
        sources: list[dict[str, Any]],
        *,
        curriculum_id: Optional[str] = None,
        change_summary: Optional[str] = None,
        alignment: Optional[dict[str, Any]] = None,
        framework_id: Optional[str] = None,
        topic: Optional[str] = None,
        origin: str = "request",
        source_request: Optional[str] = None,
        created_by: str = AGENT_ACTOR,
    ) -> dict[str, Any]:
        """
        Save a new curriculum (no curriculum_id) or a new version of one.
        The status is pending_review when validation passed, else draft --
        never approved.
        """
        if not isinstance(content, dict) or not str(content.get("title") or "").strip():
            raise CurriculumStoreError("A curriculum with a title is required.")
        status = PENDING_REVIEW if validation.get("valid") else DRAFT
        now = _now()
        with self._lock:
            if curriculum_id is None:
                curriculum_id = f"CUR-{uuid.uuid4().hex[:12]}"
                version = 1
                previous_status = None
                self._connection.execute(
                    "INSERT INTO curricula (curriculum_id, title, topic, language, level, target_audience, "
                    "total_hours, framework_id, status, current_version, origin, source_request, created_at, "
                    "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)",
                    (curriculum_id, content["title"], topic, content.get("language", "en"), content.get("level"),
                     content.get("target_audience"), content.get("total_hours"), framework_id, status, origin,
                     (source_request or "")[:1000], now, now),
                )
            else:
                check_curriculum_id(curriculum_id)
                row = self._connection.execute(
                    "SELECT current_version, status FROM curricula WHERE curriculum_id = ?", (curriculum_id,)
                ).fetchone()
                if row is None:
                    raise CurriculumStoreError(f"Unknown curriculum {curriculum_id}.")
                version, previous_status = row[0] + 1, row[1]
                self._connection.execute(
                    "UPDATE curricula SET title = ?, language = ?, level = ?, target_audience = ?, total_hours = ?, "
                    "framework_id = COALESCE(?, framework_id), status = ?, current_version = ?, updated_at = ? "
                    "WHERE curriculum_id = ?",
                    (content["title"], content.get("language", "en"), content.get("level"),
                     content.get("target_audience"), content.get("total_hours"), framework_id, status, version, now,
                     curriculum_id),
                )
            self._connection.execute(
                "INSERT INTO curriculum_versions (curriculum_id, version, status, content_json, validation_json, "
                "alignment_json, sources_json, change_summary, created_at, created_by) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (curriculum_id, version, status, _json(content), _json(validation),
                 _json(alignment) if alignment else None, _json(sources or []), change_summary, now, created_by),
            )
            self._audit(curriculum_id, version, "version_saved", previous_status, status, created_by, change_summary)
            self._connection.commit()
        return {"curriculum_id": curriculum_id, "version": version, "status": status}

    # ----------------------------------------------------------- reading

    def get(self, curriculum_id: str, version: Optional[int] = None) -> Optional[dict[str, Any]]:
        """A curriculum with one version's content (the current one by default)."""
        check_curriculum_id(curriculum_id)
        with self._lock:
            cursor = self._connection.execute(
                "SELECT * FROM curricula WHERE curriculum_id = ?", (curriculum_id,)
            )
            head = cursor.fetchone()
            if head is None:
                return None
            head = dict(zip([d[0] for d in cursor.description], head))
            row = self._connection.execute(
                "SELECT version, status, content_json, validation_json, alignment_json, sources_json, "
                "change_summary, created_at, created_by FROM curriculum_versions "
                "WHERE curriculum_id = ? AND version = ?",
                (curriculum_id, version or head["current_version"]),
            ).fetchone()
        if row is None:
            return None
        return {
            **head,
            "version": row[0], "version_status": row[1], "content": json.loads(row[2]),
            "validation": json.loads(row[3]), "alignment": json.loads(row[4]) if row[4] else None,
            "sources": json.loads(row[5]), "change_summary": row[6], "version_created_at": row[7],
            "version_created_by": row[8],
        }

    def list(self, status: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
        if status is not None and status not in STATUSES:
            raise CurriculumStoreError(f"Unknown status {status}.")
        query = "SELECT * FROM curricula" + (" WHERE status = ?" if status else "") + \
                " ORDER BY updated_at DESC LIMIT ?"
        with self._lock:
            cursor = self._connection.execute(query, [status, limit] if status else [limit])
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def versions(self, curriculum_id: str) -> list[dict[str, Any]]:
        check_curriculum_id(curriculum_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT version, status, change_summary, created_at, created_by, validation_json "
                "FROM curriculum_versions WHERE curriculum_id = ? ORDER BY version", (curriculum_id,)
            ).fetchall()
        return [{"version": v, "status": s, "change_summary": c, "created_at": t, "created_by": b,
                 "valid": json.loads(j).get("valid")} for v, s, c, t, b, j in rows]

    def audit_trail(self, curriculum_id: Optional[str] = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM curriculum_audit"
        params: list[Any] = []
        if curriculum_id:
            query += " WHERE curriculum_id = ?"
            params.append(check_curriculum_id(curriculum_id))
        with self._lock:
            cursor = self._connection.execute(query + " ORDER BY audit_id", params)
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # ------------------------------------------------------------ review

    def _audit(self, curriculum_id, version, event, previous, new, actor, notes) -> None:
        """One audit row; the caller holds the lock and commits."""
        self._connection.execute(
            "INSERT INTO curriculum_audit (curriculum_id, version, event, previous_status, new_status, actor, "
            "notes, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (curriculum_id, version, event, previous, new, actor, notes, _now()),
        )

    def _review(self, curriculum_id: str, version: int, reviewer: str, new_status: str, notes: Optional[str]):
        if not isinstance(reviewer, str) or not reviewer.strip() or reviewer.startswith("system:"):
            raise CurriculumStoreError("An administrator identity is required.")
        record = self.get(curriculum_id, version)
        if record is None:
            raise CurriculumStoreError(f"Unknown curriculum {curriculum_id} version {version}.")
        if version != record["current_version"]:
            raise CurriculumStoreError("Only the current version can be reviewed.")
        if record["status"] != PENDING_REVIEW:
            raise CurriculumStoreError(
                f"This version is {record['status']}: only a version pending review can be reviewed.")
        with self._lock:
            self._connection.execute(
                "UPDATE curricula SET status = ?, approved_version = CASE WHEN ? = 'approved' THEN ? "
                "ELSE approved_version END, updated_at = ? WHERE curriculum_id = ?",
                (new_status, new_status, version, _now(), curriculum_id),
            )
            self._connection.execute(
                "UPDATE curriculum_versions SET status = ? WHERE curriculum_id = ? AND version = ?",
                (new_status, curriculum_id, version),
            )
            self._audit(curriculum_id, version, "review", record["status"], new_status, reviewer.strip(), notes)
            self._connection.commit()
        return self.get(curriculum_id, version)

    def approve(self, curriculum_id: str, version: int, reviewer: str, notes: Optional[str] = None):
        """Admin only (AdminService checks the session first)."""
        return self._review(curriculum_id, version, reviewer, APPROVED, notes)

    def request_changes(self, curriculum_id: str, version: int, reviewer: str, notes: str):
        """Admin only. Notes saying what to change are required."""
        if not isinstance(notes, str) or not notes.strip():
            raise CurriculumStoreError("Say what should change.")
        return self._review(curriculum_id, version, reviewer, CHANGES_REQUESTED, notes.strip())

    # --------------------------------------------------------- documents

    def add_document(self, curriculum_id: str, version: int, language: str, filename: str,
                     data: bytes, pages: int) -> dict[str, Any]:
        check_curriculum_id(curriculum_id)
        document_id = f"DOC-{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._connection.execute(
                "INSERT INTO curriculum_documents (document_id, curriculum_id, version, language, filename, "
                "sha256, pages, size_bytes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (document_id, curriculum_id, version, language, filename, hashlib.sha256(data).hexdigest(),
                 pages, len(data), _now()),
            )
            self._connection.commit()
        return self.document(document_id)

    def document(self, document_id: str) -> Optional[dict[str, Any]]:
        if not isinstance(document_id, str) or not DOCUMENT_ID.match(document_id):
            raise CurriculumStoreError("Unknown document id.")
        with self._lock:
            cursor = self._connection.execute(
                "SELECT * FROM curriculum_documents WHERE document_id = ?", (document_id,))
            row = cursor.fetchone()
            columns = [d[0] for d in cursor.description]
        return dict(zip(columns, row)) if row else None

    def documents(self, curriculum_id: str) -> list[dict[str, Any]]:
        check_curriculum_id(curriculum_id)
        with self._lock:
            cursor = self._connection.execute(
                "SELECT * FROM curriculum_documents WHERE curriculum_id = ? ORDER BY created_at", (curriculum_id,))
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._connection.close()
