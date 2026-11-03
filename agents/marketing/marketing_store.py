"""
Marketing campaigns, their versions and review history (SQLite).

Agent 5's own database (marketing.sqlite3, MARKETING_DB_PATH). Curricula stay in the Curriculum Agent's store, which
nothing here reads or writes; a campaign names the curriculum id and version it was written from.

Lifecycle (per campaign; approval is per version), the same rules as Agent 4's documents:

    draft  ->  pending_review  ->  approved

- Saving a version (the agent, or an administrator's edit) gives "pending_review" when it passed validation and
  "draft" when it did not. It can never give "approved".
- Only approve() makes a version approved. It needs an administrator identity (AdminService passes one after its
  session check) and is not an agent tool.
- Versions are never changed once written: an edit or a regeneration is a new version. An approved version stays the
  approved one (approved_version) until a newer version is approved. The head row's status describes the latest version,
  so a campaign with a newer version under review still has approved_version set: anything that wants content that is
  "ready" must read get_approved(), never the head status or current_version.
- Draft and pending-review content is never ready for distribution. Nothing in this project distributes anything.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agents.sqlite_utils import connect, write_transaction

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "marketing.sqlite3"

DRAFT = "draft"
PENDING_REVIEW = "pending_review"
APPROVED = "approved"
STATUSES = (DRAFT, PENDING_REVIEW, APPROVED)
AGENT_ACTOR = "system:marketing_agent"

CAMPAIGN_ID = re.compile(r"^MKT-[0-9a-f]{12}$")


class MarketingStoreError(ValueError):
    """A request the store refuses (unknown id, status not allowed...)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def check_campaign_id(campaign_id: Any) -> str:
    if not isinstance(campaign_id, str) or not CAMPAIGN_ID.match(campaign_id):
        raise MarketingStoreError("A campaign id looks like MKT- followed by 12 hexadecimal characters.")
    return campaign_id


def _approval_problem(version: int, current_version: int, version_status: str, reviewer: Any) -> Optional[str]:
    """Why this version cannot be approved now (None if it can)."""
    if not isinstance(reviewer, str) or not reviewer.strip() or reviewer.startswith("system:"):
        return "An administrator identity is required."
    if version != current_version:
        return "Only the current version can be approved."
    if version_status != PENDING_REVIEW:
        return f"This version is {version_status}: only a version pending review can be approved."
    return None


class MarketingStore:
    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        # The default is read when the store opens, so tests can redirect it.
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = connect(self.db_path)
        self._lock = threading.Lock()
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    curriculum_id TEXT NOT NULL,
                    curriculum_version INTEGER NOT NULL,
                    language TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_version INTEGER NOT NULL,
                    approved_version INTEGER,
                    approved_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS campaign_versions (
                    campaign_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    curriculum_version INTEGER NOT NULL,
                    content_json TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    change_summary TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    PRIMARY KEY (campaign_id, version)
                );
                CREATE TABLE IF NOT EXISTS campaign_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
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

    def save_version(self, content: dict[str, Any], validation: dict[str, Any], *, campaign_id: Optional[str] = None,
                     change_summary: Optional[str] = None, created_by: str = AGENT_ACTOR) -> dict[str, Any]:
        """A new campaign (no campaign_id) or a new version of one; never approved."""
        status = PENDING_REVIEW if validation.get("valid") else DRAFT
        now = _now()
        with write_transaction(self._connection, self._lock, MarketingStoreError):
            if campaign_id is None:
                campaign_id = f"MKT-{uuid.uuid4().hex[:12]}"
                version, previous_status = 1, None
                self._connection.execute(
                    "INSERT INTO campaigns (campaign_id, curriculum_id, curriculum_version, language, title, status, "
                    "current_version, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)",
                    (campaign_id, content["curriculum_id"], content["curriculum_version"], content["language"],
                     content["title"], status, now, now))
            else:
                check_campaign_id(campaign_id)
                row = self._connection.execute(
                    "SELECT current_version, status, curriculum_id, language FROM campaigns WHERE campaign_id = ?",
                    (campaign_id,)).fetchone()
                if row is None:
                    raise MarketingStoreError(f"Unknown campaign {campaign_id}.")
                if (row[2], row[3]) != (content["curriculum_id"], content["language"]):
                    raise MarketingStoreError("A new version must be for the same course and language.")
                version, previous_status = row[0] + 1, row[1]
                self._connection.execute(
                    "UPDATE campaigns SET curriculum_version = ?, title = ?, status = ?, current_version = ?, "
                    "updated_at = ? WHERE campaign_id = ?",
                    (content["curriculum_version"], content["title"], status, version, now, campaign_id))
            self._connection.execute(
                "INSERT INTO campaign_versions (campaign_id, version, status, curriculum_version, content_json, "
                "validation_json, change_summary, created_at, created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (campaign_id, version, status, content["curriculum_version"], _json(content), _json(validation),
                 change_summary, now, created_by))
            self._audit(campaign_id, version, "version_saved", previous_status, status, created_by, change_summary)
        return {"campaign_id": campaign_id, "version": version, "status": status}

    # ----------------------------------------------------------- reading

    def get(self, campaign_id: str, version: Optional[int] = None) -> Optional[dict[str, Any]]:
        """A campaign with one version's content (the current one by default)."""
        check_campaign_id(campaign_id)
        with self._lock:
            cursor = self._connection.execute("SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,))
            head = cursor.fetchone()
            if head is None:
                return None
            head = dict(zip([d[0] for d in cursor.description], head))
            cursor = self._connection.execute(
                "SELECT * FROM campaign_versions WHERE campaign_id = ? AND version = ?",
                (campaign_id, version or head["current_version"]))
            row = cursor.fetchone()
            if row is None:
                return None
            row = dict(zip([d[0] for d in cursor.description], row))
        return {**head, "version": row["version"], "version_status": row["status"],
                "content": json.loads(row["content_json"]), "validation": json.loads(row["validation_json"]),
                "change_summary": row["change_summary"], "version_created_at": row["created_at"],
                "version_created_by": row["created_by"]}

    def get_approved(self, campaign_id: str) -> Optional[dict[str, Any]]:
        """
        The approved version of a campaign, or None if none was approved. This (not the head row's status, nor
        current_version) is the only content that may be treated as ready for distribution.
        """
        head = self.get(campaign_id)
        if head is None or head["approved_version"] is None:
            return None
        record = self.get(campaign_id, head["approved_version"])
        return record if record and record["version_status"] == APPROVED else None

    def list(self, status: Optional[str] = None, curriculum_id: Optional[str] = None,
             limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        if status is not None and status not in STATUSES:
            raise MarketingStoreError(f"Unknown status {status}.")
        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if curriculum_id:
            clauses.append("curriculum_id = ?")
            params.append(curriculum_id)
        query = "SELECT * FROM campaigns" + (" WHERE " + " AND ".join(clauses) if clauses else "") + \
                " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        with self._lock:
            cursor = self._connection.execute(query, [*params, max(1, int(limit)), max(0, int(offset))])
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def counts(self) -> dict[str, int]:
        """Campaigns per head status (all campaigns, no row limit)."""
        with self._lock:
            found = dict(self._connection.execute("SELECT status, COUNT(*) FROM campaigns GROUP BY status").fetchall())
        return {status: found.get(status, 0) for status in STATUSES}

    def versions(self, campaign_id: str) -> list[dict[str, Any]]:
        check_campaign_id(campaign_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT version, status, curriculum_version, change_summary, created_at, created_by, validation_json "
                "FROM campaign_versions WHERE campaign_id = ? ORDER BY version", (campaign_id,)).fetchall()
        return [{"version": v, "status": s, "curriculum_version": cv, "change_summary": c, "created_at": t,
                 "created_by": b, "valid": json.loads(j).get("valid")} for v, s, cv, c, t, b, j in rows]

    def audit_trail(self, campaign_id: Optional[str] = None) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM campaign_audit", []
        if campaign_id:
            query += " WHERE campaign_id = ?"
            params.append(check_campaign_id(campaign_id))
        with self._lock:
            cursor = self._connection.execute(query + " ORDER BY audit_id", params)
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # ------------------------------------------------------------ review

    def _audit(self, campaign_id, version, event, previous, new, actor, notes) -> None:
        """One audit row; the caller holds the lock and commits."""
        self._connection.execute(
            "INSERT INTO campaign_audit (campaign_id, version, event, previous_status, new_status, actor, notes, at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (campaign_id, version, event, previous, new, actor, notes, _now()))

    def check_approvable(self, campaign_id: str, version: int, reviewer: Any) -> dict[str, Any]:
        """The version's record if it can be approved by this reviewer now; otherwise MarketingStoreError."""
        record = self.get(campaign_id, version)
        if record is None:
            raise MarketingStoreError(f"Unknown campaign {campaign_id} version {version}.")
        problem = _approval_problem(version, record["current_version"], record["version_status"], reviewer)
        if problem:
            raise MarketingStoreError(problem)
        return record

    def approve(self, campaign_id: str, version: int, reviewer: str, notes: Optional[str] = None) -> dict[str, Any]:
        """Admin only (AdminService checks the session first). The checks run again inside the transaction."""
        check_campaign_id(campaign_id)
        with write_transaction(self._connection, self._lock, MarketingStoreError):
            row = self._connection.execute(
                "SELECT c.current_version, v.status FROM campaigns c "
                "JOIN campaign_versions v ON v.campaign_id = c.campaign_id AND v.version = ? "
                "WHERE c.campaign_id = ?", (version, campaign_id)).fetchone()
            if row is None:
                raise MarketingStoreError(f"Unknown campaign {campaign_id} version {version}.")
            problem = _approval_problem(version, row[0], row[1], reviewer)
            if problem:
                raise MarketingStoreError(problem)
            updated = self._connection.execute(
                "UPDATE campaign_versions SET status = 'approved' WHERE campaign_id = ? AND version = ? "
                "AND status = 'pending_review'", (campaign_id, version)).rowcount
            if updated != 1:
                raise MarketingStoreError("This version changed while it was being approved; try again.")
            now = _now()
            self._connection.execute(
                "UPDATE campaigns SET status = 'approved', approved_version = ?, approved_at = ?, updated_at = ? "
                "WHERE campaign_id = ?", (version, now, now, campaign_id))
            self._audit(campaign_id, version, "review", PENDING_REVIEW, APPROVED, reviewer.strip(), notes)
        return self.get(campaign_id, version)

    def close(self) -> None:
        with self._lock:
            self._connection.close()
