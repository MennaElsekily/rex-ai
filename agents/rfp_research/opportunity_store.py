"""
Persistent, searchable storage for discovered individual tender/RFP
opportunities (brief requirement 7: "stored in a searchable dashboard
or Google Sheet so that users can review and search the available
opportunities").

This is a separate concept from agents.conversation_store.ConversationStore:
- ConversationStore holds per-conversation agent memory (e.g. REX
  Assistant's learner profile), scoped to one conversation_id.
- OpportunityStore holds the project-wide catalog of discovered
  opportunities, independent of any particular conversation.

No Google Sheets credentials or dashboard UI exist in this project,
so this is a SQLite-backed catalog (the brief's own suggested stack
lists SQLite as an acceptable database choice) that is genuinely
queryable/searchable, not a Sheet -- see the final report for this
honestly-documented gap.

qualification_status is stored as the actual string
("qualified"/"not_qualified"/"unknown"), not collapsed into a boolean
-- an earlier version of this store used `1 if qualifies else 0`,
which silently turned an "unknown" assessment into the same value as
"not qualified." That bug is why this column is TEXT, not INTEGER.

Also supports brief requirement 8 ("monitor for newly discovered
relevant opportunities"): save() reports whether an opportunity is
new or was already on record -- the real, testable part of
"monitoring" this project implements. Actual notification delivery
(email/Slack/etc.) is not implemented (no channel/credentials
configured).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent / "rfp_opportunities.sqlite3"
)

_LIST_FIELDS = ("eligibility", "available_documents", "missing_information")


class OpportunityStore:
    """SQLite-backed catalog of discovered individual tender/RFP opportunities."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(
            str(self.db_path), check_same_thread=False
        )
        self._lock = threading.Lock()

        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS opportunities (
                    url TEXT PRIMARY KEY,
                    title TEXT,
                    reference_number TEXT,
                    organization TEXT,
                    country TEXT,
                    category TEXT,
                    scope_of_work TEXT,
                    deadline TEXT,
                    publication_date TEXT,
                    eligibility_json TEXT,
                    submission_instructions TEXT,
                    submission_url TEXT,
                    available_documents_json TEXT,
                    ai_summary TEXT,
                    qualification_status TEXT,
                    qualification_reason TEXT,
                    missing_information_json TEXT,
                    verification_status TEXT,
                    review_status TEXT,
                    first_seen_at TEXT,
                    last_seen_at TEXT
                )
                """
            )
            self._connection.commit()
            self._migrate_schema()

    def _migrate_schema(self) -> None:
        """
        Add any columns that an earlier version of this store's schema
        didn't have, without touching existing rows or data. CREATE
        TABLE IF NOT EXISTS above is a no-op on a table that already
        exists (e.g. one created by an earlier schema version), so a
        file from before a column was added needs this to read/write
        the new columns -- otherwise an old database file (which may
        hold real discovered opportunities) would have to be deleted
        to keep working, and this store never deletes existing data.
        """
        cursor = self._connection.execute("PRAGMA table_info(opportunities)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        all_columns = (
            "url", "title", "reference_number", "organization", "country",
            "category", "scope_of_work", "deadline", "publication_date",
            "eligibility_json", "submission_instructions", "submission_url",
            "available_documents_json", "ai_summary", "qualification_status",
            "qualification_reason", "missing_information_json",
            "verification_status", "review_status", "first_seen_at",
            "last_seen_at", "source_id", "source_name", "status",
            "retrieved_at",
        )

        for column in all_columns:
            if column not in existing_columns:
                self._connection.execute(
                    f"ALTER TABLE opportunities ADD COLUMN {column} TEXT"
                )

        self._connection.commit()

    def save(self, opportunity: dict[str, Any]) -> bool:
        """
        Insert or refresh one opportunity. Returns True if this record
        was not previously on record (a genuinely new discovery),
        False if it was already known.

        The record's own source_url is used as the stable key when
        present; API-sourced records that don't carry a literal URL
        fall back to a synthesized "source_name::source_id" key built
        from the source API's own stable identifier.
        """
        url = (
            opportunity.get("url")
            or opportunity.get("source_url")
            or (
                f"{opportunity['source_name']}::{opportunity['source_id']}"
                if opportunity.get("source_name") and opportunity.get("source_id")
                else None
            )
        )

        if not url:
            raise ValueError(
                "opportunity must include a url, source_url, or "
                "source_name+source_id to key on"
            )

        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            cursor = self._connection.execute(
                "SELECT first_seen_at FROM opportunities WHERE url = ?",
                (url,),
            )
            existing = cursor.fetchone()
            is_new = existing is None
            first_seen_at = existing[0] if existing else now

            self._connection.execute(
                """
                INSERT INTO opportunities (
                    url, title, reference_number, organization, country,
                    category, scope_of_work, deadline, publication_date,
                    eligibility_json, submission_instructions,
                    submission_url, available_documents_json, ai_summary,
                    qualification_status, qualification_reason,
                    missing_information_json, verification_status,
                    review_status, first_seen_at, last_seen_at,
                    source_id, source_name, status, retrieved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title = excluded.title,
                    reference_number = excluded.reference_number,
                    organization = excluded.organization,
                    country = excluded.country,
                    category = excluded.category,
                    scope_of_work = excluded.scope_of_work,
                    deadline = excluded.deadline,
                    publication_date = excluded.publication_date,
                    eligibility_json = excluded.eligibility_json,
                    submission_instructions = excluded.submission_instructions,
                    submission_url = excluded.submission_url,
                    available_documents_json = excluded.available_documents_json,
                    ai_summary = excluded.ai_summary,
                    qualification_status = excluded.qualification_status,
                    qualification_reason = excluded.qualification_reason,
                    missing_information_json = excluded.missing_information_json,
                    verification_status = excluded.verification_status,
                    review_status = excluded.review_status,
                    last_seen_at = excluded.last_seen_at,
                    source_id = excluded.source_id,
                    source_name = excluded.source_name,
                    status = excluded.status,
                    retrieved_at = excluded.retrieved_at
                """,
                (
                    url,
                    opportunity.get("title"),
                    opportunity.get("reference_number"),
                    opportunity.get("organization"),
                    opportunity.get("country"),
                    opportunity.get("category"),
                    opportunity.get("scope_of_work"),
                    opportunity.get("deadline"),
                    opportunity.get("publication_date"),
                    json.dumps(opportunity.get("eligibility") or [], ensure_ascii=False),
                    opportunity.get("submission_instructions"),
                    opportunity.get("submission_url"),
                    json.dumps(opportunity.get("available_documents") or [], ensure_ascii=False),
                    opportunity.get("ai_summary"),
                    opportunity.get("qualification_status"),
                    opportunity.get("qualification_reason"),
                    json.dumps(opportunity.get("missing_information") or [], ensure_ascii=False),
                    opportunity.get("verification_status"),
                    opportunity.get("review_status"),
                    first_seen_at,
                    now,
                    opportunity.get("source_id"),
                    opportunity.get("source_name"),
                    opportunity.get("status"),
                    opportunity.get("retrieved_at"),
                ),
            )
            self._connection.commit()

        return is_new

    def search(
        self,
        country: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search the stored catalog by country and/or a title keyword."""
        query = "SELECT * FROM opportunities WHERE 1=1"
        params: list[Any] = []

        if country:
            query += " AND country = ?"
            params.append(country)

        if keyword:
            query += " AND title LIKE ?"
            params.append(f"%{keyword}%")

        query += " ORDER BY last_seen_at DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cursor = self._connection.execute(query, params)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

        results = []

        for row in rows:
            record = dict(zip(columns, row))

            for field in _LIST_FIELDS:
                record[field] = json.loads(record.pop(f"{field}_json") or "[]")

            results.append(record)

        return results

    def close(self) -> None:
        with self._lock:
            self._connection.close()
