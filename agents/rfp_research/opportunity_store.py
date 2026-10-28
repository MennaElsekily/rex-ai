"""
The searchable tender catalogue and admin review queue (brief Agent 2,
requirements 13, 16 and 17).

Every tender the RFP Research agent finds -- from the World Bank API or
from web research -- is stored here as **pending**. An admin approves or
rejects it; approved tenders form the searchable dashboard. Every
decision is recorded in an audit table.

- Duplicates (requirement 13): the same URL is one row, and the same
  tender found on another site (same normalised title and country) is
  recognised as a duplicate instead of a second row. A tender found
  again keeps its approval decision.
- Alerts (requirement 17): tenders discovered since the admin last
  marked alerts as seen are reported by unseen_alerts().
- The Egypt/Saudi business rule is stored in review_status
  ("flagged_for_immediate_review" / "monitored_pipeline");
  approval_status is the admin's decision.

qualification_status is stored as the actual string
("qualified"/"not_qualified"/"unknown"), never collapsed into a boolean.

Applications (after approval) use two more fields, kept separate from
approval_status:

- submission_status: the submission process --
  not_started -> in_progress -> submitted / submission_failed /
  requires_manual_action / outcome_unknown.
  Every run is an application_attempts row. claim_submission() starts a
  run atomically, so the same tender is never submitted twice at once;
  an attempt whose result is unknown (e.g. a timeout) blocks new runs
  until an admin says whether it was actually sent.
- application_status: the outcome -- not_applied -> applied ->
  awarded / not_awarded / withdrawn. "applied" is set only by a
  confirmed submission or an admin's verified manual submission; a dry
  run can never be recorded as submitted.

save() (rediscovery) never touches the decision or application fields.

SQLite (listed in the brief's suggested stack) with additive schema
migration: columns are only ever added, so an existing database file
keeps all its rows.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent / "rfp_opportunities.sqlite3"
)

_LIST_FIELDS = ("eligibility", "available_documents", "missing_information")

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
APPROVAL_STATUSES = (PENDING, APPROVED, REJECTED)

FLAGGED = "flagged_for_immediate_review"
MONITORED = "monitored_pipeline"

# submission_status
NOT_STARTED = "not_started"
IN_PROGRESS = "in_progress"
SUBMITTED = "submitted"
SUBMISSION_FAILED = "submission_failed"
REQUIRES_MANUAL_ACTION = "requires_manual_action"
OUTCOME_UNKNOWN = "outcome_unknown"
SUBMISSION_STATUSES = (NOT_STARTED, IN_PROGRESS, SUBMITTED, SUBMISSION_FAILED,
                       REQUIRES_MANUAL_ACTION, OUTCOME_UNKNOWN)
# A new run may start only from these: nothing was sent before.
_CLAIMABLE = (NOT_STARTED, SUBMISSION_FAILED, REQUIRES_MANUAL_ACTION)
# How an in-progress attempt can end.
_ATTEMPT_RESULTS = (SUBMITTED, SUBMISSION_FAILED, REQUIRES_MANUAL_ACTION, OUTCOME_UNKNOWN)

# application_status
NOT_APPLIED = "not_applied"
APPLIED = "applied"
AWARDED = "awarded"
NOT_AWARDED = "not_awarded"
WITHDRAWN = "withdrawn"
APPLICATION_STATUSES = (NOT_APPLIED, APPLIED, AWARDED, NOT_AWARDED, WITHDRAWN)
_OUTCOMES = (AWARDED, NOT_AWARDED, WITHDRAWN)

SYSTEM_ACTOR = "system:application_workflow"


class InvalidTransition(ValueError):
    """A status change the rules do not allow (e.g. rejecting an applied tender)."""

# Column -> stored from this record key (lists are stored as JSON).
_FIELDS = (
    "title", "reference_number", "organization", "country", "category",
    "training_category", "scope_of_work", "deadline", "publication_date",
    "eligibility_json", "submission_instructions", "submission_url",
    "available_documents_json", "ai_summary", "qualification_status",
    "qualification_reason", "missing_information_json",
    "verification_status", "review_status", "source_id", "source_name",
    "source_domain", "status", "retrieved_at", "language", "priority_match",
    "contact_email",
)
# Decision and application columns: never written by save().
_APPLICATION_COLUMNS = (
    "submission_status", "submission_authorized", "submission_authorized_by",
    # submitted_via_url: where the application went (the discovered
    # submission_url, the tender's official link, is a separate field).
    "submission_method", "submitted_at", "submission_reference", "submitted_via_url",
    "application_status", "application_updated_at", "application_updated_by",
)
_ALL_COLUMNS = (
    "url", *_FIELDS, "first_seen_at", "last_seen_at", "opportunity_id",
    "dedupe_key", "approval_status", "decided_at", "decided_by",
    *_APPLICATION_COLUMNS,
)
_AUDIT_NEW_COLUMNS = ("event", "details")
_ATTEMPT_JSON = ("confirmation", "missing", "manual_actions")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_day(value: Any) -> str:
    """'2026-11-02' -> '2026-11-02'; anything else is refused (never guessed)."""
    try:
        return date.fromisoformat(str(value or "").strip()).isoformat()
    except ValueError:
        raise ValueError("A submission date (YYYY-MM-DD) is required.") from None


def opportunity_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


# Words sites add around the same title ("RFP - ...", "... in Egypt").
_TITLE_FILLER = {
    "rfp", "rfq", "rfi", "eoi", "tender", "tenders", "notice", "in", "for", "of", "the", "and",
    "a", "an", "egypt", "saudi", "arabia", "ksa",
}


def dedupe_key(title: Optional[str], country: Optional[str]) -> Optional[str]:
    """
    The same tender listed on two sites: same title words and country,
    ignoring filler words such as "RFP" or the country name. Short or
    missing titles ("Tender", "RFP") are too generic to compare.
    """
    words = [w for w in re.findall(r"[^\W_]+", str(title or "").casefold()) if w not in _TITLE_FILLER]
    if len(" ".join(words)) < 15:
        return None
    return f"{' '.join(words)}|{str(country or '').casefold()}"


def _org(name: Optional[str]) -> str:
    return " ".join(re.findall(r"[^\W_]+", str(name or "").casefold()))


class OpportunityStore:
    """SQLite-backed tender catalogue with an admin approval workflow."""

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        # The default is read when the store opens (not when this module
        # loads), so tests can point it at a temporary file.
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(
            str(self.db_path), check_same_thread=False
        )
        self._lock = threading.Lock()

        with self._lock:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS opportunities (url TEXT PRIMARY KEY)"
            )
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS opportunity_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    opportunity_id TEXT NOT NULL,
                    title TEXT,
                    previous_status TEXT,
                    decision TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    decided_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alert_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                CREATE TABLE IF NOT EXISTS application_attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    opportunity_id TEXT NOT NULL,
                    attempt_no INTEGER NOT NULL,
                    adapter TEXT NOT NULL,
                    dry_run INTEGER NOT NULL DEFAULT 0,
                    triggered_by TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    submission_reference TEXT,
                    submission_url TEXT,
                    confirmation_json TEXT,
                    missing_json TEXT,
                    manual_actions_json TEXT,
                    error TEXT,
                    UNIQUE (opportunity_id, attempt_no)
                );
                """
            )
            self._connection.commit()
            self._migrate_schema()

    def _migrate_schema(self) -> None:
        """
        Add columns an earlier schema did not have, without touching
        existing rows. Rows stored before the approval workflow existed
        become "pending", so an admin reviews them like any new tender.
        """
        cursor = self._connection.execute("PRAGMA table_info(opportunities)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        for column in _ALL_COLUMNS:
            if column not in existing_columns:
                self._connection.execute(
                    f"ALTER TABLE opportunities ADD COLUMN {column} TEXT"
                )

        # Audit rows written before applications existed have no event:
        # they are review decisions (read back as event "review").
        audit_columns = {row[1] for row in self._connection.execute(
            "PRAGMA table_info(opportunity_audit)").fetchall()}
        for column in _AUDIT_NEW_COLUMNS:
            if column not in audit_columns:
                self._connection.execute(
                    f"ALTER TABLE opportunity_audit ADD COLUMN {column} TEXT"
                )

        # dedupe_key is recomputed for every row, so a change to the rule
        # also applies to tenders stored before it.
        rows = self._connection.execute(
            "SELECT url, title, country FROM opportunities"
        ).fetchall()
        for url, title, country in rows:
            self._connection.execute(
                "UPDATE opportunities SET opportunity_id = COALESCE(opportunity_id, ?), "
                "dedupe_key = ?, "
                "approval_status = COALESCE(approval_status, ?) WHERE url = ?",
                (opportunity_id(url), dedupe_key(title, country), PENDING, url),
            )

        self._connection.commit()

    # ------------------------------------------------------------- saving

    @staticmethod
    def _key(opportunity: dict[str, Any]) -> str:
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
        return url

    @staticmethod
    def _values(opportunity: dict[str, Any]) -> list[Any]:
        values = []
        for column in _FIELDS:
            if column.endswith("_json"):
                values.append(json.dumps(opportunity.get(column[:-5]) or [], ensure_ascii=False))
            elif column == "priority_match":
                values.append("1" if opportunity.get("priority_match") else None)
            else:
                values.append(opportunity.get(column))
        return values

    def find_duplicate(self, opportunity: dict[str, Any]) -> Optional[str]:
        """The url of an existing row for the same tender, if any."""
        url = self._key(opportunity)
        key = dedupe_key(opportunity.get("title"), opportunity.get("country"))
        with self._lock:
            if self._connection.execute(
                "SELECT 1 FROM opportunities WHERE url = ?", (url,)
            ).fetchone():
                return url
            if key:
                organization = _org(opportunity.get("organization"))
                for other_url, other_org in self._connection.execute(
                    "SELECT url, organization FROM opportunities WHERE dedupe_key = ?", (key,)
                ).fetchall():
                    # Same title and country; different named issuers
                    # mean two different tenders with a generic title.
                    other = _org(other_org)
                    if not organization or not other or organization == other:
                        return other_url
        return None

    def save(self, opportunity: dict[str, Any]) -> bool:
        """
        Store or refresh one tender. Returns True for a new discovery,
        False when it was already on record (same url, or the same tender
        under another url). A tender found again keeps its approval
        decision; new tenders start as pending.
        """
        url = self._key(opportunity)
        now = _now()
        existing_url = self.find_duplicate(opportunity)

        with self._lock:
            if existing_url and existing_url != url:
                # Same tender, different site: keep the first row.
                self._connection.execute(
                    "UPDATE opportunities SET last_seen_at = ? WHERE url = ?",
                    (now, existing_url),
                )
                self._connection.commit()
                return False

            is_new = existing_url is None
            columns = ", ".join(_FIELDS)
            placeholders = ", ".join("?" for _ in _FIELDS)
            updates = ", ".join(f"{c} = excluded.{c}" for c in _FIELDS)
            self._connection.execute(
                f"""
                INSERT INTO opportunities (
                    url, {columns}, first_seen_at, last_seen_at,
                    opportunity_id, dedupe_key, approval_status
                ) VALUES (?, {placeholders}, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    {updates}, last_seen_at = excluded.last_seen_at
                """,
                (
                    url, *self._values(opportunity), now, now, opportunity_id(url),
                    dedupe_key(opportunity.get("title"), opportunity.get("country")),
                    PENDING,
                ),
            )
            self._connection.commit()

        return is_new

    # ------------------------------------------------------------- reading

    def _rows(self, query: str, params: list[Any]) -> list[dict[str, Any]]:
        with self._lock:
            cursor = self._connection.execute(query, params)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

        results = []
        for row in rows:
            record = dict(zip(columns, row))
            for field in _LIST_FIELDS:
                record[field] = json.loads(record.pop(f"{field}_json", None) or "[]")
            record["priority_match"] = bool(record.get("priority_match"))
            # Rows stored before applications existed read as "not started".
            record["submission_status"] = record.get("submission_status") or NOT_STARTED
            record["application_status"] = record.get("application_status") or NOT_APPLIED
            record["submission_authorized"] = bool(record.get("submission_authorized"))
            results.append(record)
        return results

    def search(
        self,
        country: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 50,
        approval_status: Optional[str] = None,
        category: Optional[str] = None,
        flagged: Optional[bool] = None,
        submission_status: Optional[str] = None,
        application_status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Search the catalogue. keyword matches title, organisation or
        summary. Egypt-flagged and priority-matching tenders come first,
        then the most recently seen.
        """
        query = "SELECT * FROM opportunities WHERE 1=1"
        params: list[Any] = []

        if country:
            query += " AND country = ?"
            params.append(country)
        if keyword:
            query += " AND (title LIKE ? OR organization LIKE ? OR ai_summary LIKE ?)"
            params += [f"%{keyword}%"] * 3
        if approval_status:
            if approval_status not in APPROVAL_STATUSES:
                raise ValueError(f"Unknown approval status: {approval_status}")
            query += " AND approval_status = ?"
            params.append(approval_status)
        if category:
            query += " AND training_category = ?"
            params.append(category)
        if flagged is not None:
            query += " AND review_status = ?" if flagged else " AND (review_status IS NULL OR review_status != ?)"
            params.append(FLAGGED)
        for column, value, allowed, default in (
            ("submission_status", submission_status, SUBMISSION_STATUSES, NOT_STARTED),
            ("application_status", application_status, APPLICATION_STATUSES, NOT_APPLIED),
        ):
            if value:
                if value not in allowed:
                    raise ValueError(f"Unknown {column}: {value}")
                query += f" AND COALESCE({column}, ?) = ?"
                params += [default, value]

        query += (
            " ORDER BY (review_status = ?) DESC, (priority_match IS NOT NULL) DESC,"
            " last_seen_at DESC LIMIT ?"
        )
        params += [FLAGGED, limit]
        return self._rows(query, params)

    def get(self, opp_id: str) -> Optional[dict[str, Any]]:
        rows = self._rows("SELECT * FROM opportunities WHERE opportunity_id = ?", [opp_id])
        return rows[0] if rows else None

    def counts(self) -> dict[str, int]:
        """
        Tenders per approval status, pending Egypt-flagged ones, approved
        tenders needing manual action, and applied tenders.
        """
        with self._lock:
            rows = self._connection.execute(
                "SELECT approval_status, review_status, submission_status, application_status, "
                "COUNT(*) FROM opportunities "
                "GROUP BY approval_status, review_status, submission_status, application_status"
            ).fetchall()
        counts = {status: 0 for status in APPROVAL_STATUSES}
        counts.update(flagged_pending=0, needs_manual_action=0, applied=0)
        for status, review_status, submission, application, count in rows:
            status = status or PENDING
            counts[status] = counts.get(status, 0) + count
            if status == PENDING and review_status == FLAGGED:
                counts["flagged_pending"] += count
            if status == APPROVED and submission in (REQUIRES_MANUAL_ACTION, OUTCOME_UNKNOWN):
                counts["needs_manual_action"] += count
            if (application or NOT_APPLIED) != NOT_APPLIED:
                counts["applied"] += count
        return counts

    # ------------------------------------------------------------- review

    @staticmethod
    def _reviewer(reviewer: Any) -> str:
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("A reviewer identity is required")
        return reviewer.strip()

    def _require_row(self, opp_id: str) -> dict[str, Any]:
        row = self.get(opp_id)
        if row is None:
            raise KeyError(f"Unknown opportunity: {opp_id}")
        return row

    def _audit(self, opp_id, title, previous, decision, reviewer, event, details=None) -> None:
        """One audit row; the caller holds the lock and commits."""
        self._connection.execute(
            "INSERT INTO opportunity_audit (opportunity_id, title, previous_status, decision, "
            "reviewer, decided_at, event, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (opp_id, title, previous, decision, reviewer, _now(), event,
             json.dumps(details, ensure_ascii=False) if details else None),
        )

    def decide(
        self, opp_id: str, decision: str, reviewer: str, authorize_submission: bool = False,
    ) -> dict[str, Any]:
        """
        Approve, reject or return a tender to pending, with an audit
        record. authorize_submission records the admin's permission for
        REX AI to submit on the academy's behalf (approval only); approval
        alone never sets it. A tender with an application in progress,
        submitted or with an unknown result cannot be rejected or
        returned to pending.
        """
        if decision not in APPROVAL_STATUSES:
            raise ValueError(f"Unknown decision: {decision}")
        reviewer = self._reviewer(reviewer)
        row = self._require_row(opp_id)

        locked = (row["submission_status"] in (IN_PROGRESS, SUBMITTED, OUTCOME_UNKNOWN)
                  or row["application_status"] != NOT_APPLIED)
        if decision != APPROVED and locked:
            raise InvalidTransition(
                "This tender has an application in progress, submitted or with an unknown "
                "result; record the application outcome instead of changing the review decision."
            )

        authorized = decision == APPROVED and bool(authorize_submission)
        with self._lock:
            self._connection.execute(
                "UPDATE opportunities SET approval_status = ?, decided_at = ?, decided_by = ?, "
                "submission_authorized = ?, submission_authorized_by = ? WHERE opportunity_id = ?",
                (decision, _now(), reviewer, "1" if authorized else None,
                 reviewer if authorized else None, opp_id),
            )
            self._audit(opp_id, row["title"], row["approval_status"] or PENDING, decision, reviewer,
                        "review", {"authorize_submission": authorized} if decision == APPROVED else None)
            self._connection.commit()
        return self.get(opp_id)

    def audit_trail(self, opp_id: Optional[str] = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM opportunity_audit"
        params: list[Any] = []
        if opp_id:
            query += " WHERE opportunity_id = ?"
            params.append(opp_id)
        query += " ORDER BY audit_id"
        with self._lock:
            cursor = self._connection.execute(query, params)
            columns = [d[0] for d in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        for row in rows:
            row["event"] = row.get("event") or "review"
            row["details"] = json.loads(row["details"]) if row.get("details") else {}
        return rows

    # ------------------------------------------------------- applications

    def authorize_submission(self, opp_id: str, reviewer: str) -> dict[str, Any]:
        """
        Give REX AI the admin's permission to submit this approved tender
        on the academy's behalf (when it was not given with the approval).
        """
        reviewer = self._reviewer(reviewer)
        row = self._require_row(opp_id)
        if row["approval_status"] != APPROVED:
            raise InvalidTransition("Only an approved tender can be authorised for submission.")
        with self._lock:
            self._connection.execute(
                "UPDATE opportunities SET submission_authorized = '1', submission_authorized_by = ? "
                "WHERE opportunity_id = ?", (reviewer, opp_id),
            )
            self._audit(opp_id, row["title"], row["approval_status"], APPROVED, reviewer, "review",
                        {"authorize_submission": True, "given_after_approval": True})
            self._connection.commit()
        return self.get(opp_id)

    def claim_submission(self, opp_id: str, adapter: str, triggered_by: str,
                         dry_run: bool = False) -> Optional[dict[str, Any]]:
        """
        Start one application run, atomically: the tender must be
        approved and its submission_status claimable (not_started,
        submission_failed or requires_manual_action -- nothing was sent).
        Returns the new in-progress attempt, or None when another run
        already owns the tender, it was already submitted, or its last
        result is unknown. One SQL UPDATE does the check and the change,
        so two simultaneous runs cannot both win.
        """
        triggered_by = self._reviewer(triggered_by)
        now = _now()
        with self._lock:
            claimed = self._connection.execute(
                "UPDATE opportunities SET submission_status = ?, application_updated_at = ? "
                f"WHERE opportunity_id = ? AND approval_status = ? "
                f"AND COALESCE(submission_status, ?) IN ({', '.join('?' * len(_CLAIMABLE))})",
                (IN_PROGRESS, now, opp_id, APPROVED, NOT_STARTED, *_CLAIMABLE),
            ).rowcount
            if not claimed:
                self._connection.rollback()
                return None
            attempt_no = self._connection.execute(
                "SELECT COALESCE(MAX(attempt_no), 0) + 1 FROM application_attempts "
                "WHERE opportunity_id = ?", (opp_id,),
            ).fetchone()[0]
            cursor = self._connection.execute(
                "INSERT INTO application_attempts (opportunity_id, attempt_no, adapter, dry_run, "
                "triggered_by, actor, started_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (opp_id, attempt_no, adapter, 1 if dry_run else 0, triggered_by,
                 SYSTEM_ACTOR, now, IN_PROGRESS),
            )
            self._connection.commit()
            attempt_id = cursor.lastrowid
        return self.attempt(attempt_id)

    def finish_attempt(
        self,
        attempt_id: int,
        status: str,
        submission_reference: Optional[str] = None,
        submission_url: Optional[str] = None,
        confirmation: Optional[dict[str, Any]] = None,
        missing: Optional[list[str]] = None,
        manual_actions: Optional[list[str]] = None,
        error: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Record how an in-progress attempt ended and update the tender.
        "submitted" (only for a non-dry-run attempt, with the mechanism's
        confirmation) also sets application_status to "applied".
        """
        if status not in _ATTEMPT_RESULTS:
            raise ValueError(f"Unknown attempt result: {status}")
        attempt = self.attempt(attempt_id)
        if attempt is None:
            raise KeyError(f"Unknown attempt: {attempt_id}")
        if attempt["status"] != IN_PROGRESS:
            raise InvalidTransition("This attempt has already finished.")
        if status == SUBMITTED and (attempt["dry_run"] or not confirmation):
            raise InvalidTransition(
                "Only a real submission with a confirmation can be recorded as submitted; "
                "a dry run never is."
            )

        now = _now()
        with self._lock:
            self._connection.execute(
                "UPDATE application_attempts SET status = ?, finished_at = ?, submission_reference = ?, "
                "submission_url = ?, confirmation_json = ?, missing_json = ?, manual_actions_json = ?, "
                "error = ? WHERE attempt_id = ?",
                (status, now, submission_reference, submission_url,
                 json.dumps(confirmation or {}, ensure_ascii=False),
                 json.dumps(missing or [], ensure_ascii=False),
                 json.dumps(manual_actions or [], ensure_ascii=False), error, attempt_id),
            )
            if status == SUBMITTED:
                self._connection.execute(
                    "UPDATE opportunities SET submission_status = ?, submission_method = ?, "
                    "submitted_at = ?, submission_reference = ?, submitted_via_url = ?, "
                    "application_status = ?, application_updated_at = ?, application_updated_by = ? "
                    "WHERE opportunity_id = ?",
                    (SUBMITTED, attempt["adapter"], now, submission_reference, submission_url,
                     APPLIED, now, SYSTEM_ACTOR, attempt["opportunity_id"]),
                )
            else:
                self._connection.execute(
                    "UPDATE opportunities SET submission_status = ?, "
                    "application_updated_at = ? WHERE opportunity_id = ?",
                    (status, now, attempt["opportunity_id"]),
                )
            self._connection.commit()
        return self.attempt(attempt_id)

    def recover_interrupted_attempts(self) -> int:
        """
        Attempts left in progress (the app stopped mid-run) become
        "outcome_unknown", never "failed": something may have been sent,
        so they block new runs until an admin resolves them.
        Returns how many were recovered.
        """
        now = _now()
        with self._lock:
            ids = [row[0] for row in self._connection.execute(
                "SELECT opportunity_id FROM application_attempts WHERE status = ?", (IN_PROGRESS,))]
            self._connection.execute(
                "UPDATE application_attempts SET status = ?, finished_at = ?, "
                "error = 'Interrupted (the application stopped during this attempt).' WHERE status = ?",
                (OUTCOME_UNKNOWN, now, IN_PROGRESS),
            )
            for opp_id in ids:
                self._connection.execute(
                    "UPDATE opportunities SET submission_status = ?, application_updated_at = ? "
                    "WHERE opportunity_id = ? AND submission_status = ?",
                    (OUTCOME_UNKNOWN, now, opp_id, IN_PROGRESS),
                )
            self._connection.commit()
        return len(ids)

    def record_manual_submission(
        self,
        opp_id: str,
        reviewer: str,
        submitted_on: str,
        reference: Optional[str] = None,
        url: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        An admin confirms the academy actually submitted this application
        themselves (on the portal, by email, ...): submission_status
        "submitted" (method "manual"), application_status "applied".
        Also resolves an unknown result as "it was sent".
        submitted_on: the date of the submission, YYYY-MM-DD.
        """
        reviewer = self._reviewer(reviewer)
        submitted_on = _iso_day(submitted_on)
        row = self._require_row(opp_id)
        if row["approval_status"] != APPROVED:
            raise InvalidTransition("Only an approved tender can be recorded as submitted.")
        if row["submission_status"] in (IN_PROGRESS, SUBMITTED):
            raise InvalidTransition(f"The submission is already {row['submission_status']}.")

        details = {"submitted_on": submitted_on, "reference": reference, "url": url, "notes": notes,
                   "resolved_unknown_outcome": row["submission_status"] == OUTCOME_UNKNOWN}
        now = _now()
        with self._lock:
            attempt_no = self._connection.execute(
                "SELECT COALESCE(MAX(attempt_no), 0) + 1 FROM application_attempts "
                "WHERE opportunity_id = ?", (opp_id,),
            ).fetchone()[0]
            self._connection.execute(
                "INSERT INTO application_attempts (opportunity_id, attempt_no, adapter, dry_run, "
                "triggered_by, actor, started_at, finished_at, status, submission_reference, "
                "submission_url, confirmation_json) VALUES (?, ?, 'manual', 0, ?, ?, ?, ?, ?, ?, ?, ?)",
                (opp_id, attempt_no, reviewer, reviewer, now, now, SUBMITTED, reference, url,
                 json.dumps({"confirmed_by_admin": True, **details}, ensure_ascii=False)),
            )
            self._connection.execute(
                "UPDATE opportunities SET submission_status = ?, submission_method = 'manual', "
                "submitted_at = ?, submission_reference = ?, submitted_via_url = ?, "
                "application_status = ?, application_updated_at = ?, application_updated_by = ? "
                "WHERE opportunity_id = ?",
                (SUBMITTED, submitted_on, reference, url, APPLIED, now, reviewer, opp_id),
            )
            self._audit(opp_id, row["title"], row["application_status"], APPLIED, reviewer,
                        "application", details)
            self._connection.commit()
        return self.get(opp_id)

    def mark_not_submitted(self, opp_id: str, reviewer: str, notes: Optional[str] = None) -> dict[str, Any]:
        """
        An admin confirms an attempt with an unknown result was NOT sent:
        submission_status becomes "submission_failed", so a new run may start.
        """
        reviewer = self._reviewer(reviewer)
        row = self._require_row(opp_id)
        if row["submission_status"] != OUTCOME_UNKNOWN:
            raise InvalidTransition("Only an unknown result can be resolved as not sent.")
        with self._lock:
            self._connection.execute(
                "UPDATE opportunities SET submission_status = ?, application_updated_at = ?, "
                "application_updated_by = ? WHERE opportunity_id = ?",
                (SUBMISSION_FAILED, _now(), reviewer, opp_id),
            )
            self._audit(opp_id, row["title"], OUTCOME_UNKNOWN, SUBMISSION_FAILED, reviewer,
                        "application", {"resolved_unknown_outcome": True, "notes": notes})
            self._connection.commit()
        return self.get(opp_id)

    def record_application_outcome(
        self, opp_id: str, outcome: str, reviewer: str, notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        After applying: awarded, not_awarded or withdrawn -- or "applied"
        again to correct a wrongly recorded outcome. Audited.
        """
        if outcome not in (*_OUTCOMES, APPLIED):
            raise ValueError(f"Unknown application outcome: {outcome}")
        reviewer = self._reviewer(reviewer)
        row = self._require_row(opp_id)
        current = row["application_status"]
        allowed = current == APPLIED if outcome in _OUTCOMES else current in _OUTCOMES
        if not allowed:
            raise InvalidTransition(f"Cannot change the application from {current} to {outcome}.")
        with self._lock:
            self._connection.execute(
                "UPDATE opportunities SET application_status = ?, application_updated_at = ?, "
                "application_updated_by = ? WHERE opportunity_id = ?",
                (outcome, _now(), reviewer, opp_id),
            )
            self._audit(opp_id, row["title"], current, outcome, reviewer, "application",
                        {"notes": notes} if notes else None)
            self._connection.commit()
        return self.get(opp_id)

    def attempt(self, attempt_id: int) -> Optional[dict[str, Any]]:
        rows = self._attempt_rows("SELECT * FROM application_attempts WHERE attempt_id = ?", [attempt_id])
        return rows[0] if rows else None

    def attempts(self, opp_id: str) -> list[dict[str, Any]]:
        """Every application attempt for a tender, oldest first."""
        return self._attempt_rows(
            "SELECT * FROM application_attempts WHERE opportunity_id = ? ORDER BY attempt_no", [opp_id])

    def _attempt_rows(self, query: str, params: list[Any]) -> list[dict[str, Any]]:
        with self._lock:
            cursor = self._connection.execute(query, params)
            columns = [d[0] for d in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        for row in rows:
            row["dry_run"] = bool(row["dry_run"])
            for field in _ATTEMPT_JSON:
                raw = row.pop(f"{field}_json", None)
                row[field] = json.loads(raw) if raw else ({} if field == "confirmation" else [])
        return rows

    # ------------------------------------------------------------- alerts

    def _alerts_seen_at(self) -> str:
        row = self._connection.execute(
            "SELECT value FROM alert_state WHERE key = 'seen_at'"
        ).fetchone()
        return row[0] if row else ""

    def unseen_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Pending tenders first seen since the alerts were last marked as seen."""
        with self._lock:
            seen_at = self._alerts_seen_at()
        return self._rows(
            "SELECT * FROM opportunities WHERE first_seen_at > ? "
            "AND (approval_status IS NULL OR approval_status = ?) "
            "ORDER BY (review_status = ?) DESC, first_seen_at DESC LIMIT ?",
            [seen_at, PENDING, FLAGGED, limit],
        )

    def mark_alerts_seen(self) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO alert_state (key, value) VALUES ('seen_at', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_now(),),
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()
