"""
Backend entry point for the Admin interface.

Every admin action checks a server-side session token here; the
User/Admin choice in the UI is not trusted for authorization.

- The passcode comes from REX_ADMIN_PASSCODE (environment or .env) and
  is compared in constant time. It is never returned or logged. If it is
  not configured, admin access is disabled.
- A correct passcode creates a random session token kept in this
  process's memory with an idle timeout. The UI stores only the token
  in its own server-side session state.
- Repeated wrong passcodes are throttled.
- Agent requests go through the Admin orchestrator (all agents except
  REX Assistant).
- Tender review: every tender the RFP agent finds (tender APIs and web
  research) is stored as pending in its OpportunityStore. The admin
  approves or rejects it here, with an audit trail; approved tenders form
  the searchable tender dashboard. New tenders raise in-app alerts until
  the admin marks them seen. Tenders are never added to the course
  knowledge base.
- Applications: approving a tender starts the application workflow
  (agents/rfp_research/applications.py) right after the approval is
  saved. Submitting on the academy's behalf needs the admin's explicit
  authorisation; preparation is dry-run only and nothing is ever sent.
  Every application change below (manual submission, outcome, retry,
  resolving an unknown result) needs the admin session. The academy's
  application profile and documents are managed here too.

- Curricula (Agent 3): the agent saves drafts and new versions as
  "draft" or "pending_review"; only here can an admin approve the current
  version or request changes (with notes), with an audit trail. PDFs are
  read here for download (rendered on demand if a version has none).

This is a single shared passcode, suitable for a demo deployment: it
proves "an admin is present", not which admin. Reviews are attributed to
the admin session.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any, Callable, Optional

from agents.config import get_secret, get_setting
from agents.curriculum_research import curriculum_tools
from agents.curriculum_research.curriculum_store import STATUSES as CURRICULUM_STATUSES, CurriculumStore
from agents.rfp_research.application_profile import DOCUMENT_TYPES, PROFILE_FIELDS, ApplicationProfile
from agents.rfp_research.applications import ApplicationWorkflow
from agents.rfp_research.opportunity_store import APPROVAL_STATUSES, OpportunityStore

SESSION_IDLE_SECONDS = 30 * 60
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 60

# The request text sent with a search-form search; the form's choices
# travel in context["search"].
TENDER_SEARCH_REQUEST = "Find education and training tenders"


class AdminAuthError(PermissionError):
    """Raised for any admin action without a valid admin session."""


class AdminService:
    def __init__(
        self,
        orchestrator_factory: Callable[[], Any],
        opportunity_store: Optional[OpportunityStore] = None,
        passcode: Optional[str] = None,
        clock: Callable[[], float] = time.monotonic,
        application_profile: Optional[ApplicationProfile] = None,
        curriculum_store: Optional[CurriculumStore] = None,
        curriculum_documents_dir: Optional[str] = None,
    ) -> None:
        """
        orchestrator_factory: builds the Admin orchestrator lazily, so
        nothing admin-related is created until an admin signs in.
        opportunity_store: for tests; defaults to the RFP agent's store.
        passcode: for tests; defaults to REX_ADMIN_PASSCODE.
        application_profile: for tests; defaults to the profile in the
        same database as the tenders.
        curriculum_store / curriculum_documents_dir: for tests; default to
        the database and PDF folder the Curriculum MCP server uses
        (CURRICULUM_DB_PATH / CURRICULUM_DOCUMENTS_DIR).
        """
        self._orchestrator_factory = orchestrator_factory
        self._orchestrator = None
        self._opportunity_store = opportunity_store
        self._application_profile = application_profile
        self._curriculum_store = curriculum_store
        self._curriculum_documents_dir = curriculum_documents_dir
        self._curriculum_toolbox: Optional[curriculum_tools.CurriculumToolbox] = None
        self._application_workflow: Optional[ApplicationWorkflow] = None
        self._passcode = passcode if passcode is not None else get_secret("REX_ADMIN_PASSCODE")
        self._clock = clock
        self._sessions: dict[str, float] = {}
        self._failed_attempts = 0
        self._locked_until = 0.0

    # ---------------------------------------------------------------- auth

    @property
    def enabled(self) -> bool:
        return bool(self._passcode)

    def login(self, candidate: str) -> Optional[str]:
        """A new session token for the right passcode, else None."""
        now = self._clock()
        if not self.enabled or now < self._locked_until:
            return None

        if not isinstance(candidate, str) or not hmac.compare_digest(
            candidate.encode("utf-8"), self._passcode.encode("utf-8")
        ):
            self._failed_attempts += 1
            if self._failed_attempts >= MAX_FAILED_ATTEMPTS:
                self._locked_until = now + LOCKOUT_SECONDS
                self._failed_attempts = 0
            return None

        self._failed_attempts = 0
        token = secrets.token_urlsafe(32)
        self._sessions[token] = now + SESSION_IDLE_SECONDS
        return token

    def is_locked(self) -> bool:
        return self._clock() < self._locked_until

    def logout(self, token: Optional[str]) -> None:
        self._sessions.pop(token or "", None)

    def is_valid(self, token: Optional[str]) -> bool:
        expires = self._sessions.get(token or "")
        if expires is None or self._clock() > expires:
            self._sessions.pop(token or "", None)
            return False
        return True

    def _require(self, token: Optional[str]) -> str:
        if not self.enabled or not self.is_valid(token):
            raise AdminAuthError("Admin session required.")
        self._sessions[token] = self._clock() + SESSION_IDLE_SECONDS  # sliding expiry
        return "admin-session-" + hashlib.sha256(token.encode()).hexdigest()[:8]

    # ------------------------------------------------------------- actions

    def _admin_orchestrator(self):
        if self._orchestrator is None:
            self._orchestrator = self._orchestrator_factory()
        return self._orchestrator

    def _tenders(self) -> OpportunityStore:
        if self._opportunity_store is None:
            self._opportunity_store = self._admin_orchestrator().rfp_research_agent.opportunity_store
        return self._opportunity_store

    def ask(self, token: str, message: str, context: Optional[dict] = None) -> dict:
        """Send a request through the Admin orchestrator (REX excluded)."""
        self._require(token)
        return self._admin_orchestrator().handle_request(message, context=context or {})

    def search_tenders(self, token: str, options: dict, context: Optional[dict] = None) -> dict:
        """
        Run a tender search from the search form (countries, categories,
        priority, eligibility_notes, deadline_from, deadline_to). Found
        tenders join the review queue as pending.
        """
        return self.ask(token, TENDER_SEARCH_REQUEST, {**(context or {}), "search": dict(options)})

    def list_tenders(
        self,
        token: str,
        status: Optional[str] = None,
        country: Optional[str] = None,
        keyword: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
        submission_status: Optional[str] = None,
        application_status: Optional[str] = None,
    ) -> list[dict]:
        """The review queue (status="pending") or the dashboard (status="approved")."""
        self._require(token)
        return self._tenders().search(
            country=country or None, keyword=keyword or None, limit=limit,
            approval_status=status or None, category=category or None,
            submission_status=submission_status or None,
            application_status=application_status or None,
        )

    def decide(
        self, token: str, opportunity_id: str, decision: str, authorize_submission: bool = False,
    ) -> dict:
        """
        Record a review decision. A new approval then starts the
        application workflow (applications.py) at once; authorize_submission
        is the admin's permission for REX AI to submit where supported.
        The approval is saved first, so a workflow failure never undoes it.
        Returns the tender, with "application" when the workflow ran.
        """
        reviewer = self._require(token)
        if decision not in APPROVAL_STATUSES:
            raise ValueError(f"Unknown decision: {decision}")
        store = self._tenders()
        before = store.get(opportunity_id)
        tender = store.decide(opportunity_id, decision, reviewer, authorize_submission=authorize_submission)
        if decision == "approved" and before and before["approval_status"] != "approved":
            application = self._workflow().start(opportunity_id, reviewer)
            tender = {**store.get(opportunity_id), "application": application}  # read after the run
        return tender

    def audit_trail(self, token: str, opportunity_id: Optional[str] = None) -> list[dict]:
        self._require(token)
        return self._tenders().audit_trail(opportunity_id)

    def tender_counts(self, token: str) -> dict:
        self._require(token)
        return self._tenders().counts()

    def alerts(self, token: str) -> list[dict]:
        """New pending tenders the admin has not seen yet (in-app alerts)."""
        self._require(token)
        return self._tenders().unseen_alerts()

    def mark_alerts_seen(self, token: str) -> None:
        self._require(token)
        self._tenders().mark_alerts_seen()

    # ----------------------------------------------------- applications

    def _workflow(self) -> ApplicationWorkflow:
        if self._application_workflow is None:
            self._application_workflow = ApplicationWorkflow(self._tenders(), self._profile())
        return self._application_workflow

    def submission_route(self, token: str, opportunity_id: str) -> dict:
        """How this tender would be applied for (shown before approving)."""
        self._require(token)
        return self._workflow().route_for(opportunity_id)

    def authorize_submission(self, token: str, opportunity_id: str) -> dict:
        """Give the submission permission after the approval."""
        admin = self._require(token)
        return self._tenders().authorize_submission(opportunity_id, admin)

    def retry_submission(self, token: str, opportunity_id: str) -> dict:
        """
        Run the workflow again (e.g. after completing the profile). Only
        when nothing was sent before: never after a submission or an
        unknown result (resolve that first).
        """
        admin = self._require(token)
        return self._workflow().start(opportunity_id, admin)

    def record_manual_submission(
        self, token: str, opportunity_id: str, submitted_on: str,
        reference: Optional[str] = None, url: Optional[str] = None, notes: Optional[str] = None,
    ) -> dict:
        """The admin confirms the academy actually submitted this application."""
        admin = self._require(token)
        return self._tenders().record_manual_submission(
            opportunity_id, admin, submitted_on, reference=reference, url=url, notes=notes)

    def mark_not_submitted(self, token: str, opportunity_id: str, notes: Optional[str] = None) -> dict:
        """The admin confirms an attempt with an unknown result was not sent."""
        admin = self._require(token)
        return self._tenders().mark_not_submitted(opportunity_id, admin, notes=notes)

    def record_application_outcome(
        self, token: str, opportunity_id: str, outcome: str, notes: Optional[str] = None,
    ) -> dict:
        admin = self._require(token)
        return self._tenders().record_application_outcome(opportunity_id, outcome, admin, notes=notes)

    def application_history(self, token: str, opportunity_id: str) -> dict:
        """Every attempt and every audited decision for one tender."""
        self._require(token)
        store = self._tenders()
        return {"attempts": store.attempts(opportunity_id), "audit": store.audit_trail(opportunity_id)}

    # ------------------------------------------------ application profile

    def _profile(self) -> ApplicationProfile:
        if self._application_profile is None:
            self._application_profile = ApplicationProfile(db_path=self._tenders().db_path)
        return self._application_profile

    def application_profile(self, token: str) -> dict:
        """The academy's details and document list (no file paths)."""
        self._require(token)
        profile = self._profile()
        return {"fields": profile.fields(), "documents": profile.documents(),
                "field_labels": dict(PROFILE_FIELDS), "document_types": dict(DOCUMENT_TYPES)}

    def update_application_profile(self, token: str, values: dict) -> dict:
        admin = self._require(token)
        return self._profile().update_fields(values, admin)

    def upload_application_document(self, token: str, doc_type: str, file_name: str, data: bytes) -> dict:
        admin = self._require(token)
        return self._profile().add_document(doc_type, file_name, data, admin)

    def remove_application_document(self, token: str, document_id: str) -> bool:
        self._require(token)
        return self._profile().remove_document(document_id)

    # ------------------------------------------------------- curricula

    def _curricula(self) -> curriculum_tools.CurriculumToolbox:
        """
        The curriculum store and PDF folder the agent's MCP server writes
        to, opened in this process for review. Only store and documents are
        used (the academy catalogue is never opened here).
        """
        if self._curriculum_toolbox is None:
            store = self._curriculum_store or CurriculumStore(get_setting(curriculum_tools.DB_SETTING, "") or None)
            self._curriculum_toolbox = curriculum_tools.CurriculumToolbox(
                store=store, catalogue=_NoCatalogue(), documents_dir=self._curriculum_documents_dir)
        return self._curriculum_toolbox

    def list_curricula(self, token: str, status: Optional[str] = None, limit: int = 100) -> list[dict]:
        self._require(token)
        if status is not None and status not in CURRICULUM_STATUSES:
            raise ValueError(f"Unknown curriculum status: {status}")
        return self._curricula().store.list(status, limit)

    def curriculum_counts(self, token: str) -> dict:
        self._require(token)
        rows = self._curricula().store.list(None, 1000)
        return {status: sum(1 for r in rows if r["status"] == status) for status in CURRICULUM_STATUSES}

    def get_curriculum(self, token: str, curriculum_id: str, version: Optional[int] = None) -> Optional[dict]:
        """One version with its version list, PDFs and audit history (None if unknown)."""
        self._require(token)
        store = self._curricula().store
        record = store.get(curriculum_id, version)
        if record is None:
            return None
        return {"curriculum": record, "versions": store.versions(curriculum_id),
                "documents": store.documents(curriculum_id), "audit": store.audit_trail(curriculum_id)}

    def approve_curriculum(self, token: str, curriculum_id: str, version: int, notes: Optional[str] = None) -> dict:
        """Approve the current version (it must be pending review); audited under this admin session."""
        admin = self._require(token)
        return self._curricula().store.approve(curriculum_id, int(version), admin, notes)

    def request_curriculum_changes(self, token: str, curriculum_id: str, version: int, notes: str) -> dict:
        """Send the current version back with the admin's notes (required)."""
        admin = self._require(token)
        return self._curricula().store.request_changes(curriculum_id, int(version), admin, notes)

    def curriculum_audit_trail(self, token: str, curriculum_id: Optional[str] = None) -> list[dict]:
        self._require(token)
        return self._curricula().store.audit_trail(curriculum_id)

    def curriculum_pdf(self, token: str, curriculum_id: str, version: Optional[int] = None) -> Optional[dict]:
        """
        {"filename", "data"} for one version's PDF, rendering it first if
        that version has none (e.g. made before PDFs existed). None if the
        curriculum or version does not exist.
        """
        self._require(token)
        tools = self._curricula()
        record = tools.store.get(curriculum_id, version)
        if record is None:
            return None
        documents = [d for d in tools.store.documents(curriculum_id) if d["version"] == record["version"]]
        data = tools.document_bytes(documents[-1]["document_id"]) if documents else None
        if data is None:
            rendered = tools.render_curriculum_pdf(curriculum_id, record["version"])
            if rendered.get("status") != "ok":
                raise ValueError(rendered.get("message") or "The PDF could not be rendered.")
            documents = [rendered["document"]]
            data = tools.document_bytes(rendered["document"]["document_id"])
        return {"filename": documents[-1]["filename"], "data": data}


class _NoCatalogue:
    """The review side never reads the academy catalogue."""

    def __getattr__(self, name):
        raise RuntimeError("The academy catalogue is not available in the admin review service.")
