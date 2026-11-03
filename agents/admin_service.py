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

- Documents (Agent 4): the agent generates drafts from approved curricula
  only. Editing, approving and downloading happen here, behind the admin
  session; a document has its own review status and audit trail, separate
  from the curriculum's. Text copied from the curriculum cannot be edited.

- Marketing campaigns (Agent 5): the agent writes pending-review campaigns from
  approved courses only. Editing, approving and downloading happen here, behind
  the admin session, with their own audit trail. Only the approved version of a
  campaign can be downloaded; nothing is published, e-mailed or sent.

This is a single shared passcode, suitable for a demo deployment: it
proves "an admin is present", not which admin. Reviews are attributed to
the admin session.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import re
import secrets
import time
from pathlib import Path
from typing import Any, Callable, Optional

from agents.config import get_secret, get_setting
from agents.curriculum_research import curriculum_tools
from agents.curriculum_research.curriculum_store import STATUSES as CURRICULUM_STATUSES, CurriculumStore
from agents.document_generation import document_tools
from agents.document_generation.document_store import STATUSES as DOCUMENT_STATUSES, DocumentStore
from agents.document_generation.schemas import EDITABLE_SOURCES, EDITABLE_TYPES
from agents.marketing import composer as marketing_composer, marketing_tools
from agents.marketing.marketing_store import MarketingStore, STATUSES as CAMPAIGN_STATUSES
from agents.rfp_research.application_profile import DOCUMENT_TYPES, PROFILE_FIELDS, ApplicationProfile
from agents.rfp_research.applications import ApplicationWorkflow
from agents.rfp_research.opportunity_store import APPROVAL_STATUSES, OpportunityStore

SESSION_IDLE_SECONDS = 30 * 60
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 60
# Failed attempts are counted per client (one browser session), so one client's mistakes cannot lock the administrators out.
# A client that keeps opening new sessions to get fresh attempts is stopped by a ceiling over all clients together.
MAX_FAILED_ATTEMPTS_ALL_CLIENTS = 25
MAX_TRACKED_CLIENTS = 500
DEFAULT_CLIENT = "default"

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
        document_store: Optional[DocumentStore] = None,
        document_output_dir: Optional[str] = None,
        marketing_store: Optional[MarketingStore] = None,
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
        document_store / document_output_dir: for tests; default to the
        database and PDF folder the Document MCP server uses
        (DOCUMENT_DB_PATH / DOCUMENT_OUTPUT_DIR).
        marketing_store: for tests; defaults to the database the Marketing MCP
        server uses (MARKETING_DB_PATH).
        """
        self._orchestrator_factory = orchestrator_factory
        self._orchestrator = None
        self._opportunity_store = opportunity_store
        self._application_profile = application_profile
        self._curriculum_store = curriculum_store
        self._curriculum_documents_dir = curriculum_documents_dir
        self._curriculum_toolbox: Optional[curriculum_tools.CurriculumToolbox] = None
        self._document_store = document_store
        self._document_output_dir = document_output_dir
        self._document_toolbox: Optional[document_tools.DocumentToolbox] = None
        self._marketing_store = marketing_store
        self._marketing_toolbox: Optional[marketing_tools.MarketingToolbox] = None
        self._application_workflow: Optional[ApplicationWorkflow] = None
        self._passcode = passcode if passcode is not None else get_secret("REX_ADMIN_PASSCODE")
        self._clock = clock
        self._sessions: dict[str, float] = {}
        self._failures: dict[str, int] = {}
        self._locked_until: dict[str, float] = {}
        self._all_failures = 0
        self._all_locked_until = 0.0

    # ---------------------------------------------------------------- auth

    @property
    def enabled(self) -> bool:
        return bool(self._passcode)

    def login(self, candidate: str, client_id: Optional[str] = None) -> Optional[str]:
        """
        A new session token for the right passcode, else None. client_id names the caller (the app passes one id per browser
        session) so that the lockout after repeated wrong passcodes applies to that caller only.
        """
        now = self._clock()
        client = client_id or DEFAULT_CLIENT
        if not self.enabled or self._is_locked(client, now):
            return None

        if not isinstance(candidate, str) or not hmac.compare_digest(
            candidate.encode("utf-8"), self._passcode.encode("utf-8")
        ):
            self._record_failure(client, now)
            return None

        self._failures.pop(client, None)
        token = secrets.token_urlsafe(32)
        self._sessions[token] = now + SESSION_IDLE_SECONDS
        return token

    def is_locked(self, client_id: Optional[str] = None) -> bool:
        return self._is_locked(client_id or DEFAULT_CLIENT, self._clock())

    def _is_locked(self, client: str, now: float) -> bool:
        return now < self._all_locked_until or now < self._locked_until.get(client, 0.0)

    def _record_failure(self, client: str, now: float) -> None:
        self._forget_expired(now)
        self._failures[client] = self._failures.get(client, 0) + 1
        if self._failures[client] >= MAX_FAILED_ATTEMPTS:
            self._locked_until[client] = now + LOCKOUT_SECONDS
            self._failures.pop(client)
        self._all_failures += 1
        if self._all_failures >= MAX_FAILED_ATTEMPTS_ALL_CLIENTS:
            self._all_locked_until = now + LOCKOUT_SECONDS
            self._all_failures = 0

    def _forget_expired(self, now: float) -> None:
        """Keeps the per-client records bounded: expired lockouts go, and the oldest records go past the limit."""
        for client in [c for c, until in self._locked_until.items() if until <= now]:
            del self._locked_until[client]
        for records in (self._failures, self._locked_until):
            while len(records) >= MAX_TRACKED_CLIENTS:
                del records[next(iter(records))]

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


    # ------------------------------------------------------- documents

    def _documents(self) -> document_tools.DocumentToolbox:
        """
        Agent 4's documents, opened in this process for review, editing and
        download. Curricula are read through the same store the review tab
        uses; approval of a curriculum is never touched here.
        """
        if self._document_toolbox is None:
            store = self._document_store or DocumentStore(get_setting(document_tools.DB_SETTING, "") or None)
            self._document_toolbox = document_tools.DocumentToolbox(
                store=store, curricula=self._curricula().store,
                documents_dir=Path(self._document_output_dir) if self._document_output_dir else None)
        return self._document_toolbox

    def list_approved_curricula(self, token: str, limit: int = 200, offset: int = 0) -> dict:
        """
        One page of the curricula that have an approved version, the only ones
        documents can be built from: {"curricula", "total", "has_more", "offset"}.
        A failure is raised (ValueError), never reported as "none approved".
        """
        self._require(token)
        result = self._documents().list_approved_curricula(limit, offset)
        if result.get("status") != "ok":
            raise ValueError(result.get("message") or "The approved curricula could not be listed.")
        return result

    def generate_document(
        self, token: str, curriculum_id: str, document_type: str, language: Optional[str] = None,
        extra: Optional[dict] = None, document_id: Optional[str] = None, context: Optional[dict] = None,
    ) -> dict:
        """
        Ask the Document agent (through the Admin orchestrator) for a document
        from the form's choices. document_id regenerates: a new version of
        that document. The agent writes a draft; nothing is approved here.
        """
        request = {"curriculum_id": curriculum_id, "document_type": document_type, "language": language or None,
                   "extra": dict(extra or {}), "document_id": document_id or None}
        return self.ask(token, f"Generate a {document_type.replace('_', ' ')} for {curriculum_id}",
                        {**(context or {}), "document_request": request})

    def list_documents(self, token: str, status: Optional[str] = None, curriculum_id: Optional[str] = None,
                       limit: int = 100, offset: int = 0) -> list[dict]:
        """One page of documents, newest first (status is the latest version's)."""
        self._require(token)
        if status is not None and status not in DOCUMENT_STATUSES:
            raise ValueError(f"Unknown document status: {status}")
        return self._documents().store.list(status, curriculum_id or None, limit, offset)

    def document_counts(self, token: str) -> dict:
        self._require(token)
        return self._documents().store.counts()

    def get_document(self, token: str, document_id: str, version: Optional[int] = None) -> Optional[dict]:
        """One version with its version list, validation and audit history (None if unknown)."""
        self._require(token)
        store = self._documents().store
        record = store.get(document_id, version)
        if record is None:
            return None
        curriculum = self._curricula().store.get(record["curriculum_id"])
        return {"document": record, "versions": store.versions(document_id),
                "audit": store.audit_trail(document_id),
                # The curriculum has since been revised or is no longer the approved version.
                "curriculum_changed": bool(curriculum and curriculum.get("approved_version") != record["curriculum_version"])}

    def edit_document(self, token: str, document_id: str, edits: list[dict], summary: Optional[str] = None) -> dict:
        """
        Save an edited copy of the current version as a new version.
        edits: [{"section": key, "block": index, "value": text or list of lines}]. Only text the model or an
        administrator wrote can be edited; text taken from the approved curriculum is refused (and would be
        rejected by the document tool's own validation too). The new version needs review again.
        """
        admin = self._require(token)
        tools = self._documents()
        record = tools.store.get(document_id)
        if record is None:
            raise ValueError("Unknown document.")
        content = copy.deepcopy(record["content"])
        sections = {s["key"]: s for s in content["sections"]}
        for edit in edits or []:
            section = sections.get(edit.get("section"))
            blocks = section["blocks"] if section else []
            index = edit.get("block")
            if not isinstance(index, int) or not 0 <= index < len(blocks):
                raise ValueError("That part of the document does not exist.")
            block = blocks[index]
            if block.get("source") not in EDITABLE_SOURCES or block.get("type") not in EDITABLE_TYPES:
                raise ValueError("Text taken from the approved curriculum cannot be edited here: revise and "
                                 "re-approve the curriculum instead.")
            value = edit.get("value")
            if block["type"] == "paragraph":
                block["text"] = " ".join(str(value or "").split())
            else:
                lines = value if isinstance(value, list) else str(value or "").splitlines()
                block["items"] = [re.sub(r"^\s*[-•*]\s*", "", str(line)).strip() for line in lines if str(line).strip()]
            block["source"] = "admin"
        saved = tools.save_admin_edit(content, document_id, admin, summary)
        if saved.get("status") != "ok":
            raise ValueError(saved.get("message") or "The edit could not be saved.")
        rendered = tools.render_document_pdf(document_id, saved["version"])
        if rendered.get("status") != "ok":
            raise ValueError(rendered.get("message") or "The PDF could not be rendered.")
        return saved

    def approve_document(self, token: str, document_id: str, version: int, notes: Optional[str] = None) -> dict:
        """
        Approve the current version (it must be pending review); audited under
        this admin session. The curriculum version it was built from must
        still be an approved one. Its PDF is created first if needed.
        """
        admin = self._require(token)
        tools = self._documents()
        record = tools.store.get(document_id, int(version))
        if record is None:
            raise ValueError("Unknown document or version.")
        if not record["filename"]:   # the review PDF (normally made at generation)
            rendered = tools.render_document_pdf(document_id, int(version))
            if rendered.get("status") != "ok":
                raise ValueError(rendered.get("message") or "The PDF could not be rendered.")
        # Makes the approved PDF first; the approval is recorded only if that worked.
        return tools.approve_document(document_id, int(version), admin, notes)

    def document_pdf(self, token: str, document_id: str, version: Optional[int] = None) -> Optional[dict]:
        """
        {"filename", "data", "approved"} for one version's PDF: the approved PDF
        of an approved version, otherwise the review PDF (rendered first if it
        has none). None if unknown.
        """
        self._require(token)
        tools = self._documents()
        record = tools.store.get(document_id, version)
        if record is None:
            return None
        # "approved" means the file is the approved PDF. A version approved before approved PDFs existed has
        # none: it is served its review PDF (and reported as such) rather than failing.
        approved = record["version_status"] == "approved"
        if not record["filename"] and not (approved and record["approved_filename"]):
            rendered = tools.render_document_pdf(document_id, record["version"])
            if rendered.get("status") != "ok":
                raise ValueError(rendered.get("message") or "The PDF could not be rendered.")
            record = rendered["document"]
        data = tools.document_bytes(document_id, record["version"])
        if data is None:
            raise ValueError("The PDF file is missing from the document folder.")
        has_approved_file = approved and bool(record["approved_filename"])
        return {"filename": record["approved_filename"] if has_approved_file else record["filename"], "data": data,
                "approved": has_approved_file}

    def approved_curriculum_pdf(self, token: str, curriculum_id: str) -> dict:
        """
        {"filename", "data", "pages", "version", "language"}: Agent 3's own PDF of this curriculum's approved version,
        for the Documents tab's "Curriculum PDF" choice. Read-only and not a generated document (no record, no review);
        raises ValueError with the reason if the curriculum has no approved version.
        """
        self._require(token)
        return self._documents().approved_curriculum_pdf(curriculum_id)

    def document_audit_trail(self, token: str, document_id: Optional[str] = None) -> list[dict]:
        self._require(token)
        return self._documents().store.audit_trail(document_id)


    # -------------------------------------------------------- marketing

    def _marketing(self) -> marketing_tools.MarketingToolbox:
        """
        Agent 5's campaigns, opened in this process for review, editing and download. Courses are read through the
        Curriculum Agent's store; approval of a curriculum is never touched here.
        """
        if self._marketing_toolbox is None:
            store = self._marketing_store or MarketingStore(get_setting(marketing_tools.DB_SETTING, "") or None)
            self._marketing_toolbox = marketing_tools.MarketingToolbox(store=store, curricula=self._curricula().store)
        return self._marketing_toolbox

    def list_approved_courses(self, token: str, limit: int = 200, offset: int = 0) -> dict:
        """
        One page of the courses that have an approved version, the only ones campaigns can be written for:
        {"courses", "total", "has_more", "offset"}. A failure is raised (ValueError), never reported as "none approved".
        """
        self._require(token)
        result = self._marketing().list_approved_courses(limit, offset)
        if result.get("status") != "ok":
            raise ValueError(result.get("message") or "The approved courses could not be listed.")
        return result

    def generate_campaign(self, token: str, request: dict, context: Optional[dict] = None) -> dict:
        """
        Ask the Marketing agent (through the Admin orchestrator) for a campaign from the form's choices: {"curriculum_id",
        "audience"?, "tone"?, "goal"?, "notes"?, "channels"?, "variations"?, "campaign_id"? (a new version of that
        campaign)}. The agent writes a pending-review campaign; nothing is approved, published or sent here.
        """
        return self.ask(token, f"Create a marketing campaign for {request.get('curriculum_id')}",
                        {**(context or {}), "marketing_request": dict(request or {})})

    def list_campaigns(self, token: str, status: Optional[str] = None, curriculum_id: Optional[str] = None,
                       limit: int = 100, offset: int = 0) -> list[dict]:
        """One page of campaigns, newest first (status is the latest version's)."""
        self._require(token)
        if status is not None and status not in CAMPAIGN_STATUSES:
            raise ValueError(f"Unknown campaign status: {status}")
        return self._marketing().store.list(status, curriculum_id or None, limit, offset)

    def campaign_counts(self, token: str) -> dict:
        self._require(token)
        return self._marketing().store.counts()

    def get_campaign(self, token: str, campaign_id: str, version: Optional[int] = None) -> Optional[dict]:
        """One version with its version list, validation and audit history (None if unknown)."""
        self._require(token)
        store = self._marketing().store
        record = store.get(campaign_id, version)
        if record is None:
            return None
        curriculum = self._curricula().store.get(record["curriculum_id"])
        return {"campaign": record, "versions": store.versions(campaign_id), "audit": store.audit_trail(campaign_id),
                # The course has since been revised or is no longer the approved version.
                "course_changed": bool(curriculum and curriculum.get("approved_version") != record["curriculum_version"])}

    def edit_campaign(self, token: str, campaign_id: str, edits: list[dict], summary: Optional[str] = None) -> dict:
        """
        Save an edited copy of the current version as a new version. edits: [{"path": [...], "value": text}] where a
        path names a text of the strategy or of one variation (marketing/composer.editable_paths); the course facts
        cannot be edited. The new version is checked again and needs review again; one that fails the checks is kept as
        a draft and cannot be approved.
        """
        admin = self._require(token)
        tools = self._marketing()
        record = tools.store.get(campaign_id)
        if record is None:
            raise ValueError("Unknown campaign.")
        content = copy.deepcopy(record["content"])
        for edit in edits or []:
            marketing_composer.apply_edit(content, list(edit.get("path") or []), edit.get("value"))
        saved = tools.save_admin_edit(content, campaign_id, admin, summary)
        if saved.get("status") != "ok":
            raise ValueError(saved.get("message") or "The edit could not be saved.")
        return saved

    def approve_campaign(self, token: str, campaign_id: str, version: int, notes: Optional[str] = None) -> dict:
        """
        Approve the current version (it must be pending review); audited under this admin session. The course version
        it was written from must still be an approved one. Approval makes the content downloadable; it does not
        publish or send anything.
        """
        admin = self._require(token)
        try:
            return self._marketing().approve_campaign(campaign_id, int(version), admin, notes)
        except (ValueError, LookupError) as exc:
            raise ValueError(str(exc)) from exc

    def campaign_export(self, token: str, campaign_id: str, fmt: str = "markdown") -> dict:
        """{"filename", "data", "mime", "version"} of the APPROVED version (markdown or json); ValueError if none is approved."""
        self._require(token)
        return self._marketing().approved_export(campaign_id, fmt)

    def campaign_audit_trail(self, token: str, campaign_id: Optional[str] = None) -> list[dict]:
        self._require(token)
        return self._marketing().store.audit_trail(campaign_id)


class _NoCatalogue:
    """The review side never reads the academy catalogue."""

    def __getattr__(self, name):
        raise RuntimeError("The academy catalogue is not available in the admin review service.")
