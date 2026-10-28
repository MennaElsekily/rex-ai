"""
The application workflow: what happens after an admin approves a tender.

    AdminService.decide(... "approved" ...)       (admin session checked)
      -> ApplicationWorkflow.start(tender, admin)
         1. detect_route(): how this tender is applied for (source-specific)
         2. OpportunityStore.claim_submission(): atomic; nothing runs twice
         3. the route's adapter checks the prerequisites and prepares it
         4. OpportunityStore.finish_attempt(): result + history

What can be automated today (see the RFP section of
docs/PROJECT_TECHNICAL_GUIDE.md for the evidence):

- Etimad (Saudi Arabia), etenders.gov.eg (Egypt), tender listing sites
  and other web pages: no submission API; Etimad needs Nafath (personal
  login) and the authorised signer. ManualRouteAdapter records the
  official steps and ends as "requires_manual_action".
- World Bank notices that request expressions of interest, with a
  contact email from the official API: EmailExpressionOfInterestAdapter
  prepares the email and its attachments from the admin's application
  profile. It runs in DRY-RUN mode only: nothing is ever sent, and the
  result is "requires_manual_action: review and send it yourself". This
  module contains no email-sending code.

Nothing is invented: the email uses only the profile's saved fields and
uploaded documents and the tender's own fields; anything missing is
listed. Web-page text never chooses the route or the recipient: the
email route needs a record verified from the World Bank API.

An application becomes "submitted"/"applied" only by a confirmed real
submission (impossible in dry-run) or by the admin recording a verified
manual submission (OpportunityStore.record_manual_submission).
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

from agents.config import get_setting
from agents.rfp_research.api_clients.world_bank import SOURCE_NAME as WORLD_BANK_SOURCE
from agents.rfp_research.application_profile import DOCUMENT_TYPES, PROFILE_FIELDS, ApplicationProfile
from agents.rfp_research.opportunity_store import (
    REQUIRES_MANUAL_ACTION,
    SUBMISSION_FAILED,
    OpportunityStore,
)
from agents.rfp_research.web_tenders import TENDER_AGGREGATORS

logger = logging.getLogger(__name__)

# Only "dry_run" exists. Any other value still means dry run: live
# sending needs a separate, explicitly approved implementation.
SUBMISSION_MODE_SETTING = "REX_SUBMISSION_MODE"

EMAIL_EOI = "email_eoi"
MANUAL = "manual"

# What an emailed expression of interest needs from the profile.
EOI_REQUIRED_FIELDS = ("legal_name", "contact_person", "contact_email")
EOI_REQUIRED_DOCUMENTS = ("company_profile",)
EOI_RECOMMENDED_DOCUMENTS = ("experience_references",)

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass
class SubmissionRoute:
    """How one tender is applied for."""

    adapter: str                     # EMAIL_EOI or MANUAL
    label: str                       # shown to the admin
    official_url: Optional[str]
    steps: list[str]                 # what has to happen, in order
    recipient: Optional[str] = None  # EMAIL_EOI only, from the official API
    automatic: bool = False          # can REX AI prepare it itself?

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttemptResult:
    status: str
    manual_actions: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    confirmation: dict[str, Any] = field(default_factory=dict)
    submission_url: Optional[str] = None
    error: Optional[str] = None


def _domain(tender: dict[str, Any]) -> str:
    domain = tender.get("source_domain") or urlparse(
        str(tender.get("source_url") or tender.get("url") or "")).netloc
    return str(domain).lower().removeprefix("www.")


def _is_world_bank_eoi(tender: dict[str, Any]) -> bool:
    return (tender.get("source_name") == WORLD_BANK_SOURCE
            and tender.get("verification_status") == "verified_from_api"
            and "expression of interest" in str(tender.get("category") or "").casefold())


def detect_route(tender: dict[str, Any]) -> SubmissionRoute:
    """The submission route for a tender, from its source (code rules only)."""
    domain = _domain(tender)
    link = tender.get("submission_url") or tender.get("source_url") or tender.get("url")
    deadline = f" before {tender['deadline']}" if tender.get("deadline") else ""

    if tender.get("source_name") == WORLD_BANK_SOURCE:
        email = str(tender.get("contact_email") or "").strip()
        if _is_world_bank_eoi(tender) and _EMAIL.match(email):
            return SubmissionRoute(
                EMAIL_EOI, "World Bank notice: expression of interest by email to the contact "
                "address from the official API", link,
                [f"Send the expression of interest to {email}{deadline}, as the notice instructs.",
                 "Keep the sent message as proof of submission."],
                recipient=email, automatic=True)
        instructions = tender.get("submission_instructions") or "the contact named in the notice"
        return SubmissionRoute(
            MANUAL, "World Bank notice: submit to the borrower's implementing agency", link,
            [f"Read the notice and submit as it instructs{deadline}: {instructions}.",
             "Bids usually follow the borrower's own process (often signed documents or a bid security)."])

    if domain.endswith("etimad.sa") or (tender.get("country") == "Saudi Arabia" and domain.endswith(".sa")):
        return SubmissionRoute(
            MANUAL, "Etimad portal (Saudi Arabia): manual only", link or "https://tenders.etimad.sa",
            ["Log in to Etimad with Nafath (the authorised representative's national ID or iqama).",
             "Make sure the academy is registered as a supplier (CR, VAT, GOSI, Zakat certificates).",
             "Buy the tender documents if required, then use 'Submit Offer' and attach the files.",
             "The authorised signer signs and submits the offer electronically."])

    if domain.endswith("etenders.gov.eg") or domain.endswith(".gov.eg"):
        return SubmissionRoute(
            MANUAL, "Egyptian government tender: manual only", link or "https://etenders.gov.eg",
            ["Make sure the academy is a registered supplier (tax ID and commercial registration).",
             f"Get the tender documents and submit as the notice instructs{deadline}.",
             "Many tenders also need sealed technical/financial envelopes and a bid bond."])

    if any(domain.endswith(site) for site in TENDER_AGGREGATORS):
        return SubmissionRoute(
            MANUAL, f"Tender listing site ({domain}): find the original notice", link,
            [f"This site only republishes tenders. Find the original notice from "
             f"{tender.get('organization') or 'the issuing organisation'}.",
             "Submit through the route the original notice gives."])

    return SubmissionRoute(
        MANUAL, "Web page: manual", link,
        ["Read the source page and follow its submission instructions."])


class ManualRouteAdapter:
    """Every source without a usable submission mechanism."""

    name = MANUAL
    dry_run = False   # it never sends anything either way

    def run(self, tender, route, profile, authorized) -> AttemptResult:
        return AttemptResult(
            REQUIRES_MANUAL_ACTION,
            manual_actions=[*route.steps,
                            "Then record the manual submission here, with the date and reference."],
            submission_url=route.official_url,
        )


class EmailExpressionOfInterestAdapter:
    """
    Prepares the expression-of-interest email for a World Bank notice.
    Dry run only: it builds the message and lists the attachments; this
    class has no way to send anything.
    """

    name = EMAIL_EOI
    dry_run = True

    def run(self, tender, route, profile: ApplicationProfile, authorized: bool) -> AttemptResult:
        fields = profile.fields()
        documents = profile.documents()
        present = {d["doc_type"] for d in documents}

        missing = [PROFILE_FIELDS[f] for f in EOI_REQUIRED_FIELDS if not fields.get(f)]
        missing += [f"Document: {DOCUMENT_TYPES[d]}" for d in EOI_REQUIRED_DOCUMENTS if d not in present]
        if not authorized:
            missing.append("Your authorisation to submit on the academy's behalf "
                           "(the checkbox when approving)")
        if missing:
            return AttemptResult(
                REQUIRES_MANUAL_ACTION, missing=missing,
                manual_actions=["Complete the application profile (and authorise submission), "
                                "then prepare again.", *route.steps],
                submission_url=route.official_url)

        email = _prepare_email(tender, route, fields, documents)
        recommended = [DOCUMENT_TYPES[d] for d in EOI_RECOMMENDED_DOCUMENTS if d not in present]
        return AttemptResult(
            REQUIRES_MANUAL_ACTION,
            confirmation={"dry_run": True, "prepared_email": email,
                          "recommended_documents_missing": recommended},
            manual_actions=[f"Dry run: nothing was sent. Review the prepared email, then send it "
                            f"yourself from {fields['contact_email']} to {route.recipient}"
                            f"{' before ' + tender['deadline'] if tender.get('deadline') else ''}.",
                            "Then record the manual submission here, with the date and reference."],
            submission_url=f"mailto:{route.recipient}")


def _prepare_email(tender, route, fields, documents) -> dict[str, Any]:
    """The email, built only from saved profile fields, documents and tender fields."""
    reference = tender.get("reference_number")
    title = tender.get("title") or "the notice"
    subject = f"Expression of Interest: {reference + ' - ' if reference else ''}{title}"[:200]

    identity = [fields["legal_name"]]
    for key, label in (("registration_number", "Commercial registration"), ("address", "Address"),
                       ("website", "Website")):
        if fields.get(key):
            identity.append(f"{label}: {fields[key]}")
    contact = ", ".join(filter(None, [fields["contact_person"], fields.get("contact_title")]))
    signature = "\n".join(filter(None, [contact, fields["legal_name"], fields["contact_email"],
                                        fields.get("contact_phone")]))
    attachments = [{"document_id": d["document_id"], "type": DOCUMENT_TYPES[d["doc_type"]],
                    "file_name": d["original_name"], "size_bytes": d["size_bytes"]} for d in documents]

    body = "\n\n".join([
        "Dear Sir or Madam,",
        f"{fields['legal_name']} wishes to express its interest in \"{title}\""
        f"{' (reference ' + reference + ')' if reference else ''}"
        f"{', issued by ' + tender['organization'] if tender.get('organization') else ''}.",
        "Our details:\n" + "\n".join(identity),
        "Please find attached: " + "; ".join(f"{a['type']} ({a['file_name']})" for a in attachments) + ".",
        "Kind regards,\n" + signature,
    ])
    return {"to": route.recipient, "from": fields["contact_email"], "subject": subject,
            "body": body, "attachments": attachments}


class ApplicationWorkflow:
    """Runs the route's adapter for one approved tender, at most once at a time."""

    def __init__(self, store: OpportunityStore, profile: ApplicationProfile) -> None:
        self.store = store
        self.profile = profile
        mode = get_setting(SUBMISSION_MODE_SETTING, "dry_run")
        if mode != "dry_run":
            logger.warning("%s=%s ignored: only dry-run preparation exists.", SUBMISSION_MODE_SETTING, mode)
        self.adapters = {MANUAL: ManualRouteAdapter(), EMAIL_EOI: EmailExpressionOfInterestAdapter()}
        # An attempt left in progress by a stopped app may have sent
        # something: it becomes "outcome_unknown" (never "failed").
        self.store.recover_interrupted_attempts()

    def route_for(self, opp_id: str) -> dict[str, Any]:
        tender = self.store.get(opp_id)
        if tender is None:
            raise KeyError(f"Unknown opportunity: {opp_id}")
        return detect_route(tender).to_dict()

    def start(self, opp_id: str, triggered_by: str) -> dict[str, Any]:
        """
        Run the workflow for an approved tender. Returns the finished
        attempt, or {"started": False, ...} when the tender is not
        approved, another run owns it, it was already submitted, or its
        last result is unknown.
        """
        tender = self.store.get(opp_id)
        if tender is None:
            raise KeyError(f"Unknown opportunity: {opp_id}")
        route = detect_route(tender)
        adapter = self.adapters[route.adapter]

        attempt = self.store.claim_submission(opp_id, adapter.name, triggered_by, dry_run=adapter.dry_run)
        if attempt is None:
            return {"started": False, "submission_status": tender["submission_status"],
                    "route": route.to_dict()}

        try:
            result = adapter.run(tender, route, self.profile, tender["submission_authorized"])
        except Exception as exc:
            # These adapters never send anything, so a crash means "not
            # submitted", and a new run is safe.
            logger.exception("Application preparation failed")
            result = AttemptResult(SUBMISSION_FAILED, error=f"Preparation failed ({type(exc).__name__}).",
                                   submission_url=route.official_url)

        finished = self.store.finish_attempt(
            attempt["attempt_id"], result.status, submission_url=result.submission_url,
            confirmation=result.confirmation, missing=result.missing,
            manual_actions=result.manual_actions, error=result.error)
        return {"started": True, **finished, "route": route.to_dict()}
