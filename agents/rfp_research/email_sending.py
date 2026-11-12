"""
Really sending the expression-of-interest email that
agents/rfp_research/applications.py prepares (dry run only, never
sent -- see that module's docstring and its own tests, which assert it
contains no sending code).

This module is the one deliberate exception: an admin who reviews the
dry-run draft (optionally edits it -- see save_draft/effective_email)
can choose to actually send it, through Resend, with safety enforced
here (not only in the UI, and not only in test mode):

- REX_EMAIL_MODE defaults to "test". Anything other than exactly
  "production" is treated as test mode, the same convention
  applications.py uses for REX_SUBMISSION_MODE.
- In BOTH modes the message goes only to the address configured in
  REX_EMAIL_TEST_RECIPIENT (.env) -- never hardcoded here, never typed by
  the admin, and never the tender's own contact (there is no fallback to
  it), so a tender contact can never receive mail from this system.
  Sending is refused, with a clear error, if that setting is missing or
  not a valid email address. The tender's contact is kept in the audit
  record (intended_recipient) for reference only.
- Production mode additionally refuses to send when the draft was built
  from a demonstration profile (see application_profile.is_demo_profile).
- Sending needs the admin's authorisation to submit on the academy's
  behalf (the same safeguard the application workflow uses), a sender
  address, and the admin's explicit confirmation in the UI.
- At most one real send per application attempt
  (OpportunityStore.record_email_send's UNIQUE constraint): the second
  click on an already-sent attempt is refused, not repeated. The draft
  can be freely edited and re-saved any number of times before that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from agents.config import get_setting
from agents.rfp_research import email_provider
from agents.rfp_research.opportunity_store import InvalidTransition, OpportunityStore

EMAIL_MODE_SETTING = "REX_EMAIL_MODE"
TEST_RECIPIENT_SETTING = "REX_EMAIL_TEST_RECIPIENT"
# Resend only sends from a domain verified in that Resend account; the
# academy's own contact address (the dry-run draft's "from") almost
# certainly is not one. When configured, mail is sent from this address
# instead, with the academy's address set as Reply-To so a response
# still reaches them.
FROM_OVERRIDE_SETTING = "RESEND_FROM_EMAIL"
PRODUCTION = "production"

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class EmailSendError(ValueError):
    """The prepared email for this tender cannot be sent right now."""


@dataclass
class SentEmail:
    email_id: int
    mode: str
    to: str
    intended_recipient: str
    redirected: bool
    provider: str
    provider_reference: Optional[str]


def email_mode() -> str:
    return (get_setting(EMAIL_MODE_SETTING, "test") or "test").strip().lower()


def test_recipient() -> str:
    """The configured recipient (REX_EMAIL_TEST_RECIPIENT), or "" if not configured. Never a hardcoded address."""
    return (get_setting(TEST_RECIPIENT_SETTING, "") or "").strip()


def configured_recipient() -> str:
    """The only address any email is sent to, in test and production mode alike ("" if not configured)."""
    return test_recipient()


def sender_override() -> str:
    return (get_setting(FROM_OVERRIDE_SETTING, "") or "").strip()


class EmailSendingService:
    """Sends the dry-run draft an application attempt already prepared (or the admin's saved edit of it)."""

    def __init__(self, store: OpportunityStore, send_fn=None) -> None:
        self.store = store
        # send_fn: for tests, so no real network call is ever made while testing this
        # service itself; defaults to the real Resend client.
        self._send = send_fn

    def _send_via_provider(self, *args, **kwargs) -> str:
        if self._send is not None:
            return self._send(*args, **kwargs)
        return email_provider.send_via_resend(*args, **kwargs)

    def _latest_attempt(self, opp_id: str) -> Optional[dict[str, Any]]:
        attempts = self.store.attempts(opp_id)
        return attempts[-1] if attempts else None

    def effective_email(self, opp_id: str, attempt: dict[str, Any]) -> Optional[dict[str, Any]]:
        """The prepared draft, with the admin's saved edit (if any) applied on top."""
        original = (attempt["confirmation"] or {}).get("prepared_email")
        if not original:
            return None
        draft = self.store.get_email_draft(opp_id, attempt["attempt_no"])
        if draft:
            return {**original, "subject": draft["subject"], "body": draft["body"], "edited": True,
                    "draft_updated_at": draft["updated_at"], "draft_updated_by": draft["updated_by"]}
        return {**original, "edited": False}

    def current_draft(self, opp_id: str) -> Optional[dict[str, Any]]:
        """The draft the admin would see on the review screen for this tender's latest attempt."""
        latest = self._latest_attempt(opp_id)
        if latest is None:
            return None
        email = self.effective_email(opp_id, latest)
        if email is None:
            return None
        return {**email, "attempt_no": latest["attempt_no"],
                "already_sent": bool(self.store.email_send_for_attempt(opp_id, latest["attempt_no"]))}

    def save_draft(self, opp_id: str, subject: str, body: str, admin: str) -> dict[str, Any]:
        latest = self._latest_attempt(opp_id)
        original = self.effective_email(opp_id, latest) if latest else None
        if not original:
            raise EmailSendError("No prepared email exists for this tender yet. Prepare the application first.")
        if self.store.email_send_for_attempt(opp_id, latest["attempt_no"]):
            raise EmailSendError("This application attempt's email has already been sent; it can no longer be edited.")
        subject, body = (subject or "").strip(), (body or "").strip()
        if not subject or not body:
            raise EmailSendError("The subject and body cannot be empty.")
        return self.store.save_email_draft(opp_id, latest["attempt_no"], subject, body, admin)

    def discard_draft(self, opp_id: str) -> bool:
        """Reverts the review screen to the originally generated text."""
        latest = self._latest_attempt(opp_id)
        if latest is None:
            return False
        return self.store.delete_email_draft(opp_id, latest["attempt_no"])

    def send_prepared_email(self, opp_id: str, admin: str) -> SentEmail:
        tender = self.store.get(opp_id)
        if tender is None:
            raise KeyError(f"Unknown opportunity: {opp_id}")

        latest = self._latest_attempt(opp_id)
        email = self.effective_email(opp_id, latest) if latest else None
        if not email:
            raise EmailSendError(
                "No prepared email exists for this tender yet. Prepare the application first.")

        if self.store.email_send_for_attempt(opp_id, latest["attempt_no"]):
            raise EmailSendError("An email has already been sent for this application attempt.")
        if not tender.get("submission_authorized"):
            raise EmailSendError("Authorise submission on the academy's behalf first (\"Authorise now\"), "
                                 "then send.")

        mode = email_mode()
        real_to = email["to"]
        is_demo = bool(email.get("is_demo_profile"))

        if mode == PRODUCTION and is_demo:
            raise EmailSendError(
                "Refused: the organisation profile used for this draft contains demo data; "
                "it cannot be used to send in production mode.")
        # The configured recipient in every mode: never the tender's contact, with no fallback to it.
        to = configured_recipient()
        if not to:
            raise EmailSendError(
                f"{TEST_RECIPIENT_SETTING} is not configured. Set it in .env to the address every "
                "application email must go to; nothing is sent until it is.")
        if not _EMAIL.match(to):
            raise EmailSendError(f"{TEST_RECIPIENT_SETTING} is not a valid email address. Check .env.")

        attachment_names = [a["file_name"] for a in email.get("attachments", [])]

        draft_from = email["from"]
        override = sender_override()
        from_addr = override or draft_from
        if not from_addr:
            raise EmailSendError("No sender address: add the contact email to the Application profile "
                                 f"(or set {FROM_OVERRIDE_SETTING} in .env), prepare again, then send.")
        send_kwargs = {"reply_to": draft_from} if override and override != draft_from else {}

        try:
            reference = self._send_via_provider(to, from_addr, email["subject"], email["body"],
                                                attachment_names, **send_kwargs)
        except email_provider.EmailProviderError as exc:
            try:
                self.store.record_email_send(
                    opp_id, latest["attempt_no"], recipient=to, intended_recipient=real_to,
                    sender=from_addr, subject=email["subject"], body=email["body"], mode=mode,
                    status="failed", created_by=admin, provider="resend", error=str(exc),
                    is_demo_profile=is_demo)
            except InvalidTransition:
                pass   # already recorded by an earlier attempt; the failure is reported either way
            raise EmailSendError(str(exc)) from exc

        record = self.store.record_email_send(
            opp_id, latest["attempt_no"], recipient=to, intended_recipient=real_to,
            sender=from_addr, subject=email["subject"], body=email["body"], mode=mode,
            status="sent", created_by=admin, provider="resend", provider_reference=reference,
            is_demo_profile=is_demo)
        return SentEmail(email_id=record["email_id"], mode=mode, to=to, intended_recipient=real_to,
                         redirected=to != real_to, provider="resend", provider_reference=reference)
