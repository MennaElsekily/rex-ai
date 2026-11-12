"""
A thin Resend client: sends exactly one email. No SDK dependency -- a
single HTTPS POST (the project already depends on httpx).

This is the only place in the project that ever puts an email on the
wire. agents/rfp_research/applications.py never imports this module;
agents/rfp_research/email_sending.py is the sole caller, and it alone
decides, with the admin's explicit click, whether this is called at
all.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from agents.config import get_secret

RESEND_API_KEY_SETTING = "RESEND_API_KEY"
RESEND_BASE_URL = "https://api.resend.com"


class EmailProviderError(Exception):
    """The email could not be handed to the provider."""


def send_via_resend(
    to: str, from_addr: str, subject: str, body: str,
    attachment_names: Optional[list[str]] = None, api_key: Optional[str] = None,
    reply_to: Optional[str] = None,
) -> str:
    """
    Sends one plain-text email through Resend. Returns the provider's
    message id. attachment_names are recorded for traceability only
    (the application profile's documents are not re-read here); Resend
    needs base64 content to actually attach a file, which this project
    does not do yet. reply_to: set when from_addr had to be a verified
    sending address different from the academy's own contact address.
    """
    key = api_key if api_key is not None else get_secret(RESEND_API_KEY_SETTING)
    if not key:
        raise EmailProviderError(f"{RESEND_API_KEY_SETTING} is not configured.")

    payload: dict[str, Any] = {"from": from_addr, "to": [to], "subject": subject, "text": body}
    if reply_to:
        payload["reply_to"] = reply_to
    if attachment_names:
        payload["text"] += "\n\n(Attachments prepared but not included in this test send: " + \
            ", ".join(attachment_names) + ")"

    try:
        response = httpx.post(
            f"{RESEND_BASE_URL}/emails", json=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=15,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise EmailProviderError(f"Resend request failed: {exc}") from exc

    data = response.json()
    message_id = data.get("id")
    if not message_id:
        raise EmailProviderError(f"Resend did not return a message id: {data}")
    return message_id
