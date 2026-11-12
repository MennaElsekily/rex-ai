"""
Really sending the expression-of-interest email
agents/rfp_research/applications.py only ever drafts (dry run, never
sent -- see tests/test_application_workflow.py). This is the one
admin action that can really put a message on the wire, so the tests
here cover its safety directly: test mode is the default and always
redirects to the configured test recipient, production mode uses only
the API-verified address the dry-run draft already carries (never an
admin-typed one), and at most one real send exists per application
attempt. The fake send_fn below never touches the network; a direct
call to email_provider.send_via_resend is exercised once, with
httpx.post monkeypatched to refuse any real request.
"""

import httpx
import pytest

from agents.admin_service import AdminAuthError, AdminService
from agents.rfp_research import email_provider, email_sending
from agents.rfp_research.application_profile import ApplicationProfile
from agents.rfp_research.email_sending import EmailSendError, EmailSendingService
from agents.rfp_research.opportunity_store import OpportunityStore
from tests.test_application_workflow import ETIMAD, WB_EOI, add, complete_profile

PASSCODE = "correct horse"
# The feature no longer hardcodes a recipient (see email_sending.py); tests
# that need one configured set this explicitly via the fixture below.
TEST_RECIPIENT = "qa-inbox@example.com"


def never_call_the_network(*args, **kwargs):
    raise AssertionError("A test must never make a real HTTP request.")


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    monkeypatch.setattr(httpx, "post", never_call_the_network)


@pytest.fixture(autouse=True)
def isolated_email_settings(monkeypatch):
    """
    agents.config.get_secret/get_setting load the project's real .env the
    first time either is called in this process (agents/config.py:
    load_env(), guarded by a module-level flag so it only ever runs
    once). If that first call happens to land inside a test that assumes
    a setting is absent (e.g. via monkeypatch.delenv before anything else
    touched config), dotenv still populates os.environ from the real
    file right after, since delenv cannot stop a load that has not
    happened yet -- the real RESEND_API_KEY and friends then leak into
    the test. Force the real .env to load once now (its values stay in
    os.environ, but every value this feature reads is then explicitly
    blanked below, so no test ever depends on a lucky call order to stay
    isolated from the developer's real credentials).
    """
    from agents import config
    config.load_env()
    for name in ("RESEND_API_KEY", "REX_EMAIL_MODE", "REX_EMAIL_TEST_RECIPIENT", "RESEND_FROM_EMAIL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def store(tmp_path):
    s = OpportunityStore(db_path=tmp_path / "tenders.sqlite3")
    yield s
    s.close()


@pytest.fixture
def profile(tmp_path):
    p = ApplicationProfile(db_path=tmp_path / "tenders.sqlite3", documents_dir=tmp_path / "docs")
    yield p
    p.close()


@pytest.fixture
def admin(store, profile):
    return AdminService(orchestrator_factory=lambda: None, opportunity_store=store,
                        application_profile=profile, passcode=PASSCODE)


@pytest.fixture
def token(admin):
    return admin.login(PASSCODE)


@pytest.fixture
def configured_recipient(monkeypatch):
    """REX_EMAIL_TEST_RECIPIENT set to a fixed, clearly-fake address for tests that send in test mode."""
    monkeypatch.setenv("REX_EMAIL_TEST_RECIPIENT", TEST_RECIPIENT)
    return TEST_RECIPIENT


class FakeProvider:
    """Records every call instead of sending anything."""

    def __init__(self, reference="msg_123", error=None):
        self.calls = []
        self.reference = reference
        self.error = error

    def __call__(self, to, from_addr, subject, body, attachment_names=None, reply_to=None):
        self.calls.append({"to": to, "from": from_addr, "subject": subject, "body": body,
                           "attachment_names": attachment_names, "reply_to": reply_to})
        if self.error:
            raise self.error
        return self.reference


def prepared(admin, token, store, profile):
    """An approved, authorised World Bank EOI tender with a complete profile: a prepared email exists."""
    complete_profile(profile)
    opp_id = add(store, WB_EOI)
    admin.decide(token, opp_id, "approved", authorize_submission=True)
    return opp_id


# --------------------------------------------------------------------
# Test mode is the default and always redirects
# --------------------------------------------------------------------

def test_test_mode_redirects_to_the_configured_recipient_and_never_calls_the_real_contact(
    admin, token, store, profile, configured_recipient,
):
    opp_id = prepared(admin, token, store, profile)
    fake = FakeProvider()
    service = EmailSendingService(store, send_fn=fake)

    sent = service.send_prepared_email(opp_id, "admin-session-x")

    assert sent.mode == "test" and sent.to == configured_recipient and sent.redirected is True
    assert sent.intended_recipient == "eoi@moe.gov.eg"
    assert fake.calls == [{"to": configured_recipient, "from": "tenders@andalusia.example",
                           "subject": "Expression of Interest: EG-MOE-123-CS - "
                                      "Consultancy for Teacher Training Programme",
                           "body": fake.calls[0]["body"], "attachment_names": ["Company Profile.pdf"],
                           "reply_to": None}]
    record = store.email_send_for_attempt(opp_id, 1)
    assert (record["recipient"], record["intended_recipient"], record["mode"], record["status"]) == (
        configured_recipient, "eoi@moe.gov.eg", "test", "sent")


def test_an_unset_or_unknown_mode_is_still_test_mode(monkeypatch, admin, token, store, profile, configured_recipient):
    monkeypatch.delenv("REX_EMAIL_MODE", raising=False)
    assert email_sending.email_mode() == "test"
    monkeypatch.setenv("REX_EMAIL_MODE", "whatever")
    assert email_sending.email_mode() == "whatever"   # read back as typed

    opp_id = prepared(admin, token, store, profile)
    fake = FakeProvider()
    sent = EmailSendingService(store, send_fn=fake).send_prepared_email(opp_id, "admin-session-x")

    assert sent.to == configured_recipient   # anything other than exactly "production" stays test mode


# --------------------------------------------------------------------
# A missing or invalid recipient refuses the send with a clear error
# --------------------------------------------------------------------

def test_sending_in_test_mode_without_a_configured_recipient_is_refused(admin, token, store, profile):
    opp_id = prepared(admin, token, store, profile)   # no configured_recipient fixture: unset

    with pytest.raises(EmailSendError, match="REX_EMAIL_TEST_RECIPIENT is not configured"):
        EmailSendingService(store, send_fn=FakeProvider()).send_prepared_email(opp_id, "admin-session-x")


def test_an_invalid_configured_recipient_is_refused(monkeypatch, admin, token, store, profile):
    monkeypatch.setenv("REX_EMAIL_TEST_RECIPIENT", "not-an-email-address")
    opp_id = prepared(admin, token, store, profile)

    with pytest.raises(EmailSendError, match="not a valid email address"):
        EmailSendingService(store, send_fn=FakeProvider()).send_prepared_email(opp_id, "admin-session-x")


# --------------------------------------------------------------------
# A demo profile can never send in production mode
# --------------------------------------------------------------------

def test_a_demo_profile_cannot_send_in_production_mode(monkeypatch, admin, token, store, profile):
    monkeypatch.setenv("REX_EMAIL_MODE", "production")
    profile.update_fields({"legal_name": "Demo Academy", "contact_person": "Demo Contact",
                           "contact_email": "demo@example.com"}, "admin-session-x")
    profile.add_document("company_profile", "Company Profile.pdf", b"%PDF-1.7\n%", "admin-session-x")
    opp_id = add(store, WB_EOI)
    admin.decide(token, opp_id, "approved", authorize_submission=True)

    with pytest.raises(EmailSendError, match="contains demo data"):
        EmailSendingService(store, send_fn=FakeProvider()).send_prepared_email(opp_id, "admin-session-x")


# --------------------------------------------------------------------
# Production mode: only the API-verified recipient, never admin-typed
# --------------------------------------------------------------------

def test_production_mode_also_sends_only_to_the_configured_recipient(
    monkeypatch, admin, token, store, profile,
):
    # Changed on the owner's instruction (2026-11-11): the configured recipient is used in BOTH
    # modes; production mode no longer sends to the tender's own verified contact.
    monkeypatch.setenv("REX_EMAIL_MODE", "production")
    monkeypatch.setenv("REX_EMAIL_TEST_RECIPIENT", "configured@andalusia.academy")
    opp_id = prepared(admin, token, store, profile)
    fake = FakeProvider()

    sent = EmailSendingService(store, send_fn=fake).send_prepared_email(opp_id, "admin-session-x")

    assert sent.mode == "production" and sent.to == "configured@andalusia.academy"
    assert fake.calls[0]["to"] == "configured@andalusia.academy"
    assert sent.intended_recipient == "eoi@moe.gov.eg" and sent.redirected is True   # kept for the audit only
    record = store.email_sends(opp_id)[0]
    assert record["recipient"] == "configured@andalusia.academy" and record["mode"] == "production"


def test_production_mode_never_falls_back_to_the_tender_contact(monkeypatch, admin, token, store, profile):
    monkeypatch.setenv("REX_EMAIL_MODE", "production")
    monkeypatch.delenv("REX_EMAIL_TEST_RECIPIENT", raising=False)
    opp_id = prepared(admin, token, store, profile)        # the tender HAS a verified contact
    fake = FakeProvider()

    with pytest.raises(EmailSendError, match="REX_EMAIL_TEST_RECIPIENT is not configured"):
        EmailSendingService(store, send_fn=fake).send_prepared_email(opp_id, "admin-session-x")
    assert fake.calls == []


def test_a_custom_test_recipient_can_be_configured(monkeypatch, admin, token, store, profile):
    monkeypatch.setenv("REX_EMAIL_TEST_RECIPIENT", "someone-else@example.com")
    opp_id = prepared(admin, token, store, profile)
    fake = FakeProvider()

    sent = EmailSendingService(store, send_fn=fake).send_prepared_email(opp_id, "admin-session-x")

    assert sent.to == "someone-else@example.com"


def test_a_verified_sender_override_is_used_with_the_academys_address_as_reply_to(
    monkeypatch, admin, token, store, profile, configured_recipient,
):
    monkeypatch.setenv("RESEND_FROM_EMAIL", "academy@verified-domain.example")
    opp_id = prepared(admin, token, store, profile)
    fake = FakeProvider()

    EmailSendingService(store, send_fn=fake).send_prepared_email(opp_id, "admin-session-x")

    assert fake.calls[0]["from"] == "academy@verified-domain.example"
    assert fake.calls[0]["reply_to"] == "tenders@andalusia.example"   # the academy's own contact, preserved
    record = store.email_send_for_attempt(opp_id, 1)
    assert record["sender"] == "academy@verified-domain.example"


def test_without_a_sender_override_the_draft_from_is_used_with_no_reply_to(
    admin, token, store, profile, configured_recipient,
):
    opp_id = prepared(admin, token, store, profile)
    fake = FakeProvider()

    EmailSendingService(store, send_fn=fake).send_prepared_email(opp_id, "admin-session-x")

    assert fake.calls[0]["from"] == "tenders@andalusia.example" and fake.calls[0]["reply_to"] is None


# --------------------------------------------------------------------
# Nothing to send / at most one real send per attempt
# --------------------------------------------------------------------

def test_sending_without_a_prepared_email_is_refused(admin, token, store):
    opp_id = add(store, ETIMAD)   # not approved: no application attempt, so no email was prepared

    with pytest.raises(EmailSendError, match="No prepared email"):
        EmailSendingService(store, send_fn=FakeProvider()).send_prepared_email(opp_id, "admin-session-x")


def test_sending_twice_for_the_same_attempt_is_refused(admin, token, store, profile, configured_recipient):
    opp_id = prepared(admin, token, store, profile)
    fake = FakeProvider()
    service = EmailSendingService(store, send_fn=fake)
    service.send_prepared_email(opp_id, "admin-session-x")

    with pytest.raises(EmailSendError, match="already been sent"):
        service.send_prepared_email(opp_id, "admin-session-x")

    assert len(fake.calls) == 1   # the provider was never called the second time


def test_a_failed_send_is_recorded_and_also_blocks_a_second_try(admin, token, store, profile, configured_recipient):
    opp_id = prepared(admin, token, store, profile)
    fake = FakeProvider(error=email_provider.EmailProviderError("Resend request failed: 500"))
    service = EmailSendingService(store, send_fn=fake)

    with pytest.raises(EmailSendError, match="Resend request failed"):
        service.send_prepared_email(opp_id, "admin-session-x")

    record = store.email_send_for_attempt(opp_id, 1)
    assert record["status"] == "failed" and "Resend request failed" in record["error"]
    with pytest.raises(EmailSendError, match="already been sent"):
        service.send_prepared_email(opp_id, "admin-session-x")


def test_retrying_the_whole_application_allows_a_fresh_send(admin, token, store, profile, configured_recipient):
    opp_id = prepared(admin, token, store, profile)
    EmailSendingService(store, send_fn=FakeProvider()).send_prepared_email(opp_id, "admin-session-x")

    admin.retry_submission(token, opp_id)   # a new attempt_no: a fresh prepared email

    fake = FakeProvider()
    sent = EmailSendingService(store, send_fn=fake).send_prepared_email(opp_id, "admin-session-x")
    assert sent.email_id and len(fake.calls) == 1


# --------------------------------------------------------------------
# The provider itself (one real call path, network refused by the fixture)
# --------------------------------------------------------------------

def test_a_missing_api_key_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)

    with pytest.raises(email_provider.EmailProviderError, match="RESEND_API_KEY is not configured"):
        email_provider.send_via_resend("a@example.com", "b@example.com", "Subject", "Body")


def test_a_provider_http_error_is_wrapped_clearly(monkeypatch):
    def failing_post(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx, "post", failing_post)

    with pytest.raises(email_provider.EmailProviderError, match="Resend request failed"):
        email_provider.send_via_resend("a@example.com", "b@example.com", "Subject", "Body", api_key="k")


# --------------------------------------------------------------------
# Through AdminService (the real call path from the UI), still test-mode by default
# --------------------------------------------------------------------

def test_the_admin_service_sends_through_the_service_and_the_response_says_it_was_redirected(
    monkeypatch, admin, token, store, profile, configured_recipient,
):
    opp_id = prepared(admin, token, store, profile)
    fake = FakeProvider()
    monkeypatch.setattr(admin, "_emailer", lambda: EmailSendingService(store, send_fn=fake))

    result = admin.send_application_email(token, opp_id)

    assert result["mode"] == "test" and result["to"] == configured_recipient and result["redirected"] is True
    assert result["intended_recipient"] == "eoi@moe.gov.eg" and result["provider_reference"] == "msg_123"


def test_sending_an_email_requires_an_admin_session(admin, store, profile):
    opp_id = prepared(admin, admin.login(PASSCODE), store, profile)
    with pytest.raises(AdminAuthError):
        admin.send_application_email("forged-token", opp_id)


# --------------------------------------------------------------------
# Review, edit and save a draft before sending (tender approval only ever
# prepares one; nothing is sent until the admin explicitly confirms)
# --------------------------------------------------------------------

def test_the_current_draft_is_the_originally_generated_text_until_edited(admin, token, store, profile):
    opp_id = prepared(admin, token, store, profile)

    draft = admin.current_email_draft(token, opp_id)

    assert draft["edited"] is False and draft["already_sent"] is False
    assert draft["subject"].startswith("Expression of Interest")
    assert "Andalusia Academy LLC" in draft["body"]


def test_saving_an_edit_is_reflected_in_the_current_draft_and_used_when_sending(
    admin, token, store, profile, configured_recipient,
):
    opp_id = prepared(admin, token, store, profile)

    admin.save_email_draft(token, opp_id, "A hand-edited subject", "A hand-edited body.")
    draft = admin.current_email_draft(token, opp_id)
    assert draft["edited"] is True and draft["subject"] == "A hand-edited subject"
    assert draft["draft_updated_by"].startswith("admin-session-")

    fake = FakeProvider()
    EmailSendingService(store, send_fn=fake).send_prepared_email(opp_id, "admin-session-x")
    assert fake.calls[0]["subject"] == "A hand-edited subject" and fake.calls[0]["body"] == "A hand-edited body."


def test_discarding_an_edit_reverts_to_the_generated_text(admin, token, store, profile):
    opp_id = prepared(admin, token, store, profile)
    original = admin.current_email_draft(token, opp_id)["subject"]
    admin.save_email_draft(token, opp_id, "Edited subject", "Edited body.")

    admin.discard_email_draft(token, opp_id)

    draft = admin.current_email_draft(token, opp_id)
    assert draft["edited"] is False and draft["subject"] == original


def test_an_empty_edit_is_refused(admin, token, store, profile):
    opp_id = prepared(admin, token, store, profile)
    with pytest.raises(EmailSendError, match="cannot be empty"):
        admin.save_email_draft(token, opp_id, "  ", "  ")


def test_editing_without_a_prepared_email_is_refused(admin, token, store):
    opp_id = add(store, ETIMAD)   # not approved yet
    with pytest.raises(EmailSendError, match="No prepared email"):
        admin.save_email_draft(token, opp_id, "Subject", "Body")


def test_a_sent_attempts_draft_can_no_longer_be_edited(admin, token, store, profile, configured_recipient):
    opp_id = prepared(admin, token, store, profile)
    EmailSendingService(store, send_fn=FakeProvider()).send_prepared_email(opp_id, "admin-session-x")

    with pytest.raises(EmailSendError, match="already been sent"):
        admin.save_email_draft(token, opp_id, "Too late", "Too late.")

    draft = admin.current_email_draft(token, opp_id)
    assert draft["already_sent"] is True


def test_the_draft_marks_a_demo_profile_clearly(admin, token, store, profile):
    profile.update_fields({"legal_name": "Demo Academy", "contact_person": "Demo Contact",
                           "contact_email": "demo@example.com"}, "admin-session-x")
    profile.add_document("company_profile", "Company Profile.pdf", b"%PDF-1.7\n%", "admin-session-x")
    opp_id = add(store, WB_EOI)
    admin.decide(token, opp_id, "approved", authorize_submission=True)

    draft = admin.current_email_draft(token, opp_id)

    assert draft["is_demo_profile"] is True and "DEMONSTRATION CONTENT" in draft["body"]


# --------------------------------------------------------------------
# Every approved tender gets a draft; sending stays explicit and guarded
# --------------------------------------------------------------------

def test_a_manual_route_tender_gets_a_draft_that_test_mode_sends_only_to_the_test_recipient(
    admin, token, store, profile, configured_recipient,
):
    complete_profile(profile)
    opp_id = add(store, ETIMAD)   # Etimad: manual-only portal, no verified contact address
    admin.decide(token, opp_id, "approved", authorize_submission=True)
    fake = FakeProvider()

    sent = EmailSendingService(store, send_fn=fake).send_prepared_email(opp_id, "admin-session-x")

    assert [c["to"] for c in fake.calls] == [configured_recipient]
    assert sent.to == configured_recipient and sent.intended_recipient is None
    record = store.email_sends(opp_id)[-1]
    assert (record["recipient"], record["status"], record["mode"]) == (configured_recipient, "sent", "test")


def test_approval_alone_never_sends_anything(admin, token, store, profile, configured_recipient, monkeypatch):
    complete_profile(profile)
    calls = []
    monkeypatch.setattr(email_provider, "send_via_resend", lambda *a, **k: calls.append(a) or "x")

    for tender in (WB_EOI, ETIMAD):
        admin.decide(token, add(store, tender), "approved", authorize_submission=True)

    assert calls == []
    assert all(store.email_sends(row["opportunity_id"]) == [] for row in store.search())


def test_sending_needs_the_admins_authorisation(admin, token, store, profile, configured_recipient):
    complete_profile(profile)
    opp_id = add(store, ETIMAD)
    admin.decide(token, opp_id, "approved")   # approved, but not authorised to submit
    fake = FakeProvider()

    with pytest.raises(EmailSendError, match="Authorise submission"):
        EmailSendingService(store, send_fn=fake).send_prepared_email(opp_id, "admin-session-x")
    assert fake.calls == []

    admin.authorize_submission(token, opp_id)
    EmailSendingService(store, send_fn=fake).send_prepared_email(opp_id, "admin-session-x")
    assert [c["to"] for c in fake.calls] == [configured_recipient]


def test_production_mode_sends_a_tender_without_a_contact_only_to_the_configured_recipient(
    admin, token, store, profile, monkeypatch,
):
    # (Before 2026-11-11 production mode refused such a tender because it sent to the tender's
    # contact; now every email goes to the configured recipient, so the contact is irrelevant.)
    monkeypatch.setenv("REX_EMAIL_MODE", "production")
    monkeypatch.setenv("REX_EMAIL_TEST_RECIPIENT", "configured@andalusia.academy")
    complete_profile(profile)
    profile.update_fields({"contact_email": "tenders@andalusia.academy"}, "admin-session-test")
    opp_id = add(store, ETIMAD)
    admin.decide(token, opp_id, "approved", authorize_submission=True)
    fake = FakeProvider()

    sent = EmailSendingService(store, send_fn=fake).send_prepared_email(opp_id, "admin-session-x")

    assert [call["to"] for call in fake.calls] == ["configured@andalusia.academy"] and sent.to == fake.calls[0]["to"]


def test_without_a_sender_address_nothing_is_sent(admin, token, store, profile, configured_recipient):
    profile.update_fields({"legal_name": "Andalusia Academy LLC"}, "admin-session-test")   # no contact email
    opp_id = add(store, ETIMAD)
    admin.decide(token, opp_id, "approved", authorize_submission=True)
    fake = FakeProvider()

    with pytest.raises(EmailSendError, match="No sender address"):
        EmailSendingService(store, send_fn=fake).send_prepared_email(opp_id, "admin-session-x")
    assert fake.calls == []
