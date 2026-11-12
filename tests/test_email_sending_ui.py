"""
The "Send now" action in the Documents.. er, Dashboard tab's application
panel (rex_ai_app.py): the one UI control that can really send an email.
httpx.post is monkeypatched to a fake Resend response, so no real
network call is ever made; the test still checks the exact payload sent,
to prove test mode redirected it away from the tender's real contact.
"""

import httpx
import pytest

from agents.admin_service import AdminService
from agents.rfp_research.application_profile import ApplicationProfile
from tests.test_application_workflow import WB_EOI, add, complete_profile
from tests.test_rex_ai_app import PASSCODE, env, goto, start, unlock  # noqa: F401  (env is a fixture)

TEST_RECIPIENT = "qa-inbox@example.com"


@pytest.fixture(autouse=True)
def isolated_email_settings(monkeypatch):
    """
    See tests/test_email_sending.py's fixture of the same name: forces
    the real .env to load once now, then blanks every email setting, so
    no test here depends on call order to stay isolated from the real
    RESEND_API_KEY and friends. Each test below sets back only the
    settings it actually needs.
    """
    from agents import config
    config.load_env()
    for name in ("RESEND_API_KEY", "REX_EMAIL_MODE", "REX_EMAIL_TEST_RECIPIENT", "RESEND_FROM_EMAIL"):
        monkeypatch.delenv(name, raising=False)


class FakeResponse:
    def __init__(self, message_id="msg_test_1"):
        self._message_id = message_id

    def raise_for_status(self):
        pass

    def json(self):
        return {"id": self._message_id}


@pytest.fixture
def prepared_eoi(env, tmp_path):
    """An approved, authorised World Bank EOI tender with a prepared (never-sent) email."""
    profile = ApplicationProfile(db_path=env["tenders"].db_path, documents_dir=tmp_path / "docs")
    complete_profile(profile)
    opp_id = add(env["tenders"], WB_EOI)

    backend = AdminService(orchestrator_factory=lambda: None, opportunity_store=env["tenders"],
                           application_profile=profile, passcode=PASSCODE)
    backend.decide(backend.login(PASSCODE), opp_id, "approved", authorize_submission=True)
    profile.close()
    return opp_id


def open_dashboard(at, title):
    goto(at, "dashboard")
    assert any(title in (e.label or "") for e in at.expander)
    return at


def unescaped(value: str) -> str:
    """plain() backslash-escapes markdown-special characters (e.g. the dots in an email address)."""
    return value.replace("\\", "")


def test_sending_in_test_mode_redirects_and_never_uses_the_real_contact(monkeypatch, env, prepared_eoi):
    monkeypatch.delenv("REX_EMAIL_MODE", raising=False)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("RESEND_API_KEY", "test-key")
    monkeypatch.setenv("REX_EMAIL_TEST_RECIPIENT", TEST_RECIPIENT)

    at = open_dashboard(unlock(start()), "Consultancy for Teacher Training Programme")
    assert any("Send now" in (b.label or "") for b in at.button)

    at.checkbox(key=f"confirm_send_{prepared_eoi}").check()
    at.button(key=f"send_now_{prepared_eoi}").click().run()

    assert not at.exception, at.exception
    assert captured["json"]["to"] == [TEST_RECIPIENT]   # never the tender's own contact
    assert "eoi@moe.gov.eg" not in str(captured["json"])
    assert any(TEST_RECIPIENT in unescaped(s.value) for s in at.success)
    assert any("eoi@moe.gov.eg" in unescaped(c.value) for c in at.caption)   # the "real contact was never used" note


def test_a_second_send_for_the_same_attempt_is_refused_by_the_ui(monkeypatch, env, prepared_eoi):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse())
    monkeypatch.setenv("RESEND_API_KEY", "test-key")
    monkeypatch.setenv("REX_EMAIL_TEST_RECIPIENT", TEST_RECIPIENT)

    at = open_dashboard(unlock(start()), "Consultancy for Teacher Training Programme")
    at.checkbox(key=f"confirm_send_{prepared_eoi}").check()
    at.button(key=f"send_now_{prepared_eoi}").click().run()

    at = open_dashboard(at, "Consultancy for Teacher Training Programme")
    assert not any(b.key == f"send_now_{prepared_eoi}" for b in at.button)   # replaced by the sent confirmation
    assert any(TEST_RECIPIENT in unescaped(s.value) for s in at.success)


def test_without_confirming_the_checkbox_nothing_is_sent(monkeypatch, env, prepared_eoi):
    def never_call_the_network(*args, **kwargs):
        raise AssertionError("must not call the network without confirmation")

    monkeypatch.setattr(httpx, "post", never_call_the_network)
    monkeypatch.setenv("REX_EMAIL_TEST_RECIPIENT", TEST_RECIPIENT)

    at = open_dashboard(unlock(start()), "Consultancy for Teacher Training Programme")
    at.button(key=f"send_now_{prepared_eoi}").click().run()

    assert not at.exception, at.exception
    assert any("confirmation" in e.value.lower() or "تأكيد" in e.value for e in at.error)


def test_a_missing_test_recipient_shows_a_clear_error_instead_of_the_send_form(env, prepared_eoi):
    at = open_dashboard(unlock(start()), "Consultancy for Teacher Training Programme")   # recipient not configured

    assert not any(b.key == f"send_now_{prepared_eoi}" for b in at.button)
    assert any("REX_EMAIL_TEST_RECIPIENT" in e.value for e in at.error)


def test_editing_and_saving_the_draft_is_used_when_sending(monkeypatch, env, prepared_eoi):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("RESEND_API_KEY", "test-key")
    monkeypatch.setenv("REX_EMAIL_TEST_RECIPIENT", TEST_RECIPIENT)

    at = open_dashboard(unlock(start()), "Consultancy for Teacher Training Programme")
    at.text_input(key=f"edit_subject_{prepared_eoi}").input("A hand-edited subject")
    at.text_area(key=f"edit_body_{prepared_eoi}").input("A hand-edited body.")
    at.button(key=f"save_draft_{prepared_eoi}").click().run()

    assert not at.exception, at.exception
    assert at.text_input(key=f"edit_subject_{prepared_eoi}").value == "A hand-edited subject"
    assert any("Edited by" in unescaped(c.value) for c in at.caption)

    at.checkbox(key=f"confirm_send_{prepared_eoi}").check()
    at.button(key=f"send_now_{prepared_eoi}").click().run()

    assert not at.exception, at.exception
    assert captured["json"]["subject"] == "A hand-edited subject"
    assert captured["json"]["text"].startswith("A hand-edited body.")


# --------------------------------------------------------------------
# The recipient shown before confirming, authorisation, and the review -> email flow
# --------------------------------------------------------------------

def test_the_actual_recipient_is_shown_before_confirming(monkeypatch, env, prepared_eoi):
    monkeypatch.setenv("REX_EMAIL_TEST_RECIPIENT", TEST_RECIPIENT)

    at = open_dashboard(unlock(start()), "Consultancy for Teacher Training Programme")

    header = next(m.value for m in at.markdown if m.value.startswith("**Will be sent to:**"))
    assert TEST_RECIPIENT in unescaped(header) and "eoi@moe.gov.eg" not in unescaped(header)
    assert any("eoi@moe.gov.eg" in unescaped(c.value) and "never used" in c.value for c in at.caption)


def test_production_mode_shows_and_uses_the_configured_recipient(monkeypatch, env, prepared_eoi):
    monkeypatch.setenv("REX_EMAIL_TEST_RECIPIENT", TEST_RECIPIENT)
    monkeypatch.setenv("REX_EMAIL_MODE", "production")

    at = open_dashboard(unlock(start()), "Consultancy for Teacher Training Programme")

    header = next(m.value for m in at.markdown if m.value.startswith("**Will be sent to:**"))
    assert TEST_RECIPIENT in unescaped(header) and "eoi@moe.gov.eg" not in unescaped(header)
    assert any(TEST_RECIPIENT in unescaped(w.value) and "never to the tender" in w.value for w in at.warning)


def test_an_unauthorised_approval_gets_a_draft_but_no_send_form(monkeypatch, env, tmp_path):
    from tests.test_application_workflow import ETIMAD

    monkeypatch.setenv("REX_EMAIL_TEST_RECIPIENT", TEST_RECIPIENT)
    profile = ApplicationProfile(db_path=env["tenders"].db_path, documents_dir=tmp_path / "docs")
    complete_profile(profile)
    opp_id = add(env["tenders"], ETIMAD)
    backend = AdminService(orchestrator_factory=lambda: None, opportunity_store=env["tenders"],
                           application_profile=profile, passcode=PASSCODE)
    backend.decide(backend.login(PASSCODE), opp_id, "approved")   # not authorised
    profile.close()

    at = open_dashboard(unlock(start()), "Leadership Training Programme 2030")

    assert any(t.key == f"edit_body_{opp_id}" for t in at.text_area)   # the draft is there to review
    assert not any(b.key == f"send_now_{opp_id}" for b in at.button)
    assert any("needs your authorisation" in i.value for i in at.info)


def test_approving_in_tender_review_leads_to_the_email_in_one_click(monkeypatch, env, tmp_path):
    monkeypatch.setenv("REX_EMAIL_TEST_RECIPIENT", TEST_RECIPIENT)
    profile = ApplicationProfile(db_path=env["tenders"].db_path, documents_dir=tmp_path / "docs")
    complete_profile(profile)
    profile.close()
    opp_id = add(env["tenders"], WB_EOI)
    at = unlock(start())

    at.button(key=f"approved_{opp_id}").click().run()
    at.checkbox(key="authorise_submission").check().run()
    at.button(key="confirm_approve").click().run()

    assert at.session_state.admin_section == "review"   # approving does not move the admin
    assert any("application email draft is ready" in s.value for s in at.success)
    at.segmented_control(key="review_filter").set_value("approved").run()
    at.button(key=f"open_email_{opp_id}").click().run()   # an explicit "open it" action

    assert at.session_state.admin_section == "dashboard"
    assert any(t.key == f"edit_body_{opp_id}" for t in at.text_area)
    assert any(b.key == f"send_now_{opp_id}" for b in at.button)
