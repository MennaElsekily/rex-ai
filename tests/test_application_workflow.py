"""
The application workflow (agents/rfp_research/applications.py) and its
admin actions: approval starts it, routes per source, dry-run email
preparation, missing requirements, authorisation, no duplicates, restart
recovery and history. Temp database and documents folder; nothing is
ever sent anywhere.
"""

from pathlib import Path

import pytest

from agents.admin_service import AdminAuthError, AdminService
from agents.rfp_research import applications
from agents.rfp_research.api_clients import world_bank
from agents.rfp_research.application_profile import ApplicationProfile
from agents.rfp_research.applications import EMAIL_EOI, MANUAL, ApplicationWorkflow, detect_route
from agents.rfp_research.opportunity_store import InvalidTransition, OpportunityStore

PASSCODE = "correct horse"
PDF = b"%PDF-1.7\n%"
WB = world_bank.SOURCE_NAME

WB_EOI = {"url": "https://search.worldbank.org/notice/OP001", "source_url": "https://search.worldbank.org/notice/OP001",
          "title": "Consultancy for Teacher Training Programme", "reference_number": "EG-MOE-123-CS",
          "organization": "Ministry of Education", "country": "Egypt", "deadline": "2030-11-30",
          "category": "Request for Expression of Interest", "source_name": WB,
          "verification_status": "verified_from_api", "contact_email": "eoi@moe.gov.eg",
          "review_status": "flagged_for_immediate_review"}
WB_BID = {**WB_EOI, "url": "https://search.worldbank.org/notice/OP002", "source_url": "https://search.worldbank.org/notice/OP002",
          "title": "Supply of Training Laboratory Equipment Lot 2", "category": "Invitation for Bids",
          "submission_instructions": "Bids to the PMU, Cairo; bid security required"}
ETIMAD = {"url": "https://tenders.etimad.sa/Tender/Details/77", "title": "Leadership Training Programme 2030",
          "country": "Saudi Arabia", "source_name": "Web: tenders.etimad.sa", "source_domain": "tenders.etimad.sa",
          "verification_status": "extracted_from_web_page", "review_status": "monitored_pipeline"}
EGYPT_PORTAL = {"url": "https://etenders.gov.eg/Tender/9", "title": "Digital Skills Training for Civil Servants",
                "country": "Egypt", "source_name": "Web: etenders.gov.eg", "source_domain": "etenders.gov.eg",
                "verification_status": "extracted_from_web_page"}
LISTING = {"url": "https://www.tendersinfo.com/x/1", "title": "E-learning Platform for Universities Tender",
           "country": "Egypt", "organization": "Cairo University", "source_name": "Web: www.tendersinfo.com",
           "source_domain": "www.tendersinfo.com", "verification_status": "extracted_from_web_page"}
# A web page that claims to be an expression of interest with an email:
# web text must never choose the route or the recipient.
WEB_CLAIMS_EOI = {**LISTING, "url": "https://evil.example/eoi", "source_domain": "evil.example",
                  "source_name": "Web: evil.example", "title": "Training Consultancy EOI Notice",
                  "category": "Request for Expression of Interest", "contact_email": "collect@evil.example"}


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


def add(store, tender):
    store.save(tender)
    return store.search(keyword=tender["title"])[0]["opportunity_id"]


def complete_profile(profile):
    profile.update_fields({"legal_name": "Andalusia Academy LLC", "contact_person": "Mona Adel",
                           "contact_email": "tenders@andalusia.example"}, "admin-session-x")
    profile.add_document("company_profile", "Company Profile.pdf", PDF, "admin-session-x")


# --------------------------------------------------------------------
# Routes per source
# --------------------------------------------------------------------

@pytest.mark.parametrize("tender, adapter, label", [
    (WB_EOI, EMAIL_EOI, "expression of interest by email"),
    (WB_BID, MANUAL, "borrower's implementing agency"),
    (ETIMAD, MANUAL, "Etimad portal"),
    (EGYPT_PORTAL, MANUAL, "Egyptian government tender"),
    (LISTING, MANUAL, "Tender listing site"),
    (WEB_CLAIMS_EOI, MANUAL, "Web page"),
])
def test_the_route_depends_on_the_source(tender, adapter, label):
    route = detect_route(tender)

    assert route.adapter == adapter and label in route.label
    assert route.automatic is (adapter == EMAIL_EOI)


def test_only_the_official_api_address_can_be_a_recipient():
    assert detect_route(WB_EOI).recipient == "eoi@moe.gov.eg"
    assert detect_route(WEB_CLAIMS_EOI).recipient is None
    assert detect_route({**WB_EOI, "verification_status": "extracted_from_web_page"}).adapter == MANUAL
    assert detect_route({**WB_EOI, "contact_email": "not an email"}).adapter == MANUAL


def test_etimad_steps_name_the_human_only_actions():
    steps = " ".join(detect_route(ETIMAD).steps)

    assert "Nafath" in steps and "authorised signer" in steps


def test_the_world_bank_connector_keeps_the_contact_email():
    record = world_bank._normalize({
        "id": "OP9", "project_ctry_name": "Arab Republic of Egypt", "notice_type": "Request for Expression of Interest",
        "bid_description": "Training consultancy", "contact_email": " eoi@moe.gov.eg ",
        "submission_deadline_date": "2030-11-30T00:00:00"})

    assert record.contact_email == "eoi@moe.gov.eg"


# --------------------------------------------------------------------
# Approval starts the workflow
# --------------------------------------------------------------------

def test_approval_starts_the_workflow_but_never_submits(admin, token, store):
    opp_id = add(store, ETIMAD)

    tender = admin.decide(token, opp_id, "approved")

    run = tender["application"]
    assert run["started"] is True and run["adapter"] == MANUAL
    assert tender["submission_status"] == "requires_manual_action"
    assert tender["application_status"] == "not_applied"
    [attempt] = store.attempts(opp_id)
    assert attempt["actor"] == "system:application_workflow"
    assert attempt["triggered_by"].startswith("admin-session-")
    assert any("Nafath" in step for step in attempt["manual_actions"])
    assert attempt["manual_actions"][-1].startswith("Then record the manual submission")


def test_re_approving_or_rejecting_starts_nothing(admin, token, store):
    approved, rejected = add(store, ETIMAD), add(store, LISTING)
    admin.decide(token, approved, "approved")

    again = admin.decide(token, approved, "approved")
    admin.decide(token, rejected, "rejected")

    assert "application" not in again
    assert len(store.attempts(approved)) == 1 and store.attempts(rejected) == []


def test_rediscovery_after_approval_starts_nothing(admin, token, store):
    opp_id = add(store, ETIMAD)
    admin.decide(token, opp_id, "approved")

    store.save(ETIMAD)  # found again by a later search

    assert len(store.attempts(opp_id)) == 1


# --------------------------------------------------------------------
# The dry-run expression of interest
# --------------------------------------------------------------------

def test_missing_profile_documents_and_authorisation_are_listed(admin, token, store):
    opp_id = add(store, WB_EOI)

    run = admin.decide(token, opp_id, "approved")["application"]

    assert run["adapter"] == EMAIL_EOI and run["status"] == "requires_manual_action"
    assert run["missing"] == ["Legal name", "Contact person", "Contact email",
                              "Document: Company profile",
                              "Your authorisation to submit on the academy's behalf (the checkbox when approving)"]
    assert run["confirmation"] == {}  # nothing was prepared


def test_a_complete_profile_gives_a_prepared_email_that_is_never_sent(admin, token, store, profile):
    complete_profile(profile)
    opp_id = add(store, WB_EOI)

    tender = admin.decide(token, opp_id, "approved", authorize_submission=True)

    run = tender["application"]
    assert run["dry_run"] is True and run["status"] == "requires_manual_action" and run["missing"] == []
    email = run["confirmation"]["prepared_email"]
    assert (email["to"], email["from"]) == ("eoi@moe.gov.eg", "tenders@andalusia.example")
    assert email["subject"] == "Expression of Interest: EG-MOE-123-CS - Consultancy for Teacher Training Programme"
    assert "Andalusia Academy LLC" in email["body"] and "Mona Adel" in email["body"]
    assert "Commercial registration" not in email["body"]  # not in the profile: not invented
    assert [a["file_name"] for a in email["attachments"]] == ["Company Profile.pdf"]
    assert run["confirmation"]["recommended_documents_missing"] == ["Experience / reference letters"]
    assert run["manual_actions"][0].startswith("Dry run: nothing was sent.")
    assert (tender["submission_status"], tender["application_status"]) == ("requires_manual_action", "not_applied")


def test_authorising_later_then_preparing_again(admin, token, store, profile):
    complete_profile(profile)
    opp_id = add(store, WB_EOI)
    first = admin.decide(token, opp_id, "approved")["application"]
    assert first["missing"][-1].startswith("Your authorisation")

    admin.authorize_submission(token, opp_id)
    second = admin.retry_submission(token, opp_id)

    assert second["attempt_no"] == 2 and second["missing"] == []
    assert "prepared_email" in second["confirmation"]
    assert store.audit_trail(opp_id)[-1]["details"] == {"authorize_submission": True, "given_after_approval": True}


def test_a_live_mode_setting_does_not_enable_sending(monkeypatch, admin, token, store, profile):
    monkeypatch.setenv("REX_SUBMISSION_MODE", "live")
    complete_profile(profile)
    opp_id = add(store, WB_EOI)

    run = admin.decide(token, opp_id, "approved", authorize_submission=True)["application"]

    assert run["dry_run"] is True and run["status"] == "requires_manual_action"
    source = Path(applications.__file__).read_text(encoding="utf-8")
    assert "smtplib" not in source and "sendmail" not in source and ".send(" not in source


# --------------------------------------------------------------------
# Failures, unknown results, restarts
# --------------------------------------------------------------------

def test_a_preparation_crash_is_recorded_as_failed_and_can_be_retried(monkeypatch, admin, token, store):
    opp_id = add(store, ETIMAD)

    def broken(self, *args):
        raise RuntimeError("bug")

    monkeypatch.setattr(applications.ManualRouteAdapter, "run", broken)
    run = admin.decide(token, opp_id, "approved")["application"]
    assert run["status"] == "submission_failed" and "RuntimeError" in run["error"]

    monkeypatch.undo()
    assert admin.retry_submission(token, opp_id)["status"] == "requires_manual_action"


def test_an_unknown_result_blocks_retries_until_the_admin_resolves_it(admin, token, store):
    opp_id = add(store, ETIMAD)
    admin.decide(token, opp_id, "approved")
    attempt = store.claim_submission(opp_id, MANUAL, "admin-session-x")
    store.finish_attempt(attempt["attempt_id"], "outcome_unknown", error="Timed out")

    blocked = admin.retry_submission(token, opp_id)
    assert blocked["started"] is False and blocked["submission_status"] == "outcome_unknown"

    admin.mark_not_submitted(token, opp_id, notes="Checked: not sent.")
    assert admin.retry_submission(token, opp_id)["started"] is True


def test_starting_the_workflow_recovers_interrupted_attempts(store, profile):
    store.save(ETIMAD)
    opp_id = store.search()[0]["opportunity_id"]
    store.decide(opp_id, "approved", "admin-session-x")
    store.claim_submission(opp_id, MANUAL, "admin-session-x")  # the app stopped here

    ApplicationWorkflow(store, profile)  # the next start

    assert store.get(opp_id)["submission_status"] == "outcome_unknown"


# --------------------------------------------------------------------
# The whole journey, history and authorisation
# --------------------------------------------------------------------

def test_from_approval_to_award(admin, token, store):
    opp_id = add(store, ETIMAD)
    admin.decide(token, opp_id, "approved")

    admin.record_manual_submission(token, opp_id, "2030-11-02", reference="ETM-OFFER-77",
                                   url="https://tenders.etimad.sa/offer/77")
    with pytest.raises(InvalidTransition):
        admin.decide(token, opp_id, "rejected")  # an applied tender keeps its approval
    admin.record_application_outcome(token, opp_id, "awarded", notes="Letter of award received")

    tender = store.get(opp_id)
    assert (tender["approval_status"], tender["submission_status"], tender["application_status"]) == (
        "approved", "submitted", "awarded")
    history = admin.application_history(token, opp_id)
    assert [a["adapter"] for a in history["attempts"]] == [MANUAL, "manual"]
    assert [(e["event"], e["decision"]) for e in history["audit"]] == [
        ("review", "approved"), ("application", "applied"), ("application", "awarded")]


def test_the_route_is_shown_before_approving(admin, token, store):
    route = admin.submission_route(token, add(store, WB_EOI))

    assert route["adapter"] == EMAIL_EOI and route["recipient"] == "eoi@moe.gov.eg"


def test_dashboard_filters_by_application_state(admin, token, store):
    manual = add(store, ETIMAD)
    add(store, LISTING)
    admin.decide(token, manual, "approved")

    assert [t["opportunity_id"] for t in admin.list_tenders(
        token, "approved", submission_status="requires_manual_action")] == [manual]
    assert admin.tender_counts(token)["needs_manual_action"] == 1


@pytest.mark.parametrize("action, args", [
    ("submission_route", ("x",)), ("authorize_submission", ("x",)), ("retry_submission", ("x",)),
    ("record_manual_submission", ("x", "2030-11-02")), ("mark_not_submitted", ("x",)),
    ("record_application_outcome", ("x", "awarded")), ("application_history", ("x",)),
])
def test_every_application_action_requires_an_admin_session(admin, action, args):
    with pytest.raises(AdminAuthError):
        getattr(admin, action)("forged-token", *args)


def test_no_tender_tool_can_start_an_application():
    """The MCP tools only search and save pending tenders."""
    from agents import tender_mcp_server

    source = Path(tender_mcp_server.__file__).read_text(encoding="utf-8")
    assert "applications" not in source and "claim_submission" not in source and "decide(" not in source
