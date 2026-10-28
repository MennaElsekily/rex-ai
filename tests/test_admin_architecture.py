"""
The User/Admin architecture:

    User UI  -> agents/user_service.py  -> REX Assistant (no orchestrator)
    Admin UI -> agents/admin_service.py (server-side session check)
             -> create_admin_orchestrator() -> admin agents, REX excluded

Mocked: REX Assistant and the admin agents' handlers. Temp SQLite files.
"""

from unittest.mock import MagicMock

import pytest

from agents.admin_service import (
    LOCKOUT_SECONDS,
    MAX_FAILED_ATTEMPTS,
    SESSION_IDLE_SECONDS,
    AdminAuthError,
    AdminService,
)
from agents.conversation_store import ConversationStore
from agents.orchestrator import Orchestrator
from agents.register_agents import create_admin_orchestrator
from agents.rfp_research.opportunity_store import OpportunityStore
from agents.user_service import RexUserService


def ok(request):
    return {
        "request_id": request["request_id"],
        "agent_name": request["agent_name"],
        "status": "success",
        "result": {"handled_by": request["agent_name"]},
        "error_message": None,
    }


# --------------------------------------------------------------------
# User workflow: REX directly, never the orchestrator
# --------------------------------------------------------------------

def test_user_messages_go_straight_to_rex(monkeypatch):
    rex = MagicMock()
    rex.handle_message.return_value = {"status": "success", "state": {}}
    handle_request = MagicMock()
    monkeypatch.setattr(Orchestrator, "handle_request", handle_request)

    service = RexUserService(agent=rex)
    service.chat("Find education tenders in Egypt", state=None, preferred_language="ar")

    rex.handle_message.assert_called_once_with(
        "Find education tenders in Egypt", state=None, preferred_language="ar"
    )
    handle_request.assert_not_called()


def test_user_service_exposes_no_admin_agents():
    service = RexUserService(agent=MagicMock())

    assert not hasattr(service, "ask")
    assert not hasattr(service, "orchestrator")


# --------------------------------------------------------------------
# Admin orchestrator: REX excluded
# --------------------------------------------------------------------

@pytest.fixture
def admin_orchestrator(tmp_path):
    rfp = MagicMock()
    rfp.AGENT_NAME = "rfp_research"
    rfp.process_request.side_effect = ok
    return create_admin_orchestrator(
        rfp_research_agent=rfp,
        conversation_store=ConversationStore(db_path=tmp_path / "conv.sqlite3"),
        opportunity_store=OpportunityStore(db_path=tmp_path / "opps.sqlite3"),
    )


def test_admin_orchestrator_registers_admin_agents_only(admin_orchestrator):
    assert set(admin_orchestrator.agents) == {"rfp_research", "curriculum_research"}
    assert "rex_assistant" in admin_orchestrator.excluded_agents


def test_rex_cannot_be_registered_on_the_admin_orchestrator(admin_orchestrator):
    with pytest.raises(ValueError):
        admin_orchestrator.register_agent("rex_assistant", ok)


@pytest.mark.parametrize("text, expected_agent", [
    ("Find education and training tenders in Egypt", "rfp_research"),
    ("ابحث عن مناقصات تدريب في مصر", "rfp_research"),
    ("Create a curriculum about data analysis", "curriculum_research"),
])
def test_admin_requests_reach_admin_agents(admin_orchestrator, text, expected_agent):
    response = admin_orchestrator.handle_request(text, context={"conversation_id": "a1"})

    assert response["status"] == "success"
    assert response["result"][0]["agent_name"] == expected_agent


@pytest.mark.parametrize("text, language, phrase", [
    ("Which course do you recommend for me?", "en", "User interface"),
    ("رشحلي كورسات مناسبة ليا", "ar", "واجهة المستخدم"),
])
def test_rex_requests_are_refused_by_the_admin_orchestrator(
    admin_orchestrator, text, language, phrase
):
    response = admin_orchestrator.handle_request(text, context={"conversation_id": "a1"})

    assert response["status"] == "error"
    assert response["result"][0]["agent_name"] == "orchestrator"
    assert phrase in response["error_message"]


def test_default_orchestrator_behaviour_is_unchanged():
    router = Orchestrator()
    router.register_agent("rex_assistant", ok)

    response = router.handle_request("Can you recommend a course for me?")

    assert response["result"][0]["agent_name"] == "rex_assistant"


def test_explicit_ui_language_reaches_admin_agents(admin_orchestrator):
    admin_orchestrator.handle_request(
        "Find tenders in Egypt", context={"preferred_language": "ar"}
    )

    request = admin_orchestrator.rfp_research_agent.process_request.call_args[0][0]
    assert request["context"]["language"] == "ar"


# --------------------------------------------------------------------
# Admin authorization (server-side)
# --------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def tenders(tmp_path):
    store = OpportunityStore(db_path=tmp_path / "tenders.sqlite3")
    yield store
    store.close()


@pytest.fixture
def service(clock, admin_orchestrator, tenders):
    return AdminService(
        orchestrator_factory=lambda: admin_orchestrator,
        opportunity_store=tenders,
        passcode="correct horse",
        clock=clock,
    )


def test_admin_is_disabled_without_a_configured_passcode(admin_orchestrator):
    service = AdminService(lambda: admin_orchestrator, passcode="")

    assert service.enabled is False
    assert service.login("") is None
    assert service.login("anything") is None


def test_wrong_passcode_gives_no_session(service):
    assert service.login("wrong") is None
    assert service.login("Correct Horse") is None


def test_right_passcode_gives_a_session(service):
    token = service.login("correct horse")

    assert token and service.is_valid(token)


@pytest.mark.parametrize("action, args", [
    ("ask", ("Find tenders in Egypt",)),
    ("search_tenders", ({"countries": ["Egypt"]},)),
    ("list_tenders", ()),
    ("decide", ("someid", "approved")),
    ("audit_trail", ()),
    ("tender_counts", ()),
    ("alerts", ()),
    ("mark_alerts_seen", ()),
])
@pytest.mark.parametrize("token", [None, "", "forged-token"])
def test_every_admin_action_requires_a_valid_session(service, action, args, token):
    with pytest.raises(AdminAuthError):
        getattr(service, action)(token, *args)


def test_sessions_expire_after_idle_time(service, clock):
    token = service.login("correct horse")
    clock.now += SESSION_IDLE_SECONDS + 1

    with pytest.raises(AdminAuthError):
        service.list_tenders(token)


def test_logout_ends_the_session(service):
    token = service.login("correct horse")
    service.logout(token)

    with pytest.raises(AdminAuthError):
        service.list_tenders(token)


def test_repeated_wrong_passcodes_are_throttled(service, clock):
    for _ in range(MAX_FAILED_ATTEMPTS):
        service.login("wrong")

    assert service.is_locked()
    assert service.login("correct horse") is None  # locked even if correct

    clock.now += LOCKOUT_SECONDS + 1
    assert service.login("correct horse")


def test_admin_requests_go_through_the_admin_orchestrator(service):
    token = service.login("correct horse")

    response = service.ask(token, "Find education tenders in Egypt")

    assert response["result"][0]["agent_name"] == "rfp_research"


def tender(url, title, country="Egypt", review_status="flagged_for_immediate_review"):
    return {"url": url, "title": title, "country": country, "organization": "Ministry of Education",
            "training_category": "Education and Training", "review_status": review_status}


def test_review_decisions_are_attributed_to_the_admin_session(service, tenders):
    tenders.save(tender("https://etenders.gov.eg/t/1", "Teacher Training Services Tender"))
    opp_id = tenders.search()[0]["opportunity_id"]
    token = service.login("correct horse")

    service.decide(token, opp_id, "approved")

    audit = service.audit_trail(token, opp_id)
    assert audit[0]["decision"] == "approved"
    assert audit[0]["reviewer"].startswith("admin-session-")
    assert token not in audit[0]["reviewer"]  # the token itself is never stored


def test_unknown_decisions_are_refused(service, tenders):
    tenders.save(tender("https://etenders.gov.eg/t/1", "Teacher Training Services Tender"))
    token = service.login("correct horse")

    with pytest.raises(ValueError):
        service.decide(token, tenders.search()[0]["opportunity_id"], "published")


def test_review_queue_and_dashboard(service, tenders):
    tenders.save(tender("https://etenders.gov.eg/t/1", "Teacher Training Services Tender"))
    tenders.save(tender("https://tenders.etimad.sa/t/2", "Digital Skills Programme Tender",
                        country="Saudi Arabia", review_status="monitored_pipeline"))
    token = service.login("correct horse")
    saudi = service.list_tenders(token, "pending", country="Saudi Arabia")[0]["opportunity_id"]

    service.decide(token, saudi, "approved")

    assert [t["title"] for t in service.list_tenders(token, "pending")] == ["Teacher Training Services Tender"]
    assert [t["title"] for t in service.list_tenders(token, "approved", keyword="Digital")] == [
        "Digital Skills Programme Tender"]
    assert service.list_tenders(token, "approved", category="Corporate Learning") == []
    counts = service.tender_counts(token)
    assert (counts["pending"], counts["approved"], counts["flagged_pending"]) == (1, 1, 1)


def test_new_tenders_raise_alerts_until_seen(service, tenders):
    token = service.login("correct horse")
    tenders.save(tender("https://etenders.gov.eg/t/1", "Teacher Training Services Tender"))

    assert len(service.alerts(token)) == 1
    service.mark_alerts_seen(token)
    assert service.alerts(token) == []


def test_search_form_choices_reach_the_rfp_agent(service, admin_orchestrator):
    token = service.login("correct horse")
    options = {"countries": ["Saudi Arabia"], "categories": ["Corporate Learning"],
               "priority": "Ministry of Health", "eligibility_notes": "", "deadline_from": None,
               "deadline_to": "2030-12-31"}

    response = service.search_tenders(token, options, context={"preferred_language": "ar"})

    assert response["result"][0]["agent_name"] == "rfp_research"
    request = admin_orchestrator.rfp_research_agent.process_request.call_args[0][0]
    assert request["context"]["search"] == options
    assert request["context"]["language"] == "ar"


def test_tenders_are_never_ingested_into_the_course_knowledge_base(service):
    assert not hasattr(service, "ingest_approved")
