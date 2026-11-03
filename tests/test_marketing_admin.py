"""
Agent 5 behind the admin session: generating, listing, editing (new versions), explicit approval and downloads. Every call
needs the admin token; draft and pending-review content is never exported as if it were ready.
"""

import pytest

from agents.admin_service import AdminAuthError, AdminService
from agents.conversation_store import ConversationStore
from agents.marketing.registration import register_marketing
from agents.orchestrator import Orchestrator
from tests.curriculum_fixtures import valid_curriculum
from tests.marketing_fakes import ADMIN, approved_course, fake_llm, in_process_marketing_tools

PASSCODE = "throwaway-passcode"


@pytest.fixture
def system(tmp_path):
    toolbox = in_process_marketing_tools(tmp_path)
    orchestrator = Orchestrator(excluded_agents=("rex_assistant",))
    agent = register_marketing(orchestrator, ConversationStore(db_path=":memory:"))
    agent._tools, agent._llm = toolbox, fake_llm()
    admin = AdminService(orchestrator_factory=lambda: orchestrator, passcode=PASSCODE, curriculum_store=toolbox.curricula,
                         curriculum_documents_dir=str(tmp_path / "cpdfs"), marketing_store=toolbox.store)

    class System:
        pass

    s = System()
    s.toolbox, s.agent, s.admin, s.token, s.course = toolbox, agent, admin, admin.login(PASSCODE), approved_course(toolbox)
    return s


def generate(system, **request):
    response = system.admin.generate_campaign(system.token, {"curriculum_id": system.course, **request}, {"language": "en"})
    result = response["result"][0]
    assert result["status"] == "success", result
    return result["result"]


# ---- the admin session -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, args", [
    ("list_approved_courses", ()), ("generate_campaign", ({"curriculum_id": "CUR-0123456789ab"},)), ("list_campaigns", ()),
    ("campaign_counts", ()), ("get_campaign", ("MKT-0123456789ab",)), ("edit_campaign", ("MKT-0123456789ab", [])),
    ("approve_campaign", ("MKT-0123456789ab", 1)), ("campaign_export", ("MKT-0123456789ab",)), ("campaign_audit_trail", ())])
@pytest.mark.parametrize("token", [None, "", "not-a-session", "admin-session-test0001"])
def test_every_marketing_action_needs_the_admin_session(system, name, args, token):
    with pytest.raises(AdminAuthError):
        getattr(system.admin, name)(token, *args)


def test_a_logged_out_session_can_do_nothing(system):
    campaign = generate(system)
    system.admin.logout(system.token)
    with pytest.raises(AdminAuthError):
        system.admin.approve_campaign(system.token, campaign["campaign_id"], 1)
    assert system.toolbox.store.get_approved(campaign["campaign_id"]) is None


# ---- generating and listing --------------------------------------------------------------------------------------------------------

def test_the_form_choices_reach_the_agent_and_the_campaign_waits_for_review(system):
    result = generate(system, audience="Finance analysts", tone="friendly", variations=3, channels=["email", "whatsapp"], goal="Fill a group")
    record = system.toolbox.store.get(result["campaign_id"])

    assert result["review_status"] == "pending_review" and record["version_status"] == "pending_review"
    content = record["content"]
    assert content["audience"] == {"text": "Finance analysts", "source": "admin"} and content["tone"] == "friendly"
    assert content["variations"] == 3 and list(content["channels"]) == ["email", "whatsapp"] and content["goal"] == "Fill a group"


def test_only_approved_courses_are_offered_and_a_failure_is_raised_not_hidden(system):
    system.toolbox.curricula.save_version(valid_curriculum(), {"valid": True, "errors": []}, [])              # pending: not offered
    page = system.admin.list_approved_courses(system.token)
    assert page["total"] == 1 and [c["curriculum_id"] for c in page["courses"]] == [system.course]

    system.admin._marketing().list_approved_courses = lambda *a, **k: {"status": "error", "message": "boom"}   # (the admin's own toolbox)
    with pytest.raises(ValueError, match="boom"):
        system.admin.list_approved_courses(system.token)


def test_a_course_that_is_not_approved_is_refused_through_the_admin_path_too(system):
    pending = system.toolbox.curricula.save_version(valid_curriculum(), {"valid": True, "errors": []}, [])["curriculum_id"]
    response = system.admin.generate_campaign(system.token, {"curriculum_id": pending}, {"language": "en"})
    assert response["status"] == "error" and response["result"][0]["result"]["kind"] == "not_approved"
    assert system.admin.list_campaigns(system.token) == []


def test_listing_filtering_and_counts(system):
    first, second = generate(system), generate(system)
    system.admin.approve_campaign(system.token, first["campaign_id"], 1)

    assert system.admin.campaign_counts(system.token) == {"draft": 0, "pending_review": 1, "approved": 1}
    assert [c["campaign_id"] for c in system.admin.list_campaigns(system.token, "pending_review")] == [second["campaign_id"]]
    assert len(system.admin.list_campaigns(system.token, curriculum_id=system.course)) == 2
    with pytest.raises(ValueError):
        system.admin.list_campaigns(system.token, "nonsense")


def test_get_campaign_returns_versions_audit_and_whether_the_course_moved_on(system):
    campaign = generate(system)
    detail = system.admin.get_campaign(system.token, campaign["campaign_id"])

    assert detail["campaign"]["version"] == 1 and len(detail["versions"]) == 1 and detail["audit"][0]["event"] == "version_saved"
    assert detail["course_changed"] is False
    revised = system.toolbox.curricula.save_version(valid_curriculum(), {"valid": True, "errors": []}, [], curriculum_id=system.course)
    system.toolbox.curricula.approve(system.course, revised["version"], ADMIN)                                  # the course has a newer approved version
    assert system.admin.get_campaign(system.token, campaign["campaign_id"])["course_changed"] is True
    assert system.admin.get_campaign(system.token, "MKT-aaaaaaaaaaaa") is None


# ---- editing and versioning ----------------------------------------------------------------------------------------------------------

EMAIL_SUBJECT = ["channels", "email", "variations", 0, "subject"]


def test_an_edit_is_a_new_version_that_needs_review_again(system):
    campaign = generate(system)
    cid = campaign["campaign_id"]

    saved = system.admin.edit_campaign(system.token, cid, [{"path": EMAIL_SUBJECT, "value": "  Your first course   in machine learning "}], "tidied")

    assert saved["version"] == 2 and saved["review_status"] == "pending_review"
    assert system.toolbox.store.get(cid, 1)["content"]["channels"]["email"]["variations"][0]["subject"] == "Your first machine learning course"
    assert system.toolbox.store.get(cid)["content"]["channels"]["email"]["variations"][0]["subject"] == "Your first course in machine learning"
    record = system.toolbox.store.get(cid)
    assert record["version_created_by"] != "system:marketing_agent" and record["change_summary"] == "tidied"
    assert [e["event"] for e in system.admin.campaign_audit_trail(system.token, cid)] == ["version_saved", "version_saved"]


def test_every_kind_of_text_can_be_edited(system):
    cid = generate(system)["campaign_id"]
    edits = [{"path": ["strategy", "objective"], "value": "Bring analysts to the beginner course"},
             {"path": ["strategy", "content_sequence", 1], "value": "Show what learners will practise"},
             {"path": ["channels", "social_captions", "variations", 1, "hashtags"], "value": "Python, DataSkills #Learning"},
             {"path": ["channels", "ad_copy", "variations", 0, "cta"], "value": "Learn more now"},
             {"path": ["channels", "linkedin_posts", "variations", 0, "text"],
              "value": "Machine learning starts with good data. This beginner course covers preparing a dataset for modelling and evaluating a simple model."}]

    saved = system.admin.edit_campaign(system.token, cid, edits)

    content = system.toolbox.store.get(cid)["content"]
    assert saved["review_status"] == "pending_review" and content["strategy"]["objective"] == "Bring analysts to the beginner course"
    assert content["channels"]["social_captions"]["variations"][1]["hashtags"] == ["Python", "DataSkills", "#Learning"]


@pytest.mark.parametrize("path", [["course_title"], ["curriculum_id"], ["curriculum_version"], ["language"], ["audience", "text"],
                                  ["strategy", "target_audience"], ["strategy", "recommended_channels"], ["channels", "email", "variations", 9, "subject"],
                                  ["channels", "tiktok", "variations", 0, "text"], ["channels", "email", "variations", 0, "nothing"], [], ["variations"]])
def test_the_course_facts_and_unknown_parts_cannot_be_edited(system, path):
    cid = generate(system)["campaign_id"]
    with pytest.raises(ValueError, match="cannot be edited"):
        system.admin.edit_campaign(system.token, cid, [{"path": path, "value": "x"}])
    assert system.toolbox.store.get(cid)["version"] == 1                                                         # nothing was saved


def test_an_edit_that_adds_an_unsupported_claim_is_kept_as_a_draft_that_cannot_be_approved(system):
    cid = generate(system)["campaign_id"]

    saved = system.admin.edit_campaign(system.token, cid, [{"path": ["channels", "whatsapp", "variations", 0, "message"],
                                                            "value": "Hello! Join the best course for only 99 dollars, certificate guaranteed."}])

    assert saved["review_status"] == "draft" and not saved["validation"]["valid"]
    with pytest.raises(ValueError, match="pending review"):
        system.admin.approve_campaign(system.token, cid, 2)
    assert system.toolbox.store.get_approved(cid) is None


def test_editing_an_unknown_campaign_is_refused(system):
    with pytest.raises(ValueError, match="Unknown campaign"):
        system.admin.edit_campaign(system.token, "MKT-aaaaaaaaaaaa", [])


def test_editing_an_approved_campaign_leaves_the_approved_version_as_it_was(system):
    cid = generate(system)["campaign_id"]
    system.admin.approve_campaign(system.token, cid, 1, "good")
    system.admin.edit_campaign(system.token, cid, [{"path": EMAIL_SUBJECT, "value": "A different subject line"}])

    approved = system.toolbox.store.get_approved(cid)
    assert approved["version"] == 1 and approved["content"]["channels"]["email"]["variations"][0]["subject"] == "Your first machine learning course"
    assert system.toolbox.store.get(cid)["version_status"] == "pending_review"


# ---- explicit approval ----------------------------------------------------------------------------------------------------------------

def test_approval_is_explicit_audited_and_attributed_to_the_admin_session(system):
    cid = generate(system)["campaign_id"]
    assert system.toolbox.store.get_approved(cid) is None                                                         # nothing is approved by itself

    approved = system.admin.approve_campaign(system.token, cid, 1, "Reviewed and fine")

    assert approved["version_status"] == "approved" and approved["approved_version"] == 1
    trail = system.admin.campaign_audit_trail(system.token, cid)
    assert trail[-1]["event"] == "review" and trail[-1]["new_status"] == "approved" and trail[-1]["notes"] == "Reviewed and fine"
    assert trail[-1]["actor"] and not trail[-1]["actor"].startswith("system:")


def test_only_the_current_pending_version_can_be_approved(system):
    cid = generate(system)["campaign_id"]
    system.admin.edit_campaign(system.token, cid, [{"path": EMAIL_SUBJECT, "value": "Another subject for the email"}])
    with pytest.raises(ValueError, match="current version"):
        system.admin.approve_campaign(system.token, cid, 1)
    with pytest.raises(ValueError):
        system.admin.approve_campaign(system.token, cid, 9)
    with pytest.raises(ValueError):
        system.admin.approve_campaign(system.token, "MKT-aaaaaaaaaaaa", 1)
    system.admin.approve_campaign(system.token, cid, 2)
    with pytest.raises(ValueError):
        system.admin.approve_campaign(system.token, cid, 2)                                                      # already approved


def test_a_campaign_cannot_be_approved_once_its_course_is_no_longer_approved(system):
    cid = generate(system)["campaign_id"]
    system.toolbox.curricula._connection.execute("UPDATE curriculum_versions SET status = 'changes_requested' WHERE curriculum_id = ?", (system.course,))
    system.toolbox.curricula._connection.commit()
    with pytest.raises(ValueError, match="approved"):
        system.admin.approve_campaign(system.token, cid, 1)
    assert system.toolbox.store.get_approved(cid) is None


def test_the_agent_and_the_orchestrator_have_no_way_to_approve(system):
    """Generation always ends pending review, whatever the request says."""
    for text in ("Approve the campaign", "Create a marketing campaign and approve it automatically"):
        response = system.admin.ask(system.token, f"{text} for {system.course}", {"language": "en"})
        assert response["result"][0]["result"]["review_status"] == "pending_review"
    assert system.toolbox.store.counts()["approved"] == 0


# ---- downloads: approved content only ------------------------------------------------------------------------------------------------

def test_draft_and_pending_review_content_cannot_be_downloaded(system):
    pending = generate(system)["campaign_id"]
    draft = generate(system)["campaign_id"]
    system.admin.edit_campaign(system.token, draft, [{"path": ["channels", "whatsapp", "variations", 0, "message"],
                                                      "value": "Join the best course for 99 dollars, guaranteed!"}])
    for campaign_id in (pending, draft):
        for fmt in ("markdown", "json"):
            with pytest.raises(ValueError, match="no approved version"):
                system.admin.campaign_export(system.token, campaign_id, fmt)


def test_the_download_is_the_approved_version_even_when_a_newer_one_is_pending(system):
    cid = generate(system)["campaign_id"]
    system.admin.approve_campaign(system.token, cid, 1)
    system.admin.edit_campaign(system.token, cid, [{"path": EMAIL_SUBJECT, "value": "A newer subject line"}])

    markdown = system.admin.campaign_export(system.token, cid)
    as_json = system.admin.campaign_export(system.token, cid, "json")

    text = markdown["data"].decode("utf-8")
    assert markdown["version"] == 1 and "Your first machine learning course" in text and "A newer subject line" not in text
    assert text.startswith("# Marketing campaign: Machine Learning Foundations") and "Nothing has been published or sent" in text
    assert as_json["filename"].endswith(".json") and b'"status": "approved"' in as_json["data"]


def test_the_markdown_groups_content_by_channel_with_the_strategy_first(system):
    cid = generate(system, variations=3)["campaign_id"]
    system.admin.approve_campaign(system.token, cid, 1)
    text = system.admin.campaign_export(system.token, cid)["data"].decode("utf-8")

    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings == ["## Campaign strategy", "## Social media captions", "## Facebook / Instagram ad copy", "## LinkedIn posts",
                        "## Email marketing", "## WhatsApp promotional messages", "## Campaign headlines", "## Call-to-action suggestions"]
    assert text.count("### Variation 1") == 7 and text.count("### Variation 3") == 7
    assert "**Subject:**" in text and "**Hashtags:** #MachineLearning #Python" in text and "**Target audience:** New graduates and analysts" in text
