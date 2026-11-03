"""
The admin "Marketing" tab (rex_ai_app.py, streamlit.testing AppTest): the real AdminService and real stores on the temporary
default paths set in tests/conftest.py; the agents are the fake admin orchestrator of tests/test_rex_ai_app.py. Campaigns are
made with the real agent and a fake model. These are automated UI tests, not a real browser.
"""

import pytest

from agents.curriculum_research.curriculum_store import CurriculumStore
from agents.marketing.marketing_store import MarketingStore
from agents.marketing.marketing_tools import MarketingToolbox
from tests.curriculum_fixtures import valid_curriculum
from tests.marketing_fakes import ADMIN, make_campaign
from tests.test_rex_ai_app import click, env, start, unlock  # noqa: F401  (env is a fixture)

AUDIENCE = "New graduates and analysts"
ALL_CHANNELS = ["social_captions", "ad_copy", "linkedin_posts", "email", "whatsapp", "headlines", "ctas"]


@pytest.fixture
def toolbox():
    """The toolbox on the (temporary) default paths the app's AdminService also uses."""
    return MarketingToolbox(store=MarketingStore(), curricula=CurriculumStore())


def approved(toolbox, content=None):
    saved = toolbox.curricula.save_version(content or valid_curriculum(), {"valid": True, "errors": []}, [])
    toolbox.curricula.approve(saved["curriculum_id"], 1, ADMIN)
    return saved["curriculum_id"]


def campaign(toolbox, **request):
    cid = approved(toolbox)
    return cid, make_campaign(toolbox, cid, **request)["result"]["campaign_id"]


def request_sent(env):
    return env["orchestrator"].handle_request.call_args.kwargs["context"]["marketing_request"]


def generate_button(at):
    return at.button[[b.label for b in at.button].index("Create campaign")]


# ---- the tab and the form ----------------------------------------------------------------------------------------------------------

def test_the_marketing_tab_is_there_and_empty_without_approved_courses(env):
    at = unlock(start())

    assert "Marketing" in [tab.label for tab in at.tabs]
    assert any("No course is approved yet" in i.value for i in at.info)
    assert any("No campaigns yet" in i.value for i in at.info)
    assert not [b for b in at.button if b.label == "Create campaign"]


def test_the_form_offers_the_approved_course_its_audience_and_the_defaults(env, toolbox):
    cid = approved(toolbox)
    toolbox.curricula.save_version(valid_curriculum(), {"valid": True, "errors": []}, [])                     # pending: not offered

    at = unlock(start())

    assert not at.exception
    courses = [s for s in at.selectbox if s.key == "mk_course"][0]
    assert len(courses.options) == 1 and cid in str(courses.options[0])
    assert at.text_input(key=f"mk_audience_{cid}").value == AUDIENCE                      # the course's own audience, to confirm or change
    assert at.selectbox(key="mk_tone").value == "professional" and at.selectbox(key="mk_variations").value == 2
    assert at.multiselect(key="mk_channels").value == ALL_CHANNELS                        # all seven content types
    assert any("not translated" in c.value for c in at.caption)


def test_generating_asks_the_marketing_agent_with_the_choices_and_the_confirmed_audience(env, toolbox):
    cid = approved(toolbox)
    env["orchestrator"].handle_request.return_value = {"status": "success", "result": [{
        "agent_name": "marketing", "status": "success", "result": {"kind": "campaign", "message": "Marketing campaign written."}}]}

    at = unlock(start())
    generate_button(at).click().run()

    assert not at.exception, at.exception
    assert request_sent(env) == {"curriculum_id": cid, "audience": AUDIENCE, "tone": "professional", "variations": 2,
                                 "channels": ALL_CHANNELS, "goal": None, "notes": None, "campaign_id": None}
    assert env["orchestrator"].handle_request.call_args.kwargs["context"]["conversation_id"]
    assert any("Marketing campaign written" in s.value for s in at.success)               # kept after the redraw


def test_the_admin_can_override_the_audience_and_choose_variations_tone_and_content(env, toolbox):
    cid = approved(toolbox)
    env["orchestrator"].handle_request.return_value = {"status": "success", "result": [{
        "agent_name": "marketing", "status": "success", "result": {"kind": "campaign", "message": "Done."}}]}

    at = unlock(start())
    at.text_input(key=f"mk_audience_{cid}").input("Finance managers moving into analytics")
    at.selectbox(key="mk_tone").select("friendly")
    at.selectbox(key="mk_variations").select(3)
    at.multiselect(key="mk_channels").set_value(["email", "whatsapp"])
    at.text_input(key="mk_goal").input("Fill the spring group")
    at.text_input(key="mk_notes").input("Starts in March")
    generate_button(at).click().run()

    assert not at.exception, at.exception
    assert request_sent(env) == {"curriculum_id": cid, "audience": "Finance managers moving into analytics", "tone": "friendly",
                                 "variations": 3, "channels": ["email", "whatsapp"], "goal": "Fill the spring group",
                                 "notes": "Starts in March", "campaign_id": None}


def test_choosing_no_content_is_refused_before_asking_the_agent(env, toolbox):
    approved(toolbox)
    at = unlock(start())
    at.multiselect(key="mk_channels").set_value([])
    generate_button(at).click().run()

    assert any("Choose at least one kind of content" in w.value for w in at.warning)
    assert not env["orchestrator"].handle_request.called


def test_a_course_without_an_audience_asks_the_admin_for_one(env, toolbox):
    content = valid_curriculum()
    content["target_audience"] = ""
    cid = approved(toolbox, content)
    at = unlock(start())

    assert at.text_input(key=f"mk_audience_{cid}").value == ""
    assert any("does not state its target audience" in c.value for c in at.caption)


def test_an_agent_question_or_error_is_shown_not_swallowed(env, toolbox):
    approved(toolbox)
    env["orchestrator"].handle_request.return_value = {"status": "error", "error_message": "x", "result": [{
        "agent_name": "marketing", "status": "error", "error_message": "WhatsApp promotional messages did not pass the checks, "
        "even after one repair attempt. Nothing was saved: ...", "result": None}]}
    at = unlock(start())
    generate_button(at).click().run()

    assert not at.exception
    assert any("did not pass the checks" in e.value and "Nothing was saved" in e.value for e in at.error)


# ---- review, edit, approve, download ------------------------------------------------------------------------------------------------

def test_a_new_campaign_shows_its_strategy_variations_and_waits_for_review(env, toolbox):
    cid, campaign_id = campaign(toolbox)
    at = unlock(start())

    assert not at.exception, at.exception
    assert any(campaign_id in m.value and "Pending review" in m.value for m in at.markdown)
    assert any(AUDIENCE in m.value for m in at.markdown)
    assert at.text_area(key=f"mk_{campaign_id}_1_strategy_objective").value.startswith("Attract new graduates")
    subject = at.text_area(key=f"mk_{campaign_id}_1_channels_email_variations_0_subject")
    other = at.text_area(key=f"mk_{campaign_id}_1_channels_email_variations_1_subject")
    assert subject.value == "Your first machine learning course" and other.value != subject.value        # variations side by side
    assert any(h in " ".join(m.value for m in at.markdown) for h in ("Social media captions", "WhatsApp promotional messages"))
    assert at.button(key=f"mk_approve_{campaign_id}")                                                      # approval is a separate, explicit step
    assert any("Nothing to download yet" in c.value for c in at.caption)                                    # pending content is not downloadable
    assert not at.get("download_button")


def test_editing_saves_a_new_version_that_needs_review_again(env, toolbox):
    cid, campaign_id = campaign(toolbox)
    at = unlock(start())
    at.text_area(key=f"mk_{campaign_id}_1_channels_email_variations_0_subject").input("A reviewed subject line")
    at.button(key=f"mk_save_{campaign_id}").click().run()

    assert not at.exception, at.exception
    stored = toolbox.store.get(campaign_id)
    assert stored["current_version"] == 2 and stored["version_status"] == "pending_review"
    assert stored["content"]["channels"]["email"]["variations"][0]["subject"] == "A reviewed subject line"
    assert toolbox.store.get(campaign_id, 1)["content"]["channels"]["email"]["variations"][0]["subject"] == "Your first machine learning course"
    assert any("Saved as a new version" in s.value for s in at.success)


def test_saving_without_changes_saves_nothing(env, toolbox):
    cid, campaign_id = campaign(toolbox)
    at = unlock(start())
    at.button(key=f"mk_save_{campaign_id}").click().run()

    assert not at.exception and any("No changes to save" in i.value for i in at.info)
    assert toolbox.store.get(campaign_id)["current_version"] == 1


def test_an_edit_with_an_unsupported_claim_becomes_a_draft_and_cannot_be_approved(env, toolbox):
    cid, campaign_id = campaign(toolbox)
    at = unlock(start())
    at.text_area(key=f"mk_{campaign_id}_1_channels_whatsapp_variations_0_message").input("Join the best course for only 99 dollars, guaranteed!")
    at.button(key=f"mk_save_{campaign_id}").click().run()

    assert not at.exception, at.exception
    stored = toolbox.store.get(campaign_id)
    assert stored["version_status"] == "draft"
    assert any("does not pass the checks" in s.value for s in [*at.success, *at.warning, *at.info])
    assert any("Problems that must be fixed" in w.value for w in at.warning)
    assert not [b for b in at.button if b.key == f"mk_approve_{campaign_id}"]                                # nothing to approve
    assert any("Only the current version, while it is pending review" in c.value for c in at.caption)


def test_approving_is_explicit_and_then_the_approved_version_can_be_downloaded(env, toolbox):
    cid, campaign_id = campaign(toolbox)
    at = unlock(start())
    assert toolbox.store.get_approved(campaign_id) is None                                                   # opening the tab approves nothing

    at.text_area(key=f"mk_notes_{campaign_id}").input("Checked with marketing")
    at.button(key=f"mk_approve_{campaign_id}").click().run()

    assert not at.exception, at.exception
    approved_record = toolbox.store.get_approved(campaign_id)
    assert approved_record["version"] == 1 and toolbox.store.get(campaign_id)["status"] == "approved"
    assert toolbox.store.audit_trail(campaign_id)[-1]["notes"] == "Checked with marketing"
    assert any("Nothing was published or sent" in s.value for s in at.success)
    assert len(at.get("download_button")) == 2                                                               # Markdown and JSON, approved version only
    assert not [b for b in at.button if b.key == f"mk_approve_{campaign_id}"]


def test_a_newer_pending_version_never_replaces_the_approved_download(env, toolbox):
    cid, campaign_id = campaign(toolbox)
    toolbox.approve_campaign(campaign_id, 1, ADMIN)
    at = unlock(start())
    at.text_area(key=f"mk_{campaign_id}_1_channels_email_variations_0_subject").input("A newer pending subject")
    at.button(key=f"mk_save_{campaign_id}").click().run()

    assert not at.exception, at.exception
    assert toolbox.store.get(campaign_id)["version_status"] == "pending_review" and toolbox.store.get_approved(campaign_id)["version"] == 1
    assert any("approved version 1 only" in m.value for m in at.markdown)
    assert any("version 1 is approved" in e.label for e in at.expander)
    assert len(at.get("download_button")) == 2


def test_regenerating_asks_the_agent_for_a_new_version_with_the_same_choices(env, toolbox):
    cid, campaign_id = campaign(toolbox, tone="inspiring", variations=3, channels=["email", "ctas"], goal="Fill a group")
    env["orchestrator"].handle_request.return_value = {"status": "success", "result": [{
        "agent_name": "marketing", "status": "success", "result": {"kind": "campaign", "message": "New version."}}]}
    at = unlock(start())
    at.button(key=f"mk_regen_{campaign_id}").click().run()

    assert not at.exception, at.exception
    assert request_sent(env) == {"curriculum_id": cid, "audience": AUDIENCE, "tone": "inspiring", "variations": 3,
                                 "channels": ["email", "ctas"], "goal": "Fill a group", "notes": None, "campaign_id": campaign_id}


def test_the_list_can_be_filtered_by_status(env, toolbox):
    cid, first = campaign(toolbox)
    second = make_campaign(toolbox, cid)["result"]["campaign_id"]
    toolbox.approve_campaign(second, 1, ADMIN)
    at = unlock(start())
    at.selectbox(key="mk_filter").select("approved").run()

    assert not at.exception
    texts = " ".join(m.value for m in at.markdown)
    assert second in texts and first not in texts


def test_the_tab_is_available_in_arabic_with_an_arabic_campaign(env, toolbox):
    cid = approved(toolbox, valid_curriculum("ar"))
    campaign_id = make_campaign(toolbox, cid, "ar")["result"]["campaign_id"]
    at = start()
    at.session_state["lang"] = "ar"
    unlock(at)

    assert not at.exception, at.exception
    assert "التسويق" in [tab.label for tab in at.tabs]
    assert any("الجمهور المستهدف" in m.value for m in at.markdown) and any(campaign_id in m.value for m in at.markdown)
    assert any("لم يُنشر ولم يُرسل شيء" in c.value or "لا يُنشر" in c.value for c in at.caption)


def test_nothing_in_the_tab_can_send_publish_or_email(env, toolbox):
    cid, campaign_id = campaign(toolbox)
    toolbox.approve_campaign(campaign_id, 1, ADMIN)
    at = unlock(start())
    buttons = [b for b in at.button if str(b.key or "").startswith("mk_")]          # (the other tabs have their own buttons)
    assert buttons
    labels = " ".join(b.label.lower() for b in buttons)

    assert not any(word in labels for word in ("publish", "send", "post now", "schedule", "e-mail now", "email now"))
