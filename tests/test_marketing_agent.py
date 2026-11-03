"""
Agent 5 end to end with a fake model: registration, routing, approved-course retrieval, audience handling, the strategy,
all seven content types, variations, the one repair, failures that save nothing, languages, and conversation memory.
"""

import copy
from unittest.mock import MagicMock

import pytest

from agents.conversation_store import ConversationStore
from agents.marketing import schemas
from agents.marketing.marketing_agent import MarketingAgent
from agents.marketing.registration import register_marketing
from agents.orchestrator import Orchestrator
from agents.register_agents import create_admin_orchestrator
from tests.curriculum_fakes import FakeLLM
from tests.curriculum_fixtures import valid_curriculum
from tests.marketing_fakes import (
    ADMIN,
    SCHEMA_OF,
    TEXTS,
    approved_course,
    channel_answer,
    fake_llm,
    in_process_marketing_tools,
    make_campaign,
    save_curriculum,
)


@pytest.fixture
def toolbox(tmp_path):
    return in_process_marketing_tools(tmp_path)


def stored(toolbox, result):
    return toolbox.store.get(result["campaign_id"])


# ---- initialization and registration ------------------------------------------------------------------------------------

def test_the_agent_starts_nothing_until_it_is_used():
    agent = MarketingAgent()
    assert agent.AGENT_NAME == "marketing" and agent._tools is None and agent._llm is None and agent._graph is None
    agent.close()                                                                   # (closing an unused agent is fine)


def test_the_agent_registers_with_an_orchestrator_and_claims_open_questions():
    orchestrator = Orchestrator(excluded_agents=("rex_assistant",))
    agent = register_marketing(orchestrator, ConversationStore(db_path=":memory:"))

    assert orchestrator.agents["marketing"] == agent.process_request and "marketing" in orchestrator._continuation_checks
    assert orchestrator._continuation_checks["marketing"] == agent.wants_follow_up


def test_the_admin_orchestrator_includes_the_marketing_agent():
    rfp = MagicMock()
    rfp.AGENT_NAME = "rfp_research"
    orchestrator = create_admin_orchestrator(rfp_research_agent=rfp, conversation_store=ConversationStore(db_path=":memory:"))

    assert {"marketing", "document_generation", "curriculum_research", "rfp_research"} <= set(orchestrator.agents)
    assert orchestrator.marketing_agent.AGENT_NAME == "marketing"


# ---- routing -----------------------------------------------------------------------------------------------------------------

@pytest.fixture
def router():
    return Orchestrator()


@pytest.mark.parametrize("text, plan", [
    ("Write a marketing campaign", ["marketing"]),
    ("Create a social media campaign", ["marketing"]),
    ("Write a marketing campaign for CUR-0123456789ab", ["marketing"]),
    ("Write a marketing campaign for the curriculum CUR-0123456789ab", ["marketing"]),       # names an approved course: Agent 5 alone
    ("Prepare a linkedin post promoting CUR-0123456789ab", ["marketing"]),
    ("اكتب حملة تسويقية للمنهج CUR-0123456789ab", ["marketing"]),
    ("Create a curriculum and a marketing campaign", ["curriculum_research", "marketing"]),    # the earlier plan is unchanged
    ("Create a curriculum and marketing campaign", ["curriculum_research", "marketing"]),
    ("Create a course about SQL and a marketing campaign", ["curriculum_research", "marketing"]),
    ("Generate a participant guide for CUR-0123456789ab", ["document_generation"]),
    ("Find education tenders in Egypt", ["rfp_research"]),
])
def test_chat_routing(router, text, plan):
    assert router._make_plan(text) == plan


def test_the_marketing_tab_request_routes_to_agent_5_without_keyword_guessing(router):
    state = router._plan_node({"request": {"user_input": "hello there", "context": {"marketing_request": {"curriculum_id": "CUR-0123456789ab"}}}})
    assert state["plan"] == ["marketing"]
    other = router._plan_node({"request": {"user_input": "hello there", "context": {"marketing_request": "nope"}}})
    assert other["plan"] == []                                                       # only an object counts


def test_a_curriculum_then_marketing_request_ignores_the_unapproved_draft(toolbox):
    """The draft the Curriculum Agent just wrote is not approved: marketing must not use it and must ask for an approved course."""
    orchestrator = Orchestrator(excluded_agents=("rex_assistant",))
    curriculum = MagicMock(return_value={"request_id": "r", "agent_name": "curriculum_research", "status": "success",
                                         "result": {"kind": "draft", "curriculum_id": "CUR-aaaaaaaaaaaa", "title": "Machine Learning Foundations"}})
    orchestrator.register_agent("curriculum_research", curriculum)
    approved = approved_course(toolbox)
    agent = register_marketing(orchestrator, ConversationStore(db_path=":memory:"))
    agent._tools, agent._llm = toolbox, FakeLLM({})

    response = orchestrator.handle_request("Create a curriculum about machine learning foundations and a marketing campaign",
                                           {"conversation_id": "c1"})

    results = response["result"]
    assert [r["agent_name"] for r in results] == ["curriculum_research", "marketing"]
    mine = results[1]
    assert mine["status"] == "success" and mine["result"]["kind"] == "clarify"
    assert "not approved" in mine["result"]["message"] and approved in mine["result"]["message"]    # the approved ones are offered instead
    assert "CUR-aaaaaaaaaaaa" not in mine["result"]["message"] and toolbox.store.list() == []


# ---- retrieval of approved courses ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("state", ["pending", "draft", "changes"])
def test_a_course_that_is_not_approved_is_refused_and_nothing_is_written(toolbox, state):
    saved = save_curriculum(toolbox, valid=state != "draft")
    if state == "changes":
        toolbox.curricula.request_changes(saved["curriculum_id"], 1, ADMIN, "fix")
    llm = fake_llm()

    result = make_campaign(toolbox, saved["curriculum_id"], llm=llm)

    assert result["status"] == "error" and result["result"]["kind"] == "not_approved"
    assert "approve" in result["error_message"].lower() and llm.calls == [] and toolbox.store.list() == []


def test_an_unknown_or_malformed_course_id_is_reported(toolbox):
    assert make_campaign(toolbox, "CUR-aaaaaaaaaaaa")["result"]["kind"] == "not_found"
    assert make_campaign(toolbox, "nonsense")["status"] == "error"


def test_a_campaign_uses_the_approved_version_even_when_a_newer_revision_waits(toolbox):
    cid = approved_course(toolbox)
    revised = valid_curriculum()
    revised["title"] = "Revised and not approved"
    save_curriculum(toolbox, revised, curriculum_id=cid)

    result = make_campaign(toolbox, cid)["result"]

    assert result["curriculum_version"] == 1 and "newer version" in result["message"]
    assert stored(toolbox, result)["content"]["course_title"] == valid_curriculum()["title"]


def test_missing_course_information_is_reported_not_invented(toolbox):
    content = valid_curriculum()
    content["modules"] = []
    cid = approved_course(toolbox, content)
    llm = fake_llm()

    result = make_campaign(toolbox, cid, llm=llm)

    assert result["result"]["kind"] == "missing_information" and "modules" in result["result"]["missing"]
    assert llm.calls == [] and toolbox.store.list() == []


def test_chat_without_a_course_lists_the_approved_ones_and_none_approved_is_said(toolbox):
    agent = MarketingAgent(tools=toolbox, llm=fake_llm(), conversation_store=ConversationStore(db_path=":memory:"))
    ask = lambda text: agent.process_request({"request_id": "r", "user_input": text, "context": {"conversation_id": "c", "language": "en"}})

    none = ask("Write a marketing campaign")
    assert none["status"] == "error" and none["result"]["kind"] == "not_approved" and "No course is approved" in none["error_message"]

    first, second = approved_course(toolbox), approved_course(toolbox, {**valid_curriculum(), "title": "SQL for Analysts"})
    listed = ask("Write a marketing campaign")
    assert listed["status"] == "success" and listed["result"]["kind"] == "clarify"
    assert first in listed["result"]["message"] and second in listed["result"]["message"]


def test_chat_picks_the_course_by_name_by_id_or_by_number(toolbox):
    store = ConversationStore(db_path=":memory:")
    agent = MarketingAgent(tools=toolbox, llm=fake_llm(), conversation_store=store)
    ask = lambda text, conv="c": agent.process_request({"request_id": "r", "user_input": text, "context": {"conversation_id": conv, "language": "en"}})
    ml = approved_course(toolbox)
    approved_course(toolbox, {**valid_curriculum(), "title": "SQL for Analysts"})

    by_name = ask("Write a marketing campaign for the machine learning course", "n")
    assert by_name["status"] == "success" and by_name["result"]["kind"] == "campaign" and by_name["result"]["curriculum_id"] == ml
    by_id = ask(f"Write a linkedin post campaign for {ml}", "i")
    assert by_id["result"]["kind"] == "campaign"

    offered = ask("Write a marketing campaign", "x")
    assert offered["result"]["kind"] == "clarify" and agent.wants_follow_up({"user_input": "2", "context": {"conversation_id": "x"}})
    picked = ask("1", "x")
    assert picked["status"] == "success" and picked["result"]["kind"] == "campaign"
    assert store.get("x", "marketing")["pending"] is None and store.get("x", "marketing")["last_campaign_id"] == picked["result"]["campaign_id"]


def test_a_follow_up_is_only_claimed_while_a_question_is_open(toolbox):
    agent = MarketingAgent(tools=toolbox, llm=fake_llm(), conversation_store=ConversationStore(db_path=":memory:"))
    probe = {"user_input": "2", "context": {"conversation_id": "never-asked"}}
    assert agent.wants_follow_up(probe) is False and agent.wants_follow_up({"user_input": "", "context": {}}) is False


# ---- the audience -------------------------------------------------------------------------------------------------------------

def test_the_courses_own_audience_is_used_and_shown(toolbox):
    cid = approved_course(toolbox)
    llm = fake_llm()
    result = make_campaign(toolbox, cid, llm=llm)["result"]
    content = stored(toolbox, result)["content"]

    assert content["audience"] == {"text": "New graduates and analysts", "source": "curriculum"}
    assert content["strategy"]["target_audience"] == "New graduates and analysts"
    assert "Target audience (confirmed): New graduates and analysts" in llm.prompts("CampaignStrategy")[0]


def test_the_admin_can_override_the_audience_before_generation(toolbox):
    cid = approved_course(toolbox)
    llm = fake_llm()
    result = make_campaign(toolbox, cid, llm=llm, audience="Finance managers moving into analytics")["result"]
    content = stored(toolbox, result)["content"]

    assert content["audience"] == {"text": "Finance managers moving into analytics", "source": "admin"}
    assert content["strategy"]["target_audience"] == "Finance managers moving into analytics"
    assert all("Finance managers moving into analytics" in p for p in llm.prompts("EmailSet"))


def test_confirming_the_courses_audience_is_not_an_override(toolbox):
    cid = approved_course(toolbox)
    content = stored(toolbox, make_campaign(toolbox, cid, audience="New graduates and analysts")["result"])["content"]
    assert content["audience"]["source"] == "curriculum"


def test_the_model_cannot_change_the_audience(toolbox):
    cid = approved_course(toolbox)
    llm = fake_llm(CampaignStrategy={**fake_llm().answers["CampaignStrategy"][0], "target_audience": "Everyone"})
    assert stored(toolbox, make_campaign(toolbox, cid, llm=llm)["result"])["content"]["strategy"]["target_audience"] == "New graduates and analysts"


def test_a_missing_audience_is_asked_for_not_invented_and_the_answer_continues_the_request(toolbox):
    content = valid_curriculum()
    content["target_audience"] = ""
    cid = approved_course(toolbox, content)
    store = ConversationStore(db_path=":memory:")
    llm = fake_llm()
    agent = MarketingAgent(tools=toolbox, llm=llm, conversation_store=store)
    ask = lambda text: agent.process_request({"request_id": "r", "user_input": text, "context": {"conversation_id": "c", "language": "en"}})

    asked = ask(f"Write a marketing campaign for {cid}")

    assert asked["status"] == "success" and asked["result"]["kind"] == "need_audience" and asked["result"]["missing"] == ["audience"]
    assert "will not invent" in asked["result"]["message"] and llm.calls == [] and toolbox.store.list() == []
    assert agent.wants_follow_up({"user_input": "Accounting graduates", "context": {"conversation_id": "c"}})

    done = ask("Accounting graduates")

    assert done["result"]["kind"] == "campaign"
    assert stored(toolbox, done["result"])["content"]["audience"] == {"text": "Accounting graduates", "source": "admin"}
    assert not agent.wants_follow_up({"user_input": "thanks", "context": {"conversation_id": "c"}})   # the question is closed


def test_the_form_can_supply_the_missing_audience_directly(toolbox):
    content = valid_curriculum()
    content["target_audience"] = "   "
    cid = approved_course(toolbox, content)
    assert make_campaign(toolbox, cid)["result"]["kind"] == "need_audience"
    assert make_campaign(toolbox, cid, audience="Accounting graduates")["result"]["kind"] == "campaign"


# ---- the campaign, its strategy and the seven content types ---------------------------------------------------------------------

def test_a_campaign_has_a_strategy_and_every_content_type_in_pending_review(toolbox):
    cid = approved_course(toolbox)
    llm = fake_llm()
    response = make_campaign(toolbox, cid, llm=llm)
    result, record = response["result"], None

    assert response["status"] == "success" and result["kind"] == "campaign" and result["review_status"] == "pending_review"
    record = stored(toolbox, result)
    assert record["version_status"] == "pending_review" and toolbox.store.get_approved(result["campaign_id"]) is None   # never approved
    content = record["content"]
    assert list(content["channels"]) == list(schemas.CHANNELS) and result["channels"] == list(schemas.CHANNELS)
    strategy = content["strategy"]
    assert set(strategy) == {"objective", "target_audience", "main_message", "recommended_channels", "content_sequence", "call_to_action"}
    assert 3 <= len(strategy["content_sequence"]) <= 6 and set(strategy["recommended_channels"]) <= set(schemas.CHANNELS)
    assert llm.steps() == ["CampaignStrategy", *SCHEMA_OF]                           # strategy first, then one call per channel
    assert "Nothing has been published or sent" in result["message"] and result["validation"]["errors"] == []


@pytest.mark.parametrize("channel", schemas.CHANNELS)
def test_each_required_content_type_is_generated_with_its_own_structure(toolbox, channel):
    cid = approved_course(toolbox)
    result = make_campaign(toolbox, cid, channels=[channel])["result"]
    variations = stored(toolbox, result)["content"]["channels"][channel]["variations"]

    assert list(stored(toolbox, result)["content"]["channels"]) == [channel] and len(variations) == 2
    for variation in variations:
        for name, (low, high) in schemas.LIMITS[channel].items():
            assert low <= len(variation[name]) <= high, (channel, name)
    if channel == "social_captions":
        assert all(v["hashtags"] for v in variations)
    if channel == "email":
        assert all(v["subject"] and v["preheader"] and v["body"] and v["cta"] for v in variations)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_the_admin_chooses_one_two_or_three_variations(toolbox, count):
    cid = approved_course(toolbox)
    llm = fake_llm()
    result = make_campaign(toolbox, cid, llm=llm, variations=count)["result"]
    content = stored(toolbox, result)["content"]

    assert content["variations"] == count and result["variations"] == count
    assert all(len(body["variations"]) == count for body in content["channels"].values())
    assert all(f"Write exactly {count} variation(s)" in p for name in SCHEMA_OF for p in llm.prompts(name))


def test_two_variations_are_the_default_and_they_really_differ(toolbox):
    cid = approved_course(toolbox)
    content = stored(toolbox, make_campaign(toolbox, cid)["result"])["content"]
    assert content["variations"] == 2
    for channel, body in content["channels"].items():
        main = schemas.MAIN_FIELD[channel]
        assert body["variations"][0][main] != body["variations"][1][main]


def test_the_admin_can_generate_only_the_channels_they_need(toolbox):
    cid = approved_course(toolbox)
    llm = fake_llm()
    result = make_campaign(toolbox, cid, llm=llm, channels=["whatsapp", "email"])["result"]

    assert list(stored(toolbox, result)["content"]["channels"]) == ["email", "whatsapp"]       # in the standard order
    assert llm.steps() == ["CampaignStrategy", "EmailSet", "WhatsAppSet"]


@pytest.mark.parametrize("options, fragment", [
    ({"tone": "sarcastic"}, "tone"), ({"channels": ["tiktok"]}, "Channels"), ({"variations": 0}, "variations"),
    ({"variations": 4}, "variations"), ({"variations": "2"}, "variations"), ({"variations": True}, "variations"),
    ({"campaign_id": "nope"}, "campaign id"), ({"audience": "x" * 400}, "audience")])
def test_invalid_choices_are_refused_before_any_model_call(toolbox, options, fragment):
    cid = approved_course(toolbox)
    llm = fake_llm()
    result = make_campaign(toolbox, cid, llm=llm, **options)

    assert result["status"] == "error" and result["result"]["kind"] == "invalid" and fragment in result["error_message"]
    assert llm.calls == [] and toolbox.store.list() == []


def test_tone_goal_and_notes_reach_the_model_and_notes_are_a_source_of_facts(toolbox):
    cid = approved_course(toolbox)
    llm = fake_llm(WhatsAppSet=lambda prompt: {"variations": [
        {"message": "Hello! Our beginner machine learning course starts in March. Want to see the modules?"},
        {"message": "Hi, if you want to practise preparing data and evaluating a simple model, this course suits you. Shall I send the outline?"}]})

    result = make_campaign(toolbox, cid, llm=llm, tone="friendly", goal="Fill the spring group", notes="Starts in March.",
                           channels=["whatsapp"])["result"]

    assert result["kind"] == "campaign"                                               # "March" is allowed: the admin supplied it
    prompt = llm.prompts("WhatsAppSet")[0]
    assert "Tone: friendly" in prompt and "Fill the spring group" in prompt and "Starts in March." in prompt
    content = stored(toolbox, result)["content"]
    assert content["tone"] == "friendly" and content["goal"] == "Fill the spring group" and content["notes"] == "Starts in March."


def test_the_prompts_carry_only_the_approved_facts_and_the_no_invention_rules(toolbox):
    cid = approved_course(toolbox)
    llm = fake_llm()
    make_campaign(toolbox, cid, llm=llm)
    prompt = llm.prompts("EmailSet")[0]

    for fact in ("Machine Learning Foundations", "Total hours: 40", "Prepare a dataset for modelling", "Cleaning data", "Python"):
        assert fact in prompt
    for rule in ("Never invent or imply", "prices", "discounts", "dates", "job, salary or promotion promises", "Write exactly 2 variation"):
        assert rule in prompt


def test_progress_is_reported_step_by_step_and_a_failing_callback_changes_nothing(toolbox):
    cid = approved_course(toolbox)
    steps = []

    def progress(*step):
        steps.append(step)
        raise RuntimeError("the page went away")

    agent = MarketingAgent(tools=toolbox, llm=fake_llm())
    result = agent.process_request({"request_id": "r", "user_input": "x", "context": {
        "language": "en", "marketing_request": {"curriculum_id": cid, "channels": ["email", "ctas"]}, "progress": progress}})

    assert result["status"] == "success"
    assert [(d, t) for d, t, _ in steps] == [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert [label for *_, label in steps] == ["Campaign strategy", "Email marketing", "Call-to-action suggestions", ""]


# ---- the one repair and failures that save nothing ----------------------------------------------------------------------------------

BAD_WHATSAPP = {"variations": [
    {"message": "Hello! Join the best course today and get a free workshop and a certificate guaranteed."},
    {"message": "Hi, if you want to practise preparing data and evaluating a simple model, this course suits you. Shall I send the outline?"}]}


def test_a_failing_channel_is_repaired_once_with_the_exact_problems(toolbox):
    cid = approved_course(toolbox)
    llm = fake_llm(WhatsAppSet=[BAD_WHATSAPP, channel_answer("whatsapp")])

    response = make_campaign(toolbox, cid, llm=llm)

    assert response["status"] == "success"
    assert llm.steps().count("WhatsAppSet") == 2 and llm.steps().count("EmailSet") == 1        # only that channel was asked again
    repair = llm.prompts("WhatsAppSet")[1]
    assert "previous answer had these problems" in repair and "WhatsApp promotional messages, variation 1" in repair
    assert "superlative" in repair and "price" in repair and "certificate" in repair and "guarantee" in repair


def test_a_channel_that_still_fails_after_the_repair_saves_nothing_and_names_the_content_type(toolbox):
    cid = approved_course(toolbox)
    llm = fake_llm(WhatsAppSet=[BAD_WHATSAPP, BAD_WHATSAPP])

    response = make_campaign(toolbox, cid, llm=llm)

    result = response["result"]
    assert response["status"] == "error" and result["kind"] == "validation_failed" and result["failed_content"] == "whatsapp"
    assert "WhatsApp promotional messages did not pass the checks, even after one repair attempt" in response["error_message"]
    assert "Nothing was saved" in response["error_message"] and toolbox.store.list() == []
    assert llm.steps().count("WhatsAppSet") == 2                                          # one repair, never a third attempt
    assert "HeadlineSet" not in llm.steps() and "CtaSet" not in llm.steps()               # the rest was not written


def test_a_failing_strategy_is_named_and_saves_nothing(toolbox):
    cid = approved_course(toolbox)
    bad = {**fake_llm().answers["CampaignStrategy"][0], "objective": "Reach 100000 learners and double their salary"}
    response = make_campaign(toolbox, cid, llm=fake_llm(CampaignStrategy=[bad, bad]))

    assert response["result"]["failed_content"] == "strategy" and "Campaign strategy did not pass the checks" in response["error_message"]
    assert toolbox.store.list() == []


def test_the_wrong_number_of_variations_is_repaired_then_refused(toolbox):
    cid = approved_course(toolbox)
    one = {"variations": TEXTS["en"]["headlines"][:1]}
    repaired = make_campaign(toolbox, cid, llm=fake_llm(HeadlineSet=[one, channel_answer("headlines")]), channels=["headlines"])
    assert repaired["status"] == "success"
    failed = make_campaign(toolbox, cid, llm=fake_llm(HeadlineSet=[one, one]), channels=["headlines"])
    assert failed["result"]["kind"] == "validation_failed" and "exactly 2" in failed["error_message"]


def test_a_model_failure_saves_nothing(toolbox):
    cid = approved_course(toolbox)
    response = make_campaign(toolbox, cid, llm=fake_llm(AdCopySet=[RuntimeError("ollama is down")] * 3))
    assert response["result"]["kind"] == "generation_failed" and "Nothing was saved" in response["error_message"]
    assert toolbox.store.list() == []


def test_the_tool_server_being_down_is_an_answer_not_a_crash():
    broken = MagicMock()
    broken.get_approved_course.side_effect = ConnectionError("server down")
    agent = MarketingAgent(tools=broken, llm=fake_llm())
    response = agent.process_request({"request_id": "r", "user_input": "x", "context": {
        "language": "en", "marketing_request": {"curriculum_id": "CUR-0123456789ab"}}})
    assert response["status"] == "error" and "could not be completed" in response["error_message"]


def test_an_empty_request_and_a_malformed_form_are_refused(toolbox):
    agent = MarketingAgent(tools=toolbox, llm=fake_llm())
    assert agent.process_request({"request_id": "r", "user_input": "  ", "context": {}})["status"] == "error"
    assert agent.process_request({"request_id": "r", "user_input": "x", "context": {"marketing_request": "no"}})["error_message"] == \
        "marketing_request must be an object."


# ---- regeneration, versions and approval --------------------------------------------------------------------------------------

def test_regenerating_is_a_new_version_and_never_touches_the_approved_one(toolbox):
    cid = approved_course(toolbox)
    first = make_campaign(toolbox, cid)["result"]
    toolbox.approve_campaign(first["campaign_id"], 1, ADMIN)
    original = copy.deepcopy(toolbox.store.get_approved(first["campaign_id"])["content"])

    second = make_campaign(toolbox, cid, campaign_id=first["campaign_id"], tone="inspiring")["result"]

    assert second["campaign_id"] == first["campaign_id"] and second["version"] == 2 and second["review_status"] == "pending_review"
    assert toolbox.store.get(first["campaign_id"])["version_status"] == "pending_review"
    assert toolbox.store.get_approved(first["campaign_id"])["version"] == 1 and toolbox.store.get_approved(first["campaign_id"])["content"] == original
    assert [v["status"] for v in toolbox.store.versions(first["campaign_id"])] == ["approved", "pending_review"]
    assert toolbox.store.get(first["campaign_id"])["version_created_by"] == "system:marketing_agent"


def test_generation_can_never_produce_approved_content(toolbox):
    cid = approved_course(toolbox)
    for _ in range(3):
        result = make_campaign(toolbox, cid)["result"]
        assert result["review_status"] == "pending_review"
    assert toolbox.store.counts() == {"draft": 0, "pending_review": 3, "approved": 0}
    assert all(toolbox.store.get_approved(c["campaign_id"]) is None for c in toolbox.store.list())


# ---- languages ------------------------------------------------------------------------------------------------------------------

def test_an_arabic_course_gets_an_arabic_campaign_and_an_english_one_stays_english(toolbox):
    ar_id, en_id = approved_course(toolbox, valid_curriculum("ar")), approved_course(toolbox)
    arabic = make_campaign(toolbox, ar_id, "ar")
    english = make_campaign(toolbox, en_id, "en")

    ar_content, en_content = stored(toolbox, arabic["result"])["content"], stored(toolbox, english["result"])["content"]
    assert ar_content["language"] == "ar" and ar_content["audience"]["text"] == "الخريجون الجدد والمهتمون بالبيانات"
    assert "تمت كتابة حملة تسويقية" in arabic["result"]["message"] and "لم يُنشر ولم يُرسل شيء" in arabic["result"]["message"]
    assert en_content["language"] == "en" and "Marketing campaign written" in english["result"]["message"]


def test_the_campaign_follows_the_courses_language_not_the_interface_language(toolbox):
    """An Arabic course gets an Arabic campaign even when the admin's interface is English: nothing is translated."""
    cid = approved_course(toolbox, valid_curriculum("ar"))
    llm = fake_llm("ar")
    response = make_campaign(toolbox, cid, "en", llm=llm)
    content = stored(toolbox, response["result"])["content"]

    assert content["language"] == "ar" and "Marketing campaign written" in response["result"]["message"]     # the message follows the interface
    assert "Write every text value in Arabic (Modern Standard Arabic)" in llm.prompts("EmailSet")[0]


def test_arabic_errors_and_questions_are_in_arabic(toolbox):
    cid = approved_course(toolbox, valid_curriculum("ar"))
    bad = {"variations": [{"message": "مرحبًا! انضم إلى أفضل دورة اليوم بسعر خاص مع شهادة مضمونة."},
                          {"message": "أهلًا، إذا أردت التدرب على تجهيز البيانات وتقييم نموذج بسيط فقد تناسبك هذه الدورة. هل أرسل لك المخطط؟"}]}
    response = make_campaign(toolbox, cid, "ar", llm=fake_llm("ar", WhatsAppSet=[bad, bad]))

    assert response["result"]["failed_content"] == "whatsapp" and "رسائل واتساب الترويجية" in response["error_message"]
    assert "لم يُحفظ شيء" in response["error_message"] and toolbox.store.list() == []


# ---- found with the real model: the strategy's channels are not the model's job -------------------------------------------------------

def test_the_recommended_channels_are_the_administrators_selection_whatever_the_model_says(toolbox):
    cid = approved_course(toolbox)
    llm = fake_llm(CampaignStrategy={**fake_llm().answers["CampaignStrategy"][0], "recommended_channels": ["tiktok", "fax"]})

    result = make_campaign(toolbox, cid, llm=llm, channels=["email", "whatsapp"])["result"]

    assert result["kind"] == "campaign"
    assert stored(toolbox, result)["content"]["strategy"]["recommended_channels"] == ["email", "whatsapp"]
    assert "recommended_channels" not in schemas.CampaignStrategy.model_fields            # the model is not asked for machine keys


@pytest.mark.parametrize("language, names", [
    ("en", "Email marketing, WhatsApp promotional messages"), ("ar", "البريد التسويقي, رسائل واتساب الترويجية")])
def test_the_strategy_prompt_names_the_channels_in_the_courses_language(toolbox, language, names):
    cid = approved_course(toolbox, valid_curriculum(language))
    llm = fake_llm(language)
    make_campaign(toolbox, cid, language, llm=llm, channels=["email", "whatsapp"])
    listed = llm.prompts("CampaignStrategy")[0].split("Channels for this campaign:")[1].strip().splitlines()[0]

    assert listed == names
    if language == "ar":
        assert "whatsapp" not in listed.lower() and "email" not in listed.lower()                   # no English keys to copy into Arabic text


# ---- call to action: short imperative phrases, not benefit sentences (found with the real model) -------------------------------------

SENTENCES = {"variations": [{"text": "Start applying machine learning to real-world problems with our course."},
                            {"text": "Gain the skills to analyze and interpret data-driven insights effectively."}]}      # what the real model wrote


def cta_prompt_text():
    from agents.marketing import prompts

    return prompts.CHANNEL_TASKS["ctas"]


def test_the_cta_instructions_ask_for_short_imperative_phrases_with_examples():
    task = cta_prompt_text()

    for example in ('"Explore the course"', '"Join the first module"', '"See the full outline"'):
        assert example in task
    assert "short, imperative, action-oriented phrases" in task and "starting with a verb" in task
    assert "Do not write sentences that describe the course or its benefits" in task and "do not end with a full stop" in task
    assert "at most 8 words and at most 60 characters" in task                           # the limits are unchanged
    assert "No urgency or discount" in task


def test_the_first_cta_request_carries_the_examples_and_no_repair_text(toolbox):
    cid = approved_course(toolbox)
    llm = fake_llm()
    make_campaign(toolbox, cid, llm=llm, channels=["ctas"])
    first = llm.prompts("CtaSet")[0]

    assert "Explore the course" in first and "Join the first module" in first and "See the full outline" in first
    assert "This is what you wrote" not in first and "previous answer had these problems" not in first


def test_a_cta_repair_explains_the_distinction_and_quotes_the_offending_output(toolbox):
    cid = approved_course(toolbox)
    llm = fake_llm(CtaSet=[SENTENCES, channel_answer("ctas")])

    response = make_campaign(toolbox, cid, llm=llm, channels=["ctas"])

    assert response["status"] == "success" and llm.steps().count("CtaSet") == 2
    repair = llm.prompts("CtaSet")[1]
    assert "previous answer had these problems" in repair and "must have at most 8" in repair and "must have 3 to 60" in repair   # the checker's problems
    assert "A call to action is a short, imperative, action-oriented phrase" in repair
    assert "It is not a sentence that describes the course or what learners will gain" in repair and "has no full stop" in repair
    assert '"Start applying machine learning to real-world problems with our course." (10 words, 71 characters)' in repair      # quotes the output
    assert '"Gain the skills to analyze and interpret data-driven insights effectively." (10 words, 74 characters)' in repair
    assert "Rewrite each one as a short action phrase" in repair


def test_the_cta_note_is_only_added_to_the_call_to_action_repair(toolbox):
    cid = approved_course(toolbox)
    bad_whatsapp = {"variations": [{"message": "Hello! Join the best course today for a free workshop."},
                                   {"message": "Hi, if you want to practise preparing data and evaluating a simple model, this course suits you."}]}
    llm = fake_llm(WhatsAppSet=[bad_whatsapp, channel_answer("whatsapp")], CtaSet=[SENTENCES, channel_answer("ctas")])

    assert make_campaign(toolbox, cid, llm=llm, channels=["whatsapp", "ctas"])["status"] == "success"
    assert "previous answer had these problems" in llm.prompts("WhatsAppSet")[1]
    assert "A call to action is a short" not in llm.prompts("WhatsAppSet")[1] and "This is what you wrote" not in llm.prompts("WhatsAppSet")[1]
    assert "This is what you wrote" in llm.prompts("CtaSet")[1]


def test_a_cta_that_is_still_wrong_after_its_repair_saves_nothing_and_is_named(toolbox):
    cid = approved_course(toolbox)
    llm = fake_llm(CtaSet=[SENTENCES, SENTENCES])

    response = make_campaign(toolbox, cid, llm=llm)

    assert response["status"] == "error" and response["result"]["failed_content"] == "ctas"
    assert "Call-to-action suggestions did not pass the checks, even after one repair attempt" in response["error_message"]
    assert "Nothing was saved" in response["error_message"] and toolbox.store.list() == []
    assert llm.steps().count("CtaSet") == 2                                              # one repair, never a third attempt


def test_the_cta_limits_and_the_other_checks_are_unchanged(toolbox):
    cid = approved_course(toolbox)
    claims = {"variations": [{"text": "Get a free certificate now"}, {"text": "Book before March"}]}
    response = make_campaign(toolbox, cid, llm=fake_llm(CtaSet=[claims, claims]), channels=["ctas"])

    assert response["result"]["kind"] == "validation_failed"
    codes_found = {p["code"] for p in response["result"]["problems"]}
    assert {"price_claim", "certificate_claim", "date_claim"} <= codes_found
    assert schemas.LIMITS["ctas"]["text"] == (3, 60) and schemas.MAX_CTA_WORDS == 8
