"""
The Marketing agent's open questions ("which course?", "who is this campaign for?"): tied to the request that asked, forgotten
after a fixed time, answered only by a real answer, never by an unrelated message, and isolated per conversation. The clock is
injected, so expiry is deterministic.
"""

import pytest

from agents.conversation_store import ConversationStore
from agents.marketing.marketing_agent import PENDING_TTL_SECONDS, MarketingAgent
from agents.marketing.registration import register_marketing
from agents.marketing.understanding import classify_reply
from agents.orchestrator import Orchestrator
from tests.curriculum_fixtures import valid_curriculum
from tests.marketing_fakes import approved_course, fake_llm, in_process_marketing_tools


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def world(tmp_path):
    toolbox = in_process_marketing_tools(tmp_path)
    content = valid_curriculum()
    content["target_audience"] = ""                                          # the course states no audience
    s = type("World", (), {})()
    s.toolbox, s.clock, s.store, s.llm = toolbox, Clock(), ConversationStore(db_path=":memory:"), fake_llm()
    s.course = approved_course(toolbox, content)
    s.agent = MarketingAgent(tools=toolbox, llm=s.llm, conversation_store=s.store, clock=s.clock)
    return s


def ask(world, text, conversation="c", language="en"):
    return world.agent.process_request({"request_id": "r", "user_input": text, "context": {"conversation_id": conversation, "language": language}})


def claims(world, text, conversation="c"):
    return world.agent.wants_follow_up({"user_input": text, "context": {"conversation_id": conversation}})


def pending(world, conversation="c"):
    return (world.store.get(conversation, "marketing") or {}).get("pending")


def open_the_question(world, conversation="c", language="en"):
    asked = ask(world, f"Write a marketing campaign for {world.course}", conversation, language)
    assert asked["status"] == "success" and asked["result"]["kind"] == "need_audience", asked
    return asked


# ---- classifying a reply -----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, kind", [
    ("Accounting graduates", "audience"), ("new graduates in accounting", "audience"), ("Students", "audience"),
    ("Help desk staff", "audience"), ("Start-up founders", "audience"), ("خريجون جدد في المحاسبة", "audience"),
    ("cancel", "cancel"), ("Never mind", "cancel"), ("stop.", "cancel"), ("إلغاء", "cancel"), ("لا", "cancel"),
    ("", "empty"), ("   ", "empty"), ("12345", "empty"), ("!!!", "empty"),
    ("ok", "small_talk"), ("Thanks!", "small_talk"), ("hello", "small_talk"), ("شكرا", "small_talk"), ("تمام", "small_talk"),
    ("What courses do you have?", "new_request"), ("Find tenders in Egypt", "new_request"), ("Create a curriculum about SQL", "new_request"),
    ("Write a marketing campaign for CUR-0123456789ab", "new_request"), ("هل يمكنك مساعدتي", "new_request"), ("اكتب حملة", "new_request"),
    ("everyone", "ambiguous"), ("Anyone", "ambiguous"), ("الجميع", "ambiguous"),
    ("see https://example.com now", "invalid"), ("contact a@b.com team", "invalid"), ("<script>x</script> people", "invalid"),
    ("word " * 20, "invalid"), ("a" * 400, "invalid"),
])
def test_classifying_a_reply(text, kind):
    assert classify_reply(text)[0] == kind


# ---- a valid answer ---------------------------------------------------------------------------------------------------------------------------------

def test_the_question_is_tied_to_the_request_and_stamped_with_the_time(world):
    open_the_question(world)
    state = pending(world)

    assert state["awaiting"] == "audience" and state["curriculum_id"] == world.course
    assert state["request_text"] == f"Write a marketing campaign for {world.course}" and state["created_at"] == world.clock.now
    assert world.llm.calls == [] and world.toolbox.store.list() == []                       # nothing is written before the answer


def test_a_valid_audience_answer_continues_the_original_request_and_only_then_generates(world):
    open_the_question(world)
    assert claims(world, "Accounting graduates") is True
    assert world.llm.calls == []

    done = ask(world, "Accounting graduates")

    assert done["status"] == "success" and done["result"]["kind"] == "campaign" and done["result"]["curriculum_id"] == world.course
    assert world.toolbox.store.get(done["result"]["campaign_id"])["content"]["audience"] == {"text": "Accounting graduates", "source": "admin"}
    assert pending(world) is None and world.llm.calls                                       # the question is closed once answered
    assert claims(world, "Accounting graduates") is False


def test_an_arabic_answer_works_too(world):
    open_the_question(world, language="ar")
    assert claims(world, "خريجون جدد في المحاسبة")
    done = ask(world, "خريجون جدد في المحاسبة", language="ar")
    assert done["result"]["kind"] == "campaign"


# ---- replies that are not an answer ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["thanks", "ok", "hello", "شكرا", "What courses do you have?", "Find tenders in Egypt", "", "   "])
def test_an_unrelated_message_is_never_taken_as_the_audience(world, text):
    open_the_question(world)

    assert claims(world, text) is False                                                      # the orchestrator will not hand it over
    assert world.toolbox.store.list() == [] and world.llm.calls == []


@pytest.mark.parametrize("text", ["thanks", "ok", "hello"])
def test_even_if_it_is_handed_over_an_acknowledgement_only_asks_again(world, text):
    open_the_question(world)
    again = ask(world, text)

    assert again["status"] == "success" and again["result"]["kind"] == "need_audience" and "Who is this campaign for" in again["result"]["message"]
    assert world.llm.calls == [] and world.toolbox.store.list() == [] and pending(world)["awaiting"] == "audience"


@pytest.mark.parametrize("text, claimed", [("everyone", True), ("الجميع", True), ("see https://example.com now", True), ("word " * 20, True)])
def test_an_ambiguous_or_invalid_audience_is_asked_again_and_the_question_stays_open(world, text, claimed):
    open_the_question(world)
    original = pending(world)["created_at"]
    assert claims(world, text) is claimed

    again = ask(world, text)

    assert again["result"]["kind"] == "need_audience" and again["result"]["missing"] == ["audience"]
    assert world.llm.calls == [] and world.toolbox.store.list() == []
    assert pending(world)["created_at"] == original                                          # asking again does not extend the time limit
    assert ask(world, "Accounting graduates")["result"]["kind"] == "campaign"                 # and a real answer still works


def test_the_form_refuses_an_audience_with_a_link_or_markup(world):
    from tests.marketing_fakes import make_campaign

    for bad in ("see https://example.com", "<b>students</b>", "mail me at a@b.com"):
        result = make_campaign(world.toolbox, world.course, audience=bad)
        assert result["status"] == "error" and result["result"]["kind"] == "invalid" and "without links" in result["error_message"]
    assert world.toolbox.store.list() == []


# ---- expiry ---------------------------------------------------------------------------------------------------------------------------------------------

def test_the_question_works_until_the_time_limit_and_not_after(world):
    open_the_question(world)
    world.clock.now += PENDING_TTL_SECONDS - 1
    assert claims(world, "Accounting graduates") is True

    world.clock.now += 2
    assert claims(world, "Accounting graduates") is False


def test_an_expired_question_never_starts_a_campaign(world):
    open_the_question(world)
    world.clock.now += PENDING_TTL_SECONDS + 60

    late = ask(world, "Accounting graduates")

    assert late["result"]["kind"] == "clarify" or late["status"] == "error"                   # read as a fresh message: "which course?" / none
    assert world.toolbox.store.list() == [] and world.llm.calls == []
    assert "Accounting graduates" not in str(world.store.get("c", "marketing") or "")           # and it was not kept as the audience


def test_a_question_without_a_time_stamp_is_treated_as_expired(world):
    world.store.set("c", "marketing", {"pending": {"awaiting": "audience", "curriculum_id": world.course, "options": {}}})
    assert claims(world, "Accounting graduates") is False
    assert ask(world, "Accounting graduates")["result"]["kind"] != "campaign"
    assert world.toolbox.store.list() == []


def test_the_course_question_expires_too(world):
    other = approved_course(world.toolbox, valid_curriculum())
    asked = ask(world, "Write a marketing campaign")
    assert asked["result"]["kind"] == "clarify" and claims(world, "1")
    world.clock.now += PENDING_TTL_SECONDS + 1
    assert claims(world, "1") is False and claims(world, other) is False


# ---- a new request or a cancellation while waiting -------------------------------------------------------------------------------------------------

def test_a_new_marketing_request_while_waiting_replaces_the_question_and_does_not_reuse_the_old_one(world):
    other = approved_course(world.toolbox, valid_curriculum())                                 # this course states its own audience
    open_the_question(world)

    new = ask(world, f"Write a marketing campaign for {other}")

    assert new["result"]["kind"] == "campaign" and new["result"]["curriculum_id"] == other
    assert world.toolbox.store.get(new["result"]["campaign_id"])["content"]["audience"]["source"] == "curriculum"
    assert pending(world) is None and claims(world, "Accounting graduates") is False


@pytest.mark.parametrize("text, language", [("cancel", "en"), ("Never mind", "en"), ("stop", "en"), ("إلغاء", "ar")])
def test_cancelling_closes_the_question_and_creates_nothing(world, text, language):
    open_the_question(world, language=language)
    assert claims(world, text) is True

    done = ask(world, text, language=language)

    assert done["status"] == "success" and done["result"]["kind"] == "cancelled"
    assert ("cancelled" in done["result"]["message"]) or ("إلغاء" in done["result"]["message"])
    assert pending(world) is None and world.toolbox.store.list() == [] and world.llm.calls == []
    assert claims(world, "Accounting graduates") is False                                      # the later message starts nothing


def test_the_course_question_can_be_cancelled(world):
    approved_course(world.toolbox, valid_curriculum())
    ask(world, "Write a marketing campaign")
    assert pending(world)["awaiting"] == "course" and claims(world, "cancel")
    assert ask(world, "cancel")["result"]["kind"] == "cancelled" and pending(world) is None


# ---- conversations stay apart ------------------------------------------------------------------------------------------------------------------------------

def test_another_conversation_can_neither_answer_nor_be_affected(world):
    open_the_question(world, "conv-a")

    assert claims(world, "Accounting graduates", "conv-b") is False
    other = ask(world, "Accounting graduates", "conv-b")
    assert other["result"]["kind"] != "campaign"                                                # read as a fresh message of its own
    assert (pending(world, "conv-b") or {}).get("awaiting") != "audience"                       # never conv-a's audience question
    assert pending(world, "conv-a")["awaiting"] == "audience" and claims(world, "Accounting graduates", "conv-a") is True
    assert world.toolbox.store.list() == []


def test_two_open_questions_in_two_conversations_are_answered_separately(world):
    open_the_question(world, "conv-a")
    open_the_question(world, "conv-b")

    a = ask(world, "Accounting graduates", "conv-a")
    assert a["result"]["kind"] == "campaign" and pending(world, "conv-a") is None and pending(world, "conv-b")["awaiting"] == "audience"
    b = ask(world, "Nursing students", "conv-b")
    assert world.toolbox.store.get(b["result"]["campaign_id"])["content"]["audience"]["text"] == "Nursing students"
    assert world.toolbox.store.get(a["result"]["campaign_id"])["content"]["audience"]["text"] == "Accounting graduates"


# ---- the Marketing tab and the orchestrator ---------------------------------------------------------------------------------------------------------------

def test_the_form_does_not_leave_a_chat_question_open(world):
    from tests.marketing_fakes import make_campaign

    result = make_campaign(world.toolbox, world.course)                                        # (conversation "c", no audience in the form)
    assert result["result"]["kind"] == "need_audience"

    agent_state = world.store.get("c", "marketing")
    assert agent_state is None                                                                  # (make_campaign uses its own agent without a store)
    form = world.agent.process_request({"request_id": "r", "user_input": "x", "context": {
        "conversation_id": "c", "language": "en", "marketing_request": {"curriculum_id": world.course}}})
    assert form["result"]["kind"] == "need_audience" and pending(world) is None and claims(world, "Accounting graduates") is False


def test_an_unrelated_message_through_the_orchestrator_starts_nothing(world):
    orchestrator = Orchestrator(excluded_agents=("rex_assistant",))
    agent = register_marketing(orchestrator, world.store)
    agent._tools, agent._llm, agent._clock = world.toolbox, world.llm, world.clock
    first = orchestrator.handle_request(f"Write a marketing campaign for {world.course}", {"conversation_id": "o1"})
    assert first["result"][0]["result"]["kind"] == "need_audience"

    stray = orchestrator.handle_request("thanks", {"conversation_id": "o1"})

    assert stray["status"] == "error" and stray["result"][0]["agent_name"] == "orchestrator"
    assert world.toolbox.store.list() == [] and world.llm.calls == []
    answer = orchestrator.handle_request("Accounting graduates", {"conversation_id": "o1"})
    assert answer["result"][0]["agent_name"] == "marketing" and answer["result"][0]["result"]["kind"] == "campaign"
