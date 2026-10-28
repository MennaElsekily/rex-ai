"""
Tests for agents.rex_assistant (Agent 1 / REX Assistant).

Mocked: the chat model (_get_llm) and the MCP client, so these tests
check REX's own graph logic -- mode selection, discovery, answer
writing from evidence, consent, language and error handling -- without
Ollama, the embedding models, or the MCP server subprocess. Live checks
against real services are run and reported separately.
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.rex_assistant.graph import _detect_consent, build_graph
from agents.rex_assistant.interface import RexAssistantAgent
from agents.rex_assistant.lead_store import LeadStore


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Profile extraction gets `profile_json`; answer writing gets `answer`."""

    def __init__(self, profile_json="{}", answer="Grounded answer [1]."):
        self.profile_json = profile_json
        self.answer = answer
        self.extraction_calls = []
        self.answer_calls = []

    def invoke(self, messages):
        last = messages[-1].content
        if "Extract any of these fields" in last:
            self.extraction_calls.append(messages)
            return FakeMessage(self.profile_json)
        self.answer_calls.append(messages)
        if isinstance(self.answer, Exception):
            raise self.answer
        return FakeMessage(self.answer)


class FakeLeadStore(LeadStore):
    def __init__(self):
        self.saved = []

    def save_lead(self, lead):
        self.saved.append(lead)
        return True


def _evidence(status="ok", chunks=None, **extra):
    evidence = {
        "status": status,
        "question": "q",
        "retrieval_query": "q",
        "query_language": "en",
        "translation": "not_needed",
        "translation_error": None,
        "chunks": [] if chunks is None else chunks,
        "error": None,
    }
    evidence.update(extra)
    return evidence


CHUNK = {
    "reference": 1,
    "text": "Entry requirements: basic computer literacy. Duration: 60 hours.",
    "source": "04_Flutter.pdf",
    "page": 3,
    "section": "Entry requirements",
    "chunk_id": 12,
    "relevance_score": 1.2,
}

FULL_PROFILE = {
    "background": "accountant",
    "career_goal": "data analyst",
    "skill_level": "beginner",
    "training_preference": "online",
    "budget": "5000 EGP",
    "available_time": "5 hours a week",
}


@pytest.fixture
def llm():
    fake = FakeLLM()
    with patch("agents.rex_assistant.graph._get_llm", return_value=fake):
        yield fake


@pytest.fixture
def mcp():
    client = MagicMock()
    client.search_knowledge_base.return_value = _evidence(chunks=[CHUNK])
    return client


@pytest.fixture
def leads():
    return FakeLeadStore()


@pytest.fixture
def agent(llm, mcp, leads):
    return RexAssistantAgent(lead_store=leads, mcp_client=mcp)


def _converse(agent, *messages, state=None):
    response = None
    for message in messages:
        response = agent.handle_message(message, state=state)
        state = response["state"]
    return response


# --------------------------------------------------------------------
# Mode selection
# --------------------------------------------------------------------

@pytest.mark.parametrize("message", ["Hello", "أهلاً", "Thanks!"])
def test_greetings_do_not_start_discovery_or_search(message, agent, mcp):
    result = _converse(agent, message)["result"]

    assert result["mode"] is None
    assert result["pending_field"] is None
    mcp.search_knowledge_base.assert_not_called()


def test_unrelated_question_is_checked_against_the_evidence_not_answered(agent, mcp):
    # "How much is a pizza" cannot be told apart from a course follow-up
    # ("How much does it cost?") by keywords, so the evidence decides:
    # nothing relevant -> the fixed "not found" message, no discovery.
    mcp.search_knowledge_base.return_value = _evidence(status="no_results", chunks=[])

    result = _converse(agent, "How much is a pizza")["result"]

    assert result["mode"] is None
    assert result["answer_status"] == "no_results"
    assert "couldn't find" in result["answer"]


@pytest.mark.parametrize("message", [
    "What courses are available?",
    "What does the Python course cover?",
    "What are the prerequisites for this course?",
    "ما هي الدورات المتاحة؟",
    "مدة الكورس كام؟",
])
def test_course_questions_are_answered_without_discovery(message, agent, mcp, llm):
    result = _converse(agent, message)["result"]

    assert result["mode"] is None
    assert result["answer_status"] == "answered"
    assert result["answer"] == "Grounded answer [1]."
    assert result["sources"][0]["source"] == "04_Flutter.pdf"
    kwargs = mcp.search_knowledge_base.call_args.kwargs
    assert kwargs["question"] == message          # original question kept
    assert llm.extraction_calls == []             # no profile extraction


@pytest.mark.parametrize("message", [
    "Recommend a course for me.",
    "I want to become a data analyst. What should I study?",
    "Can you recommend a course based on my background?",
    "رشحلي كورسات مناسبة ليا.",
    "Ana 3ayez akhod course fel programming.",
])
def test_recommendation_requests_start_discovery(message, agent, mcp):
    result = _converse(agent, message)["result"]

    assert result["mode"] == "discovery"
    assert result["pending_field"] == "background"
    mcp.search_knowledge_base.assert_not_called()


# --------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------

def test_questions_are_asked_one_at_a_time_in_order(agent):
    response = _converse(agent, "Recommend a course for me")
    asked = [response["result"]["pending_field"]]

    for answer in ("accountant", "data analyst", "beginner", "online", "5000 EGP"):
        response = _converse(agent, answer, state=response["state"])
        asked.append(response["result"]["pending_field"])

    assert asked == [
        "background", "career_goal", "skill_level",
        "training_preference", "budget", "available_time",
    ]
    assert response["result"]["answer"].endswith(
        "How much time can you dedicate to learning?"
    )


@pytest.mark.parametrize("language, opener, answers, expected_question", [
    ("en", "Recommend a course for me",
     [], "What is your educational or professional background?"),
    ("ar", "رشحلي كورسات مناسبة ليا",
     [], "ما هو مؤهلك الدراسي أو خلفيتك المهنية؟"),
    ("ar", "رشحلي كورسات مناسبة ليا",
     ["محاسب", "محلل بيانات", "مبتدئ"], "هل تفضل التدريب أونلاين أم حضوريًا؟"),
])
def test_fixed_bilingual_templates(language, opener, answers, expected_question, agent):
    response = _converse(agent, opener, *answers)

    assert response["result"]["response_language"] == language
    assert response["result"]["answer"].endswith(expected_question)


@pytest.mark.parametrize("field, reply, stored", [
    ("training_preference", "online", "online"),
    ("training_preference", "أونلاين", "online"),
    ("training_preference", "حضوري", "offline"),
    ("training_preference", "in person please", "offline"),
    ("skill_level", "مبتدئ", "beginner"),
    ("skill_level", "intermediate", "intermediate"),
    ("budget", "5000 جنيه", "5000 جنيه"),
    ("available_time", "ساعتين في اليوم", "ساعتين في اليوم"),
    ("background", "خريج تجارة", "خريج تجارة"),
])
def test_reply_is_stored_against_the_pending_field(field, reply, stored, agent):
    state = {
        "mode": "discovery", "pending_field": field, "profile": {},
        "declined_fields": [], "field_attempts": {}, "discovery_intro_shown": True,
    }

    response = agent.handle_message(reply, state=state)

    assert response["state"]["profile"][field] == stored


def test_several_facts_in_one_message_fill_several_fields(llm, agent):
    llm.profile_json = (
        '{"background": "accountant", "career_goal": "data analyst", '
        '"skill_level": "beginner"}'
    )

    response = _converse(agent, "Recommend a course for me")
    response = _converse(
        agent, "I'm an accountant, a beginner, and I want to be a data analyst",
        state=response["state"],
    )

    assert response["result"]["pending_field"] == "training_preference"


def test_extracted_closed_fields_need_support_in_the_message(llm, agent):
    # Live finding: the model added skill level and "online" to
    # "I want to become a data analyst", so those questions were skipped.
    llm.profile_json = (
        '{"career_goal": "data analyst", "skill_level": "beginner", '
        '"training_preference": "online", "budget": "5000", "available_time": "10 hours"}'
    )

    response = _converse(agent, "Recommend a course for me", "accountant")
    response = _converse(agent, "I want to become a data analyst", state=response["state"])

    assert response["state"]["profile"] == {"background": "accountant",
                                            "career_goal": "data analyst"}
    assert response["result"]["pending_field"] == "skill_level"


def test_closed_fields_stated_in_the_message_are_kept(llm, agent):
    llm.profile_json = '{"skill_level": "beginner", "training_preference": "online", "budget": "5000 EGP"}'

    response = _converse(agent, "Recommend a course for me")
    response = _converse(agent, "I'm a beginner, I prefer online, budget 5000 EGP",
                         state=response["state"])

    profile = response["state"]["profile"]
    assert profile["skill_level"] == "beginner"
    assert profile["training_preference"] == "online"
    assert profile["budget"] == "5000 EGP"


def test_answered_fields_are_not_asked_again(agent):
    state = {"profile": {"background": "accountant", "career_goal": "analyst"}}

    response = agent.handle_message("Recommend a course for me", state=state)

    assert response["result"]["pending_field"] == "skill_level"


def test_skipped_field_is_recorded_and_not_asked_again(agent):
    response = _converse(agent, "Recommend a course for me", "skip")

    assert response["state"]["declined_fields"] == ["background"]
    assert "background" not in response["state"]["profile"]
    assert response["result"]["pending_field"] == "career_goal"


def test_unclear_answer_gets_one_clarification_then_is_skipped(agent):
    response = _converse(agent, "Recommend a course for me", "accountant",
                         "analyst", "yes")       # unclear skill level
    assert response["result"]["pending_field"] == "skill_level"
    assert response["result"]["answer"].startswith("Sorry, I didn't quite get that.")

    response = _converse(agent, "no", state=response["state"])  # still unclear
    assert "skill_level" in response["state"]["declined_fields"]
    assert response["result"]["pending_field"] == "training_preference"


def test_placeholder_values_are_not_treated_as_facts(llm, agent):
    llm.profile_json = '{"background": "لا يوجد معلومات", "budget": "unknown"}'

    response = _converse(agent, "رشحلي كورسات مناسبة ليا")

    assert response["state"]["profile"] == {}


def test_no_recommendation_until_profile_is_complete(agent, mcp):
    _converse(agent, "Recommend a course for me", "accountant", "analyst")

    mcp.search_knowledge_base.assert_not_called()


def test_complete_profile_triggers_personalized_retrieval(agent, mcp, llm):
    response = agent.handle_message(
        "Recommend a course for me", state={"profile": dict(FULL_PROFILE)}
    )

    kwargs = mcp.search_knowledge_base.call_args.kwargs
    assert "data analyst" in kwargs["retrieval_query"]
    assert kwargs["question"] == "Recommend a course for me"
    prompt = llm.answer_calls[-1][-1].content
    assert "LEARNER PROFILE" in prompt
    assert "Training preference (online/offline): online" in prompt
    assert response["result"]["consent_pending"] is True


def test_recommend_now_proceeds_with_incomplete_profile_and_says_so(agent, mcp, llm):
    response = _converse(agent, "Recommend a course for me", "accountant",
                         "recommend now")

    mcp.search_knowledge_base.assert_called_once()
    prompt = llm.answer_calls[-1][-1].content
    assert "Skill level: not provided" in prompt
    assert "these suggestions use only what you've shared" in response["result"]["answer"]


def test_completed_profile_is_reused_for_later_recommendations(agent, mcp):
    response = agent.handle_message(
        "Recommend a course for me", state={"profile": dict(FULL_PROFILE)}
    )
    response = _converse(agent, "no", state=response["state"])  # consent: no
    response = _converse(agent, "Recommend another course for me",
                         state=response["state"])

    assert response["result"]["pending_field"] is None
    assert mcp.search_knowledge_base.call_count == 2


def test_changed_career_goal_updates_the_profile(llm, agent):
    llm.profile_json = '{"career_goal": "web developer"}'
    state = {"profile": dict(FULL_PROFILE)}

    response = agent.handle_message(
        "Actually I want to become a web developer, recommend a course", state=state
    )

    assert response["state"]["profile"]["career_goal"] == "web developer"


# --------------------------------------------------------------------
# Switching between modes
# --------------------------------------------------------------------

def test_course_question_mid_discovery_keeps_profile_and_resumes(agent, mcp):
    response = _converse(agent, "Recommend a course for me", "accountant")
    assert response["result"]["pending_field"] == "career_goal"

    response = _converse(agent, "Okay, but what does the Python course teach?",
                         state=response["state"])
    assert response["result"]["answer_status"] == "answered"
    assert response["state"]["profile"] == {"background": "accountant"}
    assert response["result"]["pending_field"] == "career_goal"
    assert response["result"]["answer"].endswith("What career path are you interested in?")

    response = _converse(agent, "data analyst", state=response["state"])
    assert response["state"]["profile"]["career_goal"] == "data analyst"
    assert response["result"]["pending_field"] == "skill_level"


def test_follow_up_question_uses_the_course_discussed_before(agent, mcp):
    response = _converse(agent, "What does the Python course cover?",
                         "How long does it take?")

    kwargs = mcp.search_knowledge_base.call_args.kwargs
    assert kwargs["question"] == "How long does it take?"
    assert "the Python course" in kwargs["retrieval_query"]
    assert response["result"]["answer_status"] == "answered"


def test_course_follow_up_without_any_course_is_left_to_the_evidence(agent, mcp, llm):
    # No keyword guess: retrieval runs, and grounding rule 6 makes the
    # model list the matching courses and ask which one when there are
    # several.
    response = _converse(agent, "How much does it cost?")

    assert mcp.search_knowledge_base.call_args.kwargs["question"] == "How much does it cost?"
    assert "ask which one they mean" in llm.answer_calls[-1][0].content
    assert response["result"]["answer_status"] == "answered"


@pytest.mark.parametrize("question", [
    # Live finding: these got "Which course do you mean?" or a greeting.
    "Who teaches Building LLM Applications with RAG?",
    "What is the refund policy?",
    "Do you have a course in quantum underwater basket weaving?",
    "Is there a student discount?",
    "ما هي سياسة استرداد الرسوم؟",
    "هل الدورات معتمدة؟",
])
def test_academy_questions_are_answered_from_the_knowledge_base(question, agent, mcp):
    result = _converse(agent, question)["result"]

    mcp.search_knowledge_base.assert_called_once()
    assert result["answer_status"] == "answered"
    assert result["mode"] is None


@pytest.mark.parametrize("reply", ["I can pay 5000", "I have a certificate in accounting"])
def test_statements_mentioning_academy_words_still_answer_discovery(reply, agent, mcp):
    state = _converse(agent, "Recommend a course for me")["state"]
    state["pending_field"] = "budget" if "pay" in reply else "background"

    response = agent.handle_message(reply, state=state)

    assert response["state"]["profile"].get(state["pending_field"]) == reply
    mcp.search_knowledge_base.assert_not_called()


# --------------------------------------------------------------------
# Answer writing from evidence
# --------------------------------------------------------------------

def test_answer_is_written_by_rex_from_mcp_evidence(agent, mcp, llm):
    response = _converse(agent, "What are the prerequisites for the Flutter course?")

    prompt = llm.answer_calls[-1][-1].content
    assert "Entry requirements: basic computer literacy." in prompt
    assert "Never invent facts" in llm.answer_calls[-1][0].content
    assert "E:\\" not in prompt
    assert response["result"]["answer"] == "Grounded answer [1]."


@pytest.mark.parametrize("evidence, expected_status, phrase", [
    (_evidence("no_results"), "no_results", "couldn't find relevant information"),
    (_evidence("empty_index", error="empty"), "empty_index", "no documents are indexed"),
    (_evidence("error", error="boom"), "unavailable", "temporarily unavailable"),
    ({"unexpected": True}, "unavailable", "temporarily unavailable"),
])
def test_missing_evidence_and_tool_errors_give_clear_messages(
    evidence, expected_status, phrase, agent, mcp, llm
):
    mcp.search_knowledge_base.return_value = evidence

    result = _converse(agent, "What courses are available?")["result"]

    assert result["answer_status"] == expected_status
    assert phrase in result["answer"]
    assert result["sources"] == []
    assert llm.answer_calls == []  # the model is not asked to improvise


def test_mcp_connection_failure_gives_clear_message(agent, mcp):
    mcp.search_knowledge_base.side_effect = RuntimeError("MCP server unavailable")

    result = _converse(agent, "What courses are available?")["result"]

    assert result["answer_status"] == "unavailable"
    assert "temporarily unavailable" in result["answer"]


def test_model_failure_gives_clear_message(agent, llm):
    llm.answer = RuntimeError("ollama down")

    result = _converse(agent, "What courses are available?")["result"]

    assert result["answer_status"] == "generation_failed"
    assert result["sources"] == []


def test_errors_are_bilingual(agent, mcp):
    mcp.search_knowledge_base.return_value = _evidence("error", error="boom")

    result = _converse(agent, "ما هي الدورات المتاحة؟")["result"]

    assert "تعذّر الوصول إلى قاعدة المعرفة" in result["answer"]


def test_failed_recommendation_does_not_ask_for_consent(agent, mcp):
    mcp.search_knowledge_base.return_value = _evidence("error", error="boom")

    result = agent.handle_message(
        "Recommend a course for me", state={"profile": dict(FULL_PROFILE)}
    )["result"]

    assert result["consent_pending"] is False


def test_previous_sources_do_not_leak_into_the_next_turn(agent):
    response = _converse(agent, "What courses are available?", "Hello")

    assert response["result"]["sources"] == []
    assert "evidence" not in response["state"]


# --------------------------------------------------------------------
# Consent and persistence
# --------------------------------------------------------------------

@pytest.mark.parametrize("reply, expected", [
    ("yes", True),
    ("Yes please, save them", True),
    ("ok sure", True),
    ("yes, I know my email", True),
    ("no problem", True),
    ("نعم", True),
    ("نعم الان", True),
    ("نعم، الاسم أحمد", True),
    ("موافقة", True),
    ("لا مانع", True),
    ("no", False),
    ("No thanks", False),
    ("I don't agree", False),
    ("لا شكرا", False),
    ("لأ", False),
    ("مش موافق", False),
    ("Let me look at the courses first", None),
    ("maybe later, looks fine", None),
    ("I'm not sure", None),
    ("yes, but maybe later", None),
    ("ممكن بعدين", None),
])
def test_consent_detection_uses_whole_words(reply, expected):
    assert _detect_consent(reply) is expected


def _pending_consent_state():
    return {"profile": dict(FULL_PROFILE), "consent_pending": True}


def test_consent_yes_saves_only_the_profile(agent, leads, llm):
    response = agent.handle_message("yes", state=_pending_consent_state())

    assert leads.saved == [{"profile": FULL_PROFILE}]
    assert response["result"]["lead_captured"] is True
    assert llm.extraction_calls == []  # "yes" is not read as a profile fact


@pytest.mark.parametrize("reply", ["maybe later", "hmm"])
def test_unclear_short_consent_reply_asks_again_and_never_saves(reply, agent, leads):
    response = agent.handle_message(reply, state=_pending_consent_state())

    assert leads.saved == []
    assert response["result"]["consent_pending"] is True


@pytest.mark.parametrize("reply", [
    "Let me look at the courses first",
    "about 5 hours a week",
])
def test_moving_on_from_consent_never_saves_and_does_not_nag(reply, agent, leads):
    response = agent.handle_message(reply, state=_pending_consent_state())

    assert leads.saved == []
    assert response["result"]["consent_pending"] is False
    assert response["state"].get("consent") is None
    assert response["result"]["lead_captured"] is False
    assert "yes or no" not in response["result"]["answer"]


def test_course_question_while_consent_is_pending_is_answered(agent, leads, mcp):
    # Live finding: this used to get "Sorry, could you confirm with yes or no?"
    response = agent.handle_message(
        "What are the prerequisites of the data analysis course?",
        state=_pending_consent_state(),
    )

    mcp.search_knowledge_base.assert_called_once()
    assert response["result"]["answer"] == "Grounded answer [1]."
    assert leads.saved == []


def test_consent_no_saves_nothing_and_says_so(agent, leads):
    response = agent.handle_message("no", state=_pending_consent_state())

    assert leads.saved == []
    assert response["result"]["answer"] == "No problem, nothing has been saved."


def test_failed_save_is_not_reported_as_saved(agent, leads):
    leads.save_lead = MagicMock(side_effect=OSError("disk full"))

    response = agent.handle_message("yes", state=_pending_consent_state())

    assert response["result"]["lead_captured"] is False
    assert "couldn't save" in response["result"]["answer"]


def test_arabic_consent_saves(agent, leads):
    state = {**_pending_consent_state(), "conversation_language": "ar"}

    response = agent.handle_message("نعم الان", state=state)

    assert len(leads.saved) == 1
    assert response["result"]["answer"] == "شكرًا لك! تم حفظ بياناتك للمتابعة."


def test_nothing_is_saved_during_discovery(agent, leads):
    _converse(agent, "Recommend a course for me", "accountant", "data analyst")

    assert leads.saved == []


# --------------------------------------------------------------------
# Language
# --------------------------------------------------------------------

@pytest.mark.parametrize("short_reply", ["ok", "5000", "yes please"])
def test_short_reply_keeps_arabic_conversation_language(short_reply, agent):
    response = _converse(agent, "ما هي رسوم الدورة؟", short_reply)

    assert response["result"]["response_language"] == "ar"
    assert response["state"]["preferred_language"] is None


def test_clear_switch_to_english_is_followed(agent):
    response = _converse(
        agent, "ما هي رسوم الدورة؟", "What are the prerequisites for the Flutter course?"
    )

    assert response["result"]["response_language"] == "en"


def test_explicit_language_request_is_kept_as_preference(agent):
    response = _converse(
        agent, "ما هي رسوم الدورة؟", "Please answer in English", "ما هي مدة الدورة؟"
    )

    assert response["state"]["preferred_language"] == "en"
    assert response["result"]["response_language"] == "en"


def test_arabizi_message_gets_arabic_reply(agent):
    response = _converse(agent, "Ana 3ayez akhod course fel programming.")

    assert response["result"]["response_language"] == "ar"
    assert response["result"]["answer"].endswith("ما هو مؤهلك الدراسي أو خلفيتك المهنية؟")


def test_answer_is_written_in_the_conversation_language(agent, llm):
    _converse(agent, "ما هي الدورات المتاحة؟")

    system_prompt = llm.answer_calls[-1][0].content
    assert "Modern Standard Arabic" in system_prompt


def test_arabic_discovery_reply_keeps_arabic(agent):
    response = _converse(agent, "رشحلي كورسات مناسبة ليا", "online")

    assert response["result"]["response_language"] == "ar"


# --------------------------------------------------------------------
# Interface contract
# --------------------------------------------------------------------

def test_interface_rejects_empty_message(agent):
    response = agent.handle_message("   ")

    assert response["status"] == "error"
    assert response["result"] is None


def test_interface_returns_common_response_shape(agent):
    response = agent.handle_message("What courses are available?", request_id="req-1")

    assert response["request_id"] == "req-1"
    assert response["agent_name"] == "REX Assistant"
    assert response["status"] == "success"
    assert {"answer", "sources", "profile", "mode", "pending_field"} <= set(response["result"])
    assert "state" in response


def test_interface_close_delegates_to_mcp_client(agent, mcp):
    agent.close()

    mcp.close.assert_called_once()


def test_graph_can_be_built_with_injected_client(leads, mcp):
    graph, client = build_graph(lead_store=leads, mcp_client=mcp)

    assert client is mcp
    assert graph is not None


def test_catalogue_listing_questions_get_the_search_hint(agent, mcp):
    # Moved from the generic engine (knowledge_base/evidence.py) to REX.
    agent.handle_message("Which courses do you offer?")

    query = mcp.search_knowledge_base.call_args.kwargs["retrieval_query"]
    from agents.course_intent import CATALOGUE_LISTING_HINT
    assert query == f"Which courses do you offer?. {CATALOGUE_LISTING_HINT}"


def test_ordinary_course_questions_are_searched_as_asked(agent, mcp):
    agent.handle_message("What are the prerequisites of the Flutter course?")

    assert mcp.search_knowledge_base.call_args.kwargs["retrieval_query"] is None


@pytest.mark.parametrize("question", [
    # Live finding: answered with a greeting, no search.
    "How experienced is Karim Nabil",
    "Who is Hana Ibrahim?",
    "مين هي هنا إبراهيم؟",
])
def test_any_question_is_checked_against_the_knowledge_base(question, agent, mcp):
    result = _converse(agent, question)["result"]

    mcp.search_knowledge_base.assert_called_once()
    assert result["answer_status"] == "answered"


@pytest.mark.parametrize("message", ["How are you?", "Who are you?", "What can you do?", "ازيك؟"])
def test_small_talk_questions_get_the_greeting_without_a_search(message, agent, mcp):
    result = _converse(agent, message)["result"]

    mcp.search_knowledge_base.assert_not_called()
    assert result["answer_status"] is None


def test_a_question_back_during_discovery_is_still_a_clarification(agent, mcp):
    response = _converse(agent, "Recommend a course for me", "what do you mean?")

    mcp.search_knowledge_base.assert_not_called()
    assert response["result"]["pending_field"] == "background"


@pytest.mark.parametrize("reply", ["not specific time", "any time", "مش محدد"])
def test_flexible_time_answers_are_accepted(reply, agent):
    state = {"profile": {k: v for k, v in FULL_PROFILE.items() if k != "available_time"},
             "mode": "discovery", "pending_field": "available_time",
             "declined_fields": [], "field_attempts": {}, "discovery_intro_shown": True}

    response = agent.handle_message(reply, state=state)

    assert response["state"]["profile"]["available_time"] == reply


def test_rex_uses_qwen_3b_by_default(monkeypatch, mcp, leads):
    monkeypatch.delenv("REX_LLM_MODEL", raising=False)
    monkeypatch.setattr("agents.config.get_secret", lambda name: None)

    assert RexAssistantAgent(lead_store=leads, mcp_client=mcp).model == "qwen2.5:3b"


def test_the_model_can_be_changed_in_the_environment(monkeypatch, mcp, leads):
    monkeypatch.setattr("agents.config.get_secret",
                        lambda name: "qwen2.5:7b" if name == "REX_LLM_MODEL" else None)

    assert RexAssistantAgent(lead_store=leads, mcp_client=mcp).model == "qwen2.5:7b"


def test_the_chat_model_reads_the_whole_prompt():
    from agents.rex_assistant import graph as rex_graph

    llm = rex_graph._get_llm.__wrapped__("qwen2.5:7b")

    assert llm.num_ctx == rex_graph.LLM_CONTEXT_TOKENS == 8192
