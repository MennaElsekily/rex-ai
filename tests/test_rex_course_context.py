"""
Two-mode REX, Phase 4: deterministic course clarification and follow-up
course context (agents/rex_assistant/course_context.py and graph.py).

Mocked: the chat model and the RAG MCP server. The fake server keeps the
real tool contract -- list_courses, and search_knowledge_base returning
course-labelled chunks, an English retrieval_query for Arabic questions
and metadata-filtered searches -- with six of the real course names.
"""

from unittest.mock import patch

import pytest

from agents.conversation_store import ConversationStore
from agents.register_agents import REX_ASSISTANT_AGENT_NAME, make_rex_assistant_handler
from agents.rex_assistant import course_context as cc
from agents.rex_assistant.graph import CourseCatalogue
from agents.rex_assistant.interface import RexAssistantAgent
from agents.rex_assistant.lead_store import LeadStore

COURSES = [
    {"course_id": "AI-101", "course_name": "Introduction to Artificial Intelligence and Python"},
    {"course_id": "DA-101", "course_name": "Excel for Business Analysis"},
    {"course_id": "DA-301", "course_name": "Python for Data Analysis"},
    {"course_id": "MOB-101", "course_name": "Flutter Mobile App Development"},
    {"course_id": "MOB-201", "course_name": "Advanced Flutter with Firebase"},
    {"course_id": "SW-301", "course_name": "Backend APIs with Python and Django"},
]
PYTHON_COURSES = ["AI-101", "DA-301", "SW-301"]


def _course_chunk(n, course):
    return {
        "text": f"{course['course_name']}: duration {n + 4} weeks, fee {n}000 EGP.",
        "course_id": course["course_id"], "course_name": course["course_name"],
        "source": "catalogue.pdf", "page": n, "section": course["course_name"],
        "section_path": course["course_name"], "record_id": None, "chunk_id": n,
        "relevance_score": 1.0,
    }


CHUNKS = {course["course_id"]: _course_chunk(n, course) for n, course in enumerate(COURSES, start=1)}
GENERAL = {"text": "Refunds are possible within 14 days of the start date.", "course_id": None,
           "course_name": None, "source": "faqs.json", "page": None, "section": "Refunds",
           "section_path": "What is the refund policy?", "record_id": None, "chunk_id": 100,
           "relevance_score": 0.5}


class FakeRag:
    """The RAG MCP tools; unfiltered searches return passages of many courses."""

    def __init__(self, order=("AI-101", "SW-301", "DA-301", "DA-101", "MOB-101"), translations=None,
                 courses=COURSES, list_fails=False):
        self.order, self.translations, self.courses = list(order), translations or {}, courses
        self.list_fails = list_fails
        self.calls, self.list_calls, self.listing_calls = [], 0, []

    def list_courses(self):
        self.list_calls += 1
        if self.list_fails:
            raise RuntimeError(r"MCP server down at E:\secret")
        return {"status": "ok", "courses": [dict(c) for c in self.courses], "error": None}

    def get_course_records(self, query=None, level=None):
        """The verified catalogue records: the same courses (no other facts stated), always complete."""
        self.listing_calls.append({"query": query, "level": level})
        return {"status": "ok", "records": [dict(c) for c in self.courses], "total_in_catalogue": len(self.courses),
                "complete": True, "filter": {}, "match_method": "all", "error": None}

    def search_knowledge_base(self, question, retrieval_query=None, metadata_filter=None, course_ids=None,
                              top_k=None):
        self.calls.append({"question": question, "retrieval_query": retrieval_query,
                           "metadata_filter": metadata_filter, "course_ids": course_ids, "top_k": top_k})
        if course_ids:   # the RAG tool limits the search to these courses' passages
            chunks = [CHUNKS[i] for i in course_ids]
        else:
            chunks = [CHUNKS[i] for i in self.order] + [GENERAL]
        chunks = [{**chunk, "reference": n} for n, chunk in enumerate(chunks, start=1)]
        courses = []
        for chunk in chunks:
            course = {"course_id": chunk["course_id"], "course_name": chunk["course_name"]}
            if chunk["course_id"] and course not in courses:
                courses.append(course)
        return {"status": "ok", "question": question,
                "retrieval_query": self.translations.get(question, retrieval_query or question),
                "query_language": "en", "translation": "not_needed", "translation_error": None,
                "chunks": chunks, "courses": courses, "error": None}


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, answer="The course lasts 7 weeks [1]."):
        self.answer = answer
        self.answer_calls = []

    def invoke(self, messages):
        if "Extract any of these fields" in messages[-1].content:
            return FakeMessage("{}")
        self.answer_calls.append(messages)
        return FakeMessage(self.answer)

    @property
    def prompt(self):
        return self.answer_calls[-1][-1].content


class NoLeads(LeadStore):
    def save_lead(self, lead):
        raise AssertionError("no lead may be saved in these tests")


@pytest.fixture
def llm():
    fake = FakeLLM()
    with patch("agents.rex_assistant.graph._get_llm", return_value=fake):
        yield fake


@pytest.fixture
def rag():
    return FakeRag()


@pytest.fixture
def agent(llm, rag):
    return RexAssistantAgent(lead_store=NoLeads(), mcp_client=rag)


def _converse(agent, *messages, state=None):
    response = None
    for message in messages:
        response = agent.handle_message(message, state=state)
        state = response["state"]
    return response


def _ids(courses):
    return [c["course_id"] for c in courses]


def _source_courses(result):
    return {s["course_id"] for s in result["sources"]}


def _passages_in_prompt(llm):
    return [cid for cid, chunk in CHUNKS.items() if chunk["text"] in llm.prompt]


# --------------------------------------------------------------------
# 1. A direct question with one matching course
# --------------------------------------------------------------------

@pytest.mark.parametrize("question, course", [
    ("How long is the Python for Data Analysis course?", "DA-301"),
    ("What does DA-301 cost?", "DA-301"),
    ("Tell me about the Advanced Flutter course", "MOB-201"),
    ("Which topics does the Django course cover?", "SW-301"),
])
def test_one_matching_course_is_answered_from_its_own_passages(question, course, agent, rag, llm):
    result = _converse(agent, question)["result"]

    assert result["answer_status"] == "answered"
    assert result["course_options"] == []
    assert result["selected_course"]["course_id"] == course
    # Other courses' passages never reach the model or the sources.
    assert _passages_in_prompt(llm) == [course]
    assert _source_courses(result) <= {course, None}
    assert f"The question is about: {course}" in llm.prompt
    # One search, then the course's own passages (the RAG tool limits that
    # search to the course): changed after a live-index trace, where the
    # course's heading ranked but its entry requirements did not.
    assert [call["course_ids"] for call in rag.calls] == [None, [course]]


# --------------------------------------------------------------------
# 2. An ambiguous question with several matching courses
# --------------------------------------------------------------------

def test_ambiguous_question_asks_which_course_instead_of_guessing(agent, rag, llm):
    response = _converse(agent, "What does the Python course cover?")
    result = response["result"]

    assert result["answer"].startswith("Which course do you mean?")
    assert "1. AI-101 Introduction to Artificial Intelligence and Python" in result["answer"]
    assert "2. DA-301 Python for Data Analysis" in result["answer"]
    assert "3. SW-301 Backend APIs with Python and Django" in result["answer"]
    assert _ids(result["course_options"]) == PYTHON_COURSES
    assert result["sources"] == [] and result["selected_course"] is None
    assert rag.calls == [] and llm.answer_calls == []      # nothing guessed, nothing searched
    assert response["state"]["course_choice"]["question"] == "What does the Python course cover?"


def test_two_courses_with_the_same_name_word_are_ambiguous(agent):
    result = _converse(agent, "Does the Flutter course give students a laptop?")["result"]

    assert _ids(result["course_options"]) == ["MOB-101", "MOB-201"]


# --------------------------------------------------------------------
# 3. The user selects one of the offered courses
# --------------------------------------------------------------------

@pytest.mark.parametrize("reply", ["2", "the second one", "Python for Data Analysis", "data analysis", "#2"])
def test_selection_answers_the_original_question_for_that_course(reply, agent, rag, llm):
    response = _converse(agent, "What does the Python course cover?", reply)
    result = response["result"]

    assert result["selected_course"]["course_id"] == "DA-301"
    assert rag.calls[0]["question"] == "What does the Python course cover?"
    assert "(course: DA-301 Python for Data Analysis)" in rag.calls[0]["retrieval_query"]
    assert "What does the Python course cover?" in llm.prompt
    assert _passages_in_prompt(llm) == ["DA-301"]
    assert response["state"]["course_choice"] is None


def test_selection_by_a_name_word_and_by_last(agent):
    assert _converse(agent, "What does the Python course cover?", "the Django one")[
        "result"]["selected_course"]["course_id"] == "SW-301"
    assert _converse(agent, "What does the Python course cover?", "the last one")[
        "result"]["selected_course"]["course_id"] == "SW-301"


def test_selection_with_its_own_question_answers_that_question(agent, rag):
    _converse(agent, "What does the Python course cover?", "The second one, how much is it?")

    assert rag.calls[0]["question"] == "The second one, how much is it?"


def test_reply_that_narrows_but_still_matches_several_asks_again(agent, rag):
    state = {"course_choice": {"question": "How long is it?", "candidates": []}}

    response = agent.handle_message("Flutter", state=state)

    assert _ids(response["result"]["course_options"]) == ["MOB-101", "MOB-201"]
    assert response["state"]["course_choice"]["question"] == "How long is it?"
    assert rag.calls == []


# --------------------------------------------------------------------
# 4. Follow-ups refer to the selected course
# --------------------------------------------------------------------

@pytest.mark.parametrize("follow_up", ["How long is it?", "What does it cost?", "What are its prerequisites?",
                                       "Is it online?", "Is there a discount for it?"])
def test_follow_up_uses_the_selected_course(follow_up, agent, rag, llm):
    response = _converse(agent, "How long is the Python for Data Analysis course?", follow_up)
    result = response["result"]

    call = rag.calls[-1]
    assert call["question"] == follow_up
    assert call["retrieval_query"] == f"{follow_up} (course: DA-301 Python for Data Analysis)"
    assert result["selected_course"]["course_id"] == "DA-301"
    assert _passages_in_prompt(llm) == ["DA-301"]
    assert "Refunds are possible" in llm.prompt          # general passages stay available


def test_follow_up_does_not_switch_course_when_others_rank_higher(llm):
    # Every unfiltered search ranks other courses first.
    rag = FakeRag(order=("AI-101", "SW-301", "DA-301", "MOB-101", "DA-101"))
    agent = RexAssistantAgent(lead_store=NoLeads(), mcp_client=rag)

    for follow_up in ("How long is it?", "What does it cost?"):
        state = _converse(agent, "Tell me about DA-101").get("state")
        result = _converse(agent, follow_up, state=state)["result"]
        assert result["selected_course"]["course_id"] == "DA-101"
        assert _passages_in_prompt(llm) == ["DA-101"]
        assert _source_courses(result) <= {"DA-101", None}


def test_course_absent_from_the_results_is_searched_with_a_course_filter(llm):
    rag = FakeRag(order=("AI-101", "SW-301", "MOB-101"))   # DA-301 never ranks
    agent = RexAssistantAgent(lead_store=NoLeads(), mcp_client=rag)

    result = _converse(agent, "How long is the Python for Data Analysis course?")["result"]

    # REX names the course; the RAG tool builds the database filter
    # (knowledge_base/evidence.course_filter, tested in tests/test_course_filter_chroma.py).
    assert rag.calls[-1]["course_ids"] == ["DA-301"] and rag.calls[-1]["metadata_filter"] is None
    assert _passages_in_prompt(llm) == ["DA-301"]
    assert "Refunds are possible" in llm.prompt          # general passage kept from the first search
    assert result["selected_course"]["course_id"] == "DA-301"


def test_naming_another_course_switches_the_selection(agent, llm):
    response = _converse(agent, "How long is the Python for Data Analysis course?", "And the Excel course?")

    assert response["result"]["selected_course"]["course_id"] == "DA-101"
    assert _passages_in_prompt(llm) == ["DA-101"]


def test_a_new_topic_keeps_the_course_but_is_not_limited_to_it(agent, rag, llm):
    response = _converse(agent, "How long is the Python for Data Analysis course?", "What is the refund policy?")

    assert rag.calls[-1]["retrieval_query"] is None             # searched on its own
    assert len(_passages_in_prompt(llm)) > 1                    # not limited to DA-301
    assert response["result"]["selected_course"]["course_id"] == "DA-301"


def test_catalogue_listing_is_never_clarified_and_replaces_the_course(agent, rag):
    response = _converse(agent, "How long is the Python for Data Analysis course?",
                         "What Python courses are available?")

    assert response["result"]["course_options"] == []
    assert response["result"]["answer_status"] == "answered"
    # The listing comes from the catalogue tool, and the courses it lists
    # (not the earlier DA-301) are what a follow-up refers to.
    assert rag.listing_calls[-1] == {"query": "What Python courses are available?", "level": None}
    assert [c["course_id"] for c in response["state"]["course_focus"]] == sorted(c["course_id"] for c in COURSES)


# --------------------------------------------------------------------
# 5. Missing, expired or conflicting context
# --------------------------------------------------------------------

def test_follow_up_without_context_asks_which_course_then_answers_it(agent, rag, llm):
    response = _converse(agent, "How long is it?")
    assert response["result"]["answer"].startswith("Which course are you asking about?")
    assert response["result"]["course_options"] == [] and rag.calls == []

    response = _converse(agent, "Python for Data Analysis", state=response["state"])

    assert rag.calls[0]["question"] == "How long is it?"
    assert response["result"]["selected_course"]["course_id"] == "DA-301"
    assert response["result"]["answer_status"] == "answered"


def test_an_unrelated_reply_drops_the_open_choice(agent, rag):
    response = _converse(agent, "What does the Python course cover?", "Hello")
    assert response["state"]["course_choice"] is None

    response = _converse(agent, "2", state=response["state"])   # no longer a choice

    assert response["result"]["selected_course"] is None
    assert rag.calls == []


@pytest.mark.parametrize("bad_state", [
    {"course_focus": "DA-301"},
    {"course_focus": [{"course_id": "ZZ-999", "course_name": "Removed course"}]},
    {"course_focus": [None, 7, {"course_id": ""}]},
    {"course_choice": {"candidates": "x"}},
    {"course_choice": "2"},
])
def test_malformed_or_stale_course_state_is_ignored(bad_state, agent, rag):
    response = agent.handle_message("How long is it?", state=bad_state)

    assert response["status"] == "success"
    assert response["result"]["answer"].startswith("Which course are you asking about?")
    assert rag.calls == []


def test_a_new_question_while_a_choice_is_open_is_answered_normally(agent, rag):
    response = _converse(agent, "What does the Python course cover?", "What is the refund policy?")

    assert response["result"]["answer_status"] == "answered"
    assert rag.calls[0]["question"] == "What is the refund policy?"
    assert response["state"]["course_choice"] is None


def test_course_list_unavailable_falls_back_to_the_courses_in_the_evidence(llm):
    rag = FakeRag(list_fails=True)
    agent = RexAssistantAgent(lead_store=NoLeads(), mcp_client=rag)

    result = _converse(agent, "What does the Python course cover?")["result"]

    # The evidence held AI-101 and SW-301 (DA-301 too): still asked, not guessed.
    assert set(_ids(result["course_options"])) <= set(PYTHON_COURSES)
    assert len(result["course_options"]) >= 2
    assert llm.answer_calls == []
    assert "secret" not in repr(result)


# --------------------------------------------------------------------
# 6. Arabic and Arabizi
# --------------------------------------------------------------------

def test_arabic_clarification_selection_and_follow_up(agent, rag, llm):
    response = _converse(agent, "كم سعر كورس بايثون؟")
    result = response["result"]
    assert result["response_language"] == "ar"
    assert result["answer"].startswith("أي دورة تقصد؟")
    assert "2. DA-301 Python for Data Analysis" in result["answer"]
    assert result["answer"].endswith("اكتب رقم الدورة أو اسمها.")

    response = _converse(agent, "التانية", state=response["state"])
    assert response["result"]["selected_course"]["course_id"] == "DA-301"
    assert response["result"]["response_language"] == "ar"
    assert rag.calls[-1]["question"] == "كم سعر كورس بايثون؟"

    response = _converse(agent, "مدته كام؟", state=response["state"])
    assert response["result"]["selected_course"]["course_id"] == "DA-301"
    assert rag.calls[-1]["retrieval_query"] == "مدته كام؟ (course: DA-301 Python for Data Analysis)"
    assert _passages_in_prompt(llm) == ["DA-301"]


def test_arabic_description_is_matched_on_the_english_search_query(llm):
    question = "كم سعر دورة تحليل البيانات؟"
    rag = FakeRag(translations={question: "How much is the data analysis course?"})
    agent = RexAssistantAgent(lead_store=NoLeads(), mcp_client=rag)

    result = _converse(agent, question)["result"]

    assert result["selected_course"]["course_id"] == "DA-301"
    assert _passages_in_prompt(llm) == ["DA-301"]
    # The first search identified DA-301; its own passages are then searched by id.
    assert [call["course_ids"] for call in rag.calls] == [None, ["DA-301"]]


def test_arabic_follow_up_without_context_asks_in_arabic(agent):
    result = _converse(agent, "الكورس ده بكام؟")["result"]

    assert result["answer"].startswith("عن أي دورة تسأل؟")


def test_arabizi_question_is_clarified_too(agent):
    result = _converse(agent, "el python course b kam?")["result"]

    assert _ids(result["course_options"]) == PYTHON_COURSES


# --------------------------------------------------------------------
# 7. Recommendations never trigger course clarification
# --------------------------------------------------------------------

FULL_PROFILE = {"background": "accountant", "career_goal": "data analyst", "skill_level": "beginner",
                "training_preference": "online", "budget": "5000 EGP", "available_time": "5 hours a week"}


@pytest.mark.parametrize("request_text", ["Recommend a course for me", "Recommend a Python course for me",
                                          "رشحلي كورس بايثون"])
def test_recommendations_are_not_clarified(request_text, agent, rag, llm):
    response = agent.handle_message(request_text, state={"profile": dict(FULL_PROFILE)})
    result = response["result"]

    assert result["course_options"] == []
    assert "LEARNER PROFILE" in llm.prompt          # the recommendation task, unchanged
    assert rag.calls[0]["metadata_filter"] is None
    assert rag.calls[0]["retrieval_query"].startswith("Courses suitable for a learner with")


def test_follow_up_after_a_recommendation_asks_which_recommended_course(agent, rag, llm):
    llm.answer = "AI-101 :: It covers the basics.\nSW-301 :: It is the next step."   # ids from the catalogue list
    response = agent.handle_message("Recommend a course for me", state={"profile": dict(FULL_PROFILE)})
    response = _converse(agent, "no", state=response["state"])               # consent: no
    focus = _ids(response["state"]["course_focus"])
    assert focus == ["AI-101", "SW-301"]

    result = _converse(agent, "How long is it?", state=response["state"])["result"]

    assert _ids(result["course_options"]) == focus


# --------------------------------------------------------------------
# 8. Sources after clarification
# --------------------------------------------------------------------

def test_sources_after_a_choice_belong_to_the_chosen_course_or_general_passages(agent, llm):
    # [1] is the chosen course's catalogue record, [2] its passage, [3] a general passage.
    llm.answer = "It covers pandas [2]. Refunds within 14 days [3]."

    result = _converse(agent, "What does the Python course cover?", "3")["result"]

    assert [(s["reference"], s["course_id"]) for s in result["sources"]] == [(2, "SW-301"), (3, None)]


def test_not_found_after_a_choice_lists_no_sources(agent, llm):
    llm.answer = "The context does not contain information about a free laptop. [1]"

    result = _converse(agent, "Does the Flutter course give students a laptop?", "1")["result"]

    assert result["sources"] == [] and "[1]" not in result["answer"]
    assert result["selected_course"]["course_id"] == "MOB-101"


def test_not_found_answer_naming_the_course_code_lists_no_sources(agent, llm):
    # Live finding: "MOB-201 ... There is no mention of laptop provision"
    # kept three sources, because the code's digits looked like a fact.
    llm.answer = ("The Flutter course, MOB-201 Advanced Flutter with Firebase, has no mention of "
                  "laptop provision in the provided information.")

    result = _converse(agent, "Does the Flutter course give students a laptop?", "2")["result"]

    assert result["sources"] == []


def test_real_answers_with_numbers_keep_their_sources(agent, llm):
    llm.answer = "MOB-201 costs 9000 EGP; there is no mention of a discount."

    result = _converse(agent, "Does the Flutter course give students a laptop?", "2")["result"]

    assert _source_courses(result) == {"MOB-201"}


# --------------------------------------------------------------------
# 9. Conversation isolation
# --------------------------------------------------------------------

def test_one_conversations_course_never_affects_another(agent, rag):
    first = _converse(agent, "How long is the Python for Data Analysis course?")
    second = _converse(agent, "How long is it?")         # a different conversation

    assert second["result"]["answer"].startswith("Which course are you asking about?")
    assert first["state"]["course_focus"][0]["course_id"] == "DA-301"
    assert _converse(agent, "What does it cost?", state=first["state"])[
        "result"]["selected_course"]["course_id"] == "DA-301"


def test_saved_conversations_keep_their_own_course(tmp_path, agent):
    store = ConversationStore(db_path=tmp_path / "conversations.sqlite3")
    handler = make_rex_assistant_handler(agent, store)

    def ask(conversation, text):
        handler({"request_id": text, "user_input": text, "context": {"conversation_id": conversation}})
        return store.get(conversation, REX_ASSISTANT_AGENT_NAME)

    ask("learner-a", "How long is the Python for Data Analysis course?")
    ask("learner-b", "What does the Python course cover?")
    state_b = ask("learner-b", "3")
    state_a = ask("learner-a", "What does it cost?")

    assert _ids(state_a["course_focus"]) == ["DA-301"]
    assert _ids(state_b["course_focus"]) == ["SW-301"]
    assert state_a.get("course_choice") is None


# --------------------------------------------------------------------
# 10. Regressions: Phases 1-3 behaviour around the new step
# --------------------------------------------------------------------

def test_question_mid_discovery_is_clarified_answered_and_discovery_resumes(agent):
    response = _converse(agent, "Recommend a course for me", "accountant")
    assert response["result"]["pending_field"] == "career_goal"

    response = _converse(agent, "What does the Python course cover?", state=response["state"])
    assert response["result"]["course_options"]
    assert response["result"]["pending_field"] == "career_goal"

    response = _converse(agent, "2", state=response["state"])
    assert response["result"]["selected_course"]["course_id"] == "DA-301"
    assert response["result"]["answer"].endswith("What career path are you interested in?")
    assert response["state"]["profile"] == {"background": "accountant"}   # "2" is not a profile fact

    response = _converse(agent, "data analyst", state=response["state"])
    assert response["state"]["profile"]["career_goal"] == "data analyst"


@pytest.mark.parametrize("question", ["Hello", "Thanks!", "What is the refund policy?",
                                      "Is there a student discount?", "Do you have a course in basket weaving?"])
def test_questions_without_a_course_are_not_clarified(question, agent):
    assert _converse(agent, question)["result"]["course_options"] == []


def test_course_list_is_cached_and_not_fetched_for_greetings(agent, rag):
    _converse(agent, "Hello")
    assert rag.list_calls == 0

    _converse(agent, "How long is the Python for Data Analysis course?", "What does it cost?", "How long is it?")
    assert rag.list_calls == 1


def test_catalogue_cache_refreshes_after_its_lifetime_and_retries_failures():
    now = [0.0]
    rag = FakeRag(list_fails=True)
    catalogue = CourseCatalogue(rag, clock=lambda: now[0])

    assert catalogue.courses() == [] and rag.list_calls == 1
    assert catalogue.courses() == [] and rag.list_calls == 1        # not retried at once
    now[0] = 61
    rag.list_fails = False
    assert _ids(catalogue.courses()) == _ids(COURSES) and rag.list_calls == 2
    now[0] = 61 + 599
    catalogue.courses()
    assert rag.list_calls == 2
    now[0] = 61 + 601
    rag.list_fails = True
    assert _ids(catalogue.courses()) == _ids(COURSES)               # last good list kept


# --------------------------------------------------------------------
# course_context on the full 14-course catalogue of the index
# --------------------------------------------------------------------

CATALOGUE = COURSES + [
    {"course_id": "AI-201", "course_name": "Machine Learning Practitioner"},
    {"course_id": "AI-301", "course_name": "Building LLM Applications with RAG"},
    {"course_id": "DA-201", "course_name": "SQL and Power BI for Data Analysts"},
    {"course_id": "MKT-101", "course_name": "Digital Marketing Fundamentals"},
    {"course_id": "MKT-201", "course_name": "Social Media and Paid Advertising"},
    {"course_id": "MKT-301", "course_name": "Marketing Analytics and Campaign Optimization"},
    {"course_id": "SW-101", "course_name": "Web Development Fundamentals"},
    {"course_id": "SW-201", "course_name": "Full Stack Web Development with React and Node.js"},
]


@pytest.mark.parametrize("text, kind, ids", [
    ("How long is the Python for Data Analysis course?", cc.NAMED, ["DA-301"]),
    ("What does the Python course cover?", cc.AMBIGUOUS, ["AI-101", "DA-301", "SW-301"]),
    ("What's the difference between DA-101 and DA-201?", cc.NAMED, ["DA-101", "DA-201"]),
    ("Excel for Business Analysis or Python for Data Analysis?", cc.NAMED, ["DA-101", "DA-301"]),
    ("Who teaches Building LLM Applications with RAG?", cc.NAMED, ["AI-301"]),
    ("How much is the web development course?", cc.AMBIGUOUS, ["SW-101", "SW-201"]),
    ("Full Stack Web Development price?", cc.NAMED, ["SW-201"]),
    ("social media course fees", cc.NAMED, ["MKT-201"]),
    ("machine learning duration", cc.NAMED, ["AI-201"]),
    ("Which AI courses are there?", cc.AMBIGUOUS, ["AI-101", "AI-201", "AI-301"]),
    ("كم سعر كورس بايثون؟", cc.AMBIGUOUS, ["AI-101", "DA-301", "SW-301"]),
    # Common words that happen to be in course names name no course.
    ("Is the course full?", cc.NONE, []),
    ("Is it paid?", cc.NONE, []),
    ("What are the fundamentals I need?", cc.NONE, []),
    ("el course da b kam?", cc.NONE, []),
    ("What is the refund policy?", cc.NONE, []),
    # Generalization check T2: every word written before "course" must
    # belong to the course's name, so an unknown course is not taken for
    # a similarly named one.
    ("How long is the Python Basics course?", cc.NONE, []),
    ("the Kubernetes course", cc.NONE, []),
    ("Tell me about the Advanced Flutter course", cc.NAMED, ["MOB-201"]),
    ("Is the Excel course good for accountants?", cc.NAMED, ["DA-101"]),
    ("Price of the Full Stack Web Development with React and Node.js course?", cc.NAMED, ["SW-201"]),
    ("How long is the AI course?", cc.AMBIGUOUS, ["AI-101", "AI-201", "AI-301"]),
    ("el python course b kam?", cc.AMBIGUOUS, ["AI-101", "DA-301", "SW-301"]),
    ("Is the Excel course or the Django course better?", cc.NAMED, ["DA-101", "SW-301"]),
])
def test_resolve_reference(text, kind, ids):
    reference = cc.resolve_reference(text, CATALOGUE)

    assert (reference.kind, _ids(reference.courses)) == (kind, ids)


@pytest.mark.parametrize("text, back, needs", [
    ("How long is it?", True, True),
    ("How much does the course cost?", True, True),
    ("What are the prerequisites for this course?", True, True),
    ("مدة الكورس كام؟", True, True),
    ("الكورس ده بكام؟", True, True),
    ("el course da b kam?", True, True),
    ("Is there a discount for it?", True, False),   # follow-up, but not "which course?" on its own
    ("What are the prerequisites?", True, False),
    ("the course on Kubernetes", False, False),
    ("What is the refund policy?", False, False),
])
def test_references_back_to_the_course(text, back, needs):
    assert cc.refers_back(text) is back
    assert cc.needs_a_course(text) is needs


def test_unknown_course_with_a_shared_word_is_answered_unscoped(llm):
    # T2: "Python Basics" is described only in a brochure (no course id);
    # DA-301 shares the word "Python". REX must not tell the model the
    # question is about DA-301, and the brochure passage must reach it.
    brochure = {"reference": 1, "text": "Python Basics (evening workshop): 4 weeks.", "course_id": None,
                "course_name": None, "source": "brochure.pdf", "chunk_id": 9}

    class BrochureRag(FakeRag):
        def search_knowledge_base(self, question, retrieval_query=None, metadata_filter=None, **kwargs):
            result = super().search_knowledge_base(question, retrieval_query, metadata_filter, **kwargs)
            result["chunks"] = [brochure, *[{**c, "reference": n} for n, c in enumerate(result["chunks"], start=2)]]
            return result

    agent = RexAssistantAgent(lead_store=NoLeads(), mcp_client=BrochureRag())
    result = _converse(agent, "How long is the Python Basics course?")["result"]

    assert result["selected_course"] is None and result["course_options"] == []
    assert "The question is about" not in llm.prompt
    assert brochure["text"] in llm.prompt


# --------------------------------------------------------------------
# Live finding: a learner's real conversation got the greeting twice
# --------------------------------------------------------------------

def test_the_reported_conversation_gets_course_help_not_the_greeting(llm):
    # No course list (the live index had no course ids): the old fallback
    # paths must still give course help.
    rag = FakeRag(list_fails=True)
    agent = RexAssistantAgent(lead_store=NoLeads(), mcp_client=rag)
    greeting = "Hello! I'm REX"

    first = _converse(agent, "Iam a computer engineer student and i need course python")
    assert first["result"]["mode"] == "discovery"                  # recommendation mode starts
    assert not first["result"]["answer"].startswith(greeting)

    listing = _converse(agent, "what courses do u offer", state=first["state"])
    assert listing["result"]["answer_status"] == "answered"

    # The questionnaire is open, and without a course list "python
    # foundations" cannot be told from an answer: it is taken as one, and
    # the next question follows -- never the greeting.
    picked = _converse(agent, "python foundations", state=listing["state"])
    assert not picked["result"]["answer"].startswith(greeting)
    assert picked["result"]["mode"] == "discovery"


def test_a_course_named_mid_questionnaire_is_answered_and_the_questionnaire_resumes(agent, rag):
    response = _converse(agent, "Recommend a course for me", "accountant")
    assert response["result"]["pending_field"] == "career_goal"

    response = _converse(agent, "Python for Data Analysis", state=response["state"])

    assert response["result"]["selected_course"]["course_id"] == "DA-301"
    assert response["result"]["answer_status"] == "answered"
    assert response["state"]["profile"] == {"background": "accountant"}
    assert response["result"]["answer"].endswith("What career path are you interested in?")


def test_a_career_goal_that_resembles_a_course_name_is_still_the_answer(agent, rag):
    response = _converse(agent, "Recommend a course for me", "accountant", "data analyst")

    assert response["state"]["profile"]["career_goal"] == "data analyst"
    assert rag.calls == []


def test_a_course_name_on_its_own_is_answered(agent, rag):
    result = _converse(agent, "Python for Data Analysis")["result"]

    assert result["selected_course"]["course_id"] == "DA-301"
    assert result["answer_status"] == "answered"


def test_an_ambiguous_course_name_on_its_own_asks_which_one(agent, rag):
    result = _converse(agent, "python")["result"]

    assert _ids(result["course_options"]) == PYTHON_COURSES


@pytest.mark.parametrize("reply", ["Hello", "Thanks!", "ok", "شكرا", "مرحبا"])
def test_greetings_and_thanks_after_a_course_answer_still_get_the_greeting(reply, agent, rag):
    state = _converse(agent, "How long is the Python for Data Analysis course?")["state"]
    calls = len(rag.calls)

    result = _converse(agent, reply, state=state)["result"]

    assert len(rag.calls) == calls                                  # nothing searched
    assert result["answer_status"] is None


def test_a_word_that_names_no_course_without_context_still_gets_the_greeting(agent, rag):
    assert _converse(agent, "accountant")["result"]["answer_status"] is None
    assert rag.calls == []


def test_clean_courses_drops_invalid_entries_and_duplicates():
    assert cc.clean_courses([{"course_id": " DA-301 ", "course_name": "Python for Data Analysis"},
                             {"course_id": "da-301"}, {"course_name": ""}, "x", None]) == [
        {"course_id": "DA-301", "course_name": "Python for Data Analysis"}]
    assert cc.clean_courses("DA-301") == []
