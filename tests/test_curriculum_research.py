"""
Direct tests for the Curriculum Research Agent (no orchestrator):
search quality against the bundled sample data, draft generation with
hours/audience parameters, the unsupported-request response, and
refinement through the shared ConversationStore.
"""

import sqlite3

import pytest

from agents.conversation_store import ConversationStore
from agents.curriculum_research import CurriculumData, CurriculumResearchAgent

AGENT_KEY = "curriculum_research"


def req(text, conv="conv-1", language=None, **context):
    ctx = dict(context)
    if conv is not None:
        ctx["conversation_id"] = conv
    if language is not None:
        ctx["language"] = language
    return {"request_id": "req-1", "user_input": text, "context": ctx}


def titles(response):
    return [c["title"] for c in response["result"]["curricula"]]


@pytest.fixture
def store():
    return ConversationStore(db_path=":memory:")


@pytest.fixture
def agent():
    """Stateless agent: no conversation store."""
    return CurriculumResearchAgent()


@pytest.fixture
def stateful(store):
    return CurriculumResearchAgent(conversation_store=store)


def module_hours(draft):
    return [module["hours"] for module in draft["modules"]]


# --------------------------------------------------------------------
# Search (against the 5 bundled sample curricula)
# --------------------------------------------------------------------

@pytest.mark.parametrize("query, expected", [
    ("Find a curriculum about data analysis", ["Introductory Data Analysis"]),
    ("Find a curriculum about AI", ["AI Literacy for Educators"]),
    ("find a curriculum about ai", ["AI Literacy for Educators"]),
    (
        "I need a training curriculum for artificial intelligence",
        ["AI Literacy for Educators"],
    ),
    ("Find a curriculum about cybersecurity", ["Cybersecurity Awareness"]),
    ("Show me a syllabus for project management",
     ["Project Management Essentials"]),
    ("curriculum about instructional design",
     ["Instructional Design Basics"]),
    ("curriculum for teachers", ["AI Literacy for Educators"]),
    ("Find a 40 hour curriculum about cybersecurity",
     ["Cybersecurity Awareness"]),
    ("Cybersecurity Awareness", ["Cybersecurity Awareness"]),
])
def test_english_search_finds_only_the_right_example(agent, query, expected):
    response = agent.process_request(req(query))

    assert response["status"] == "success"
    assert titles(response) == expected


@pytest.mark.parametrize("query, expected", [
    ("أريد منهج تدريبي لتحليل البيانات", ["Introductory Data Analysis"]),
    ("أريد منهج في الذكاء الاصطناعي", ["AI Literacy for Educators"]),
    ("ابحث عن منهج الأمن السيبراني", ["Cybersecurity Awareness"]),
    ("منهج في إدارة المشاريع", ["Project Management Essentials"]),
])
def test_arabic_search_finds_the_right_example(agent, query, expected):
    response = agent.process_request(req(query, language="ar"))

    assert response["status"] == "success"
    assert titles(response) == expected
    assert "العثور" in response["result"]["message"]


@pytest.mark.parametrize("query", [
    "Find a curriculum about cooking",
    "show me a python curriculum",
    "Find a curriculum about quantum physics",
    "main email",
])
def test_unrelated_topics_find_nothing(agent, query):
    response = agent.process_request(req(query))

    assert response["status"] == "success"
    assert titles(response) == []
    assert "No matching examples" in response["result"]["message"]


def test_short_words_do_not_match_inside_other_words(agent):
    # "ai" must not match "main", "email", "training", "maintain"...
    data = CurriculumData()

    assert [r["title"] for r in data.search("ai")] == [
        "AI Literacy for Educators"
    ]
    assert data.search("mai") == []


def test_partial_matches_are_labelled_not_presented_as_exact(agent):
    response = agent.process_request(
        req("Find a curriculum about python data analysis")
    )

    assert titles(response) == ["Introductory Data Analysis"]
    assert "closest" in response["result"]["message"]


def test_searches_are_not_excessively_broad(agent):
    for query in (
        "data analysis", "ai", "artificial intelligence", "cybersecurity",
        "project management", "instructional design", "management",
        "teachers", "phishing",
    ):
        response = agent.process_request(req(f"Find a curriculum about {query}"))
        assert len(titles(response)) <= 2, query


def test_search_with_only_generic_words_lists_examples(agent):
    response = agent.process_request(req("show me a curriculum"))

    assert len(titles(response)) == 5


def test_data_search_back_compat_exact_phrase_and_empty_query():
    data = CurriculumData()

    assert [r["title"] for r in data.search("cybersecurity awareness")] == [
        "Cybersecurity Awareness"
    ]
    assert len(data.search("")) == 5
    assert len(data.get_all()) == 5


# --------------------------------------------------------------------
# Creation: hours, audience, title
# --------------------------------------------------------------------

def test_creation_uses_hours_and_audience_from_the_request(agent):
    response = agent.process_request(req(
        "Create a curriculum about Python for 40 hours for beginners"
    ))
    draft = response["result"]["curricula"][0]

    assert response["status"] == "success"
    assert draft["title"] == "Python Training Curriculum"
    assert draft["training_hours"] == 40
    assert draft["target_audience"] == "Beginners"
    assert sum(module_hours(draft)) == 40
    assert "Duration: 40 hours" in response["result"]["message"]


def test_parameters_never_appear_in_title_or_module_titles(agent):
    response = agent.process_request(req(
        "Design a 25-hour Python course for high school students"
    ))
    draft = response["result"]["curricula"][0]
    everything = draft["title"] + " ".join(m["title"] for m in draft["modules"])

    assert draft["title"] == "Python Training Curriculum"
    assert draft["target_audience"] == "High school students"
    assert "25" not in everything
    assert "hour" not in everything.lower()
    assert "school" not in everything.lower()


def test_arabic_creation_with_hours_and_audience(agent):
    response = agent.process_request(req(
        "أنشئ منهج عن تحليل البيانات لمدة 40 ساعة للمبتدئين",
        language="ar",
    ))
    draft = response["result"]["curricula"][0]

    assert response["status"] == "success"
    assert "تم إنشاء مسودة" in response["result"]["message"]
    assert draft["title"] == "تحليل البيانات Training Curriculum"
    assert draft["training_hours"] == 40
    assert draft["target_audience"] == "المبتدئين"
    assert "ساعة" not in draft["title"]
    assert "للمبتدئين" not in draft["title"]


def test_arabic_indic_digits_are_understood(agent):
    response = agent.process_request(req(
        "صمم دورة عن الأمن السيبراني ٣٠ ساعة", language="ar"
    ))

    assert response["result"]["curricula"][0]["training_hours"] == 30


def test_defaults_apply_when_nothing_is_specified(agent):
    draft = agent.process_request(
        req("Create a curriculum about SQL")
    )["result"]["curricula"][0]

    assert draft["training_hours"] == 30
    assert draft["target_audience"] == "Beginners"
    assert draft["status"] == "draft"


def test_text_overrides_context_and_context_still_works(agent):
    from_context = agent.process_request(req(
        "Create a curriculum about SQL",
        training_hours=20, target_audience="Nurses",
    ))["result"]["curricula"][0]
    from_text = agent.process_request(req(
        "Create a curriculum about SQL for 50 hours for teachers",
        training_hours=20, target_audience="Nurses",
    ))["result"]["curricula"][0]

    assert (from_context["training_hours"], from_context["target_audience"]) == (
        20, "Nurses"
    )
    assert (from_text["training_hours"], from_text["target_audience"]) == (
        50, "Teachers"
    )


@pytest.mark.parametrize("text", [
    "Create a curriculum about SQL for 2 hours",
    "Create a curriculum about SQL for 5000 hours",
])
def test_unusable_hours_are_rejected_clearly(agent, text):
    response = agent.process_request(req(text))

    assert response["status"] == "error"
    assert "between 4 and 1000" in response["error_message"]


def test_bad_context_hours_are_rejected_clearly(agent):
    response = agent.process_request(
        req("Create a curriculum about SQL", training_hours="lots")
    )

    assert response["status"] == "error"
    assert "between 4 and 1000" in response["error_message"]


@pytest.mark.parametrize("text", [
    "Create a curriculum and marketing campaign",
    "Create a curriculum",
    "أنشئ منهج تدريبي ثم حملة تسويقية",
])
def test_missing_topic_asks_instead_of_inventing_one(agent, text):
    response = agent.process_request(req(text))

    assert response["status"] == "error"
    assert response["result"] is None
    assert "topic" in response["error_message"] or "موضوع" in response[
        "error_message"
    ]


def test_topic_before_a_marketing_clause_is_used(agent):
    response = agent.process_request(
        req("Create a curriculum about SQL and a marketing campaign")
    )

    assert response["result"]["curricula"][0]["title"] == (
        "SQL Training Curriculum"
    )


def test_empty_request_is_an_error(agent):
    assert agent.process_request(req("   "))["status"] == "error"


# --------------------------------------------------------------------
# Accreditation / gap analysis: honest "unsupported"
# --------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Do an accreditation gap analysis for my AI curriculum",
    "Is this curriculum accredited?",
    "Align the curriculum with the standards",
])
def test_accreditation_requests_are_unsupported_not_empty_success(agent, text):
    response = agent.process_request(req(text))

    assert response["status"] == "error"
    assert response["result"] == {
        "code": "unsupported_request",
        "capability": "accreditation_gap_analysis",
    }
    assert "not supported" in response["error_message"]
    assert "No matching examples" not in response["error_message"]


def test_accreditation_unsupported_message_in_arabic(agent):
    response = agent.process_request(
        req("أريد تحليل الفجوات لمنهج الأمن السيبراني", language="ar")
    )

    assert response["status"] == "error"
    assert response["result"]["code"] == "unsupported_request"
    assert "غير مدعوم" in response["error_message"]


def test_creating_a_curriculum_mentioning_accreditation_adds_a_caveat(agent):
    response = agent.process_request(
        req("Create an accredited curriculum about SQL")
    )

    assert response["status"] == "success"
    assert response["result"]["curricula"][0]["title"] == (
        "SQL Training Curriculum"
    )
    assert "NOT performed" in response["result"]["notice"]


# --------------------------------------------------------------------
# Refinement through the shared ConversationStore
# --------------------------------------------------------------------

def create(agent, conv="conv-1", text="Create a curriculum about data analysis"):
    return agent.process_request(req(text, conv=conv))


def test_draft_is_stored_under_the_curriculum_key_only(stateful, store):
    response = create(stateful)

    state = store.get("conv-1", AGENT_KEY)
    assert state["draft"]["title"] == "data analysis Training Curriculum"
    assert state["draft"]["revision"] == 1
    assert store.get("conv-1", "rex_assistant") is None
    assert store.get("conv-1", "rfp_research") is None
    assert "make it 40 hours" in response["result"]["message"]


def test_change_duration_redistributes_hours(stateful, store):
    create(stateful)

    response = stateful.process_request(req("make it 40 hours"))
    draft = response["result"]["curricula"][0]

    assert response["status"] == "success"
    assert draft["training_hours"] == 40
    assert module_hours(draft) == [10, 10, 10, 10]
    assert draft["revision"] == 2
    assert "from 30 to 40" in response["result"]["message"]
    assert store.get("conv-1", AGENT_KEY)["draft"]["training_hours"] == 40


def test_add_capstone_adds_a_module_and_keeps_total_hours(stateful):
    create(stateful, text="Create a curriculum about data analysis for 40 hours")

    draft = stateful.process_request(
        req("Add a capstone to that curriculum")
    )["result"]["curricula"][0]

    assert [m["title"] for m in draft["modules"]][-1] == "Capstone Project"
    assert draft["training_hours"] == 40
    assert sum(module_hours(draft)) == 40
    assert [m["module_number"] for m in draft["modules"]] == [1, 2, 3, 4, 5]
    assert "Capstone project" in " ".join(
        e.capitalize() for e in draft["practical_exercises"]
    ) or any("capstone" in e.lower() for e in draft["practical_exercises"])


def test_add_module_and_change_audience(stateful):
    create(stateful)

    draft = stateful.process_request(
        req("add a module on SQL")
    )["result"]["curricula"][0]
    assert draft["modules"][-1]["title"] == "SQL"

    draft = stateful.process_request(
        req("change the audience to teachers")
    )["result"]["curricula"][0]
    assert draft["target_audience"] == "Teachers"
    assert draft["modules"][-1]["title"] == "SQL"
    assert draft["revision"] == 3


def test_several_changes_in_one_message(stateful):
    create(stateful)

    response = stateful.process_request(
        req("add a module on SQL and make it 50 hours")
    )
    draft = response["result"]["curricula"][0]

    assert draft["modules"][-1]["title"] == "SQL"
    assert draft["training_hours"] == 50
    assert sum(module_hours(draft)) == 50
    assert len(response["result"]["changes"]) == 2


def test_arabic_refinements(stateful):
    stateful.process_request(req(
        "أنشئ منهج عن تحليل البيانات", language="ar"
    ))

    hours = stateful.process_request(req("اجعلها 60 ساعة", language="ar"))
    assert hours["result"]["curricula"][0]["training_hours"] == 60
    assert "تم تحديث مسودة المنهج" in hours["result"]["message"]

    capstone = stateful.process_request(req("أضف مشروعًا ختاميًا", language="ar"))
    assert capstone["result"]["curricula"][0]["modules"][-1]["title"] == (
        "Capstone Project"
    )

    module = stateful.process_request(req("أضف وحدة عن SQL", language="ar"))
    assert module["result"]["curricula"][0]["modules"][-1]["title"] == "SQL"

    audience = stateful.process_request(
        req("غيّر الفئة المستهدفة إلى المعلمين", language="ar")
    )
    assert audience["result"]["curricula"][0]["target_audience"] == "المعلمين"


def test_repeating_a_change_is_harmless(stateful, store):
    create(stateful)
    stateful.process_request(req("add a capstone"))
    before = store.get("conv-1", AGENT_KEY)["draft"]

    again = stateful.process_request(req("add a capstone"))

    assert again["status"] == "success"
    assert "already has a Capstone Project module" in again["result"]["message"]
    assert store.get("conv-1", AGENT_KEY)["draft"] == before
    assert len(before["modules"]) == 5


def test_a_failed_change_leaves_the_draft_untouched(stateful, store):
    create(stateful)
    before = store.get("conv-1", AGENT_KEY)["draft"]

    response = stateful.process_request(
        req("add a capstone and make it 2 hours")
    )

    assert response["status"] == "error"
    assert "at least" in response["error_message"]
    assert store.get("conv-1", AGENT_KEY)["draft"] == before


def test_hours_below_the_module_count_are_rejected(stateful):
    create(stateful)
    stateful.process_request(req("add a capstone"))  # now 5 modules

    response = stateful.process_request(req("make it 4 hours"))

    assert response["status"] == "error"
    assert "at least 5" in response["error_message"]


def test_removal_and_other_edits_are_reported_as_unsupported(stateful, store):
    create(stateful)
    before = store.get("conv-1", AGENT_KEY)["draft"]

    for text in ("remove the capstone", "Add a module on SQL and remove the first module"):
        response = stateful.process_request(req(text))
        assert response["status"] == "error"
        assert response["result"]["code"] == "unsupported_request"

    # "make it more fun" matches no supported edit but is an edit attempt.
    response = stateful.process_request(req("make it more fun"))
    assert response["status"] == "error"
    assert response["result"]["capability"] == "curriculum_edit"
    assert store.get("conv-1", AGENT_KEY)["draft"] == before


def test_conversations_are_isolated(stateful, store):
    create(stateful, conv="A")
    create(stateful, conv="B", text="Create a curriculum about SQL")

    stateful.process_request(req("make it 40 hours", conv="A"))

    assert store.get("A", AGENT_KEY)["draft"]["training_hours"] == 40
    assert store.get("B", AGENT_KEY)["draft"]["training_hours"] == 30
    assert store.get("B", AGENT_KEY)["draft"]["title"] == "SQL Training Curriculum"


def test_refinement_without_a_draft_is_not_applied(stateful):
    response = stateful.process_request(req("make it 40 hours"))

    # No draft in this conversation: handled as an ordinary search.
    assert response["status"] == "success"
    assert titles(response) == []


def test_search_does_not_disturb_the_stored_draft(stateful, store):
    create(stateful)
    before = store.get("conv-1", AGENT_KEY)["draft"]

    stateful.process_request(req("Find a curriculum about cybersecurity"))

    assert store.get("conv-1", AGENT_KEY)["draft"] == before


def test_creating_again_replaces_the_draft(stateful, store):
    create(stateful)
    create(stateful, text="Create a curriculum about SQL")

    assert store.get("conv-1", AGENT_KEY)["draft"]["title"] == (
        "SQL Training Curriculum"
    )


def test_agent_without_a_store_is_stateless(agent, store):
    create(agent)

    response = agent.process_request(req("make it 40 hours"))

    assert titles(response) == []
    assert "refine this draft" not in agent.process_request(
        req("Create a curriculum about SQL")
    )["result"]["message"]


def test_no_conversation_id_means_no_memory(stateful, store):
    stateful.process_request(req("Create a curriculum about SQL", conv=None))

    assert stateful.has_refinable_draft(req("make it 40 hours", conv=None)) is False


def test_store_failures_do_not_break_requests():
    class BrokenStore:
        def get(self, *_):
            raise RuntimeError("disk error")

        def set(self, *_):
            raise RuntimeError("disk error")

    agent = CurriculumResearchAgent(conversation_store=BrokenStore())

    response = agent.process_request(req("Create a curriculum about SQL"))

    assert response["status"] == "success"
    assert "refine this draft" not in response["result"]["message"]
    assert agent.has_refinable_draft(req("make it 40 hours")) is False


def test_has_refinable_draft_needs_a_draft_and_a_refinement(stateful):
    assert stateful.has_refinable_draft(req("make it 40 hours")) is False

    create(stateful)

    assert stateful.has_refinable_draft(req("make it 40 hours")) is True
    assert stateful.has_refinable_draft(req("add a capstone")) is True
    assert stateful.has_refinable_draft(req("remove the capstone")) is True
    assert stateful.has_refinable_draft(req("yes")) is False
    assert stateful.has_refinable_draft(req("Hello there")) is False
    assert stateful.has_refinable_draft(req("make it 40 hours", conv="other")) is False


def test_drafts_survive_a_new_agent_and_store_on_the_same_database(tmp_path):
    db_path = tmp_path / "conversations.sqlite3"

    first = CurriculumResearchAgent(
        conversation_store=ConversationStore(db_path=db_path)
    )
    first.process_request(req("Create a curriculum about SQL"))

    second = CurriculumResearchAgent(
        conversation_store=ConversationStore(db_path=db_path)
    )
    response = second.process_request(req("make it 60 hours"))

    assert response["result"]["curricula"][0]["training_hours"] == 60


def test_conversation_store_schema_is_unchanged(tmp_path):
    db_path = tmp_path / "conversations.sqlite3"
    agent = CurriculumResearchAgent(
        conversation_store=ConversationStore(db_path=db_path)
    )
    agent.process_request(req("Create a curriculum about SQL"))

    connection = sqlite3.connect(db_path)
    columns = [
        row[1]
        for row in connection.execute("PRAGMA table_info(conversation_state)")
    ]
    rows = connection.execute(
        "SELECT conversation_id, agent_name FROM conversation_state"
    ).fetchall()
    connection.close()

    assert columns == ["conversation_id", "agent_name", "state_json"]
    assert rows == [("conv-1", AGENT_KEY)]
