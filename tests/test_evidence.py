"""
knowledge_base/evidence.py: the shared retrieval step (no answer writing).

Mocked: the Retriever (and its vector store) and the translator, so these
tests need no embedding model, Chroma index, MarianMT model or network.
Real-service checks against a copy of the index are run separately.
"""

from unittest.mock import MagicMock

import pytest

import knowledge_base.language as language
from knowledge_base.evidence import (
    STATUS_EMPTY_INDEX,
    STATUS_ERROR,
    STATUS_INVALID_REQUEST,
    STATUS_NO_RESULTS,
    STATUS_OK,
    EvidenceRetriever,
    safe_list_courses,
    safe_search,
)


class FakeTranslator:
    def __init__(self, reply="What courses exist?", fail=False):
        self.reply = reply
        self.fail = fail
        self.calls = []

    def translate_to_english(self, text):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("translation model unavailable")
        return self.reply


@pytest.fixture(autouse=True)
def translator():
    fake = FakeTranslator()
    language._translator_singleton = fake
    yield fake
    language._translator_singleton = None


def _hit(text="Entry requirements: basic computer literacy.", **overrides):
    hit = {
        "text": text,
        "metadata": {
            "source": r"E:\private\academy\knowledge_base\documents\04_Flutter.pdf",
            "data_source": r"E:\private\academy\knowledge_base\documents\04_Flutter.pdf",
            "document_id": r"E:\private\academy\knowledge_base\documents\04_Flutter.pdf",
            "page": 3,
        },
        "source": r"E:\private\academy\knowledge_base\documents\04_Flutter.pdf",
        "page": 3,
        "section": "Entry requirements",
        "chunk_id": 12,
        "distance": 0.21,
        "relevance_score": 1.23456,
    }
    hit.update(overrides)
    return hit


def _evidence(hits=None, indexed=623):
    retriever = MagicMock()
    retriever.vector_store.collection.count.return_value = indexed
    retriever.retrieve.return_value = [] if hits is None else hits
    return EvidenceRetriever(retriever=retriever)


def test_english_question_returns_structured_evidence(translator):
    evidence = _evidence([_hit(), _hit(text="Duration: 60 hours.", chunk_id=13)])

    result = evidence.search("What are the entry requirements for Flutter?")

    assert result["status"] == STATUS_OK
    assert result["question"] == "What are the entry requirements for Flutter?"
    assert result["retrieval_query"] == result["question"]
    assert result["query_language"] == "en"
    assert result["translation"] == "not_needed"
    assert translator.calls == []
    assert result["error"] is None
    assert "answer" not in result
    assert [chunk["reference"] for chunk in result["chunks"]] == [1, 2]
    assert result["chunks"][0] == {
        "reference": 1,
        "text": "Entry requirements: basic computer literacy.",
        "source": "04_Flutter.pdf",
        "page": 3,
        "section": "Entry requirements",
        "chunk_id": 12,
        "relevance_score": 1.2346,
        # Added compatibly: provenance and structured fields.
        "page_end": None,
        "section_path": None,
        "record_id": None,
        "fields": {},
        # Added compatibly (two-mode REX, Phase 2): the course, when the
        # metadata names one -- this hit has no course metadata.
        "course_id": None,
        "course_name": None,
        # Added compatibly (MCP trace): the similarity-search distance.
        "distance": 0.21,
    }
    assert result["courses"] == []


def test_source_references_never_expose_filesystem_paths():
    evidence = _evidence([_hit(), _hit(source="/srv/app/docs/02_Data.pdf")])

    result = evidence.search("What are the entry requirements?")

    flat = repr(result)
    assert "E:\\" not in flat and "private" not in flat and "/srv" not in flat
    assert [chunk["source"] for chunk in result["chunks"]] == [
        "04_Flutter.pdf", "02_Data.pdf",
    ]


def test_arabic_question_is_translated_and_original_preserved(translator):
    evidence = _evidence([_hit()])

    result = evidence.search("ما هي متطلبات الالتحاق؟")

    assert result["question"] == "ما هي متطلبات الالتحاق؟"
    assert result["retrieval_query"] == "What courses exist?"
    assert result["query_language"] == "ar"
    assert result["translation"] == "translated"
    kwargs = evidence.retriever.retrieve.call_args.kwargs
    assert kwargs["query"] == "ما هي متطلبات الالتحاق؟"
    assert kwargs["retrieval_query"].startswith("What courses exist?")


def test_caller_supplied_retrieval_query_is_used_and_translated(translator):
    translator.reply = "learner goal data analyst beginner"
    evidence = _evidence([_hit()])

    result = evidence.search(
        "Which course do you recommend?",
        retrieval_query="متعلم هدفه محلل بيانات مبتدئ",
    )

    assert translator.calls == ["متعلم هدفه محلل بيانات مبتدئ"]
    assert result["question"] == "Which course do you recommend?"
    assert result["retrieval_query"] == "learner goal data analyst beginner"
    assert evidence.retriever.retrieve.call_args.kwargs["retrieval_query"] == (
        "learner goal data analyst beginner"
    )


def test_arabizi_question_is_transliterated_then_translated(translator):
    translator.reply = "I want to take a programming course"
    evidence = _evidence([_hit()])

    result = evidence.search("Ana 3ayez akhod course fel programming")

    assert result["query_language"] == "arabizi"
    assert result["translation"] == "translated"
    assert translator.calls == ["انا أريد اخد دورة تدريبية في"]
    assert "programming" in result["retrieval_query"]


def test_translation_failure_is_reported_and_nothing_is_retrieved(translator):
    # Changed on purpose: searching the English-only index with untranslated
    # Arabic returned unrelated passages, so a failed translation now stops
    # the search instead of silently searching the Arabic text.
    translator.fail = True
    evidence = _evidence([_hit()])

    result = evidence.search("ما هي متطلبات الالتحاق؟")

    assert result["translation"] == "failed"
    assert "translation model unavailable" in result["translation_error"]
    assert result["question"] == "ما هي متطلبات الالتحاق؟"
    assert result["status"] == "translation_failed" and result["chunks"] == []
    evidence.retriever.retrieve.assert_not_called()


def test_the_generic_engine_adds_no_domain_specific_hint():
    # Course-catalogue search hints belong to REX (agents/rex_assistant),
    # not to the shared retrieval engine.
    evidence = _evidence([_hit()])

    result = evidence.search("What courses are available?")

    searched = evidence.retriever.retrieve.call_args.kwargs["retrieval_query"]
    assert searched == "What courses are available?"
    assert result["retrieval_query"] == "What courses are available?"


def test_chunks_carry_section_path_record_id_and_fields():
    hit = _hit()
    hit["metadata"].update({"section_path": "Catalogue > Python Basics", "record_id": "PY-101",
                            "page_end": 4, "field_level": "Beginner", "field_fee": 3000})
    evidence = _evidence([hit])

    chunk = evidence.search("fees?")["chunks"][0]

    assert chunk["section_path"] == "Catalogue > Python Basics"
    assert chunk["record_id"] == "PY-101" and chunk["page_end"] == 4
    assert chunk["fields"] == {"level": "Beginner", "fee": 3000}


def test_metadata_filter_is_passed_to_the_retriever():
    evidence = _evidence([_hit()])

    evidence.search("Duration?", metadata_filter={"file_type": ".pdf"})

    assert evidence.retriever.retrieve.call_args.kwargs["metadata_filter"] == {
        "file_type": ".pdf"
    }


def test_no_relevant_chunks_is_no_results_not_an_error():
    result = _evidence([]).search("underwater basket weaving course")

    assert result["status"] == STATUS_NO_RESULTS
    assert result["chunks"] == []
    assert result["error"] is None


def test_chunks_without_text_are_dropped():
    result = _evidence([_hit(text="   ")]).search("What are the fees?")

    assert result["status"] == STATUS_NO_RESULTS
    assert result["chunks"] == []


def test_empty_index_is_reported_and_retrieval_is_skipped():
    evidence = _evidence([_hit()], indexed=0)

    result = evidence.search("What courses are available?")

    assert result["status"] == STATUS_EMPTY_INDEX
    assert result["error"]
    assert result["chunks"] == []
    evidence.retriever.retrieve.assert_not_called()


@pytest.mark.parametrize("kwargs", [
    {"question": "   "},
    {"question": "ok", "retrieval_query": "  "},
    {"question": "ok", "metadata_filter": "not a dict"},
])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        _evidence().search(**kwargs)

    result = safe_search(_evidence(), **kwargs)
    assert result["status"] == STATUS_INVALID_REQUEST
    assert result["chunks"] == []


def test_retrieval_failure_becomes_an_error_status_without_internals():
    evidence = _evidence()
    evidence.retriever.retrieve.side_effect = RuntimeError(
        r"chroma failure at E:\private\knowledge_base\vector_db"
    )

    result = safe_search(evidence, "What courses are available?")

    assert result["status"] == STATUS_ERROR
    assert result["error"] == "The knowledge-base search failed."
    assert "private" not in repr(result)
    assert result["chunks"] == []


def test_unavailable_index_becomes_an_error_status():
    evidence = _evidence()
    evidence.retriever.vector_store.collection.count.side_effect = RuntimeError(
        "database is locked"
    )

    result = safe_search(evidence, "What courses are available?")

    assert result["status"] == STATUS_ERROR
    assert result["question"] == "What courses are available?"


# --------------------------------------------------------------------
# Two-mode REX, Phase 2: course identity, filter validation, failures
# --------------------------------------------------------------------

def _course_hit(text, course_id=None, name=None, section_path=None, code=None):
    metadata = {"source": "knowledge_base/documents/catalogue.pdf", "page": 2}
    if course_id:
        metadata["field_course_id"] = course_id
    if code:
        metadata["field_course_code"] = code
    if name:
        metadata["field_course_name"] = name
    if section_path:
        metadata["section_path"] = section_path
    return _hit(text=text, metadata=metadata, source=metadata["source"])


def test_chunks_name_their_course_and_results_list_the_courses():
    evidence = _evidence([
        _course_hit("Python for Data Analysis: 40 hours.", "DA-201", "Python for Data Analysis"),
        _course_hit("Covers pandas and charts.", "DA-201", section_path="Python for Data Analysis > Topics"),
        _course_hit("Python Programming Fundamentals: 30 hours.", None, "Python Programming Fundamentals",
                    code="SW-101"),
        _course_hit("Refunds within 14 days.", section_path="FAQ > Refunds"),
    ])

    result = evidence.search("How long is the Python course?")

    chunks = result["chunks"]
    assert (chunks[0]["course_id"], chunks[0]["course_name"]) == ("DA-201", "Python for Data Analysis")
    # The name comes from the section path when only the id is indexed.
    assert (chunks[1]["course_id"], chunks[1]["course_name"]) == ("DA-201", "Python for Data Analysis")
    assert (chunks[2]["course_id"], chunks[2]["course_name"]) == ("SW-101", "Python Programming Fundamentals")
    # No course metadata: nothing is guessed from the section path alone.
    assert (chunks[3]["course_id"], chunks[3]["course_name"]) == (None, None)
    assert result["courses"] == [
        {"course_id": "DA-201", "course_name": "Python for Data Analysis"},
        {"course_id": "SW-101", "course_name": "Python Programming Fundamentals"},
    ]


@pytest.mark.parametrize("metadata_filter", [
    {"field_course_id": "DA-201"},
    {"field_level": {"$in": ["Beginner", "Intermediate"]}},
    {"$and": [{"field_department": "Data Analytics"}, {"field_fee_egp": {"$lte": 5000}}]},
])
def test_valid_metadata_filters_reach_the_retriever(metadata_filter):
    evidence = _evidence([_hit()])

    assert safe_search(evidence, "Courses?", metadata_filter=metadata_filter)["status"] == STATUS_OK
    assert evidence.retriever.retrieve.call_args.kwargs["metadata_filter"] == metadata_filter


@pytest.mark.parametrize("metadata_filter", [
    {}, {"field_level": ["Beginner"]}, {"field_level": {"$like": "Beg%"}},
    {"field_level": {"$in": []}}, {"$or": [{"a": 1}]}, {"$drop": 1}, {"field_level": {"$eq": {"x": 1}}},
])
def test_malformed_metadata_filters_are_invalid_requests_not_failures(metadata_filter):
    evidence = _evidence([_hit()])

    result = safe_search(evidence, "Courses?", metadata_filter=metadata_filter)

    assert result["status"] == STATUS_INVALID_REQUEST and result["chunks"] == [] and result["courses"] == []
    evidence.retriever.retrieve.assert_not_called()


def test_course_list_reads_distinct_courses_from_the_metadata():
    # Two-mode REX, Phase 4: the indexed courses, read-only.
    evidence = _evidence()
    evidence.retriever.vector_store.collection.get.return_value = {"metadatas": [
        {"field_course_id": "DA-301", "section_path": "Python for Data Analysis > Fees"},
        {"field_course_id": "DA-301", "field_course_name": "Python for Data Analysis"},
        {"field_course_code": "AI-101", "section_path": "Introduction to AI and Python"},
        {"section_path": "FAQ > Refunds"},   # no course
        None,
    ]}

    result = evidence.list_courses()

    assert result == {"status": STATUS_OK, "error": None, "courses": [
        {"course_id": "AI-101", "course_name": "Introduction to AI and Python"},
        {"course_id": "DA-301", "course_name": "Python for Data Analysis"},
    ]}
    collection = evidence.retriever.vector_store.collection
    assert collection.get.call_args_list[0].kwargs == {"include": ["metadatas"]}
    for write in ("add", "upsert", "update", "delete"):   # read-only
        getattr(collection, write).assert_not_called()


def test_course_list_reports_an_empty_index_and_failures_safely():
    assert _evidence(indexed=0).list_courses()["status"] == STATUS_EMPTY_INDEX

    broken = _evidence()
    broken.retriever.vector_store.collection.get.side_effect = RuntimeError(r"C:\secret\vector_db locked")
    result = safe_list_courses(broken)

    assert result == {"status": STATUS_ERROR, "courses": [], "error": "The course list is unavailable."}


def test_every_status_has_the_same_keys():
    keys = set(_evidence([_hit()]).search("Courses?"))
    empty = _evidence([_hit()], indexed=0).search("Courses?")
    no_results = _evidence([]).search("Courses?")
    broken = _evidence([_hit()])
    broken.retriever.retrieve.side_effect = RuntimeError(r"C:\secret\path failed")
    failed = safe_search(broken, "Courses?")

    assert empty["status"] == STATUS_EMPTY_INDEX and no_results["status"] == STATUS_NO_RESULTS
    assert failed["status"] == STATUS_ERROR and "secret" not in failed["error"]
    for result in (empty, no_results, failed):
        assert set(result) == keys
