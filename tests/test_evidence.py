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
    }


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


def test_translation_failure_is_reported_and_search_still_runs(translator):
    translator.fail = True
    evidence = _evidence([_hit()])

    result = evidence.search("ما هي متطلبات الالتحاق؟")

    assert result["translation"] == "failed"
    assert "translation model unavailable" in result["translation_error"]
    assert result["retrieval_query"] == "ما هي متطلبات الالتحاق؟"
    assert result["status"] == STATUS_OK
    evidence.retriever.retrieve.assert_called_once()


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
