"""
Tests for knowledge_base.answer_generator.RAGAnswerGenerator.

The retriever and ollama.chat are mocked so these tests validate the
answer generator's own logic (language handling, context assembly,
fallback behavior) without requiring ChromaDB data, the embedding
model, or a running Ollama server.
"""

from unittest.mock import MagicMock, patch

import pytest

from knowledge_base.answer_generator import RAGAnswerGenerator


def _result(text="Entry requirements: basic computer literacy.", **overrides):
    base = {
        "text": text,
        "source": "doc.pdf",
        "page": 1,
        "section": "Entry requirements",
        "chunk_id": 1,
    }
    base.update(overrides)
    return base


def _make_generator(retriever_results):
    retriever = MagicMock()
    retriever.retrieve.return_value = retriever_results
    return RAGAnswerGenerator(retriever=retriever)


def test_question_must_be_non_empty_string():
    generator = _make_generator([])

    with pytest.raises(ValueError):
        generator.answer("   ")


def test_no_results_returns_english_fallback_by_default():
    generator = _make_generator([])

    result = generator.answer("What is the capital of France?")

    assert result["sources"] == []
    assert "couldn't find relevant information" in result["answer"]


def test_no_results_returns_arabic_fallback_when_question_is_arabic():
    generator = _make_generator([])

    result = generator.answer("ما هي الدورات المتاحة؟")

    assert "لم أجد معلومات" in result["answer"]


def test_no_results_honors_explicit_preferred_language_over_detected():
    generator = _make_generator([])

    # English question, but the user prefers Arabic responses.
    result = generator.answer(
        "What is the capital of France?",
        preferred_language="ar",
    )

    assert "لم أجد معلومات" in result["answer"]


@patch("knowledge_base.answer_generator.ollama")
def test_successful_answer_includes_sources_and_model(mock_ollama):
    mock_ollama.chat.return_value = {
        "message": {"content": "Basic computer literacy is required. [1]"}
    }

    generator = _make_generator([_result()])

    result = generator.answer("What are the entry requirements?")

    assert result["model"] == "qwen2.5:3b"
    assert result["sources"] == [
        {
            "reference": 1,
            "source": "doc.pdf",
            "page": 1,
            "section": "Entry requirements",
            "chunk_id": 1,
        }
    ]
    assert "Basic computer literacy" in result["answer"]
    mock_ollama.chat.assert_called_once()


@patch("knowledge_base.answer_generator.ollama")
def test_oversized_chunk_is_skipped_not_truncated(mock_ollama):
    mock_ollama.chat.return_value = {"message": {"content": "answer"}}

    generator = RAGAnswerGenerator(
        retriever=MagicMock(
            retrieve=MagicMock(return_value=[_result(text="x" * 50)])
        ),
        max_context_chars=10,
    )

    result = generator.answer("What are the entry requirements?")

    # The chunk cannot fit in the tiny context budget, so it must be
    # skipped entirely rather than silently truncated.
    assert result["sources"] == []
    assert "couldn't prepare usable evidence" in result["answer"]


@patch("knowledge_base.answer_generator.ollama")
def test_preferred_language_is_passed_into_the_prompt(mock_ollama):
    captured = {}

    def fake_chat(model, messages, options):
        captured["system"] = messages[0]["content"]
        captured["user"] = messages[1]["content"]
        return {"message": {"content": "answer"}}

    mock_ollama.chat.side_effect = fake_chat

    generator = _make_generator([_result()])
    generator.answer(
        "What are the entry requirements?",
        preferred_language="ar",
    )

    assert "Arabic" in captured["system"]
    assert "preferred response language" in captured["user"]
