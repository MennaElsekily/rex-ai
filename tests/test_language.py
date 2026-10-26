"""
Tests for knowledge_base.language: detection, retrieval-query
construction, and preferred-response-language resolution.

These tests mock the translator so they do not require downloading or
running the MarianMT translation model.
"""

import pytest

import knowledge_base.language as language


@pytest.fixture(autouse=True)
def reset_translator_singleton():
    """Avoid leaking a mocked translator between tests."""
    language._translator_singleton = None
    yield
    language._translator_singleton = None


def test_detect_language_english():
    assert language.detect_language("What courses are available?") == "en"


def test_detect_language_arabic():
    assert language.detect_language("ما هي الدورات المتاحة؟") == "ar"


def test_detect_language_mixed():
    assert language.detect_language("hello ما هي الدورات") == "mixed"


def test_detect_language_empty():
    assert language.detect_language("") == "empty"
    assert language.detect_language("   ") == "empty"


def test_detect_language_other_when_no_letters():
    assert language.detect_language("12345 !!!") == "other"


def test_build_retrieval_query_passes_english_through_unchanged():
    class ShouldNotBeCalled:
        def translate_to_english(self, text):
            raise AssertionError("English text must not be translated")

    language._translator_singleton = ShouldNotBeCalled()

    query, detected = language.build_retrieval_query("What courses exist?")

    assert query == "What courses exist?"
    assert detected == "en"


def test_build_retrieval_query_translates_arabic():
    class FakeTranslator:
        def translate_to_english(self, text):
            return "What courses exist?"

    language._translator_singleton = FakeTranslator()

    query, detected = language.build_retrieval_query("ما هي الدورات؟")

    assert query == "What courses exist?"
    assert detected == "ar"


def test_build_retrieval_query_falls_back_on_translation_failure():
    class BrokenTranslator:
        def translate_to_english(self, text):
            raise RuntimeError("translation model unavailable")

    language._translator_singleton = BrokenTranslator()

    original = "ما هي الدورات؟"
    query, detected = language.build_retrieval_query(original)

    assert query == original
    assert detected == "ar"


def test_resolve_response_language_prefers_explicit_preference():
    assert language.resolve_response_language("ar", "en") == "en"
    assert language.resolve_response_language("en", "ar") == "ar"


def test_resolve_response_language_falls_back_to_detected():
    assert language.resolve_response_language("ar", None) == "ar"
    assert language.resolve_response_language("en", None) == "en"


def test_resolve_response_language_unknown_without_preference():
    assert language.resolve_response_language("other", None) is None
    assert language.resolve_response_language("mixed", None) is None
