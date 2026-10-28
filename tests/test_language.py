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


# --------------------------------------------------------------------
# Conversation language (what the user is speaking)
# --------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("What courses are available for beginners?", "en"),
    ("ما هي الدورات المتاحة؟", "ar"),                       # MSA
    ("أنا عايز آخد كورس في البرمجة، تنصحني بإيه؟", "ar"),   # Egyptian
    ("Ana 3ayez akhod course fel programming.", "ar"),      # Arabizi
    ("أنا عايز أتعلم Python عشان أشتغل Software Developer", "ar"),  # mixed
    ("تمام", "ar"),
    ("أيوه", "ar"),
    # No confident signal: short Latin replies, numbers, empty.
    ("ok", None),
    ("yes", None),
    ("yes please", None),
    ("online", None),
    ("5000", None),
    ("", None),
    # English with numbers is not Arabizi.
    ("I can study 5 hours a week", "en"),
])
def test_detect_conversation_language(text, expected):
    assert language.detect_conversation_language(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Please answer in English", "en"),
    ("Can you reply in Arabic?", "ar"),
    ("كلمني بالعربي لو سمحت", "ar"),
    ("ممكن ترد بالإنجليزي؟", "en"),
    ("I study English literature", None),
    ("ما هي الدورات المتاحة؟", None),
])
def test_detect_explicit_language_request(text, expected):
    assert language.detect_explicit_language_request(text) == expected


def test_short_reply_keeps_the_conversation_language():
    assert language.resolve_conversation_language("ok", previous_language="ar") == "ar"
    assert language.resolve_conversation_language("5000", previous_language="ar") == "ar"
    assert language.resolve_conversation_language("yes", previous_language="en") == "en"


def test_clear_switch_changes_the_conversation_language():
    assert language.resolve_conversation_language(
        "Actually, can you tell me about the Flutter course?", previous_language="ar"
    ) == "en"
    assert language.resolve_conversation_language(
        "طيب عايز أعرف أكتر", previous_language="en"
    ) == "ar"


def test_explicit_preference_wins_over_detection():
    assert language.resolve_conversation_language(
        "ما هي الدورات المتاحة؟", previous_language="ar", preferred_language="en"
    ) == "en"


def test_no_signal_and_no_history_defaults_to_english():
    assert language.resolve_conversation_language("ok") == "en"


# --------------------------------------------------------------------
# Retrieval-query preparation (translation status is reported)
# --------------------------------------------------------------------

class _RecordingTranslator:
    def __init__(self, reply="What courses exist?"):
        self.reply = reply
        self.calls = []

    def translate_to_english(self, text):
        self.calls.append(text)
        return self.reply


def test_prepare_english_query_is_not_translated():
    translator = _RecordingTranslator()
    language._translator_singleton = translator

    prepared = language.prepare_retrieval_query("What courses exist?")

    assert prepared.query == "What courses exist?"
    assert prepared.translation == "not_needed"
    assert prepared.language == "en"
    assert translator.calls == []


def test_prepare_arabic_query_is_translated_and_original_kept():
    language._translator_singleton = _RecordingTranslator()

    prepared = language.prepare_retrieval_query("ما هي الدورات؟")

    assert prepared.original == "ما هي الدورات؟"
    assert prepared.query == "What courses exist?"
    assert prepared.translation == "translated"
    assert prepared.language == "ar"


def test_prepare_arabizi_query_is_transliterated_before_translation():
    translator = _RecordingTranslator("I want to take a course in programming")
    language._translator_singleton = translator

    prepared = language.prepare_retrieval_query("Ana 3ayez akhod course fel programming")

    assert prepared.language == "arabizi"
    assert prepared.translation == "translated"
    # Arabizi -> Arabic script -> Egyptian words mapped to MSA. Only the
    # Arabic part is translated; the Latin word is kept verbatim.
    assert translator.calls == ["انا أريد اخد دورة تدريبية في"]
    assert prepared.query == "I want to take a course in programming"
    assert prepared.original == "Ana 3ayez akhod course fel programming"


def test_egyptian_words_are_mapped_to_msa_before_translation():
    translator = _RecordingTranslator("What does the data analysis course cover?")
    language._translator_singleton = translator

    prepared = language.prepare_retrieval_query("كورس تحليل البيانات بيغطي إيه؟")

    # Live run without this step: "The data analysis choir covers eh?";
    # with it: "A data analysis course covering what?"
    assert translator.calls == ["دورة تدريبية تحليل البيانات يغطي ماذا؟"]
    assert prepared.original == "كورس تحليل البيانات بيغطي إيه؟"


def test_egyptian_mapping_only_replaces_whole_words():
    # "كورسات" and "الكورس" have their own entries; unrelated words that
    # merely contain a mapped word are left alone.
    assert language.egyptian_to_msa("الكورسات") == "الدورات التدريبية"
    assert language.egyptian_to_msa("ديسمبر") == "ديسمبر"
    assert language.egyptian_to_msa("ما هي الدورات؟") == "ما هي الدورات؟"


def test_prepare_reports_translation_failure_and_keeps_original():
    class BrokenTranslator:
        def translate_to_english(self, text):
            raise RuntimeError("translation model unavailable")

    language._translator_singleton = BrokenTranslator()

    prepared = language.prepare_retrieval_query("ما هي الدورات؟")

    assert prepared.translation == "failed"
    assert prepared.query == "ما هي الدورات؟"
    assert "translation model unavailable" in prepared.error


def test_prepare_reports_empty_translation_as_failure():
    language._translator_singleton = _RecordingTranslator(reply="  ")

    prepared = language.prepare_retrieval_query("ما هي الدورات؟")

    assert prepared.translation == "failed"
    assert prepared.query == "ما هي الدورات؟"


def test_importing_the_language_module_does_not_load_the_translator():
    import subprocess
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable, "-B", "-c",
            "import sys, knowledge_base.language; print('torch' in sys.modules)",
        ],
        cwd=project_root, capture_output=True, text=True, check=True,
    )

    assert result.stdout.strip() == "False"


# --------------------------------------------------------------------
# Mixed Arabic/English queries: Latin names, codes and numbers are kept
# --------------------------------------------------------------------

@pytest.mark.parametrize("query, sent, expected_parts", [
    ("ما هي رسوم دورة Flutter؟", "ما هي رسوم دورة ؟", ["Flutter"]),
    ("كم سعر PY-101 و Python Basics", "كم سعر و", ["PY-101", "Python Basics"]),
    ("دورة بسعر ٣٠٠٠ جنيه", "دورة بسعر جنيه", ["3000"]),
])
def test_mixed_queries_translate_only_the_arabic_part(query, sent, expected_parts):
    translator = _RecordingTranslator("what is the price of the course")
    language._translator_singleton = translator

    prepared = language.prepare_retrieval_query(query)

    assert translator.calls == [sent]
    assert all(part in prepared.query for part in expected_parts)
    assert prepared.original == query


def test_latin_only_content_after_normalization_is_not_translated():
    translator = _RecordingTranslator("unused")
    language._translator_singleton = translator

    prepared = language.prepare_retrieval_query("٢٠٢٦ PY-101")

    assert translator.calls == []
    assert prepared.query == "2026 PY-101"


def test_untranslated_output_is_a_failure_not_a_query():
    translator = _RecordingTranslator("ما هي الرسوم")  # model echoed Arabic
    language._translator_singleton = translator

    prepared = language.prepare_retrieval_query("ما هي الرسوم")

    assert prepared.translation == "failed"
    assert prepared.query == "ما هي الرسوم"


@pytest.mark.parametrize("arabic, latin", [
    ("كم سعر كورس فلاتر؟", "Flutter"),
    ("عايز اتعلم بايثون", "Python"),
    ("كورس الباور بي آي", "Power BI"),
])
def test_arabic_spelled_tech_names_are_kept_in_latin_script(arabic, latin):
    translator = _RecordingTranslator("how much is the course")
    language._translator_singleton = translator

    prepared = language.prepare_retrieval_query(arabic)

    assert latin in prepared.query
    assert latin not in translator.calls[0]  # sent around the translator, not through it


def test_tech_terms_only_match_whole_words():
    assert "Flutter" not in language.egyptian_to_msa("فلاترات")
