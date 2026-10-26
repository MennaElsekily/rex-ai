"""
Regression tests for the heading-detection fix in knowledge_base.chunker.

A single capitalized word (e.g. a PDF line-wrap fragment like "Learning"
on its own line) must not be treated as a section heading: doing so
split real headings in half and mislabeled the "section" metadata used
for citations.
"""

from knowledge_base.chunker import chunk_document, _is_heading


def test_single_capitalized_word_is_not_a_heading():
    assert _is_heading("Learning") is False
    assert _is_heading("Preprocessing") is False
    assert _is_heading("Classification") is False


def test_multi_word_title_case_is_still_a_heading():
    assert _is_heading("Entry Requirements") is True
    assert _is_heading("Capstone Project") is True


def test_all_caps_heading_is_still_detected():
    assert _is_heading("CATALOGUE OVERVIEW") is True


def test_markdown_heading_is_still_detected():
    assert _is_heading("## Programme Summary") is True


def test_wrapped_multiword_title_is_not_split_by_trailing_word():
    document = {
        "source": "doc.pdf",
        "file_type": ".pdf",
        "pages": [
            {
                "page": 1,
                "text": (
                    "Applied Artificial Intelligence & Machine\n"
                    "Learning\n"
                    "CATALOGUE OVERVIEW\n"
                    "A structured learning portfolio covering Python "
                    "foundations and applied machine learning."
                ),
            }
        ],
    }

    chunks = chunk_document(document, max_chars=2500)

    sections = {chunk["section"] for chunk in chunks}

    assert "Learning" not in sections
    assert any(
        section and "Applied Artificial Intelligence" in section
        for section in sections
    )
