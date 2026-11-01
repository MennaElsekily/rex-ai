"""
Several records listed under one heading (knowledge_base/chunker.py,
_split_records): each record keeps its own fields, so no chunk holds one
record's text under another record's id. Generalization check T1.
"""

import pytest

from knowledge_base.chunker import chunk_document
from knowledge_base.metadata_extractor import extract_metadata


def _chunks(text, max_chars=1000):
    document = {"source": "short_courses.md", "file_type": ".md", "pages": [{"page": 1, "text": text}]}
    return [(chunk, extract_metadata(chunk)["explicit_fields"]) for chunk in chunk_document(document, max_chars=max_chars)]


TWO_COURSES = (
    "# Short Courses\n"
    "These short courses run every summer.\n\n"
    "Course ID: AA-100\nCourse Name: Alpha Analytics\nDuration: 2 weeks\n"
    "Alpha Analytics teaches dashboards for small shops.\n\n"
    "Course ID: BB-200\nCourse Name: Beta Branding\nDuration: 6 weeks\n"
    "Beta Branding teaches logo design and brand voice."
)


@pytest.mark.parametrize("max_chars", [1000, 160])
def test_each_record_under_one_heading_keeps_its_own_fields(max_chars):
    labelled = [(fields.get("course_id"), chunk["text"]) for chunk, fields in _chunks(TWO_COURSES, max_chars)]

    for course_id, text in labelled:
        if "Alpha" in text:
            assert course_id == "AA-100" and "Beta" not in text
        if "Beta" in text:
            assert course_id == "BB-200" and "Alpha" not in text
    assert {course_id for course_id, _ in labelled} >= {"AA-100", "BB-200"}


def test_records_keep_the_heading_path_and_the_preamble_has_no_course():
    chunks = _chunks(TWO_COURSES)

    assert all(chunk["section_path"] == "Short Courses" for chunk, _ in chunks)
    [(preamble, fields)] = [(c, f) for c, f in chunks if "every summer" in c["text"]]
    assert "course_id" not in fields and "Alpha" not in preamble["text"]


def test_a_name_written_above_the_id_stays_with_its_record():
    text = ("# Short Courses\n"
            "Course Name: Alpha Analytics\nCourse ID: AA-100\nAlpha teaches dashboards.\n\n"
            "Course Name: Beta Branding\nCourse ID: BB-200\nBeta teaches logos.")

    names = {fields.get("course_id"): fields.get("course_name") for _, fields in _chunks(text)}

    assert names == {"AA-100": "Alpha Analytics", "BB-200": "Beta Branding"}


def test_one_course_with_subsections_still_inherits_its_id():
    text = ("# Alpha Analytics\nCourse ID: AA-100\nAn introduction.\n\n"
            "## Fees\nFee: 3000 EGP\n\n## Schedule\nStarts in June.")

    chunks = _chunks(text, max_chars=130)   # small enough to give each subsection its own chunk

    assert len(chunks) > 1
    assert {fields.get("course_id") for _, fields in chunks} == {"AA-100"}


@pytest.mark.parametrize("text", [
    # A repeated field that is not an identity field does not split.
    "# Alpha Analytics\nCourse ID: AA-100\nModule 1\nDuration: 2 weeks\nModule 2\nDuration: 3 weeks",
    # The same id written twice is still one record.
    "# Alpha Analytics\nCourse ID: AA-100\nOverview of the course.\nCourse ID: AA-100\nMore details.",
])
def test_sections_with_one_identity_value_are_not_split(text):
    chunks = _chunks(text)

    assert len(chunks) == 1
    assert chunks[0][0]["chunk_strategy"] != "record"
