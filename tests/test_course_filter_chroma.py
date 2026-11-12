"""
The RAG service's course filter (knowledge_base/evidence.field_course_filter;
moved from REX, which no longer builds database filters) against a real
Chroma collection created in a temporary folder -- never
the knowledge-base index. Generalization check T4: ingestion stores a
numeric course id as a number, and Chroma compares types strictly.
"""

import chromadb
import pytest

from knowledge_base.evidence import _check_filter, field_course_filter


def course_filter(courses):
    return field_course_filter([c["course_id"] for c in courses])
from knowledge_base.metadata_extractor import normalize_field_value


@pytest.fixture
def collection(tmp_path):
    collection = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_or_create_collection("course_filter")
    rows = {
        "numeric-as-number": {"field_course_id": normalize_field_value("1010")},   # stored as int 1010
        "numeric-as-text": {"field_course_id": "1010"},
        "intake-code": {"field_course_code": normalize_field_value("1010")},
        "text-id": {"field_course_id": "DA-301"},
        "intake-of-text-id": {"field_course_code": "DA-301"},
        "general": {"source": "faqs.json"},
    }
    collection.add(ids=list(rows), metadatas=list(rows.values()), documents=list(rows),
                   embeddings=[[float(n), 1.0] for n in range(len(rows))])   # no embedding model needed
    return collection


def _found(collection, course_id):
    return sorted(collection.get(where=course_filter([{"course_id": course_id}]))["ids"])


def test_numeric_course_id_is_found_whether_stored_as_number_or_text(collection):
    assert normalize_field_value("1010") == 1010
    assert _found(collection, "1010") == ["intake-code", "numeric-as-number", "numeric-as-text"]


def test_text_course_id_filter_is_unchanged_and_exact(collection):
    assert course_filter([{"course_id": "DA-301"}]) == {"$or": [
        {"field_course_id": {"$in": ["DA-301"]}}, {"field_course_code": {"$in": ["DA-301"]}}]}
    assert _found(collection, "DA-301") == ["intake-of-text-id", "text-id"]


def test_numeric_filter_passes_the_evidence_filter_validation():
    _check_filter(course_filter([{"course_id": "1010"}, {"course_id": "DA-301"}]))
