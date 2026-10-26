"""
Tests for knowledge_base.retriever, focused on the relevance-score
threshold added to stop clearly unrelated nearest-neighbor matches
from being treated as usable evidence (see answer_generator fabrication
fix: an out-of-domain question previously still received chunks, and
the LLM filled the gap with an unsupported answer).

The vector store and cross-encoder are stubbed so these tests do not
require ChromaDB data or downloading the reranking model.
"""

from unittest.mock import patch

from knowledge_base.retriever import Retriever


class FakeVectorStore:
    def __init__(self, candidates):
        self._candidates = candidates

    def search(self, query_embedding, top_k=5, where=None):
        return self._candidates[:top_k]


def _candidate(text, source="doc.pdf", page=1, section=None, chunk_id=1):
    return {
        "text": text,
        "metadata": {
            "source": source,
            "page": page,
            "section": section,
            "chunk_id": chunk_id,
        },
        "distance": 0.1,
    }


@patch("knowledge_base.retriever.embed_texts", return_value=[[0.1, 0.2]])
@patch("knowledge_base.retriever._load_cross_encoder")
def test_low_scoring_candidates_are_dropped(mock_load_cross_encoder, _mock_embed):
    mock_cross_encoder = mock_load_cross_encoder.return_value
    mock_cross_encoder.predict.return_value = [-10.5, -10.9]

    candidates = [
        _candidate("Unrelated passage about something else entirely here."),
        _candidate("Another unrelated passage with different wording too."),
    ]

    retriever = Retriever(vector_store=FakeVectorStore(candidates), top_k=5)
    results = retriever.retrieve("What is the capital of France?")

    assert results == []


@patch("knowledge_base.retriever.embed_texts", return_value=[[0.1, 0.2]])
@patch("knowledge_base.retriever._load_cross_encoder")
def test_high_scoring_candidates_are_kept(mock_load_cross_encoder, _mock_embed):
    mock_cross_encoder = mock_load_cross_encoder.return_value
    mock_cross_encoder.predict.return_value = [5.2, -10.9]

    candidates = [
        _candidate("Entry requirements: basic computer literacy is needed."),
        _candidate("Unrelated passage with completely different wording."),
    ]

    retriever = Retriever(vector_store=FakeVectorStore(candidates), top_k=5)
    results = retriever.retrieve("What are the entry requirements?")

    assert len(results) == 1
    assert "Entry requirements" in results[0]["text"]


@patch("knowledge_base.retriever.embed_texts", return_value=[[0.1, 0.2]])
@patch("knowledge_base.retriever._load_cross_encoder")
def test_threshold_can_be_disabled(mock_load_cross_encoder, _mock_embed):
    mock_cross_encoder = mock_load_cross_encoder.return_value
    mock_cross_encoder.predict.return_value = [-10.5]

    candidates = [_candidate("Unrelated passage with different wording here.")]

    retriever = Retriever(
        vector_store=FakeVectorStore(candidates),
        top_k=5,
        min_relevance_score=None,
    )
    results = retriever.retrieve("What is the capital of France?")

    assert len(results) == 1
