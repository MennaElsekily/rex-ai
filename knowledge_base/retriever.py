import re
from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

from knowledge_base.embedding_service import embed_texts
from knowledge_base.vector_store import VectorStore

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Calibrated against this specific cross-encoder's raw logit output
# (not a dataset-specific value): queries unrelated to the indexed
# corpus consistently score below -9.5, while genuinely relevant
# matches score well above it, even when wording differs from the
# source text. Chunks below this score are dropped so the answer
# generator does not treat unrelated nearest-neighbor matches as
# usable evidence.
MIN_RELEVANCE_SCORE = -9.5


@lru_cache(maxsize=1)
def _load_cross_encoder() -> CrossEncoder:
    """Load the reranking model once and reuse it."""
    return CrossEncoder(CROSS_ENCODER_MODEL)


def _normalize_text(text: str) -> str:
    """Normalize whitespace and case for duplicate detection."""
    return " ".join(text.casefold().split())


def _is_low_information(text: str) -> bool:
    """
    Identify chunks that contain too little information to answer
    most questions usefully, without relying on specific document names.
    """
    normalized = _normalize_text(text)
    words = re.findall(r"\b\w+\b", normalized)

    if len(words) < 5:
        return True

    # Ignore repeated headings with no meaningful descriptive content.
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(lines) >= 2 and len(set(
        line.casefold() for line in lines
    )) == 1:
        return True

    return False


class Retriever:
    """Generic semantic retrieval with Cross-Encoder reranking."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        top_k: int = 5,
        candidate_multiplier: int = 10,
        min_relevance_score: float | None = MIN_RELEVANCE_SCORE,
    ) -> None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if (
            not isinstance(candidate_multiplier, int)
            or isinstance(candidate_multiplier, bool)
        ):
            raise TypeError("candidate_multiplier must be an integer")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")

        if min_relevance_score is not None and not isinstance(
            min_relevance_score, (int, float)
        ):
            raise TypeError("min_relevance_score must be a number or None")

        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k
        self.candidate_multiplier = candidate_multiplier
        self.min_relevance_score = min_relevance_score

    def retrieve(
        self,
        query: str,
        retrieval_query: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks and rerank them by query relevance."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty")

        if retrieval_query is None:
            retrieval_query = query

        if not isinstance(retrieval_query, str):
            raise TypeError("retrieval_query must be a string")

        retrieval_query = retrieval_query.strip()

        if not retrieval_query:
            raise ValueError("retrieval_query cannot be empty")

        query_embedding = embed_texts([retrieval_query])[0]

        candidates = self.vector_store.search(
            query_embedding,
            top_k=self.top_k * self.candidate_multiplier,
            where=metadata_filter,
        )

        if not candidates:
            return []

        # Remove duplicate and low-information chunks before reranking,
        # so repeated headings do not occupy the candidate pool.
        filtered_candidates = []
        seen_texts = set()

        for item in candidates:
            text = item.get("text", "")

            if not isinstance(text, str) or not text.strip():
                continue

            normalized_text = _normalize_text(text)

            if normalized_text in seen_texts:
                continue

            seen_texts.add(normalized_text)

            if _is_low_information(text):
                continue

            filtered_candidates.append(item)

        if not filtered_candidates:
            return []

        cross_encoder = _load_cross_encoder()

        scores = cross_encoder.predict(
            [
                (retrieval_query, item["text"])
                for item in filtered_candidates
            ],
            batch_size=16,
            show_progress_bar=False,
        )

        ranked_results = []

        for item, score in zip(filtered_candidates, scores):
            metadata = item.get("metadata") or {}

            ranked_results.append({
                "text": item["text"],
                "metadata": metadata,
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "section": metadata.get("section"),
                "chunk_id": metadata.get("chunk_id"),
                "distance": float(item["distance"]),
                "relevance_score": float(score),
            })

        ranked_results.sort(
            key=lambda item: item["relevance_score"],
            reverse=True,
        )

        if self.min_relevance_score is not None:
            ranked_results = [
                item
                for item in ranked_results
                if item["relevance_score"] >= self.min_relevance_score
            ]

        return ranked_results[:self.top_k]
