"""
Retrieval evaluation: Recall@k and MRR over a question set.

An evaluation item is {"question": str, "must_contain": [str, ...]},
optionally with "retrieval_query" when the calling agent searches with
its own query text (e.g. REX's hint for course-listing questions). A
retrieved chunk is relevant when its text contains every must_contain
string (case-insensitive) -- e.g. a course name AND its fee, so a chunk
stating "Fee: 7000" without saying which course it belongs to does not
count. Dataset-specific question sets live next to their datasets.

    python -m knowledge_base.evaluation --eval-set PATH [--store DIR] [--k 5]

This measures the full retrieval stage used by REX (query translation,
embedding, vector search, reranking, relevance threshold), not answer
quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def is_relevant(chunk_text: str, must_contain: list[str]) -> bool:
    text = chunk_text.casefold()
    return all(term.casefold() in text for term in must_contain)


def evaluate(search, items: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    """
    search(question, retrieval_query=None) -> list of chunk texts in rank
    order (best first).
    Returns recall@k, MRR (first relevant chunk within the top k) and the
    rank found for each item (None when not found).
    """
    rows = []
    for item in items:
        texts = list(search(item["question"], item.get("retrieval_query")))[:k]
        rank = next(
            (index for index, text in enumerate(texts, start=1) if is_relevant(text, item["must_contain"])),
            None,
        )
        rows.append({"question": item["question"], "language": item.get("language", "en"), "rank": rank})

    found = [row for row in rows if row["rank"] is not None]
    n = len(rows) or 1
    by_language: dict[str, dict[str, float]] = {}
    for language in sorted({row["language"] for row in rows}):
        subset = [row for row in rows if row["language"] == language]
        hits = [row for row in subset if row["rank"] is not None]
        by_language[language] = {
            "n": len(subset),
            "recall_at_k": round(len(hits) / len(subset), 3),
            "mrr": round(sum(1 / row["rank"] for row in hits) / len(subset), 3),
        }

    return {
        "k": k,
        "n": len(rows),
        "recall_at_k": round(len(found) / n, 3),
        "mrr": round(sum(1 / row["rank"] for row in found) / n, 3),
        "by_language": by_language,
        "items": rows,
    }


def main(argv: list[str] | None = None) -> int:
    from knowledge_base.evidence import EvidenceRetriever
    from knowledge_base.retriever import Retriever
    from knowledge_base.vector_store import VectorStore

    parser = argparse.ArgumentParser(description="Evaluate retrieval with Recall@k and MRR.")
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--store", default=None, help="vector store directory (default: knowledge_base/vector_db)")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)

    items = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))["items"]
    store = VectorStore(persist_directory=args.store) if args.store else VectorStore()
    evidence = EvidenceRetriever(Retriever(vector_store=store, top_k=args.k))

    def search(question: str, retrieval_query: str | None = None) -> list[str]:
        return [chunk["text"] for chunk in evidence.search(question, retrieval_query=retrieval_query)["chunks"]]

    print(json.dumps(evaluate(search, items, args.k), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
