from pathlib import Path
from typing import Any

import chromadb


DEFAULT_STORE_PATH = str(
    Path(__file__).resolve().parent / "vector_db"
)


def _flat_value(value: Any) -> str | int | float | bool | None:
    """A metadata value as a Chroma scalar, or None to leave it out."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value if item not in (None, "")]
        return "; ".join(parts) if parts else None
    return None


class VectorStore:
    """Store and retrieve document chunks using ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_STORE_PATH,
        collection_name: str = "rex_knowledge",
    ) -> None:
        self.persist_directory = str(
            Path(persist_directory).resolve()
        )

        Path(self.persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        """Store chunks and their precomputed embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Each chunk must have exactly one embedding"
            )

        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk, embedding in zip(chunks, embeddings):
            text = chunk.get("text")
            chunk_id = chunk.get("chunk_id")
            source = chunk.get("source")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Every chunk must contain non-empty text"
                )

            if chunk_id is None or not source:
                raise ValueError(
                    "Every chunk must have chunk_id and source"
                )

            ids.append(f"{source}::chunk_{chunk_id}")
            documents.append(text)

            # ChromaDB accepts scalar metadata values, not nested dicts.
            metadata = {
                key: value
                for key, value in chunk.items()
                if key not in {"text", "chunk_id", "extracted_metadata"}
                and isinstance(value, (str, int, float, bool))
                and value is not None
            }

            metadata["chunk_id"] = int(chunk_id)

            # Store extracted fields as flat, searchable metadata. Chroma
            # accepts scalars only: lists become "a; b" strings, and the
            # document's explicit fields become "field_<name>" entries
            # (usable in metadata filters, never colliding with these keys).
            extracted = chunk.get("extracted_metadata", {})

            if isinstance(extracted, dict):
                for key, value in extracted.items():
                    if not isinstance(key, str) or key in metadata:
                        continue
                    if key == "explicit_fields" and isinstance(value, dict):
                        for field, field_value in value.items():
                            flat = _flat_value(field_value)
                            if flat is not None:
                                metadata.setdefault(f"field_{field}", flat)
                        continue
                    flat = _flat_value(value)
                    if flat is not None:
                        metadata[key] = flat

            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(ids)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve similar chunks, optionally filtered by metadata."""
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")

        if self.collection.count() == 0:
            return []

        query_options = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }

        if where:
            query_options["where"] = where

        result = self.collection.query(**query_options)

        matches = []

        for i, text in enumerate(result["documents"][0]):
            matches.append({
                "text": text,
                "metadata": result["metadatas"][0][i],
                "distance": result["distances"][0][i],
            })

        return matches

    def delete_source(self, source: str) -> int:
        """Delete all stored chunks belonging to one source file."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")

        results = self.collection.get(
            where={"source": source},
            include=["metadatas"],
        )

        ids = results.get("ids", [])

        if ids:
            self.collection.delete(ids=ids)

        return len(ids)