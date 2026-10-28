
from functools import lru_cache

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load each embedding model once and reuse it."""
    return SentenceTransformer(model_name)


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[list[float]]:
    """Convert multiple text passages into embedding vectors."""
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings")

    if any(not isinstance(text, str) for text in texts):
        raise TypeError("Every item in texts must be a string")

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    if not texts:
        return []

    model = _load_model(model_name)

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return vectors.tolist()
