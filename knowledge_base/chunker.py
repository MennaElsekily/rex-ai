import re
from typing import Any

from knowledge_base.embedding_service import embed_texts


SEMANTIC_SIMILARITY_THRESHOLD = 0.30

HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+\S"),
    re.compile(r"^\d+(?:\.\d+)*(?:[.)])?\s+\S"),
    re.compile(r"^[A-Z][A-Z0-9 &/(),:'-]{2,}$"),
    re.compile(
        r"^[A-Z][a-zA-Z0-9]*(?:[ \t]+[A-Za-z0-9&/(),:'-]+){1,7}$"
    ),
]


# ==================================================
# 1. DOCUMENT-AWARE CHUNKING
# ==================================================

def _is_heading(line: str) -> bool:
    """Identify likely headings without assuming a specific document type."""
    line = line.strip()

    if not line or len(line) > 150:
        return False

    # Avoid interpreting simple duration values as headings.
    if re.fullmatch(
        r"\d+(?:\.\d+)?\s+"
        r"(?:weeks?|months?|days?|hours?|years?)",
        line,
        re.IGNORECASE,
    ):
        return False

    return any(pattern.match(line) for pattern in HEADING_PATTERNS)


def _split_sections(text: str) -> list[dict[str, str]]:
    """Split a document page into sections while preserving headings."""
    sections: list[dict[str, str]] = []
    heading = ""
    lines: list[str] = []

    def save_section() -> None:
        content = "\n".join(lines).strip()

        if content:
            sections.append({
                "heading": heading,
                "text": content,
            })

    for line in text.splitlines():
        stripped = line.strip()

        if stripped and _is_heading(stripped):
            save_section()
            heading = stripped
            lines = []
        else:
            lines.append(line)

    save_section()

    # Preserve ordinary text when no headings are detected.
    if not sections and text.strip():
        sections.append({
            "heading": "",
            "text": text.strip(),
        })

    return sections


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs using blank lines."""
    return [
        part.strip()
        for part in re.split(r"\n\s*\n+", text.strip())
        if part.strip()
    ]


def _split_sentences(text: str) -> list[str]:
    """Split text into approximate sentences."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


# ==================================================
# 2. SEMANTIC CHUNKING
# ==================================================

def _semantic_group_sentences(
    sentences: list[str],
    max_chars: int,
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> list[str]:
    """Group adjacent sentences according to embedding similarity."""
    if not sentences:
        return []

    if len(sentences) == 1:
        return sentences

    vectors = None

    try:
        vectors = embed_texts(sentences)

        if (
            len(vectors) != len(sentences)
            or not vectors
            or not vectors[0]
            or any(
                len(vector) != len(vectors[0])
                for vector in vectors
            )
        ):
            vectors = None

    except Exception:
        # Fall back to size-based grouping if embedding generation fails.
        vectors = None

    groups: list[str] = []
    current = sentences[0]

    for index in range(1, len(sentences)):
        sentence = sentences[index]
        candidate = f"{current} {sentence}"

        similarity = None

        if vectors is not None:
            left = vectors[index - 1]
            right = vectors[index]

            dot_product = sum(
                float(a) * float(b)
                for a, b in zip(left, right)
            )

            left_norm = sum(
                float(value) ** 2 for value in left
            ) ** 0.5

            right_norm = sum(
                float(value) ** 2 for value in right
            ) ** 0.5

            if left_norm > 0 and right_norm > 0:
                similarity = dot_product / (
                    left_norm * right_norm
                )

        should_group = (
            len(candidate) <= max_chars
            and (
                similarity is None
                or similarity >= similarity_threshold
            )
        )

        if should_group:
            current = candidate
        else:
            groups.append(current)
            current = sentence

    groups.append(current)

    return groups


# ==================================================
# 3. RECURSIVE CHUNKING
# ==================================================

def _recursive_split(
    text: str,
    max_chars: int,
) -> list[str]:
    """Split oversized text using progressively smaller separators."""
    text = text.strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    for separator in separators:
        parts = text.split(separator)

        if len(parts) <= 1:
            continue

        result: list[str] = []
        current = ""

        for part in parts:
            if not part:
                continue

            candidate = (
                f"{current}{separator}{part}"
                if current
                else part
            )

            if len(candidate) <= max_chars:
                current = candidate
                continue

            if current:
                result.append(current.strip())

            if len(part) > max_chars:
                result.extend(
                    _recursive_split(part, max_chars)
                )
                current = ""
            else:
                current = part

        if current:
            result.append(current.strip())

        if result and all(
            len(item) <= max_chars for item in result
        ):
            return [item for item in result if item]

    # Necessary fallback for text without useful separators.
    return [
        text[index:index + max_chars]
        for index in range(0, len(text), max_chars)
    ]


# ==================================================
# OVERLAP
# ==================================================

def _add_overlap(
    chunks: list[str],
    overlap_chars: int,
    max_chars: int,
) -> list[str]:
    """Add preceding text as overlap without splitting words."""
    if overlap_chars <= 0 or len(chunks) < 2:
        return chunks

    result = [chunks[0]]

    for index in range(1, len(chunks)):
        previous = chunks[index - 1]
        current = chunks[index]

        available = max_chars - len(current) - 1
        if available <= 0:
            result.append(current)
            continue

        overlap_limit = min(overlap_chars, available)
        start = max(0, len(previous) - overlap_limit)

        # Move forward until the overlap starts at a word boundary.
        while (
            start < len(previous)
            and start > 0
            and not previous[start - 1].isspace()
        ):
            start += 1

        overlap = previous[start:].strip()

        # If the overlap is empty, try the last complete word.
        if not overlap:
            words = previous.split()
            if words and len(words[-1]) <= available:
                overlap = words[-1]

        if not overlap:
            result.append(current)
            continue

        combined = f"{overlap}\n{current}"

        if len(combined) <= max_chars:
            result.append(combined)
        else:
            result.append(current)

    return result

# ==================================================
# MAIN CHUNKING FUNCTION
# ==================================================

def chunk_document(
    document: dict[str, Any],
    max_chars: int = 2500,
    overlap_chars: int = 200,
) -> list[dict[str, Any]]:
    """
    Create document-aware, semantic, and recursively split chunks.

    Each chunk retains source, file type, page number, section,
    and a unique chunk ID.
    """
    if not isinstance(document, dict):
        raise TypeError("document must be a dictionary")

    if not isinstance(max_chars, int) or isinstance(max_chars, bool):
        raise TypeError("max_chars must be an integer")

    if max_chars < 1:
        raise ValueError("max_chars must be greater than zero")

    if (
        not isinstance(overlap_chars, int)
        or isinstance(overlap_chars, bool)
    ):
        raise TypeError("overlap_chars must be an integer")

    if not 0 <= overlap_chars < max_chars:
        raise ValueError(
            "overlap_chars must be between zero and max_chars - 1"
        )

    pages = document.get("pages")

    if not isinstance(pages, list):
        raise TypeError("document['pages'] must be a list")

    chunks: list[dict[str, Any]] = []

    for page in pages:
        if (
            not isinstance(page, dict)
            or not isinstance(page.get("text"), str)
        ):
            raise TypeError(
                "Each page must contain text as a string"
            )

        page_number = page.get("page")

        for section in _split_sections(page["text"]):
            heading = section["heading"]
            paragraphs = _split_paragraphs(section["text"])

            # Add the heading once to provide context for retrieval.
            prefix = f"{heading}\n\n" if heading else ""

            if len(prefix) >= max_chars:
                prefix = (
                    f"{heading[:max_chars - 2]}\n\n"
                    if heading
                    else ""
                )

            content_limit = max_chars - len(prefix)

            if content_limit <= 0:
                prefix = ""
                content_limit = max_chars

            content_chunks: list[str] = []

            for paragraph in paragraphs:
                sentences = _split_sentences(paragraph)

                groups = _semantic_group_sentences(
                    sentences or [paragraph],
                    content_limit,
                )

                for group in groups:
                    content_chunks.extend(
                        _recursive_split(
                            group,
                            content_limit,
                        )
                    )

            # Keep a heading-only section when it has no body text.
            if not content_chunks and heading:
                prefix = ""
                content_chunks = [heading]

            content_chunks = _add_overlap(
                content_chunks,
                min(
                    overlap_chars,
                    max(0, content_limit - 1),
                ),
                content_limit,
            )

            for content in content_chunks:
                chunk_text = f"{prefix}{content}".strip()

                for final_text in _recursive_split(
                    chunk_text,
                    max_chars,
                ):
                    if not final_text:
                        continue

                    chunks.append({
                        "text": final_text,
                        "source": document.get("source"),
                        "file_type": document.get("file_type"),
                        "page": page_number,
                        "section": heading or None,
                    })

    for chunk_id, chunk in enumerate(chunks, start=1):
        chunk["chunk_id"] = chunk_id

    return chunks