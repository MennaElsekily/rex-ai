
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


STOP_WORDS = {
    "the", "and", "for", "are", "was", "were", "with",
    "from", "this", "that", "into", "have", "has", "had",
    "will", "can", "you", "your", "our", "their", "its",
    "is", "of", "to", "in", "on", "at", "by", "an", "a",
    "or", "as", "be", "it", "we", "they", "which", "who",
    "what", "when", "where", "how", "not", "than", "then",
    "also", "all", "any", "more", "most", "other", "such",
}


def extract_dates(text: str) -> list[str]:
    """Extract common date formats without assuming a document domain."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    pattern = (
        r"\b\d{4}-\d{2}-\d{2}\b"
        r"|\b\d{1,2}/\d{1,2}/\d{4}\b"
        r"|\b\d{1,2}-\d{1,2}-\d{4}\b"
    )

    return list(dict.fromkeys(re.findall(pattern, text)))


def extract_urls(text: str) -> list[str]:
    """Extract URLs appearing explicitly in the document."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    matches = re.findall(
        r"https?://[^\s<>\"']+",
        text,
        flags=re.IGNORECASE,
    )

    return list(dict.fromkeys(
        url.rstrip(".,;:!?)\u005d}") for url in matches
    ))


def extract_keywords(
    text: str,
    max_keywords: int = 10,
) -> list[str]:
    """Extract frequent English words as lightweight descriptive keywords."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not isinstance(max_keywords, int) or isinstance(max_keywords, bool):
        raise TypeError("max_keywords must be an integer")

    if max_keywords < 1:
        raise ValueError("max_keywords must be at least 1")

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z'-]*\b",
        text.lower(),
    )

    frequencies = Counter(
        word
        for word in words
        if len(word) > 2 and word not in STOP_WORDS
    )

    return [
        word
        for word, _ in sorted(
            frequencies.items(),
            key=lambda item: (-item[1], item[0]),
        )[:max_keywords]
    ]


def extract_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Extract technical metadata and generic, rule-based content signals.

    Content signals are observations from the text, not claims that the
    document belongs to a particular academy or follows a fixed schema.
    """
    if not isinstance(chunk, dict):
        raise TypeError("chunk must be a dictionary")

    text = chunk.get("text", "")

    if not isinstance(text, str):
        raise TypeError("chunk text must be a string")

    source = str(chunk.get("source") or "")
    source_path = Path(source) if source else None

    metadata: dict[str, Any] = {
        "document_id": source,
        "chunk_id": chunk.get("chunk_id"),
        "filename": source_path.name if source_path else None,
        "file_type": chunk.get("file_type"),
        "page": chunk.get("page"),
        "section": chunk.get("section"),
        "data_source": source,
        "dates": extract_dates(text),
        "urls": extract_urls(text),
        "keywords": extract_keywords(text),
        "extraction_method": "rule_based",
    }

    # Capture explicit key-value patterns when documents contain them.
    # The field names are discovered from the text, not preselected
    # academy-specific fields.
    explicit_fields = {}

    for line in text.splitlines():
        match = re.match(
            r"^\s*([A-Za-z][A-Za-z0-9 _/-]{0,60})\s*:\s*(.*?)\s*$",
            line,
        )

        if match:
            field_name = re.sub(
                r"\s+",
                "_",
                match.group(1).strip().lower(),
            )
            field_value = match.group(2).strip()

            if field_value:
                explicit_fields[field_name] = field_value

    metadata["explicit_fields"] = explicit_fields

    metadata["extraction_status"] = (
        "partial"
        if metadata["dates"]
        or metadata["urls"]
        or metadata["keywords"]
        or explicit_fields
        else "technical_only"
    )

    return metadata


def save_metadata_json(
    metadata_records: list[dict[str, Any]],
    output_path: str | Path = (
        "knowledge_base/metadata/chunk_metadata.json"
    ),
) -> Path:
    """Save extracted metadata records as a UTF-8 JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata_records,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return path
