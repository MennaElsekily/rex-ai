
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


_FIELD_LINE = re.compile(
    r"^\s*(?:[-*•]\s*)?([A-Za-z؀-ۿ][A-Za-z0-9؀-ۿ _/()&-]{0,48}?)\s*[:：]\s*(\S.{0,300}?)\s*$"
)
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
_RESERVED = {
    "source", "page", "page_end", "section", "section_path", "record_id",
    "chunk_id", "chunk_strategy", "file_type", "filename", "document_id",
}


def normalize_field_name(name: str) -> str:
    """'Delivery Mode' -> 'delivery_mode'; Arabic letters are kept."""
    key = re.sub(r"[^0-9A-Za-z؀-ۿ]+", "_", str(name).strip().casefold()).strip("_")
    return key[:40]


def normalize_field_value(value: Any) -> Any:
    """Numbers stay numbers ("4500" -> 4500); everything else is a trimmed string."""
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    if _NUMBER.match(text):
        return float(text) if "." in text else int(text)
    return text[:500]


def extract_explicit_fields(text: str) -> dict[str, Any]:
    """
    "Key: value" lines written in the text itself (e.g. "Level: Beginner",
    "Duration: 6 weeks"). Field names are discovered from the document,
    never preselected, and nothing is inferred beyond what is written.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    fields: dict[str, Any] = {}
    for line in text.splitlines():
        match = _FIELD_LINE.match(line)
        if not match or match.group(1).strip().count(" ") > 4:
            continue
        key = normalize_field_name(match.group(1))
        value = normalize_field_value(match.group(2))
        if key and value is not None and key not in fields:
            fields[key] = value
    return fields


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
        "document_id": chunk.get("document_id") or source,
        "chunk_id": chunk.get("chunk_id"),
        "filename": source_path.name if source_path else None,
        "file_type": chunk.get("file_type"),
        "page": chunk.get("page"),
        "page_end": chunk.get("page_end"),
        "section": chunk.get("section"),
        "section_path": chunk.get("section_path"),
        "record_id": chunk.get("record_id"),
        "chunk_strategy": chunk.get("chunk_strategy"),
        "data_source": source,
        "dates": extract_dates(text),
        "urls": extract_urls(text),
        "keywords": extract_keywords(text),
        "extraction_method": "rule_based",
    }

    # Explicit fields: structured record fields and fields inherited from
    # the enclosing sections (set by the parser/chunker), then "key: value"
    # lines in this chunk's own text. Field names come from the data.
    explicit_fields: dict[str, Any] = {}
    for key, value in (chunk.get("fields") or {}).items():
        name, normalized = normalize_field_name(key), normalize_field_value(value)
        if name and normalized is not None:
            explicit_fields[name] = normalized
    for key, value in extract_explicit_fields(text).items():
        explicit_fields.setdefault(key, value)

    metadata["explicit_fields"] = {
        key: value for key, value in explicit_fields.items() if key not in _RESERVED
    }
    explicit_fields = metadata["explicit_fields"]

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
