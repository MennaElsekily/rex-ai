"""
Chunking: document-aware first, recursive when a part is too large, and
semantic splitting where it helps.

1. Document-aware. Headings (Markdown "#", numbered "1.2", ALL CAPS,
   Title Case) build a section hierarchy that carries across pages. Small
   sections under the same top-level heading are packed together up to
   max_chars, so a chunk is a meaningful unit rather than a lone line.
   Structured records (CSV/XLSX/JSON rows) are never merged with each other.
2. Recursive. A unit that is still larger than max_chars is split at
   paragraph, then line, then sentence, then word boundaries, with a small
   word-aligned overlap between the pieces.
3. Semantic. For long running prose, sentence embeddings locate topic
   shifts (adjacent-sentence distance in the top percentile) and the text
   is split there instead of at an arbitrary length. Strategy "auto" (the
   default) uses it only for oversized prose sections; "semantic" uses it
   for every multi-sentence section; "document" never calls the model.

Every chunk starts with its context header (the section path, or the
record title), so a passage such as "Entry requirements: ..." still says
which item it belongs to when retrieved on its own. Chunks keep source,
file type, page, section, section path, record id and the explicit
fields of their section or record.

max_chars defaults to 1000 because the embedding model (all-MiniLM-L6-v2)
truncates input at 256 tokens; longer chunks would be only partly embedded.
"""

import re
from typing import Any

from knowledge_base.embedding_service import embed_texts
from knowledge_base.metadata_extractor import extract_explicit_fields

DEFAULT_MAX_CHARS = 1000
DEFAULT_MIN_CHARS = 250
DEFAULT_OVERLAP_CHARS = 120

STRATEGY_DOCUMENT = "document"
STRATEGY_SEMANTIC = "semantic"
STRATEGY_AUTO = "auto"
CHUNK_STRATEGIES = (STRATEGY_DOCUMENT, STRATEGY_SEMANTIC, STRATEGY_AUTO)

# Adjacent sentences whose embedding distance is in this top percentile
# of the section mark a topic shift (relative, so no corpus-specific
# similarity threshold is needed).
SEMANTIC_BREAKPOINT_PERCENTILE = 85
SEMANTIC_MIN_SENTENCES = 4

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)(?:[.)]\s+|\s+)(\S.*)$")
_SMALL_WORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at", "by",
    "with", "vs", "via", "&", "per", "from", "into",
}
_DURATION = re.compile(
    r"^\d+(?:\.\d+)?\s+(?:weeks?|months?|days?|hours?|years?)\b", re.IGNORECASE
)


# ==================================================
# 1. DOCUMENT-AWARE CHUNKING
# ==================================================

def _is_title_case(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    significant = [w for w in words if w.casefold() not in _SMALL_WORDS]
    if len(words) < 2 or len(words) > 8 or not significant:
        return False
    capitalized = sum(w[0].isupper() for w in significant)
    return text[0].isupper() and capitalized / len(significant) >= 0.5


def _is_all_caps(line: str) -> bool:
    letters = [ch for ch in line if ch.isalpha()]
    return len(letters) >= 3 and all(ch.isupper() for ch in letters)


def _heading_level(line: str) -> int | None | bool:
    """
    The heading level of a line: an int for explicit levels (Markdown,
    numbering), None for a heading without an explicit level (ALL CAPS,
    Title Case), or False when the line is not a heading.
    """
    line = line.strip()

    if not line or len(line) > 120:
        return False

    markdown = _MARKDOWN_HEADING.match(line)
    if markdown:
        return len(markdown.group(1))

    if _DURATION.match(line) or ":" in line or line.endswith((".", ",", ";", "?", "!", "؟")):
        return False

    numbered = _NUMBERED_HEADING.match(line)
    if numbered:
        number, rest = numbered.groups()
        # "1. Introduction" or "1.2 Scope" -- not "50 guided learning hours".
        explicit = line[len(number)] in ".)" or "." in number
        if explicit and rest[:1].isupper() and len(rest.split()) <= 10:
            return number.count(".") + 1
        return False

    letters = [ch for ch in line if ch.isalpha()]
    if len(letters) >= 3 and all(ch.isupper() for ch in letters) and len(line.split()) <= 10:
        return None

    if _is_title_case(line):
        return None

    return False


def _is_heading(line: str) -> bool:
    """Identify likely headings without assuming a specific document type."""
    return _heading_level(line) is not False


def _heading_text(line: str) -> str:
    """'## 1.2 Entry Requirements' -> 'Entry Requirements'."""
    text = line.strip()
    markdown = _MARKDOWN_HEADING.match(text)
    if markdown:
        text = markdown.group(2).strip()
    numbered = _NUMBERED_HEADING.match(text)
    if numbered and (text[len(numbered.group(1))] in ".)" or "." in numbered.group(1)):
        text = numbered.group(2).strip()
    return text


def _split_sections(text: str) -> list[dict[str, Any]]:
    """
    Split a page into sections. Each section has "heading" (the line as
    written), "level" (int, or None when not explicit) and "text".
    """
    sections: list[dict[str, Any]] = []
    heading, level, lines = "", None, []

    def save_section() -> None:
        content = "\n".join(lines).strip()
        if content or heading:
            sections.append({"heading": heading, "level": level, "text": content})

    for line in text.splitlines():
        stripped = line.strip()
        found = _heading_level(stripped) if stripped else False
        if found is not False:
            save_section()
            heading, level, lines = stripped, found, []
        else:
            lines.append(line)

    save_section()
    return [s for s in sections if s["text"] or s["heading"]]


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs using blank lines."""
    return [
        part.strip()
        for part in re.split(r"\n\s*\n+", text.strip())
        if part.strip()
    ]


def _split_sentences(text: str) -> list[str]:
    """Split text into approximate sentences (Latin and Arabic punctuation)."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?؟])\s+", text)
        if sentence.strip()
    ]


# ==================================================
# 2. SEMANTIC CHUNKING
# ==================================================

def _cosine(left: list[float], right: list[float]) -> float | None:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    norm = (sum(float(a) ** 2 for a in left) ** 0.5) * (sum(float(b) ** 2 for b in right) ** 0.5)
    return dot / norm if norm > 0 else None


def _semantic_breakpoints(sentences: list[str], percentile: float) -> set[int] | None:
    """
    Indices i where a new group should start before sentences[i], or None
    when embeddings are unavailable (callers then split recursively).
    """
    try:
        vectors = embed_texts(sentences)
    except Exception:
        return None

    if len(vectors) != len(sentences) or not vectors or not vectors[0]:
        return None

    distances = []
    for index in range(1, len(vectors)):
        similarity = _cosine(vectors[index - 1], vectors[index])
        if similarity is None:
            return None
        distances.append(1 - similarity)

    ordered = sorted(distances)
    cut = ordered[min(len(ordered) - 1, int(len(ordered) * percentile / 100))]
    return {index + 1 for index, distance in enumerate(distances) if distance >= cut and distance > 0}


def _semantic_group_sentences(
    sentences: list[str],
    max_chars: int,
    min_chars: int = DEFAULT_MIN_CHARS,
    percentile: float = SEMANTIC_BREAKPOINT_PERCENTILE,
) -> list[str]:
    """
    Group consecutive sentences, starting a new group at a topic shift
    (once the current group is at least min_chars) or when max_chars
    would be exceeded. Falls back to size-only grouping without embeddings.
    """
    if not sentences:
        return []

    if len(sentences) == 1:
        return sentences

    breakpoints = _semantic_breakpoints(sentences, percentile) or set()

    groups: list[str] = []
    current = sentences[0]

    for index in range(1, len(sentences)):
        sentence = sentences[index]
        candidate = f"{current} {sentence}"
        topic_shift = index in breakpoints and len(current) >= min_chars

        if len(candidate) <= max_chars and not topic_shift:
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

    separators = ["\n\n", "\n", ". ", "؟ ", "; ", ", ", " "]

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
# UNITS: sections and records with their context
# ==================================================

def _is_prose(text: str) -> bool:
    """Running sentences, as opposed to lists, tables or key: value lines."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    structured = sum(
        bool(re.match(r"^\s*(?:[-*•]|\d+[.)])\s", line)) or ":" in line[:50]
        for line in lines
    )
    return structured / len(lines) < 0.5 and len(_split_sentences(text)) >= SEMANTIC_MIN_SENTENCES


def _document_units(document: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Turn pages into units: each section (with its heading path and the
    explicit fields of its ancestors) or each structured record.
    """
    units: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []   # [{"title", "level", "fields"}]
    last_explicit_level = 0

    # Without Markdown or numbered levels, the common convention ranks
    # ALL CAPS headings above Title Case ones.
    caps_seen = False

    for page in document["pages"]:
        page_number = page.get("page")

        if page.get("record_id") is not None or page.get("fields"):
            title = page.get("section") or f"Record {page.get('record_id')}"
            units.append({
                "kind": "record",
                "path": [title],
                "group": ("record", page.get("record_id"), len(units)),
                "text": page["text"].strip(),
                "fields": dict(page.get("fields") or {}),
                "record_id": page.get("record_id"),
                "page": page_number,
            })
            continue

        for section in _split_sections(page["text"]):
            if section["heading"]:
                level = section["level"]
                if level is None:
                    if last_explicit_level:
                        level = last_explicit_level + 1
                    elif _is_all_caps(section["heading"]):
                        level, caps_seen = 1, True
                    else:
                        level = 2 if caps_seen else 1
                else:
                    last_explicit_level = level
                while stack and stack[-1]["level"] >= level:
                    stack.pop()
                stack.append({
                    "title": _heading_text(section["heading"]),
                    "level": level,
                    "fields": extract_explicit_fields(section["text"]),
                })
            elif stack:
                stack[-1]["fields"].update(extract_explicit_fields(section["text"]))

            if not section["text"]:
                continue

            fields: dict[str, Any] = {}
            for entry in stack:
                fields.update(entry["fields"])

            units.append({
                "kind": "section",
                "path": [entry["title"] for entry in stack],
                "group": None,
                "text": section["text"].strip(),
                "fields": fields,
                "record_id": None,
                "page": page_number,
            })

    # Units are packed only within the same branch: the first heading
    # level where the document divides (a lone document title shared by
    # every section is skipped, so two courses under "# Catalogue" are
    # still separate groups).
    sections = [u for u in units if u["kind"] == "section"]
    if sections:
        shared = _common_path([u["path"] for u in sections])
        depth = len(shared) + 1
        for unit in sections:
            unit["group"] = ("section", tuple(unit["path"][:depth]))

    return units


def _header(path: list[str]) -> str:
    return " > ".join(part for part in path if part)


def _common_path(paths: list[list[str]]) -> list[str]:
    common = list(paths[0])
    for path in paths[1:]:
        size = 0
        while size < min(len(common), len(path)) and common[size] == path[size]:
            size += 1
        common = common[:size]
    return common


def _unit_body(unit: dict[str, Any], header_path: list[str]) -> str:
    """The unit's text, led by the headings below the chunk's header path."""
    below = unit["path"][len(header_path):] if unit["kind"] == "section" else []
    return "\n".join([*below, unit["text"]]).strip()


# ==================================================
# MAIN CHUNKING FUNCTION
# ==================================================

def chunk_document(
    document: dict[str, Any],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    strategy: str = STRATEGY_AUTO,
) -> list[dict[str, Any]]:
    """
    Create document-aware chunks, split recursively or semantically when
    needed (see the module docstring).

    Each chunk retains source, file type, page (and page_end when it spans
    pages), section, section_path, record_id, the explicit fields of its
    section or record, the strategy used, and a unique chunk ID.
    """
    if not isinstance(document, dict):
        raise TypeError("document must be a dictionary")

    for name, value in (("max_chars", max_chars), ("min_chars", min_chars)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")

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

    if strategy not in CHUNK_STRATEGIES:
        raise ValueError(f"strategy must be one of {CHUNK_STRATEGIES}")

    pages = document.get("pages")

    if not isinstance(pages, list):
        raise TypeError("document['pages'] must be a list")

    for page in pages:
        if (
            not isinstance(page, dict)
            or not isinstance(page.get("text"), str)
        ):
            raise TypeError(
                "Each page must contain text as a string"
            )

    chunks: list[dict[str, Any]] = []

    def emit(units: list[dict[str, Any]], text_pieces: list[str], header_path: list[str], used: str) -> None:
        header = _header(header_path)
        pages_seen = [u["page"] for u in units if u["page"] is not None]
        fields: dict[str, Any] = {}
        for unit in units:
            for key, value in unit["fields"].items():
                fields.setdefault(key, value)
        sections = [u["path"][-1] for u in units if u["path"]]
        section = sections[0] if len(set(sections)) == 1 else (
            header_path[-1] if header_path else (sections[0] if sections else None)
        )
        for piece in text_pieces:
            body = piece.strip()
            if not body:
                continue
            text = f"{header}\n{body}" if header and not body.startswith(header) else body
            for final_text in _recursive_split(text, max_chars):
                chunks.append({
                    "text": final_text,
                    "source": document.get("source"),
                    "file_type": document.get("file_type"),
                    "page": pages_seen[0] if pages_seen else None,
                    "page_end": pages_seen[-1] if len(set(pages_seen)) > 1 else None,
                    "section": section,
                    "section_path": header or None,
                    "record_id": units[0]["record_id"] if units[0]["kind"] == "record" else None,
                    "fields": fields,
                    "chunk_strategy": used,
                })

    def split_unit(unit: dict[str, Any], header_path: list[str]) -> None:
        """One unit too large for a chunk: split it on its own."""
        header_room = len(_header(header_path)) + 1
        limit = max(100, max_chars - header_room)
        body = _unit_body(unit, header_path)

        use_semantic = unit["kind"] == "section" and (
            strategy == STRATEGY_SEMANTIC
            or (strategy == STRATEGY_AUTO and _is_prose(unit["text"]))
        )

        pieces: list[str] = []
        if use_semantic:
            for paragraph in _split_paragraphs(body):
                sentences = _split_sentences(paragraph)
                if len(sentences) >= 2:
                    for group in _semantic_group_sentences(sentences, limit, min_chars):
                        pieces.extend(_recursive_split(group, limit))
                else:
                    pieces.extend(_recursive_split(paragraph, limit))
            used = STRATEGY_SEMANTIC
        else:
            pieces = _recursive_split(body, limit)
            used = "recursive"

        pieces = _add_overlap(pieces, min(overlap_chars, max(0, limit - 1)), limit)
        emit([unit], pieces, header_path, "record" if unit["kind"] == "record" and used == "recursive" else used)

    # Pack consecutive units of the same group up to max_chars.
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        header_path = _common_path([u["path"] for u in pending])
        if pending[0]["kind"] == "record":
            header_path = pending[0]["path"]
        body = "\n\n".join(_unit_body(u, header_path) for u in pending)
        emit(list(pending), [body], header_path, "record" if pending[0]["kind"] == "record" else STRATEGY_DOCUMENT)
        pending.clear()

    def packed_size(units: list[dict[str, Any]]) -> int:
        header_path = _common_path([u["path"] for u in units])
        return len(_header(header_path)) + 1 + sum(len(_unit_body(u, header_path)) + 2 for u in units)

    for unit in _document_units(document):
        alone = packed_size([unit])

        # Semantic strategy: split every multi-sentence prose section at
        # topic shifts, even when it would fit in one chunk.
        semantic_now = (
            strategy == STRATEGY_SEMANTIC
            and unit["kind"] == "section"
            and len(_split_sentences(unit["text"])) >= SEMANTIC_MIN_SENTENCES
        )

        if alone > max_chars or semantic_now:
            flush()
            split_unit(unit, unit["path"] if unit["kind"] == "record" else unit["path"])
            continue

        if pending and (
            unit["kind"] == "record"
            or pending[0]["kind"] == "record"
            or unit["group"] != pending[0]["group"]
            or packed_size([*pending, unit]) > max_chars
        ):
            flush()

        pending.append(unit)

        if unit["kind"] == "record":
            flush()

    flush()

    for chunk_id, chunk in enumerate(chunks, start=1):
        chunk["chunk_id"] = chunk_id

    return chunks
