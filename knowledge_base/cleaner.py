
import re
from collections import Counter
from typing import Any


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.strip(" \t") for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse three or more consecutive newlines into two.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


_SENTENCE_END = re.compile(r"[.!?:;؟،…)\]\"']$")
_LIST_START = re.compile(r"^\s*(?:[-*•▪◦]|\d+[.)]\s|[a-z][.)]\s)")
_HEADING_START = re.compile(r"^\s*(?:#{1,6}\s|\d+(?:\.\d+)*[.)]?\s+[A-Z])")


def unwrap_lines(text: str) -> str:
    """
    Re-join lines that a PDF layout broke mid-sentence ("explain each\\n
    transformation.") while keeping headings, list items, key: value
    lines and paragraph breaks on their own lines.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        previous = out[-1].rstrip() if out else ""
        if (
            previous
            and stripped
            and not _SENTENCE_END.search(previous)
            and not _LIST_START.match(stripped)
            and not _HEADING_START.match(stripped)
            and (stripped[0].islower() or stripped[0].isdigit())
        ):
            out[-1] = f"{previous} {stripped}"
        else:
            out.append(line)
    return "\n".join(out)


def is_usable_text(text: str, min_chars: int = 20) -> bool:
    """
    False for text that should not be indexed: (almost) empty, as from a
    scanned PDF with no text layer, or mostly unreadable characters, as
    from a broken text extraction.
    """
    if not isinstance(text, str):
        return False
    visible = [ch for ch in text if not ch.isspace()]
    if len(visible) < min_chars:
        return False
    readable = sum(ch.isalnum() or ch in ".,;:!?()-%/&'\"#*@+–—·•" for ch in visible)
    replacement = text.count("�")
    return readable / len(visible) >= 0.6 and replacement / len(visible) < 0.05


def _edge_key(line: str) -> str:
    # Page numbers differ on every page ("Page 3", "3 / 12"), so they
    # are ignored when comparing repeated header and footer lines.
    return re.sub(r"\d+", "#", line.strip().casefold())


def remove_repeated_headers_footers(
    pages: list[dict[str, Any]],
    min_repetition_ratio: float = 0.6,
) -> list[dict[str, Any]]:
    """
    Remove repeated lines near the top or bottom of multiple pages.

    Each page must contain:
        {"page": page_number, "text": page_text}

    For pages with four or more non-empty lines, consider the first
    two and last two lines. For shorter pages, consider only the
    first and last lines to protect middle content.
    """
    if not isinstance(min_repetition_ratio, (int, float)):
        raise TypeError("min_repetition_ratio must be a number")

    if not 0 < min_repetition_ratio <= 1:
        raise ValueError(
            "min_repetition_ratio must be between 0 and 1"
        )

    if not isinstance(pages, list):
        raise TypeError("pages must be a list")

    if not pages:
        return []

    if any(
        not isinstance(page, dict)
        or not isinstance(page.get("text"), str)
        for page in pages
    ):
        raise TypeError(
            "Every page must be a dictionary with text as a string"
        )

    # A single page is insufficient to identify repeated headers/footers.
    if len(pages) < 2:
        return [page.copy() for page in pages]

    candidates = Counter()
    page_lines = []
    page_edge_indices = []

    for page in pages:
        lines = page["text"].splitlines()
        page_lines.append(lines)

        non_empty_indices = [
            index
            for index, line in enumerate(lines)
            if line.strip()
        ]

        if len(non_empty_indices) >= 4:
            # Multi-line header blocks (academy name, catalogue title,
            # notice, page number) need a wider window on longer pages;
            # a line is still removed only if it repeats across pages.
            window = max(2, min(4, len(non_empty_indices) // 2))
            edge_indices = (
                set(non_empty_indices[:window])
                | set(non_empty_indices[-window:])
            )
        else:
            edge_indices = set()

            if non_empty_indices:
                edge_indices.add(non_empty_indices[0])
                edge_indices.add(non_empty_indices[-1])

        page_edge_indices.append(edge_indices)

        # Count a candidate at most once per page.
        page_candidates = {
            _edge_key(lines[index])
            for index in edge_indices
            if lines[index].strip()
        }

        candidates.update(page_candidates)

    required_pages = max(
        2,
        int(len(pages) * min_repetition_ratio + 0.999999),
    )

    repeated_lines = {
        line
        for line, count in candidates.items()
        if count >= required_pages
    }

    cleaned_pages = []

    for page, lines, edge_indices in zip(
        pages,
        page_lines,
        page_edge_indices,
    ):
        cleaned_lines = [
            line
            for index, line in enumerate(lines)
            if not (
                index in edge_indices
                and _edge_key(line) in repeated_lines
            )
        ]

        cleaned_page = page.copy()
        cleaned_page["text"] = clean_text(
            "\n".join(cleaned_lines)
        )
        cleaned_pages.append(cleaned_page)

    return cleaned_pages
