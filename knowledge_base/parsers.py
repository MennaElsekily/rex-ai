
import csv
from pathlib import Path

from docx import Document
from pptx import Presentation
from pypdf import PdfReader

from knowledge_base.file_loader import get_file_type


def parse_document(file_path: str | Path) -> dict:
    """Extract text and relevant content from a supported document."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    extension = get_file_type(path)

    parsers = {
        ".pdf": _parse_pdf,
        ".docx": _parse_docx,
        ".pptx": _parse_pptx,
        ".txt": _parse_text,
        ".md": _parse_text,
        ".csv": _parse_csv,
    }

    return parsers[extension](path)


def _parse_pdf(path: Path) -> dict:
    """Extract text from each PDF page."""
    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        pages.append({
            "page": page_number,
            "text": page.extract_text() or "",
        })

    return {
        "source": str(path),
        "file_type": ".pdf",
        "pages": pages,
    }


def _parse_docx(path: Path) -> dict:
    """Extract paragraphs and tables from a Word document."""
    document = Document(str(path))

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    tables = [
        [
            [cell.text.strip() for cell in row.cells]
            for row in table.rows
        ]
        for table in document.tables
    ]

    return {
        "source": str(path),
        "file_type": ".docx",
        "pages": [{
            "page": None,
            "text": "\n".join(paragraphs),
        }],
        "tables": tables,
    }


def _parse_pptx(path: Path) -> dict:
    """Extract text and table content from each PowerPoint slide."""
    presentation = Presentation(str(path))
    slides = []

    for slide_number, slide in enumerate(presentation.slides, start=1):
        elements = []

        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text.strip()

                if text:
                    elements.append(text)

            if shape.has_table:
                for row in shape.table.rows:
                    elements.append(
                        " | ".join(
                            cell.text.strip()
                            for cell in row.cells
                        )
                    )

        slides.append({
            "page": slide_number,
            "text": "\n".join(elements),
        })

    return {
        "source": str(path),
        "file_type": ".pptx",
        "pages": slides,
    }


def _parse_text(path: Path) -> dict:
    """Read a plain-text or Markdown document."""
    text = path.read_text(encoding="utf-8-sig")

    return {
        "source": str(path),
        "file_type": path.suffix.lower(),
        "pages": [{
            "page": None,
            "text": text,
        }],
    }


def _parse_csv(path: Path) -> dict:
    """Read CSV rows and convert them into searchable text."""
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.reader(file))

    text = "\n".join(
        " | ".join(row)
        for row in rows
    )

    return {
        "source": str(path),
        "file_type": ".csv",
        "pages": [{
            "page": None,
            "text": text,
        }],
    }
