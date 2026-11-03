"""
PDFs for generated documents (ReportLab), English and Arabic.

Everything that makes Arabic render correctly is reused from the
Curriculum Agent's renderer (agents/curriculum_research/pdf.py): the
bundled Noto fonts, letter shaping, right-to-left ordering and the
mixed-script TextBlock. This module only lays out the generic document
content (sections of paragraph, bullet, heading and table blocks), so one
renderer serves every document type.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle

from agents.curriculum_research.pdf import (
    ACCENT,
    GREY,
    LIGHT,
    PAGE_MARGIN,
    TextBlock,
    _fonts,
    display_line,
    runs,
)
from agents.document_generation.templates import STRINGS, field_label, type_label

LABELS = {
    "en": {"document": "Document", "curriculum": "Curriculum", "version": "Document version", "status": "Review status",
           "hours": "Total hours", "level": "Level", "language": "Language", "generated": "Generated",
           "field": "Field", "value": "Value", "page": "Page", "missing": "Information not specified in the curriculum",
           "id": "Document ID", "curriculum_ref": "Curriculum (version)", "approved_on": "Approved on"},
    "ar": {"document": "الوثيقة", "curriculum": "المنهج", "version": "نسخة الوثيقة", "status": "حالة المراجعة",
           "hours": "إجمالي الساعات", "level": "المستوى", "language": "اللغة", "generated": "تاريخ الإنشاء",
           "field": "الحقل", "value": "القيمة", "page": "صفحة",
           "missing": "معلومات غير محددة في المنهج", "id": "معرّف الوثيقة", "curriculum_ref": "المنهج (النسخة)", "approved_on": "تاريخ الاعتماد"},
}
STATUS = {
    "en": {"draft": "Draft (problems remain)", "pending_review": "Pending review (not approved)",
           "approved": "Approved by the academy"},
    "ar": {"draft": "مسودة (توجد مشكلات)", "pending_review": "بانتظار المراجعة (غير معتمد)",
           "approved": "معتمد من الأكاديمية"},
}
LEVELS = {"ar": {"Beginner": "مبتدئ", "Intermediate": "متوسط", "Advanced": "متقدم"}}
LANGUAGE_NAMES = {"en": {"en": "English", "ar": "Arabic"}, "ar": {"en": "الإنجليزية", "ar": "العربية"}}
# Letters shown in front of a multiple-choice question's options (in the PDF only; the stored options have no label).
OPTION_LETTERS = {"en": "ABCDEFGH", "ar": "أبجدهوزح"}


class _Builder:
    def __init__(self, record: dict[str, Any], approval: Optional[dict[str, str]] = None) -> None:
        self.record = record
        self.approval = approval
        self.content = record["content"]
        self.language = "ar" if self.content.get("language") == "ar" else "en"
        self.rtl = self.language == "ar"
        self.labels = LABELS[self.language]
        self.story: list[Any] = []
        self.width = A4[0] - 2 * PAGE_MARGIN

    def text(self, value, **kwargs) -> TextBlock:
        return TextBlock(value, rtl=self.rtl, **kwargs)

    def table(self, header: list[str], rows: list[list[Any]], widths: list[float]) -> None:
        def cell(value, bold=False):
            return TextBlock(value, rtl=self.rtl, size=9, bold=bold, space_after=0)

        data = [[cell(h, True) for h in header]] + [[cell(v) for v in row] for row in rows]
        col_widths = [w * self.width for w in widths]
        if self.rtl:  # first column on the right
            data = [list(reversed(row)) for row in data]
            col_widths = list(reversed(col_widths))
        grid = Table(data, colWidths=col_widths, repeatRows=1)
        grid.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "RexLatin"),
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B7C3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
        self.story += [grid, Spacer(1, 6)]

    def cover(self) -> None:
        c, r, L = self.content, self.record, self.labels
        # The approved copy says so (and when); the review copy shows the status the version had when it was made.
        status = "approved" if self.approval else (r.get("version_status") or r.get("status"))
        level = LEVELS.get(self.language, {}).get(c.get("level"), c.get("level") or "")
        self.story += [Spacer(1, 28 * mm),
                       self.text(type_label(c["document_type"], self.language), size=12, color=GREY),
                       self.text(c.get("course_title") or c.get("title"), size=22, bold=True, color=ACCENT, space_after=10),
                       Spacer(1, 6)]
        self.table([L["field"], L["value"]], [
            [L["id"], r.get("document_id")], [L["version"], r.get("version")],
            [L["status"], STATUS[self.language].get(status, status)],
            [L["curriculum_ref"], f"{c.get('curriculum_id')} (v{c.get('curriculum_version')})"],
            [L["language"], LANGUAGE_NAMES[self.language].get(self.language)], [L["level"], level],
            [L["hours"], c.get("total_hours")], [L["generated"], datetime.now(timezone.utc).strftime("%Y-%m-%d")],
            *([[L["approved_on"], self.approval["approved_at"][:10]]] if self.approval else []),
        ], [0.32, 0.68])
        if c.get("missing"):
            fields = ("، " if self.rtl else ", ").join(
                sorted({field_label(m.get("field", ""), self.language) for m in c["missing"]}))
            self.story.append(self.text(f"{L['missing']}: {fields}", size=9, color=GREY))
        self.story.append(self.text(STRINGS[self.language]["disclaimer"], size=9, color=GREY))
        self.story.append(PageBreak())

    @staticmethod
    def _with_next(*flowables: Any) -> list[Any]:
        """
        Flowables that must stay on the same page as whatever follows them (ReportLab's keepWithNext). Every heading goes
        through this, so no heading is ever left alone at the bottom of a page, in any document type. The spacer in front
        of a heading is marked too: otherwise the heading would only be kept with that spacer.
        """
        for flowable in flowables:
            flowable.keepWithNext = True
        return list(flowables)

    def _with_letter(self, answer: Any, options: list[str]) -> Any:
        """The correct answer with the letter of the option it repeats; unchanged when it matches none (or more than one)."""
        squash = lambda value: " ".join(str(value or "").split())
        matches = [i for i, option in enumerate(options) if squash(option) == squash(answer)]
        return f"{OPTION_LETTERS[self.language][matches[0]]}. {answer}" if len(matches) == 1 else answer

    def _labelled_options(self, items: list[str]) -> None:
        for letter, item in zip(OPTION_LETTERS[self.language], items):
            self.story.append(self.text(f"{letter}. {item}", space_after=2))

    def sections(self) -> None:
        for section in self.content.get("sections") or []:
            options: list[str] = []                       # the options of the multiple-choice question being shown
            self.story += self._with_next(Spacer(1, 6), self.text(section.get("heading"), size=14, bold=True, color=ACCENT, space_after=4))
            for block in section.get("blocks") or []:
                kind = block.get("type")
                if kind == "paragraph":
                    text = block.get("text")
                    if block.get("role") == "q_answer":   # name the correct option by its letter, when it is one of them
                        text = self._with_letter(text, options)
                        options = []
                    self.story.append(self.text(text))
                elif kind == "heading":
                    self.story += self._with_next(Spacer(1, 3), self.text(block.get("text"), size=11.5, bold=True, color=ACCENT, space_after=3))
                elif kind == "bullets" and block.get("role") == "q_options":
                    options = [str(i) for i in block.get("items") or []]
                    self._labelled_options(options)
                elif kind == "bullets":
                    for item in block.get("items") or []:
                        self.story.append(self.text(item, bullet="•", space_after=2))
                elif kind == "table":
                    columns = len(block.get("header") or [1])
                    # (a table of four or more columns is a score table: the first column holds the criterion's text)
                    widths = {2: [0.7, 0.3], 3: [0.42, 0.38, 0.2]}.get(
                        columns, [0.30, *[0.70 / (columns - 1)] * (columns - 1)] if columns > 3 else [1 / columns] * columns)
                    self.table(block.get("header") or [], block.get("rows") or [], widths)

    def on_page(self, canvas, doc) -> None:
        canvas.saveState()
        footer = f"{self.record.get('document_id')} · v{self.record.get('version')} · {self.labels['page']} {doc.page}"
        parts = runs(display_line(footer, self.rtl))
        total = sum(pdfmetrics.stringWidth(t, f, 8) for t, f in parts)
        x = (A4[0] - total) / 2
        canvas.setFillColor(GREY)
        for text, font in parts:
            canvas.setFont(font, 8)
            canvas.drawString(x, 10 * mm, text)
            x += pdfmetrics.stringWidth(text, font, 8)
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(0.6)
        canvas.line(PAGE_MARGIN, A4[1] - 12 * mm, A4[0] - PAGE_MARGIN, A4[1] - 12 * mm)
        canvas.restoreState()


def render_document_pdf(record: dict[str, Any], approval: Optional[dict[str, str]] = None) -> tuple[bytes, int]:
    """
    The PDF of one stored document version (a DocumentStore.get record): (bytes, page count).
    approval={"approved_at": ISO time} makes the approved copy, whose cover says "Approved by the academy".
    """
    _fonts()
    builder = _Builder(record, approval)
    builder.cover()
    builder.sections()
    buffer = io.BytesIO()
    pages = {"count": 0}

    def on_page(canvas, doc):
        pages["count"] = doc.page
        builder.on_page(canvas, doc)

    content = record["content"]
    SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN, topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN, title=str(content.get("title") or "Document"),
        author="REX AI Document Generation Agent", subject=f"{record.get('document_id')} version {record.get('version')}",
        creator="REX AI", lang=builder.language, initialFontName="RexLatin",
    ).build(builder.story, onFirstPage=on_page, onLaterPages=on_page)
    return buffer.getvalue(), pages["count"]
