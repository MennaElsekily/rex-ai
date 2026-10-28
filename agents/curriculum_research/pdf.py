"""
Curriculum PDFs in English and Arabic (ReportLab).

Arabic needs three things ReportLab does not do by itself:
1. letter joining -- ``arabic_reshaper`` turns letters into their
   contextual forms;
2. right-to-left order -- ``python-bidi`` reorders each line for display
   (numbers and Latin words such as "Python" keep their own order);
3. two fonts -- the bundled Noto Naskh Arabic has no Latin letters and
   Noto Sans has no Arabic, so each line is drawn in runs, each run in a
   font that has its glyphs (chosen from the fonts' own character maps).

Lines are wrapped in *logical* order first and only then shaped and
reordered, one line at a time; wrapping already-reordered text would put
the end of a sentence on the first line. ``TextBlock`` does this for both
directions, so mixed text is handled the same way in English documents.

Fonts are bundled (agents/curriculum_research/fonts, SIL Open Font
Licence) and embedded in every PDF; machine fonts are never used.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle

FONTS_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FILES = {
    "RexLatin": "NotoSans-Regular.ttf", "RexLatin-Bold": "NotoSans-Bold.ttf",
    "RexArabic": "NotoNaskhArabic-Regular.ttf", "RexArabic-Bold": "NotoNaskhArabic-Bold.ttf",
}
ACCENT = colors.HexColor("#1F4E79")
LIGHT = colors.HexColor("#E8EEF5")
GREY = colors.HexColor("#555555")
PAGE_MARGIN = 18 * mm

DISCLAIMER = {
    "en": ("This curriculum was drafted with AI assistance from the sources listed and must be reviewed by the "
           "academy before use. Comparison with a framework is not accreditation, certification or official approval."),
    "ar": ("أُعدّ هذا المنهج بمساعدة الذكاء الاصطناعي اعتمادًا على المصادر المذكورة، ويجب أن تراجعه الأكاديمية قبل "
           "استخدامه. المقارنة مع أي إطار ليست اعتمادًا أو شهادة أو موافقة رسمية."),
}
LABELS = {
    "en": {
        "curriculum": "Curriculum", "id": "Curriculum ID", "version": "Version", "status": "Review status",
        "language": "Language", "level": "Level", "hours": "Total hours", "audience": "Target audience",
        "generated": "Generated", "overview": "Overview", "rationale": "Rationale", "objectives": "Objectives",
        "prerequisites": "Prerequisites", "none": "None", "outcomes": "Learning outcomes", "modules": "Modules",
        "module": "Module", "topics": "Topics", "activities": "Activities", "outcome_ids": "Outcomes",
        "assessments": "Assessments", "type": "Type", "description": "Description", "capstone": "Capstone project",
        "deliverables": "Deliverables", "no_capstone": "No capstone project", "tools": "Tools and resources",
        "competencies": "Skills and competencies", "sources": "Sources", "alignment": "Framework alignment",
        "requirement": "Requirement", "result": "Result", "evidence": "Evidence", "alignment_none":
        "No framework comparison was performed: no approved framework document was selected, so alignment is "
        "unable to determine.", "gaps": "Gaps and recommendations", "references": "References",
        "references_none": "No external research sources were used.", "academy": "Academy catalogue evidence",
        "field": "Field", "value": "Value", "source": "Source", "assumptions": "Assumptions and limitations",
        "questions": "Open questions", "validation": "Validation", "valid": "All validation rules passed.",
        "invalid": "Validation problems remain (this version is a draft):", "page": "Page", "hours_short": "h",
        "data_notice": "Data notice", "disclaimer": "Important", "retrieved": "retrieved", "summary": "Summary",
        "proposed": "proposed (no source)", "change": "Change in this version",
    },
    "ar": {
        "curriculum": "منهج", "id": "معرّف المنهج", "version": "النسخة", "status": "حالة المراجعة",
        "language": "اللغة", "level": "المستوى", "hours": "إجمالي الساعات", "audience": "الجمهور المستهدف",
        "generated": "تاريخ الإنشاء", "overview": "نظرة عامة", "rationale": "المبررات", "objectives": "الأهداف",
        "prerequisites": "المتطلبات السابقة", "none": "لا يوجد", "outcomes": "مخرجات التعلم", "modules": "الوحدات",
        "module": "الوحدة", "topics": "الموضوعات", "activities": "الأنشطة", "outcome_ids": "المخرجات",
        "assessments": "التقييمات", "type": "النوع", "description": "الوصف", "capstone": "المشروع الختامي",
        "deliverables": "المخرجات المطلوبة", "no_capstone": "لا يوجد مشروع ختامي", "tools": "الأدوات والموارد",
        "competencies": "المهارات والكفاءات", "sources": "المصادر", "alignment": "المواءمة مع الإطار",
        "requirement": "المتطلب", "result": "النتيجة", "evidence": "الدليل", "alignment_none":
        "لم تُجرَ أي مقارنة مع إطار: لم تُحدَّد وثيقة إطار معتمدة، لذا يتعذر تحديد المواءمة.",
        "gaps": "الفجوات والتوصيات", "references": "المراجع", "references_none": "لم تُستخدم مصادر بحث خارجية.",
        "academy": "أدلة من كتالوج الأكاديمية", "field": "الحقل", "value": "القيمة", "source": "المصدر",
        "assumptions": "الافتراضات والقيود", "questions": "أسئلة مفتوحة", "validation": "التحقق",
        "valid": "اجتاز المنهج جميع قواعد التحقق.", "invalid": "تبقى مشكلات تحقق (هذه النسخة مسودة):",
        "page": "صفحة", "hours_short": "س", "data_notice": "تنبيه حول البيانات", "disclaimer": "تنبيه مهم",
        "retrieved": "تاريخ الاسترجاع", "summary": "ملخص", "proposed": "مقترح (دون مصدر)",
        "change": "التعديل في هذه النسخة",
    },
}
STATUS = {
    "en": {"draft": "Draft (validation problems remain)", "pending_review": "Pending review (not approved)",
           "changes_requested": "Changes requested (not approved)", "approved": "Approved by the academy"},
    "ar": {"draft": "مسودة (توجد مشكلات تحقق)", "pending_review": "بانتظار المراجعة (غير معتمد)",
           "changes_requested": "مطلوب تعديلات (غير معتمد)", "approved": "معتمد من الأكاديمية"},
}
ALIGNMENT_LABELS = {
    "en": {"covered": "Covered", "partially_covered": "Partially covered", "not_covered": "Not covered",
           "unable_to_determine": "Unable to determine"},
    "ar": {"covered": "مغطى", "partially_covered": "مغطى جزئيًا", "not_covered": "غير مغطى",
           "unable_to_determine": "يتعذر التحديد"},
}
LEVELS = {"ar": {"Beginner": "مبتدئ", "Intermediate": "متوسط", "Advanced": "متقدم"}}
LANGUAGE_NAMES = {"en": {"en": "English", "ar": "Arabic"}, "ar": {"en": "الإنجليزية", "ar": "العربية"}}


# ---------------------------------------------------------------------
# Fonts and mixed-script runs
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def _fonts() -> dict[str, set[int]]:
    """Register the bundled fonts once; returns each font's character set."""
    coverage = {}
    for name, filename in FONT_FILES.items():
        font = TTFont(name, str(FONTS_DIR / filename))
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(font)
        coverage[name] = set(font.face.charToGlyph)
    return coverage


def _is_arabic(char: str) -> bool:
    code = ord(char)
    return 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0x08A0 <= code <= 0x08FF \
        or 0xFB50 <= code <= 0xFDFF or 0xFE70 <= code <= 0xFEFF


def _is_letter(char: str) -> bool:
    return char.isalpha() or _is_arabic(char)


def runs(visual: str, bold: bool = False) -> list[tuple[str, str]]:
    """
    Split a display-ordered line into (text, font) runs. Arabic letters use
    the Arabic font, other letters the Latin font; neutral characters
    (spaces, digits, punctuation) stay in the current run's font when it has
    them. A character neither font has becomes "?".
    """
    coverage = _fonts()
    arabic, latin = ("RexArabic-Bold", "RexLatin-Bold") if bold else ("RexArabic", "RexLatin")
    out: list[list[str]] = []
    for char in visual:
        code = ord(char)
        if _is_letter(char):
            preferred = arabic if _is_arabic(char) else latin
        else:
            preferred = out[-1][1] if out else latin
        if code not in coverage[preferred]:
            other = latin if preferred == arabic else arabic
            if code in coverage[other]:
                preferred = other
            else:
                char, preferred = "?", (out[-1][1] if out else latin)
        if out and out[-1][1] == preferred:
            out[-1][0] += char
        else:
            out.append([char, preferred])
    return [(text, font) for text, font in out]


def display_line(logical: str, rtl: bool) -> str:
    """One logical line shaped and reordered for drawing left to right."""
    shaped = arabic_reshaper.reshape(logical) if any(_is_arabic(c) for c in logical) else logical
    return get_display(shaped, base_dir="R" if rtl else "L")


def _width(visual: str, size: float, bold: bool) -> float:
    return sum(pdfmetrics.stringWidth(text, font, size) for text, font in runs(visual, bold))


class TextBlock(Flowable):
    """A wrapped paragraph of mixed Arabic/Latin text, left- or right-aligned."""

    def __init__(self, text: Any, *, rtl: bool, size: float = 10, bold: bool = False,
                 color=colors.black, leading: Optional[float] = None, space_after: float = 3,
                 bullet: Optional[str] = None, latin_base: Optional[bool] = None) -> None:
        """
        rtl: right-aligned, right-to-left paragraph. latin_base: in an RTL
        document, lay the line out left to right (still right-aligned) --
        for mostly-Latin text such as references or an English framework
        requirement, where an RTL base direction would scramble brackets
        and punctuation. By default it is decided from the text itself.
        """
        super().__init__()
        self.text = " ".join(str(text if text is not None else "").split())
        self.rtl, self.size, self.bold, self.color = rtl, size, bold, color
        if latin_base is None:
            arabic = sum(1 for c in self.text if _is_arabic(c) and c.isalpha())
            latin = sum(1 for c in self.text if c.isascii() and c.isalpha())
            latin_base = latin > 0 and arabic < 0.3 * (arabic + latin)
        self.base_rtl = rtl and not latin_base
        self.leading = leading or size * (1.75 if rtl else 1.4)
        self.space_after = space_after
        self.bullet = bullet
        self.lines: list[str] = []

    def _fits(self, text: str, width: float) -> bool:
        return _width(display_line(text, self.base_rtl), self.size, self.bold) <= width

    def _split_long(self, word: str, width: float) -> list[str]:
        """A word wider than the line (e.g. a URL) is cut into pieces that fit."""
        pieces, current = [], ""
        for char in word:
            if current and not self._fits(current + char, width):
                pieces.append(current)
                current = ""
            current += char
        return pieces + ([current] if current else [])

    def _wrap_logical(self, width: float) -> list[str]:
        lines, current = [], ""
        for word in self.text.split(" "):
            pieces = self._split_long(word, width) if not self._fits(word, width) else [word]
            for piece in pieces:
                candidate = f"{current} {piece}".strip()
                if not current or self._fits(candidate, width):
                    current = candidate
                    continue
                lines.append(current)
                current = piece
        if current:
            lines.append(current)
        return lines or [""]

    def wrap(self, available_width, available_height):
        self.width = available_width
        indent = self.size * 1.2 if self.bullet else 0
        self.lines = [display_line(line, self.base_rtl) for line in self._wrap_logical(available_width - indent)]
        self.height = len(self.lines) * self.leading + self.space_after
        return available_width, self.height

    def draw(self):
        canvas = self.canv
        canvas.setFillColor(self.color)
        indent = self.size * 1.2 if self.bullet else 0
        y = self.height - self.space_after - self.size * (1.15 if self.rtl else 1.0)
        for index, line in enumerate(self.lines):
            parts = runs(line, self.bold)
            total = sum(pdfmetrics.stringWidth(t, f, self.size) for t, f in parts)
            x = (self.width - indent - total) if self.rtl else indent
            if self.bullet and index == 0:
                canvas.setFont("RexLatin", self.size)
                canvas.drawString(self.width - indent * 0.75 if self.rtl else 0, y, self.bullet)
            for text, font in parts:
                canvas.setFont(font, self.size)
                canvas.drawString(x, y, text)
                x += pdfmetrics.stringWidth(text, font, self.size)
            y -= self.leading


# ---------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------

class _Builder:
    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        self.content = record["content"]
        self.language = "ar" if self.content.get("language") == "ar" else "en"
        self.rtl = self.language == "ar"
        self.labels = LABELS[self.language]
        self.story: list[Any] = []
        self.width = A4[0] - 2 * PAGE_MARGIN

    # -- small helpers
    def text(self, value, **kwargs) -> TextBlock:
        return TextBlock(value, rtl=self.rtl, **kwargs)

    def heading(self, key_or_text: str, level: int = 1) -> None:
        label = self.labels.get(key_or_text, key_or_text)
        self.story += [Spacer(1, 6 if level == 1 else 3),
                       self.text(label, size=14 if level == 1 else 11.5, bold=True, color=ACCENT, space_after=4)]

    def para(self, value) -> None:
        if value:
            self.story.append(self.text(value))

    def bullets(self, items) -> None:
        items = [i for i in items or [] if i]
        if not items:
            self.para(self.labels["none"])
        for item in items:
            self.story.append(self.text(item, bullet="•", space_after=2))

    def table(self, header: list[str], rows: list[list[Any]], widths: list[float]) -> None:
        def cell(value, bold=False):
            return TextBlock(value, rtl=self.rtl, size=9, bold=bold, space_after=0)

        data = [[cell(h, True) for h in header]] + [[cell(v) for v in row] for row in rows]
        col_widths = [w * self.width for w in widths]
        if self.rtl:  # first column on the right
            data = [list(reversed(row)) for row in data]
            col_widths = list(reversed(col_widths))
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "RexLatin"),  # the table default would be Helvetica
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B7C3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
        self.story += [table, Spacer(1, 6)]

    def level(self, value) -> str:
        return LEVELS.get(self.language, {}).get(value, value or "")

    # -- sections
    def cover(self) -> None:
        c, r, L = self.content, self.record, self.labels
        status = r.get("version_status") or r.get("status")
        self.story += [Spacer(1, 30 * mm), self.text(L["curriculum"], size=12, color=GREY),
                       self.text(c.get("title"), size=22, bold=True, color=ACCENT, space_after=10)]
        self.para(c.get("overview"))
        self.story.append(Spacer(1, 8))
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.table([L["field"], L["value"]], [
            [L["id"], r.get("curriculum_id")], [L["version"], r.get("version")],
            [L["status"], STATUS[self.language].get(status, status)],
            [L["language"], LANGUAGE_NAMES[self.language].get(self.language)],
            [L["level"], self.level(c.get("level"))], [L["hours"], c.get("total_hours")],
            [L["audience"], c.get("target_audience")], [L["generated"], generated],
        ], [0.32, 0.68])
        if r.get("change_summary") and r.get("version", 1) > 1:
            self.story.append(self.text(f"{L['change']}: {r['change_summary']}", size=9, color=GREY))
        if c.get("data_notice"):
            self.story.append(self.text(f"{L['data_notice']}: {c['data_notice']}", size=9, color=GREY))
        self.story.append(self.text(f"{L['disclaimer']}: {DISCLAIMER[self.language]}", size=9, color=GREY))
        self.story.append(PageBreak())

    def body(self) -> None:
        c, L = self.content, self.labels
        self.heading("rationale")
        self.para(c.get("rationale"))
        self.heading("objectives")
        self.bullets(c.get("objectives"))
        self.heading("audience")
        self.para(c.get("target_audience"))
        self.heading("prerequisites")
        self.bullets(c.get("prerequisites"))
        self.heading("outcomes")
        self.bullets(f"{o.get('id')}: {o.get('statement')}" for o in c.get("learning_outcomes") or [])

        self.heading("modules")
        modules = c.get("modules") or []
        self.table([L["module"], L["topics"], L["hours"], L["outcome_ids"]],
                   [[f"{m.get('id')} {m.get('title')}", "، ".join(m.get("topics") or []) if self.rtl
                     else ", ".join(m.get("topics") or []), m.get("hours"), ", ".join(m.get("outcome_ids") or [])]
                    for m in modules], [0.34, 0.40, 0.11, 0.15])
        for module in modules:
            self.heading(f"{module.get('id')} {module.get('title')} ({module.get('hours')} {L['hours_short']})", 2)
            self.para(module.get("description"))
            activities = [f"{a.get('type')}: {a.get('description')}" for a in module.get("activities") or []]
            if activities:
                self.story.append(self.text(L["activities"], bold=True, size=10, space_after=1))
                self.bullets(activities)

        self.heading("assessments")
        self.table([L["id"].split()[-1] if self.language == "en" else "المعرّف", L["type"], L["description"],
                    L["outcome_ids"]],
                   [[a.get("id"), a.get("type"), a.get("description"), ", ".join(a.get("outcome_ids") or [])]
                    for a in c.get("assessments") or []], [0.08, 0.17, 0.57, 0.18])
        self.heading("capstone")
        capstone = c.get("capstone")
        if isinstance(capstone, dict):
            self.story.append(self.text(capstone.get("title"), bold=True))
            self.para(capstone.get("description"))
            self.para(f"{L['outcome_ids']}: {', '.join(capstone.get('outcome_ids') or [])}")
            self.story.append(self.text(L["deliverables"], bold=True, size=10, space_after=1))
            self.bullets(capstone.get("deliverables"))
        else:
            self.para(f"{L['no_capstone']}: {c.get('capstone_omitted_reason') or ''}")
        self.heading("tools")
        self.bullets(c.get("tools_resources"))
        self.heading("competencies")
        self.bullets(f"{comp.get('name')} — {', '.join(comp['source_ids']) if comp.get('source_ids') else L['proposed']}"
                     for comp in c.get("competencies") or [])
        self.alignment()
        self.references()
        self.academy()
        self.heading("assumptions")
        self.bullets(c.get("assumptions"))
        if c.get("open_questions"):
            self.heading("questions")
            self.bullets(c.get("open_questions"))
        self.validation()

    def alignment(self) -> None:
        L = self.labels
        self.heading("alignment")
        alignment = self.content.get("alignment") or self.record.get("alignment")
        if not alignment or not alignment.get("items"):
            self.para((alignment or {}).get("message") or L["alignment_none"])
            return
        framework = alignment.get("framework") or {}
        self.para(f"{framework.get('name')} — {framework.get('publisher')}, {framework.get('version')} "
                  f"({framework.get('source_url')}, {L['retrieved']} {framework.get('retrieved_on')})")
        counts = alignment.get("summary") or {}
        if counts:
            self.para(f"{L['summary']}: " + " · ".join(
                f"{ALIGNMENT_LABELS[self.language][k]} {v}" for k, v in counts.items()))
        self.table([L["requirement"], L["result"], L["evidence"]],
                   [[f"{i.get('requirement_id')}: {i.get('requirement') or ''}",
                     ALIGNMENT_LABELS[self.language].get(i.get("status"), i.get("status")),
                     (", ".join(i.get("element_ids") or []) + (" — " if i.get("element_ids") else "")
                      + (i.get("explanation") or ""))]
                    for i in alignment["items"]], [0.42, 0.18, 0.40])
        gaps = [i for i in alignment["items"] if i.get("status") in ("not_covered", "partially_covered")]
        if gaps:
            self.heading("gaps", 2)
            self.bullets(f"{g['requirement_id']}: {g.get('recommendation') or g.get('explanation') or ''}"
                         for g in gaps)
        self.story.append(self.text(DISCLAIMER[self.language], size=9, color=GREY))

    def references(self) -> None:
        L = self.labels
        self.heading("references")
        references = self.content.get("references") or []
        if not references:
            self.para(L["references_none"])
        for ref in references:
            self.story.append(self.text(
                f"[{ref.get('source_id')}] {ref.get('title')} — {ref.get('publisher')}. {ref.get('url')} "
                f"({L['retrieved']} {str(ref.get('retrieved_at') or '')[:10]})", size=9, bullet="•"))

    def academy(self) -> None:
        evidence = self.content.get("academy_evidence") or []
        if not evidence:
            return
        L = self.labels
        self.heading("academy")
        self.table([L["id"].split()[-1] if self.language == "en" else "المقرر", L["field"], L["value"], L["source"]],
                   [[e.get("course_id"), e.get("field"), e.get("value"), e.get("source")] for e in evidence],
                   [0.12, 0.2, 0.43, 0.25])

    def validation(self) -> None:
        L = self.labels
        self.heading("validation")
        validation = self.record.get("validation") or {}
        errors = validation.get("errors") or []
        if not errors:
            self.para(L["valid"])
            return
        self.para(L["invalid"])
        self.bullets(e.get("message") for e in errors)

    # -- page decoration
    def on_page(self, canvas, doc) -> None:
        canvas.saveState()
        footer = f"{self.record.get('curriculum_id')} · v{self.record.get('version')} · {self.labels['page']} {doc.page}"
        line = display_line(footer, self.rtl)
        parts = runs(line)
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


def render_pdf(record: dict[str, Any]) -> tuple[bytes, int]:
    """
    The PDF of one stored curriculum version (a CurriculumStore.get record).
    Returns (pdf bytes, page count).
    """
    _fonts()
    builder = _Builder(record)
    builder.cover()
    builder.body()
    buffer = io.BytesIO()
    pages = {"count": 0}

    def on_page(canvas, doc):
        pages["count"] = doc.page
        builder.on_page(canvas, doc)

    content = record["content"]
    document = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN, topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN, title=str(content.get("title") or "Curriculum"),
        author="REX AI Curriculum Agent", subject=f"{record.get('curriculum_id')} version {record.get('version')}",
        creator="REX AI", lang="ar" if builder.rtl else "en",
        initialFontName="RexLatin")  # never reference a machine font (Helvetica by default)
    document.build(builder.story, onFirstPage=on_page, onLaterPages=on_page)
    return buffer.getvalue(), pages["count"]
