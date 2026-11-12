"""
Agent 3's PDFs (agents/curriculum_research/pdf.py), read back with pypdf:
English and Arabic content, Arabic letter joining and line order, mixed
Arabic/Latin runs, bundled embedded fonts only, status labels, alignment
and validation sections, and the render tool (safe names, checksums).
"""

import hashlib
import io
import re
import unicodedata

import pytest
from pypdf import PdfReader

from agents.curriculum_research.pdf import TextBlock, display_line, render_pdf, runs
from tests.curriculum_fakes import in_process_tools
from tests.curriculum_fixtures import valid_curriculum


def record(language="en", status="pending_review", validation=None, **content_changes):
    content = valid_curriculum(language)
    content.update(content_changes)
    return {"curriculum_id": "CUR-0123456789ab", "version": 2, "version_status": status, "content": content,
            "validation": validation or {"valid": True, "errors": []}, "change_summary": "Hours changed."}


def read(data: bytes) -> tuple[PdfReader, str]:
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join(page.extract_text() for page in reader.pages)
    return reader, text


def fonts(reader: PdfReader) -> dict[str, bool]:
    """Font name -> embedded (has a font file)."""
    out = {}
    for page in reader.pages:
        for ref in page["/Resources"]["/Font"].values():
            font = ref.get_object()
            descendant = font["/DescendantFonts"][0].get_object() if "/DescendantFonts" in font else font
            descriptor = descendant.get("/FontDescriptor")
            out[str(font["/BaseFont"])] = bool(descriptor and "/FontFile2" in descriptor.get_object())
    return out


def test_english_pdf_contains_the_curriculum():
    data, pages = render_pdf(record())

    reader, text = read(data)
    content = valid_curriculum()
    assert pages == len(reader.pages) >= 2
    for expected in (content["title"], "Machine Learning Foundations", "Data preparation", "Models and evaluation",
                     "LO1: Prepare a dataset for modelling", "Capstone project", "scikit-learn",
                     "CUR-0123456789ab", "Pending review (not approved)", "Page 2"):
        assert expected in text
    assert "not accreditation" in text  # the disclaimer
    assert reader.metadata.title == content["title"] and reader.metadata.author == "REX AI Curriculum Agent"


def test_only_the_bundled_fonts_are_used_and_embedded():
    for language, expected in (("en", {"NotoSans-Regular", "NotoSans-Bold"}),
                               ("ar", {"NotoSans-Regular", "NotoNaskhArabic-Regular", "NotoNaskhArabic-Bold"})):
        reader, _ = read(render_pdf(record(language))[0])
        used = fonts(reader)
        names = {name.split("+")[-1] for name in used}
        assert expected <= names and all(name.startswith(("NotoSans", "NotoNaskhArabic")) for name in names)
        assert all(used.values())  # embedded, never a machine font


def test_arabic_pdf_is_shaped_and_reads_in_order():
    data, _ = render_pdf(record("ar"))

    _, raw = read(data)
    # Letters were joined: the text uses Arabic presentation forms.
    assert any(0xFE70 <= ord(char) <= 0xFEFF for char in raw)
    text = unicodedata.normalize("NFKC", raw)
    content = valid_curriculum("ar")
    # (pypdf puts bracketed runs on their own line, so the status is checked in parts.)
    for expected in (content["title"], "تجهيز البيانات", "بناء النماذج وتقييمها", "مخرجات التعلم", "المشروع الختامي",
                     "بانتظار المراجعة", "غير معتمد"):
        assert expected in text  # read back in logical order
    assert "scikit-learn" in text and "Python" in text  # Latin terms keep their own order


def test_mixed_lines_are_drawn_in_runs_with_a_font_that_has_the_glyphs():
    line = display_line("يستخدم المتدربون Python 3.11 بنسبة 100%", rtl=True)

    parts = runs(line)
    assert {font for _, font in parts} == {"RexArabic", "RexLatin"}
    assert ("Python" in "".join(t for t, f in parts if f == "RexLatin"))
    assert all(not any("a" <= c.lower() <= "z" for c in t) for t, f in parts if f == "RexArabic")
    assert "%" in "".join(t for t, f in parts if f == "RexLatin")  # the Arabic font has no "%"
    assert runs("☃") == [("?", "RexLatin")]  # a character no bundled font has


def test_arabic_paragraphs_wrap_in_reading_order():
    # Words without the lam-alef ligature, so reversing a visual line gives the logical text back.
    words = ["واحد", "اثنان", "تمرين", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة"] * 3
    block = TextBlock(" ".join(words), rtl=True, size=11)

    block.wrap(120, 1000)

    assert len(block.lines) > 2
    logical = [unicodedata.normalize("NFKC", line)[::-1] for line in block.lines]
    assert logical[0].startswith("واحد") and logical[-1].endswith("عشرة")
    assert " ".join(logical).split() == words  # every word, in reading order


def test_mostly_latin_text_in_an_arabic_pdf_keeps_its_own_direction():
    """References or English framework requirements would get scrambled brackets in an RTL layout."""
    assert TextBlock("[SRC-1] NIST NICE Framework (https://www.nist.gov/nice)", rtl=True).base_rtl is False
    assert TextBlock("يطبق المتدرب مكتبة Python في تحليل البيانات", rtl=True).base_rtl is True
    assert TextBlock("Python", rtl=False).base_rtl is False


def test_long_words_such_as_urls_are_broken_to_fit():
    block = TextBlock("https://example.org/" + "a" * 300, rtl=False, size=9)

    block.wrap(200, 1000)

    assert len(block.lines) > 1
    assert "".join(block.lines) == "https://example.org/" + "a" * 300


@pytest.mark.parametrize("status, label", [("draft", "Draft (validation problems remain)"),
                                           ("approved", "Approved by the academy"),
                                           ("changes_requested", "Changes requested (not approved)")])
def test_the_review_status_is_printed(status, label):
    _, text = read(render_pdf(record(status=status))[0])
    assert label in text


def test_validation_problems_and_missing_alignment_are_stated():
    validation = {"valid": False, "errors": [{"code": "outcome_unlinked", "message": "LO2 is not assessed."}]}
    _, text = read(render_pdf(record(validation=validation))[0])

    assert "Validation problems remain" in text and "LO2 is not assessed." in text
    assert "alignment is unable to determine" in text.replace("\n", " ")


def test_the_alignment_table_and_gaps_are_printed():
    alignment = {"framework": {"framework_id": "test-framework", "name": "Test Competency Framework",
                               "publisher": "Test Body", "version": "1.0", "source_url": "https://example.org/f",
                               "retrieved_on": "2026-10-27"},
                 "summary": {"covered": 1, "partially_covered": 0, "not_covered": 1, "unable_to_determine": 1},
                 "items": [{"requirement_id": "R1", "requirement": "Prepare and clean a dataset", "status": "covered",
                            "element_ids": ["LO1", "M1"], "explanation": "Taught in M1."},
                           {"requirement_id": "R3", "requirement": "Deploy a model", "status": "not_covered",
                            "element_ids": [], "explanation": "Not taught.", "recommendation": "Add a deployment module."},
                           {"requirement_id": "R2", "requirement": "Evaluate models", "status": "unable_to_determine",
                            "element_ids": [], "explanation": "Unclear."}]}
    _, text = read(render_pdf(record(alignment=alignment))[0])
    flat = text.replace("\n", " ")

    for expected in ("Test Competency Framework", "R1: Prepare and clean a dataset", "Covered", "Not covered",
                     "Unable to determine", "LO1, M1", "Add a deployment module.", "Gaps and recommendations"):
        assert expected in flat


def test_the_render_tool_stores_a_safe_named_file_with_its_checksum(tmp_path):
    tools = in_process_tools(tmp_path)
    saved = tools.save_curriculum_draft(valid_curriculum("ar"), topic="ml")
    tools.store.approve(saved["curriculum_id"], 1, "admin-test")

    result = tools.render_curriculum_pdf(saved["curriculum_id"])

    document = result["document"]
    assert result["status"] == "ok"
    assert re.fullmatch(rf"{saved['curriculum_id']}_v1_ar_[0-9a-f]{{8}}\.pdf", document["filename"])
    assert document["pages"] >= 2
    data = tools.document_bytes(document["document_id"])
    assert data.startswith(b"%PDF") and hashlib.sha256(data).hexdigest() == document["sha256"]
    assert (tmp_path / "docs" / document["filename"]).is_file()
    assert tools.render_curriculum_pdf(saved["curriculum_id"], language="en")["status"] == "invalid_request"


def test_no_pdf_is_rendered_for_a_version_that_is_not_approved(tmp_path):
    tools = in_process_tools(tmp_path)
    pending = tools.save_curriculum_draft(valid_curriculum("en"), topic="ml")
    invalid = valid_curriculum("en")
    invalid["learning_outcomes"][0]["statement"] = "The topic in general"
    draft = tools.save_curriculum_draft(invalid, topic="ml")

    for saved in (pending, draft):
        result = tools.render_curriculum_pdf(saved["curriculum_id"])
        assert result["status"] == "invalid_request" and "after an administrator approves" in result["message"]
        assert tools.store.documents(saved["curriculum_id"]) == []
    assert not (tmp_path / "docs").exists() or not any((tmp_path / "docs").iterdir())


def test_every_render_gets_its_own_file_and_earlier_records_keep_theirs(tmp_path):
    tools = in_process_tools(tmp_path)
    cid = tools.save_curriculum_draft(valid_curriculum("en"), topic="ml")["curriculum_id"]
    tools.store.approve(cid, 1, "admin-test")

    first = tools.render_curriculum_pdf(cid)["document"]
    first_bytes = tools.document_bytes(first["document_id"])
    second = tools.render_curriculum_pdf(cid)["document"]

    assert first["filename"] != second["filename"]
    assert tools.document_bytes(first["document_id"]) == first_bytes   # never overwritten
    assert hashlib.sha256(first_bytes).hexdigest() == first["sha256"]
