"""
PDF layout of generated documents: a heading is never left alone at the bottom of a page (one central rule in the
renderer, for every document type), Arabic stays right-to-left, multiple-choice options carry A/B/C/D (Arabic: أ ب ج د)
labels in the PDF only, and the wide rubric table gives its criterion column room.
"""

import io

import pytest
from pypdf import PdfReader

from agents.document_generation import pdf as document_pdf
from agents.document_generation.pdf import render_document_pdf


def record(language, sections, document_type="course_description"):
    return {"document_id": "GDOC-TEST", "version": 1, "version_status": "draft", "content": {
        "language": language, "document_type": document_type, "title": "Layout test", "course_title": "Layout test",
        "curriculum_id": "CUR-1", "curriculum_version": 1, "level": "Beginner", "total_hours": 40, "sections": sections}}


def section(blocks, heading="Section"):
    return {"key": "s", "heading": heading, "blocks": blocks}


def para(text):
    return {"type": "paragraph", "text": text, "source": "generated"}


def head(text):
    return {"type": "heading", "text": text, "source": "curriculum"}


def pages_of(data):
    return [page.extract_text() for page in PdfReader(io.BytesIO(data)).pages]


def with_filler(n, language="en", marker="X"):
    """n one-line paragraphs, then a heading and its text: sweeping n moves the heading through every position on a page."""
    filler = [para(("Filler line %d." if language == "en" else "سطر %d.") % i) for i in range(n)]
    return record(language, [section(filler + [head(f"{'Heading' if language == 'en' else 'عنوان'} H{marker}{n}"),
                                               para(f"{'Body' if language == 'en' else 'نص'} B{marker}{n}")])])


def heading_stranded(language, n):
    """True when the heading's page does not also hold its text (the heading is stranded at the bottom of a page)."""
    pages = pages_of(render_document_pdf(with_filler(n, language))[0])
    heading_page = next(i for i, text in enumerate(pages) if f"HX{n}" in text)
    return f"BX{n}" not in pages[heading_page]


SWEEP = range(30, 52)                    # around where the first content page fills up (one page holds about 40 lines)


def test_the_sweep_can_detect_a_stranded_heading(monkeypatch):
    # the same sweep with the rule switched off finds stranded headings, so a passing sweep below means something
    monkeypatch.setattr(document_pdf._Builder, "_with_next", staticmethod(lambda *flowables: list(flowables)))
    assert any(heading_stranded("en", n) for n in SWEEP)


@pytest.mark.parametrize("n", SWEEP)
def test_a_heading_is_never_left_alone_at_the_bottom_of_a_page(n):
    assert not heading_stranded("en", n)


@pytest.mark.parametrize("n", SWEEP)
def test_the_same_holds_for_arabic_documents(n):
    assert not heading_stranded("ar", n)


def test_every_heading_and_the_spacer_before_it_is_marked_keep_with_next():
    blocks = [para("one"), head("A heading"), para("two"), head("Second"), {"type": "bullets", "items": ["x", "y"], "source": "generated"}]
    builder = document_pdf._Builder(record("en", [section(blocks, "The section")]))
    builder.sections()
    story = builder.story
    headings = [i for i, f in enumerate(story) if getattr(f, "text", None) in ("The section", "A heading", "Second")]

    assert len(headings) == 3
    for index in headings:
        assert story[index].keepWithNext and story[index - 1].keepWithNext          # the heading and its spacer
    assert not any(getattr(f, "keepWithNext", False) for f in story if getattr(f, "text", None) in ("one", "two", "x", "y"))


def test_a_chain_of_headings_moves_with_the_first_block_that_follows():
    # section heading + two block headings in a row, then text: none of the three may be left behind
    for n in SWEEP:
        data = record("en", [section([*[para(f"Filler {i}.") for i in range(n)], head(f"Group G{n}"), head(f"Subgroup S{n}"), para(f"Text T{n}")], "Top")])
        pages = pages_of(render_document_pdf(data)[0])
        page = next(text for text in pages if f"S{n}" in text)
        assert f"T{n}" in page and f"G{n}" in page


def test_margins_page_numbers_and_arabic_direction_are_unchanged():
    data, pages = render_document_pdf(with_filler(40, "ar"))
    texts = pages_of(data)

    assert data.startswith(b"%PDF") and pages == len(texts) >= 2
    assert all("GDOC-TEST" in text for text in texts)                 # the footer (document id and page number) on every page
    assert "1" in texts[0] and str(pages) in texts[-1]


# ---- A/B/C/D labels (PDF only) ----------------------------------------------------------------------

def options_block(items):
    return {"type": "bullets", "items": items, "source": "generated", "role": "q_options"}


def answer_block(text):
    return {"type": "paragraph", "text": text, "source": "generated", "role": "q_answer"}


def story_texts(language, blocks, document_type="question_bank"):
    builder = document_pdf._Builder(record(language, [section(blocks)], document_type))
    builder.sections()
    return [f.text for f in builder.story if hasattr(f, "text")]


# the second multiple-choice question of the inspected English Question Bank (GDOC-a690070818b6, page 2)
INSPECTED_EN = ["Apply feature scaling", "Handle missing values", "Remove duplicates", "Identify and correct data entry errors"]
# the first question of the inspected Arabic Question Bank (GDOC-865b6e1f785a, page 2)
INSPECTED_AR = ["تحديد البيانات التي يجب استبعادها", "استخدام الفلاتر لاستبعاد البيانات غير المرغوب فيها",
                "تحليل البيانات لتحديد الأنماط والعيوب", "تحديد العناصر غير الصحيحة في البيانات"]


def test_english_options_are_labelled_a_to_d_and_the_answer_names_its_letter():
    texts = story_texts("en", [options_block(INSPECTED_EN), head("Correct answer"), answer_block("Identify and correct data entry errors")])

    assert [t for t in texts if t[:3] in ("A. ", "B. ", "C. ", "D. ")] == [
        "A. Apply feature scaling", "B. Handle missing values", "C. Remove duplicates", "D. Identify and correct data entry errors",
        "D. Identify and correct data entry errors"]                                  # four options, then the answer: letter D


def test_arabic_options_use_arabic_letters_and_the_answer_names_its_letter():
    texts = story_texts("ar", [options_block(INSPECTED_AR), head("الإجابة الصحيحة"), answer_block(INSPECTED_AR[0])])

    assert texts[-1] == f"أ. {INSPECTED_AR[0]}"
    assert [t[:2] for t in texts if t[:2] in ("أ.", "ب.", "ج.", "د.")] == ["أ.", "ب.", "ج.", "د.", "أ."]
    assert not any(t[:2] in ("A.", "B.") for t in texts)


def test_labels_are_added_when_drawing_only_the_stored_content_is_not_touched():
    blocks = [options_block(list(INSPECTED_EN)), answer_block("Remove duplicates")]
    data = record("en", [section(blocks)], "question_bank")
    before = repr(data)

    pdf, _ = render_document_pdf(data)

    assert repr(data) == before and "A. " not in repr(data)                          # nothing was written into the content
    text = " ".join(pages_of(pdf))
    assert "A. Apply feature scaling" in text and "C. Remove duplicates" in text


def test_an_answer_that_is_not_one_of_the_options_gets_no_letter():
    texts = story_texts("en", [options_block(INSPECTED_EN), answer_block("Something else entirely")])
    assert texts[-1] == "Something else entirely"
    twice = story_texts("en", [options_block(["Same", "Same", "Other", "More"]), answer_block("Same")])
    assert twice[-1] == "Same"                                                       # ambiguous: no letter rather than a wrong one


def test_options_of_the_next_question_are_not_mixed_up_with_the_previous_one():
    texts = story_texts("en", [options_block(["One", "Two", "Three", "Four"]), answer_block("Two"),
                               options_block(["Five", "Six", "Seven", "Eight"]), answer_block("Eight")])
    # each option list is labelled A-D from its own start, and each answer takes the letter of ITS question's option
    assert [t for t in texts if t[1:3] == ". "] == ["A. One", "B. Two", "C. Three", "D. Four", "B. Two",
                                                    "A. Five", "B. Six", "C. Seven", "D. Eight", "D. Eight"]


def test_ordinary_bullets_keep_their_bullet_and_no_letters():
    builder = document_pdf._Builder(record("en", [section([{"type": "bullets", "items": ["x", "y"], "source": "generated"}])]))
    builder.sections()
    bullets = [f for f in builder.story if getattr(f, "bullet", None)]
    assert [b.text for b in bullets] == ["x", "y"] and all(b.bullet == "•" for b in bullets)


# ---- the rubric score table -------------------------------------------------------------------------

def table_widths(language, columns):
    header = [f"C{i}" for i in range(columns)]
    builder = document_pdf._Builder(record(language, [section([{"type": "table", "header": header, "rows": [["x"] * columns], "source": "curriculum"}])]))
    builder.sections()
    grid = next(f for f in builder.story if hasattr(f, "_colWidths"))
    return list(grid._colWidths)


def test_the_score_table_gives_its_criterion_column_room():
    widths = table_widths("en", 6)
    assert widths[0] > 2 * widths[1] and abs(sum(widths) - document_pdf._Builder(record("en", [])).width) < 1
    assert len(set(round(w, 3) for w in widths[1:])) == 1                             # the five score columns are equal


def test_the_arabic_score_table_is_mirrored_with_the_criterion_column_on_the_right():
    widths = table_widths("ar", 6)
    assert widths[-1] > 2 * widths[0]


def test_two_and_three_column_tables_keep_their_widths():
    total = document_pdf._Builder(record("en", [])).width
    assert [round(w / total, 2) for w in table_widths("en", 2)] == [0.7, 0.3]
    assert [round(w / total, 2) for w in table_widths("en", 3)] == [0.42, 0.38, 0.2]
