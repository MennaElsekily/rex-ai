"""
Checks for Agent 4: the request, the curriculum it is built from, the
model's narrative parts, and the finished document content.

The document is validated again by the tool that saves it
(document_tools.py): a content dict passed in is never trusted. The key
rule is that every block copied from the curriculum (source "curriculum"
or "missing") must be exactly what composing the document from the
approved curriculum would produce, so an edit or a faulty caller cannot
change approved course facts. Only "generated" and "admin" blocks are
free text, and those are checked for accreditation claims and for hour
figures that contradict the curriculum.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Iterator, Optional

from agents.curriculum_research.validation import _ACCREDITATION_AR, _ACCREDITATION_EN
from agents.document_generation import schemas
from agents.document_generation.composer import compose
from agents.document_generation.templates import (
    EXTRA_FIELDS,
    SECTION_HEADINGS,
    TEMPLATES,
    DocumentTemplate,
)
from agents.document_generation import terms, units
from agents.text_utils import normalize_for_matching

MAX_TEXT = 4000
MAX_ITEMS = 60
MAX_BLOCKS_PER_SECTION = 1000
# Two texts this alike are treated as the same question / the same performance description.
SIMILAR = 0.88
# "12 hours", "12-hour", "12h" / "12 ساعة", "12 ساعات"
_HOURS = re.compile(r"(\d+(?:[.,]\d+)?)\s*[- ]?\s*(?:hours?|hrs?|h\b|ساع[ةه]|ساعات)", re.IGNORECASE)
# "45 minutes", "10-min", "15 دقيقة", "20 دقائق": the curriculum records hours only, so minute timings are invented.
_MINUTES = re.compile(r"(\d+(?:[.,]\d+)?)\s*[- ]?\s*(?:minutes?|mins?\b|دقيق[ةه]|دقائق)", re.IGNORECASE)
# An option that stands for the other options. Applied to text normalised by normalize_for_matching (alef/hamza, ya and
# ta marbuta folded), so the Arabic forms below are written in that folded spelling.
_NONE_OF_THE_ABOVE = re.compile(
    r"\b(?:all|none|both|neither)\s+(?:of\s+)?(?:the\s+)?(?:above|these|options|answers|choices)\b|"
    r"\ball\s+(?:answers|options|choices|of\s+them)\s+are\s+(?:correct|true)\b|"
    r"(?:جميع|كل)\s+(?:ما\s+)?(?:سبق|ذكر|ورد|الاطراف|الخيارات|الاختيارات|الاجابات|العبارات|الاعلاه)|"
    r"(?:جميع|كل)\s+\S+\s+اعلاه|(?:الخيارات|الاختيارات|الاجابات|العبارات)\s+(?:السابقه|اعلاه|جميعها|كلها)|"
    r"لا\s+(?:شيء|احد|واحد)\s+مما\s+(?:سبق|ذكر|ورد)|لا\s+شيء\s+من\s+(?:ذلك|ما\s+سبق)|"
    r"^\s*(?:كلاهما|كلا\s+الخيارين)\s*$", re.IGNORECASE)


# Scripts that never belong in an Arabic or English document: Chinese, Japanese, Korean, Cyrillic, Thai, Devanagari.
# Latin and Arabic are not listed, so English technical terms in an Arabic document are fine.
_FOREIGN_SCRIPT = re.compile(r"[Ѐ-ӿ฀-๿ऀ-ॿ぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]+")
# A position or label written into an option's text: "…، 0", "…, 2", "… - 3" at the end; "A) ", "1. ", "(2) ", "أ) " at the start.
_OPTION_INDEX_END = re.compile(r"(?:[,،;:|\-–—]\s*[0-4]\s*[.)\]]?|[(\[]\s*[0-4]\s*[)\]])\s*$")
_OPTION_INDEX_START = re.compile(r"^\s*[(\[]?\s*(?:[A-Da-d]|[1-4]|[أبجد])\s*[).:\]\-–]\s+")
_SENTENCE_END = re.compile(r"[.!?؟۔;؛\n]+")
MIN_REPEATED_WORDS = 3


def issue(code: str, message: str, where: str = "", target: str = "") -> dict[str, str]:
    """target (optional): the item of a unit the problem is about, e.g. "multiple_choice:1" (see generator._targeted_repair)."""
    return {"code": code, "where": where, "message": message, **({"target": target} if target else {})}


# -- the curriculum ---------------------------------------------------------

def missing_curriculum_fields(content: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    """Required curriculum fields that are empty."""
    missing = []
    for field in required:
        value = content.get(field)
        if value is None or value == "" or value == [] or (field == "total_hours" and not isinstance(value, int)):
            missing.append(field)
    return missing


def allowed_hours(content: dict[str, Any]) -> set[float]:
    hours = {float(content["total_hours"])} if isinstance(content.get("total_hours"), int) else set()
    hours |= {float(m["hours"]) for m in content.get("modules") or [] if isinstance(m.get("hours"), (int, float))}
    return hours


# -- the request ------------------------------------------------------------

def clean_extra(extra: Any) -> tuple[dict[str, str], list[dict[str, str]]]:
    """The administrator's extra information, trimmed; unknown or oversized values are reported."""
    if extra in (None, {}):
        return {}, []
    if not isinstance(extra, dict):
        return {}, [issue("extra_invalid", "Additional information must be a set of named values.")]
    clean, problems = {}, []
    for key, value in extra.items():
        if key not in EXTRA_FIELDS:
            problems.append(issue("extra_unknown", f"Unknown additional information: {key}.", key))
            continue
        text = " ".join(str(value or "").split())
        if len(text) > EXTRA_FIELDS[key]:
            problems.append(issue("extra_too_long", f"{key} is longer than {EXTRA_FIELDS[key]} characters.", key))
        elif text:
            clean[key] = text
    return clean, problems


# -- free text --------------------------------------------------------------

def text_problems(text: str, hours: set[float], where: str = "", minutes: bool = False) -> list[dict[str, str]]:
    """
    Accreditation claims and hour figures that the curriculum does not support; with minutes=True also minute-level
    timings (the curriculum records hours only).
    """
    problems = []
    normalized = normalize_for_matching(text)
    foreign = _FOREIGN_SCRIPT.findall(text)
    if foreign:
        problems.append(issue("foreign_script", f"The text contains characters from another writing system "
                              f"(\"{' '.join(foreign)[:30]}\"). Write only in the document's language; technical terms "
                              "such as Python or SQL may stay in English.", where))
    if minutes and _MINUTES.search(normalized):
        problems.append(issue("minutes_unsupported", "The text states a timing in minutes, which the approved curriculum "
                              "does not record. Refer to the module's hours only.", where))
    if _ACCREDITATION_EN.search(text) or _ACCREDITATION_AR.search(normalized):
        problems.append(issue("accreditation_claim", "The text must not claim accreditation, certification or "
                              "official approval.", where))
    for match in _HOURS.finditer(normalized):
        value = float(match.group(1).replace(",", "."))
        if hours and value not in hours:
            problems.append(issue("hours_mismatch", f"The text mentions {match.group(1)} hours, which the approved "
                                  "curriculum does not state.", where))
    return problems


def placeholder_parts(template: DocumentTemplate, content: dict[str, Any]) -> Any:
    """Parts with placeholder text, so the curriculum-sourced blocks can be recomposed and compared."""
    schema = template.parts_schema
    if schema is None:
        return None
    x = "x"
    if schema is schemas.DescriptionParts:
        return schema.model_construct(summary=x, assessment_approach=x)
    if schema is schemas.ProposalParts:
        return schema.model_construct(executive_summary=x, background_need=x, methodology=x, expected_deliverables=[x])
    if schema is schemas.GuideParts:
        return schema.model_construct(introduction=x, assessment_instructions=x, modules=[
            schemas.ModuleGuide.model_construct(module_id=m.get("id"), key_concepts=[x], exercise=x)
            for m in content.get("modules") or []])
    modules = content.get("modules") or []
    if schema is schemas.InstructorParts:
        return schema.model_construct(preparation=x, assessment_guidance=x, modules=[
            schemas.ModuleTeaching.model_construct(module_id=m.get("id"), teaching_methods=[x], instructor_activities=[x],
                                                   instructor_notes=x, discussion_questions=[x]) for m in modules])
    if schema is schemas.SlidesParts:
        return schema.model_construct(closing_summary=x, modules=[
            schemas.ModuleSlides.model_construct(module_id=m.get("id"), key_points=[x], example=x, discussion_prompt=x,
                                                 visual=x) for m in modules])
    if schema is schemas.QuestionBankParts:
        mc = schemas.McQuestion.model_construct(question=x, options=[x] * units.MC_OPTIONS, correct_index=0, rationale=x)
        tf = schemas.TfQuestion.model_construct(statement=x, answer=True, rationale=x)
        sa = schemas.SaQuestion.model_construct(question=x, model_answer=x, rationale=x)
        return schema.model_construct(modules=[schemas.ModuleQuestions.model_construct(
            module_id=m.get("id"), multiple_choice=[mc, mc], true_false=[tf], short_answer=[sa]) for m in modules])
    if schema is schemas.RubricParts:
        return schema.model_construct(rubrics=[schemas.AssessmentRubric.model_construct(
            assessment_id=spec["id"], instructions=x, criteria=[schemas.CriterionLevels.model_construct(
                outcome_id=c["outcome_id"], beginning=x, developing=x, proficient=x, advanced=x) for c in spec["criteria"]])
            for spec in units.rubric_specs(content) if spec["criteria"]])
    return schema.model_construct(introduction=x, evaluation_guidance=x)


def check_parts(parts: Any, template: DocumentTemplate, content: dict[str, Any]) -> list[dict[str, str]]:
    """Checks the model's narrative parts against the approved curriculum."""
    if parts is None:
        return []
    if template.unit_schema is not None:
        return check_unit_parts(parts, template, content)
    problems: list[dict[str, str]] = []
    hours = allowed_hours(content)
    for name, value in parts.model_dump().items():
        if name == "modules":
            continue
        texts = value if isinstance(value, list) else [value]
        for text in texts:
            problems += text_problems(str(text), hours, name)
    if isinstance(parts, schemas.GuideParts):
        expected = [m.get("id") for m in content.get("modules") or []]
        given = [m.module_id for m in parts.modules]
        if sorted(given) != sorted(expected) or len(set(given)) != len(given):
            problems.append(issue("module_ids", f"Write exactly one entry for each module id: {', '.join(expected)} "
                                  f"(got {', '.join(given) or 'none'}).", "modules"))
        for module in parts.modules:
            for text in [*module.key_concepts, module.exercise]:
                problems += text_problems(text, hours, module.module_id)
    return problems


# -- documents written one module (or assessment) at a time ----------------------

def repeated_sentence(text: str) -> Optional[str]:
    """The first sentence (at least MIN_REPEATED_WORDS words) that the text says more than once, else None."""
    seen: set[str] = set()
    for sentence in _SENTENCE_END.split(text):
        key = units.normalized_text(sentence)
        if len(key.split()) < MIN_REPEATED_WORDS:
            continue
        if key in seen:
            return sentence.strip()
        seen.add(key)
    return None


NEAR_SENTENCE = 0.9          # two sentences of one text this alike are the same sentence said twice
NEAR_MIN_WORDS = 6
OPENING_REPEATS = 4         # this many sentences of one text starting with the same two words is a loop ("يجب أيضا ... يجب أيضا ...")


def _sentences(text: str) -> list[str]:
    return [units.normalized_text(part) for part in _SENTENCE_END.split(text) if units.normalized_text(part)]


def rambling(text: str) -> Optional[str]:
    """Why a text looks like a model looping (a phrase, sentence or sentence opening said again and again), else None."""
    sentences = _sentences(text)
    long = [s for s in sentences if len(s.split()) >= NEAR_MIN_WORDS]
    for index, sentence in enumerate(long):
        if any(SequenceMatcher(None, sentence, other).ratio() >= NEAR_SENTENCE for other in long[:index]):
            return " ".join(sentence.split()[:8])
    words = units.normalized_text(text).split()
    for size, times in ((4, 3), (6, 2)):
        counts: dict[tuple[str, ...], int] = {}
        for start in range(len(words) - size + 1):
            gram = tuple(words[start:start + size])
            counts[gram] = counts.get(gram, 0) + 1
            if counts[gram] >= times:
                return " ".join(gram)
    openings: dict[str, int] = {}
    for sentence in sentences:
        parts = sentence.split()
        if len(parts) >= 3:
            openings[" ".join(parts[:2])] = openings.get(" ".join(parts[:2]), 0) + 1
    repeated = [opening for opening, count in openings.items() if count >= OPENING_REPEATS]
    return repeated[0] if repeated else None


def _repetition_problems(text: str, where: str) -> list[dict[str, str]]:
    repeated = repeated_sentence(text)
    if repeated is not None:
        return [issue("repeated_text", f"A sentence is repeated within the same text: \"{repeated[:60]}\". Say each thing "
                      "once and keep the answer short.", where)]
    loop = rambling(text)
    if loop is not None:
        return [issue("repeated_text", f"The text keeps repeating itself (\"{loop[:60]}\"). Say each thing once, in "
                      "one or two short sentences.", where)]
    return []


# ---- incomplete text and English words in Arabic text -----------------------------------------------------------------

_ARABIC_LETTER = "\u0621-\u063a\u0641-\u064a"
_GLUED = re.compile(rf"\S*(?:[{_ARABIC_LETTER}][A-Za-z]|[A-Za-z][{_ARABIC_LETTER}])\S*")
_LOWER_WORD = re.compile(r"(?<![A-Za-z0-9_\-])([a-z]{3,})(?![A-Za-z0-9_\-])")
_ENDS_OPEN = re.compile(r"[,،;؛:]\s*$")
_TERMINAL = (".", "?", "؟", "!", "…")


def curriculum_words(content: dict[str, Any]) -> set[str]:
    """The lowercase English words that the approved curriculum itself uses (they may appear in Arabic text)."""
    found: set[str] = set()
    for text in _walk_texts(content):
        found.update(word.lower() for word in re.findall(r"[A-Za-z]{3,}", text))
    return found


def foreign_word_problems(text: str, allowed: set[str], where: str = "") -> list[dict[str, str]]:
    """
    English in Arabic text that is not a technical term: a Latin word glued to Arabic letters ("المodule"), or a plain
    lowercase English word that is neither in the approved curriculum nor in terms.TECHNICAL_TERMS. Capitalised,
    upper-case, hyphenated and digit-containing terms (Python, SQL, scikit-learn, F1) are never reported.
    """
    glued = _GLUED.findall(text)
    if glued:
        shown = glued[0].strip(".,;:!?()\"'،؛؟")
        return [issue("foreign_word", f"Latin letters are attached to Arabic letters (\"{shown[:30]}\"). Write the whole word in "
                      "Arabic; only technical terms such as Python or SQL may stay in English.", where)]
    words = [w for w in _LOWER_WORD.findall(text) if w not in allowed and w not in terms.TECHNICAL_TERMS]
    if words:
        return [issue("foreign_word", f"The Arabic text contains the English word \"{words[0]}\". Write it in Arabic; only "
                      "technical terms (and terms from the approved curriculum) may stay in English.", where)]
    return []


def incomplete_problems(text: str, where: str = "", stem: bool = False) -> list[dict[str, str]]:
    """
    Text that looks cut off: it ends on a comma, colon or semicolon, or on a word no sentence ends with (and, or, the,
    و, في, ...). stem=True (a question or a true/false statement) must also end with terminal punctuation. A word cut off
    in the middle cannot be recognised without a dictionary and is not reported.
    """
    stripped = text.rstrip()
    if not stripped:
        return []
    if _ENDS_OPEN.search(stripped):
        return [issue("incomplete_text", f"The text ends in the middle of a sentence (\"...{stripped[-25:]}\"). Finish it.", where)]
    if stem and not stripped.endswith(_TERMINAL):
        return [issue("incomplete_text", f"The question does not end with a question mark or a full stop "
                      f"(\"...{stripped[-25:]}\"); it may be cut off. Write it in full.", where)]
    last = units.normalized_text(stripped.rstrip("".join(_TERMINAL) + " )\"'")).split()
    if last and (last[-1] in terms.DANGLING_ENGLISH or last[-1] in terms.DANGLING_ARABIC):
        return [issue("incomplete_text", f"The text ends on \"{last[-1]}\", so it looks cut off (\"...{stripped[-25:]}\"). "
                      "Finish the sentence.", where)]
    return []


_SKIP = ("module_id", "assessment_id", "outcome_id", "correct_index", "answer")


def _walk_paths(value: Any, path: tuple = ()) -> Iterator[tuple[tuple, str]]:
    """Every text of a unit with its path, e.g. (("short_answer", 0, "model_answer"), "text")."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in _SKIP:
                yield from _walk_paths(item, (*path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_paths(item, (*path, index))


def _walk_texts(value: Any) -> Iterator[str]:
    for _, text in _walk_paths(value):
        yield text


def target_of(path: tuple) -> str:
    """"multiple_choice:1" for a text inside the second multiple-choice question; "" when it is not inside a list item."""
    return f"{path[0]}:{path[1]}" if len(path) >= 2 and isinstance(path[1], int) else ""


def _unit_id_key(template: DocumentTemplate) -> str:
    return "assessment_id" if template.unit_field == "rubrics" else "module_id"


def check_unit(unit: Any, template: DocumentTemplate, content: dict[str, Any], earlier: tuple[Any, ...] = ()) -> list[dict[str, str]]:
    """
    Checks one unit of the model's text (one module's questions, one rubric...) against the approved curriculum.
    earlier: the units already accepted, so a question repeated in another module is found.
    """
    unit_id = getattr(unit, _unit_id_key(template))
    hours = allowed_hours(content)
    allowed_words = curriculum_words(content) if content.get("language") == "ar" else None
    problems: list[dict[str, str]] = []
    for path, text in _walk_paths(unit.model_dump()):
        found: list[dict[str, str]] = []
        if not text.strip():
            found.append(issue("empty_text", "Every field must be filled in.", unit_id))
        found += text_problems(text, hours, unit_id, minutes=template.forbid_minutes)
        found += _repetition_problems(text, unit_id)
        found += incomplete_problems(text, unit_id, stem=_is_stem(path))
        if allowed_words is not None:
            found += foreign_word_problems(text, allowed_words, unit_id)
        if template.targeted_repair and target_of(path):   # (only the Question Bank repairs item by item)
            found = [{**p, "target": target_of(path)} for p in found]
        problems += found
    if isinstance(unit, schemas.ModuleQuestions):
        problems += _check_questions(unit, content, earlier)
    elif isinstance(unit, schemas.AssessmentRubric):
        problems += _check_rubric(unit, content)
    return problems


def check_unit_parts_frame(frame: Any, template: DocumentTemplate, content: dict[str, Any]) -> list[dict[str, str]]:
    """The text written once for the whole document: filled in, no accreditation claims or unsupported figures."""
    problems: list[dict[str, str]] = []
    hours = allowed_hours(content)
    allowed_words = curriculum_words(content) if content.get("language") == "ar" else None
    for text in _walk_texts(frame.model_dump()):
        if not text.strip():
            problems.append(issue("empty_text", "Every field must be filled in.", "overview"))
        problems += text_problems(text, hours, "overview", minutes=template.forbid_minutes)
        problems += _repetition_problems(text, "overview")
        problems += incomplete_problems(text, "overview")
        if allowed_words is not None:
            problems += foreign_word_problems(text, allowed_words, "overview")
    return problems


def check_unit_parts(parts: Any, template: DocumentTemplate, content: dict[str, Any]) -> list[dict[str, str]]:
    """The whole container: the ids cover every unit exactly once, the frame text and every unit pass their checks."""
    problems: list[dict[str, str]] = []
    hours = allowed_hours(content)
    for name, value in parts.model_dump().items():
        if name != template.unit_field:
            for text in _walk_texts(value):
                problems += text_problems(text, hours, name, minutes=template.forbid_minutes)
    key = _unit_id_key(template)
    unit_list = list(getattr(parts, template.unit_field))
    expected = units.unit_ids(template.unit_field, content)
    given = [getattr(u, key) for u in unit_list]
    if sorted(given) != sorted(expected) or len(set(given)) != len(given):
        problems.append(issue("unit_ids", f"Write exactly one entry for each of: {', '.join(expected)} "
                              f"(got {', '.join(given) or 'none'}).", template.unit_field))
        return problems
    accepted: list[Any] = []
    for unit in unit_list:
        problems += check_unit(unit, template, content, tuple(accepted))
        accepted.append(unit)
    return problems


def _similar(a: str, b: str) -> bool:
    na, nb = units.normalized_text(a), units.normalized_text(b)
    return bool(na) and bool(nb) and (na == nb or SequenceMatcher(None, na, nb).ratio() >= SIMILAR)


def _is_stem(path: tuple) -> bool:
    """Is this text a question or a true/false statement (which must end with terminal punctuation)?"""
    return bool(path) and path[-1] in ("question", "statement") and path[0] in ("multiple_choice", "true_false", "short_answer")


MAX_ANSWER_WORDS = 80        # a model answer of one or two sentences
MAX_EXPLANATION_WORDS = 60   # the explanation of a question


def _too_long(text: str, limit: int) -> bool:
    return len(text.split()) > limit


def _unit_stems(unit: Any) -> list[str]:
    return [*(q.question for q in unit.multiple_choice), *(q.statement for q in unit.true_false),
            *(q.question for q in unit.short_answer)]


def _stem_targets(unit: Any) -> list[str]:
    """The target of each stem of _unit_stems, in the same order."""
    return [*(f"multiple_choice:{i}" for i in range(len(unit.multiple_choice))),
            *(f"true_false:{i}" for i in range(len(unit.true_false))),
            *(f"short_answer:{i}" for i in range(len(unit.short_answer)))]


def _same_pairs(normalized: list[str]) -> list[tuple[int, int]]:
    """For each option that repeats an earlier one: (index of the earlier option, index of the repeat)."""
    first: dict[str, int] = {}
    pairs = []
    for index, value in enumerate(normalized):
        if value in first:
            pairs.append((first[value], index))
        else:
            first[value] = index
    return pairs


def _check_questions(unit: schemas.ModuleQuestions, content: dict[str, Any], earlier: tuple[Any, ...]) -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    module = units.find_module(content, unit.module_id)
    if module is None:
        return [issue("unit_ids", f"Unknown module {unit.module_id}.", unit.module_id)]
    wanted = units.module_terms(content, module)
    where = unit.module_id
    option_sets = [(f"module {other.module_id}", [{units.normalized_text(o) for o in q.options} for q in other.multiple_choice])
                   for other in earlier]
    seen_here: list[set[str]] = []
    for number, q in enumerate(unit.multiple_choice, start=1):
        options = [o.strip() for o in q.options]
        normalized = [units.normalized_text(o) for o in options]
        if len(options) != units.MC_OPTIONS or not all(normalized):
            problems.append(issue("question_options", f"Multiple-choice question {number} needs exactly "
                                  f"{units.MC_OPTIONS} non-empty options.", where, f"multiple_choice:{number - 1}"))
            continue
        for first, second in _same_pairs(normalized):
            problems.append(issue("question_options", f"Multiple-choice question {number} has options that are the same: "
                                  f"option {first + 1} and option {second + 1} (counting from 1) are both "
                                  f"\"{options[first][:60]}\". Replace option {second + 1} with a genuinely different, "
                                  "complete and meaningful statement; all four options must differ.", where, f"multiple_choice:{number - 1}"))
        for position, option in enumerate(options, start=1):
            if _NONE_OF_THE_ABOVE.search(normalize_for_matching(option)):
                problems.append(issue("question_options", f"Multiple-choice question {number}, option {position} "
                                      f"(\"{option[:60]}\") must not use \"all/none of the above\": exactly one option must "
                                      "be correct on its own. Replace it with a specific, different statement.", where, f"multiple_choice:{number - 1}"))
            if _OPTION_INDEX_END.search(option) or _OPTION_INDEX_START.search(option):
                problems.append(issue("question_options", f"Multiple-choice question {number}, option {position} "
                                      f"(\"{option[:60]}\") contains a number or letter that labels its position. Write the "
                                      "option text only, without numbering or an index.", where, f"multiple_choice:{number - 1}"))
        if not units.shares_a_term(f"{q.question} {options[q.correct_index]}", wanted):
            problems.append(issue("question_ungrounded", f"Multiple-choice question {number} is not about this module's "
                                  "topics.", where, f"multiple_choice:{number - 1}"))
        mine = set(normalized)
        reused = next((f"multiple-choice question {i} of this module" for i, other in enumerate(seen_here, start=1)
                       if len(mine & other) >= units.MC_OPTIONS - 1), None) or next(
            (f"a multiple-choice question of {label}" for label, sets in option_sets for other in sets
             if len(mine & other) >= units.MC_OPTIONS - 1), None)
        if reused:
            problems.append(issue("question_options", f"Multiple-choice question {number} uses (nearly) the same options as "
                                  f"{reused}, possibly in a different order. Write new options that fit this question.",
                                  where, f"multiple_choice:{number - 1}"))
        seen_here.append(mine)
        if _too_long(q.rationale, MAX_EXPLANATION_WORDS):
            problems.append(_explanation_too_long(f"Multiple-choice question {number}", q.rationale, where, f"multiple_choice:{number - 1}"))
    for q in unit.true_false:
        if _too_long(q.rationale, MAX_EXPLANATION_WORDS):
            problems.append(_explanation_too_long("The true/false question", q.rationale, where, "true_false:0"))
    for q in unit.short_answer:
        if _too_long(q.rationale, MAX_EXPLANATION_WORDS):
            problems.append(_explanation_too_long("The short-answer question", q.rationale, where, "short_answer:0"))
        if _too_long(q.model_answer, MAX_ANSWER_WORDS):
            problems.append(issue("answer_too_long", f"The model answer is too long ({len(q.model_answer.split())} words). Give "
                                  f"a model answer of one or two sentences, at most {MAX_ANSWER_WORDS} words.", where, "short_answer:0"))
    for q in unit.true_false:
        if not units.shares_a_term(q.statement, wanted):
            problems.append(issue("question_ungrounded", "The true/false statement is not about this module's topics.", where, "true_false:0"))
    for q in unit.short_answer:
        if not units.shares_a_term(q.question, wanted):
            problems.append(issue("question_ungrounded", "The short-answer question is not about this module's topics.", where, "short_answer:0"))
    stems, targets = _unit_stems(unit), _stem_targets(unit)
    seen = [stem for other in earlier for stem in _unit_stems(other)]
    for index, stem in enumerate(stems):
        if any(_similar(stem, other) for other in [*stems[:index], *seen]):
            problems.append(issue("question_duplicate", f"A question is repeated: \"{stem[:60]}\". Write different questions.",
                                  where, targets[index]))
    return problems


def _explanation_too_long(label: str, text: str, where: str, target: str) -> dict[str, str]:
    return issue("explanation_too_long", f"{label}: the explanation is too long ({len(text.split())} words). Keep it to "
                 f"at most {MAX_EXPLANATION_WORDS} words.", where, target)


def _check_rubric(unit: schemas.AssessmentRubric, content: dict[str, Any]) -> list[dict[str, str]]:
    spec = units.rubric_spec(content, unit.assessment_id)
    if spec is None or not spec["criteria"]:
        return [issue("unit_ids", f"Unknown assessment {unit.assessment_id}.", unit.assessment_id)]
    expected = [c["outcome_id"] for c in spec["criteria"]]
    given = [c.outcome_id for c in unit.criteria]
    if sorted(given) != sorted(expected) or len(set(given)) != len(given):
        return [issue("criteria_ids", f"Write exactly one criterion for each learning outcome: {', '.join(expected)} "
                      f"(got {', '.join(given) or 'none'}).", unit.assessment_id)]
    problems = []
    purpose = " ".join(str(spec.get(key) or "") for key in ("description",)).strip() or str(spec.get("title") or "")
    if _repeats(unit.instructions, purpose):
        problems.append(issue("instructions_repeat_purpose", "The assessor instructions repeat the assessment's purpose "
                              f"(\"{purpose[:60]}\"). Explain how to assess: what evidence to look for and how to apply "
                              "the four levels.", unit.assessment_id))
    for criterion in unit.criteria:
        problems += _distinct_levels([criterion.beginning, criterion.developing, criterion.proficient, criterion.advanced],
                                     f"{unit.assessment_id}/{criterion.outcome_id}")
    return problems


def _repeats(text: str, purpose: str) -> bool:
    """Is the text the purpose again (the same words, or nothing but words of the purpose)?"""
    words, other = set(units.normalized_text(text).split()), set(units.normalized_text(purpose).split())
    return bool(words) and bool(other) and (_similar(text, purpose) or words <= other)


MIN_DIFFERING_WORDS = 3


def _same_description(a: str, b: str) -> bool:
    """
    Level descriptions of one criterion legitimately share most of their wording ("Demonstrates limited ..." /
    "Demonstrates partial ..."), so text similarity would reject good rubrics. They are the same description only when
    they are identical or differ by fewer than MIN_DIFFERING_WORDS words.
    """
    words_a, words_b = set(units.normalized_text(a).split()), set(units.normalized_text(b).split())
    # (one description made of all the words of the other, plus a clause, is the same description said again: this is
    # how a level was "distinguished" from the one below it before)
    return not words_a or not words_b or words_a <= words_b or words_b <= words_a or len(words_a ^ words_b) < MIN_DIFFERING_WORDS


def _distinct_levels(texts: list[str], where: str) -> list[dict[str, str]]:
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if _same_description(texts[i], texts[j]):
                return [issue("descriptors_similar", "The four performance levels of a criterion must be described "
                              "differently from one another.", where)]
    return []


# -- checks that still apply to a stored document after an administrator's edit ------

def content_problems(template: DocumentTemplate, content: dict[str, Any]) -> list[dict[str, str]]:
    """
    Rules a stored document must keep whoever edited it: a multiple-choice question has four distinct options and its
    answer is exactly one of them, true/false answers are True or False, no question is repeated, and the four
    descriptions of a rubric criterion differ.
    """
    problems: list[dict[str, str]] = []
    language = content.get("language") if content.get("language") in ("ar", "en") else "en"
    words = (normalize_for_matching(templates_strings(language)["true"]), normalize_for_matching(templates_strings(language)["false"]))
    for section in content.get("sections") or []:
        key = section.get("key")
        options: Optional[list[str]] = None
        stems: list[str] = []
        descriptors: list[str] = []
        for index, block in enumerate(section.get("blocks") or []):
            role, where = block.get("role"), f"{key}[{index}]"
            if template.doc_type == "question_bank":
                if role == "q_options":
                    options = [i.strip() for i in block.get("items") or []]
                    normalized = [units.normalized_text(o) for o in options]
                    if len(options) != units.MC_OPTIONS or not all(normalized) or len(set(normalized)) != len(normalized):
                        problems.append(issue("question_options", f"A multiple-choice question needs exactly {units.MC_OPTIONS} "
                                              "different, non-empty options.", where))
                elif role == "q_answer":
                    answer = units.normalized_text(block.get("text"))
                    if options is None or [units.normalized_text(o) for o in options].count(answer) != 1:
                        problems.append(issue("question_answer", "The correct answer must be exactly one of the question's "
                                              "options. If you changed an option, change the answer to match.", where))
                    options = None
                elif role == "tf_answer":
                    if normalize_for_matching(str(block.get("text") or "")).strip() not in words:
                        problems.append(issue("question_answer", "A true/false answer must be True or False.", where))
                elif role == "q_stem":
                    text = str(block.get("text") or "")
                    if any(_similar(text, other) for other in stems):
                        problems.append(issue("question_duplicate", f"A question is repeated: \"{text[:60]}\".", where))
                    stems.append(text)
            elif template.doc_type == "assessment_rubric" and role == "descriptor":
                descriptors.append(str(block.get("text") or ""))
                if len(descriptors) == len(units.PERFORMANCE_LEVELS):
                    problems += _distinct_levels(descriptors, where)
                    descriptors = []
    return problems


def templates_strings(language: str) -> dict[str, Any]:
    from agents.document_generation.templates import STRINGS
    return STRINGS[language]


# -- the finished document --------------------------------------------------

def _locked(content: dict[str, Any]) -> list[tuple[str, str]]:
    return [(section.get("key"), json.dumps(block, sort_keys=True, ensure_ascii=False))
            for section in content.get("sections") or [] for block in section.get("blocks") or []
            if isinstance(block, dict) and block.get("source") in ("curriculum", "missing")]


def _skeleton(content: dict[str, Any]) -> list[tuple[str, str, str]]:
    # Generated text and administrator text are both "free": editing a generated block makes it an admin block.
    return [(section.get("key"), block.get("type"),
             "free" if block.get("source") in schemas.EDITABLE_SOURCES else block.get("source"))
            for section in content.get("sections") or [] for block in section.get("blocks") or []]


def validate_content(content: Any, record: dict[str, Any]) -> dict[str, Any]:
    """
    {"valid", "errors", "warnings"} for one document against its approved
    curriculum record (a CurriculumStore.get result for the approved version).
    """
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not isinstance(content, dict):
        return {"valid": False, "errors": [issue("structure", "The document content must be an object.")], "warnings": []}
    template = TEMPLATES.get(content.get("document_type"))
    if template is None:
        return {"valid": False, "errors": [issue("document_type", "Unknown document type.")], "warnings": []}
    language = content.get("language")
    curriculum = record["content"]
    if language not in ("ar", "en") or language != curriculum.get("language"):
        errors.append(issue("language", "The document language must be the approved curriculum's language."))
    if content.get("curriculum_id") != record["curriculum_id"] or content.get("curriculum_version") != record["version"]:
        errors.append(issue("traceability", "The document does not name the curriculum version it is built from."))
    cleaned_extra, extra_problems = clean_extra(content.get("extra"))
    if extra_problems:
        errors += [{**p, "where": p.get("where") or "extra"} for p in extra_problems]
    elif (content.get("extra") or {}) != cleaned_extra:
        errors.append(issue("extra_invalid", "Additional information must be trimmed single-line text.", "extra"))
    sections = content.get("sections")
    if not isinstance(sections, list) or [s.get("key") for s in sections if isinstance(s, dict)] != list(template.sections):
        errors.append(issue("sections", "The sections must match the document template."))
        return {"valid": False, "errors": errors, "warnings": warnings}

    hours = allowed_hours(curriculum)
    for section in sections:
        blocks = section.get("blocks")
        if not isinstance(blocks, list) or len(blocks) > MAX_BLOCKS_PER_SECTION:
            errors.append(issue("structure", "A section needs a list of blocks.", section.get("key", "")))
            continue
        for index, block in enumerate(blocks):
            where = f"{section['key']}[{index}]"
            if not isinstance(block, dict) or block.get("type") not in schemas.BLOCK_TYPES \
                    or block.get("source") not in schemas.BLOCK_SOURCES:
                errors.append(issue("structure", "A block has an unknown type or source.", where))
                continue
            texts = _block_texts(block)
            if texts is None or len(texts) > MAX_ITEMS or any(len(t) > MAX_TEXT for t in texts):
                errors.append(issue("structure", "A block is malformed or too long.", where))
                continue
            if block["source"] in schemas.EDITABLE_SOURCES:
                if block["type"] not in schemas.EDITABLE_TYPES and block["type"] != "heading":
                    errors.append(issue("structure", "Only paragraphs and bullet lists can be written.", where))
                if not any(t.strip() for t in texts):
                    errors.append(issue("empty", "This text is empty.", where))
                for text in texts:
                    errors += text_problems(text, hours, where, minutes=template.forbid_minutes)

    if not errors:
        extra = content.get("extra") if isinstance(content.get("extra"), dict) else {}
        parts = placeholder_parts(template, curriculum)
        reference = compose(template, record, language, parts, extra)
        if _skeleton(content) != _skeleton(reference):
            errors.append(issue("sections", "The blocks of each section must match the document template."))
        elif not _matches_a_composition(content, template, record, language, parts, extra, reference):
            errors.append(issue("curriculum_changed", "Text taken from the approved curriculum was changed. Revise "
                                "and re-approve the curriculum instead."))
        else:
            errors += content_problems(template, content)
    for gap in content.get("missing") or []:
        warnings.append(issue("missing_information", f"Not specified in the curriculum: {gap.get('field')}.",
                              str(gap.get("section", ""))))
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def _matches_a_composition(content: dict[str, Any], template: DocumentTemplate, record: dict[str, Any], language: str,
                           parts: Any, extra: dict[str, str], reference: dict[str, Any]) -> bool:
    """
    The curriculum-sourced text must be exactly what composing the document produces today, or what it produced before
    an earlier wording change: documents made then keep validating against their own wording and are never rewritten.
    """
    locked = _locked(content)
    if locked == _locked(reference):
        return True
    for localize_level, legacy in ((False, False), (True, True), (False, True)):
        if locked == _locked(compose(template, record, language, parts, extra, localize_level=localize_level, legacy=legacy)):
            return True
    return False


def _block_texts(block: dict[str, Any]) -> Optional[list[str]]:
    kind = block["type"]
    if kind in ("paragraph", "heading"):
        return [block["text"]] if isinstance(block.get("text"), str) else None
    if kind == "bullets":
        items = block.get("items")
        return list(items) if isinstance(items, list) and all(isinstance(i, str) for i in items) else None
    header, rows = block.get("header"), block.get("rows")
    if not isinstance(header, list) or not isinstance(rows, list) or not all(isinstance(r, list) for r in rows):
        return None
    return [str(c) for c in header] + [str(c) for r in rows for c in r]


__all__ = ["SECTION_HEADINGS", "allowed_hours", "check_parts", "clean_extra", "issue", "missing_curriculum_fields",
           "placeholder_parts", "text_problems", "validate_content"]
