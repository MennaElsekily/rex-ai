"""
The bilingual companion a freshly generated curriculum gets automatically
(never on a revision -- see curriculum_agent.py, which calls this only
when graph.py's state has no "base", i.e. a brand-new curriculum).

This is a TRANSLATION of the curriculum graph.py already generated and
saved, not a second research-and-generate run: every number, hour,
module count, id and outcome-to-module mapping is copied unchanged, and
only the human-readable text is given to the model to translate, a few
items per call. This is what keeps both languages "the same curriculum"
rather than two independently-written ones, as the brief requires.

Nothing here invents content. If the model's answer cannot be matched
one-to-one to the collected texts, the companion is not created and the
error is reported; the translated curriculum must also pass the same
validation as any other. The already-saved curriculum in the admin's own
language is never affected either way.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Callable, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.curriculum_research.llm import CurriculumModelError, structured
from agents.curriculum_research.safe_repair import apply_safe_fixes
from agents.curriculum_research.validation import _VAGUE_AR, first_verb, is_measurable, is_wrong_language
from agents.text_utils import normalize_for_matching

COMPANION_LANGUAGE = {"en": "ar", "ar": "en"}
LANGUAGE_NAMES = {"en": "English", "ar": "Arabic (Modern Standard Arabic)"}
# How many texts one translation call asks for at once -- see translate_curriculum().
MAX_ITEMS_PER_CALL = 5

_TRANSLATE = ChatPromptTemplate.from_messages([
    ("system", "You translate a training curriculum's text into {language_name}, for a professional training "
               "academy. Translate every item faithfully and completely; do not add, remove, summarize or "
               "invent anything, and never leave any word untranslated or copied from the source language "
               "(the only exception is a widely used technical term such as Python, SQL or Power BI, which "
               "stays as-is even inside Arabic text). Keep every number, id and measurement exactly as implied "
               "by the source (do not change hours, counts or quantities). If an item is a learning outcome or "
               "objective describing what a learner will be able to DO, it must open with a strong, specific "
               "action verb in {language_name} (for Arabic: a present-tense verb such as يطبق، يصمم، يبني، "
               "ينشئ، يقيم، يحلل، ينفذ، يطور، يكتب، يختبر — never a vague verb equivalent to 'understand', "
               "'know' or 'learn' such as يفهم or يعرف). Answer with exactly {count} translations, in the "
               "exact same order as the numbered items given."),
    ("human", "Translate each of these {count} items into {language_name}:\n{items_block}"),
])


class TranslationBatch(BaseModel):
    translations: list[str] = Field(description="One translation per source item, in the same order.")


def _mark(slots: list[tuple[Any, Any, int]], texts: list[str], container: Any, key: Any) -> None:
    value = container[key] if isinstance(container, list) else container.get(key)
    if isinstance(value, str) and value.strip():
        texts.append(value)
        slots.append((container, key, len(texts) - 1))


def _collect_texts(content: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[tuple[Any, Any, int]]]:
    """
    A deep copy of content, the human-readable texts it contains (title,
    overview, module/assessment/capstone text, ...), and where each one
    came from, so the model's translations can be written back in place.
    Numbers, ids, hours and the controlled "level" value are never
    collected (level stays its English literal even in Arabic content --
    the same convention request_parser.py and the PDF renderer use).
    """
    out = copy.deepcopy(content)
    texts: list[str] = []
    slots: list[tuple[Any, Any, int]] = []

    for key in ("title", "overview", "rationale", "target_audience", "capstone_omitted_reason"):
        _mark(slots, texts, out, key)

    for key in ("objectives", "prerequisites"):
        for i in range(len(out.get(key) or [])):
            _mark(slots, texts, out[key], i)

    for outcome in out.get("learning_outcomes") or []:
        _mark(slots, texts, outcome, "statement")

    for module in out.get("modules") or []:
        _mark(slots, texts, module, "title")
        _mark(slots, texts, module, "description")
        for i in range(len(module.get("topics") or [])):
            _mark(slots, texts, module["topics"], i)
        for activity in module.get("activities") or []:
            _mark(slots, texts, activity, "type")
            _mark(slots, texts, activity, "description")

    for assessment in out.get("assessments") or []:
        _mark(slots, texts, assessment, "type")
        _mark(slots, texts, assessment, "description")

    capstone = out.get("capstone")
    if isinstance(capstone, dict):
        _mark(slots, texts, capstone, "title")
        _mark(slots, texts, capstone, "description")
        for i in range(len(capstone.get("deliverables") or [])):
            _mark(slots, texts, capstone["deliverables"], i)

    return out, texts, slots


def translate_curriculum(llm: Any, content: dict[str, Any]) -> dict[str, Any]:
    """
    A new curriculum content dict: the same structure, numbers and ids as
    `content`, with every collected text translated into the companion
    language (English <-> Arabic) and "language" set to it. Raises
    CurriculumModelError if the model's answer cannot be matched back to
    the source texts even after one retry.
    """
    language = content.get("language") or "en"
    target = COMPANION_LANGUAGE.get(language, "en")
    out, texts, slots = _collect_texts(content)
    out["language"] = target
    if not texts:
        return out

    # One real small local model (qwen2.5:3b), asked to translate a whole
    # curriculum's text (title, objectives, every module, every
    # assessment...) in a SINGLE call, was observed to lose track partway
    # through: later items came back as bare list numbers ("11.", "12.",
    # ...) instead of a translation. Chunking keeps every call to a few
    # items, the same "one small focused task" principle graph.py already
    # uses for the English generation itself (a separate call each for
    # the outline, the module details and the extras).
    translated: list[str] = []
    for start in range(0, len(texts), MAX_ITEMS_PER_CALL):
        translated.extend(_translate_chunk(llm, texts[start:start + MAX_ITEMS_PER_CALL], target))

    # The same model also sometimes left an item in the source language
    # ("Formulas: IF, XLOOKUP, SUMIFS"), rendered an outcome's action verb as
    # a vague one ("فهم"), or kept the source's "The learner can ..." as
    # "يمكن للمتعلم ..." (no action verb first); validation would then make
    # the whole companion a draft. Such an item gets up to MAX_RETRIES
    # focused re-translations on its own, each shown its previous answer and
    # what is wrong with it. Validation still has the final word: whatever
    # remains wrong keeps the companion a draft.
    # Two DIFFERENT source texts coming back as the very same translation is a lost item too.
    outcomes = {index for _, key, index in slots if key == "statement"}
    if target == "ar":
        for index in outcomes:
            translated[index] = _without_modal_lead_in(translated[index])
    source_of: dict[str, str] = {}

    def same_source_translation(index: int) -> Optional[str]:
        # The model answers the same prompt the same way, so a retry often repeats a bad answer; but the
        # very same source text elsewhere in the curriculum (a module title repeated as its first topic)
        # may already have an acceptable translation -- the same text gets the same translation.
        return next((translated[other] for other in range(len(texts)) if other != index
                     and texts[other] == texts[index]
                     and _acceptable(translated[other], target, index in outcomes, texts[index])), None)

    for index, text in enumerate(translated):
        is_outcome = index in outcomes
        duplicate = source_of.setdefault(text, texts[index]) != texts[index]
        if _acceptable(text, target, is_outcome, texts[index]) and not duplicate:
            continue
        reused = same_source_translation(index)
        if reused is not None:
            translated[index] = reused
            continue
        hint = (f"This is a learning outcome: start with one measurable action verb in {LANGUAGE_NAMES[target]}. "
                if is_outcome else "")
        previous = text
        for _ in range(MAX_RETRIES):
            feedback = _problem_note(previous, target, is_outcome, texts[index]) if previous != text or not duplicate else ""
            try:
                retry = _translate_one(llm, texts[index], target,
                                       note=f"({hint}{feedback}Answer entirely in {LANGUAGE_NAMES[target]}.)")
            except CurriculumModelError:
                # A retry is a second chance, never a reason to lose the whole companion: keep what we have
                # (validation then decides). A real 3B model once answered one retry with 4 items.
                break
            if target == "ar" and is_outcome:
                retry = _without_modal_lead_in(retry)
            if _acceptable(retry, target, is_outcome, texts[index]) and source_of.get(retry, texts[index]) == texts[index]:
                translated[index] = retry
                source_of[retry] = texts[index]
                break
            previous = retry

    # An item fixed only by a later retry can still lend its translation to an identical source text.
    for index in range(len(translated)):
        if not _acceptable(translated[index], target, index in outcomes, texts[index]):
            reused = same_source_translation(index)
            if reused is not None:
                translated[index] = reused

    # Last resort for an outcome whose Arabic verb is still wrong after every retry ("Apply ..." came back as
    # "ينطبق ..." or a cut-off "ين ..."): the English source states the verb, so its standard Arabic verb is used.
    if target == "ar":
        for index in outcomes:
            if not is_measurable(translated[index], "ar"):
                translated[index] = _with_source_verb(translated[index], texts[index])

    for container, key, index in slots:
        container[key] = translated[index]
    return out


# The standard Arabic verb for each measurable English verb an outcome can start with (validation.py's lists).
_AR_VERB_FOR = {
    "apply": "يطبق", "analyse": "يحلل", "analyze": "يحلل", "assess": "يقيم", "build": "يبني", "calculate": "يحسب",
    "choose": "يختار", "classify": "يصنف", "clean": "ينظف", "compare": "يقارن", "compute": "يحسب", "conduct": "ينفذ",
    "configure": "يجهز", "construct": "يبني", "create": "ينشئ", "debug": "يصحح", "define": "يحدد", "demonstrate": "يعرض",
    "deploy": "ينشر", "describe": "يصف", "design": "يصمم", "develop": "يطور", "diagnose": "يحلل", "differentiate": "يميز",
    "distinguish": "يميز", "document": "يوثق", "estimate": "يقدر", "evaluate": "يقيم", "examine": "يحلل", "execute": "ينفذ",
    "explain": "يشرح", "extract": "يستخرج", "formulate": "يصوغ", "identify": "يحدد", "implement": "ينفذ", "integrate": "يدمج",
    "interpret": "يفسر", "justify": "يبرر", "manage": "يدير", "measure": "يقيس", "model": "ينمذج", "monitor": "يراقب",
    "optimise": "يحسن", "optimize": "يحسن", "organise": "ينظم", "organize": "ينظم", "plan": "يخطط", "prepare": "يعد",
    "present": "يعرض", "prioritise": "يرتب", "prioritize": "يرتب", "produce": "ينشئ", "program": "يبرمج", "recommend": "يقترح",
    "report": "يعرض", "secure": "يحمي", "select": "يختار", "solve": "يحل", "summarise": "يلخص", "summarize": "يلخص",
    "test": "يختبر", "train": "يدرب", "transform": "يحول", "troubleshoot": "يعالج", "use": "يستخدم", "validate": "يتحقق",
    "visualise": "يعرض", "visualize": "يعرض", "write": "يكتب",
}


def _with_source_verb(arabic: str, english: str) -> str:
    """
    The Arabic outcome with its mangled leading verb ("ينطبق", a cut-off "ين", or the English verb itself) replaced
    by the standard Arabic verb for the English source's own verb, when that makes it measurable. An outcome with no
    verb at all is left as it is (nothing is added to make it pass), so validation still reports it.
    """
    verb = _AR_VERB_FOR.get(first_verb(english, "en"))
    words = str(arabic).split()
    if not verb or not words:
        return arabic
    lead = words[0]
    if normalize_for_matching(lead) in _VAGUE_AR:
        return arabic          # a vague verb ("يفهم") is safe_repair.py's to fix, with a note the admin sees
    if re.fullmatch(r"[A-Za-z][\w-]*", lead) or (lead.startswith(("ي", "ت")) and len(lead) <= 6):
        rest = words[1:]
        if rest and rest[0].casefold().strip(".,:;") == first_verb(english, "en"):
            rest = rest[1:]                       # "ين Analyze ..." -> the English verb the model left behind goes too
        fixed = " ".join([verb, *rest])
    else:
        return arabic
    return fixed if is_measurable(fixed, "ar") and not is_wrong_language(fixed, "ar") else arabic


# Re-translations of one item that still fails the checks (see translate_curriculum).
MAX_RETRIES = 3

# "يمكن للمتعلم أن يطبق" / "يستطيع الطالب استخدام" / "سيتمكن المتدرب من تحليل": the source's "The learner can ..."
# kept as a lead-in before the action verb. Only removed when what follows is a measurable verb.
_MODAL_LEAD_IN = re.compile(
    r"^\s*(?:(?:ال)?(?:متعلم|متدرب|طالب|طلاب|مشارك)\w*\s+)?"
    r"(?:يمكنه|يمكنها|يمكنهم|يمكن|يستطيع|تستطيع|يتمكن|سيتمكن|سيكون\s+قادرا\s+على|سيكون\s+قادراً\s+على)\s+"
    r"(?:(?:لل|ال)\w+\s+)?(?:من\s+|أن\s+|ان\s+)?")


def _without_modal_lead_in(text: str) -> str:
    stripped = _MODAL_LEAD_IN.sub("", str(text or ""), count=1).strip()
    return stripped if stripped and stripped != str(text).strip() and is_measurable(stripped, "ar") else text


def _problem_note(previous: str, target: str, is_outcome: bool, source: str = "") -> str:
    """What was wrong with the previous answer, quoted, so the retry can correct exactly that."""
    if not str(previous).strip():
        return "Your previous answer was empty. "
    if _runaway(previous, source):
        return ("Your previous answer repeated the same words over and over and was far too long. Give ONE short "
                f"translation, about as long as the source ({len(source)} characters). ")
    if is_wrong_language(previous, target):
        keep = "names of software, functions or programming languages (for example Excel, SQL, XLOOKUP, Python)"
        return (f"Your previous translation was \"{_quote(previous)}\": it is still mostly not in {LANGUAGE_NAMES[target]}. "
                f"Translate every ordinary word; only {keep} stay in Latin letters. ")
    if is_outcome and not is_measurable(previous, target):
        verbs = "يطبق، يحلل، يصمم، ينشئ، يستخدم، يقيم" if target == "ar" else "Apply, Analyze, Design, Create, Use"
        return (f"Your previous translation was \"{_quote(previous)}\": it does not start with a measurable action verb. "
                f"Start directly with a verb such as {verbs}. ")
    return f"Your previous translation was \"{_quote(previous)}\". "


def _acceptable(text: str, target: str, is_outcome: bool, source: str = "") -> bool:
    return (bool(str(text).strip()) and not is_wrong_language(text, target)
            and (not is_outcome or is_measurable(text, target)) and not _runaway(text, source))


def _runaway(text: str, source: str) -> bool:
    """A "translation" many times longer than its source: the real model once looped, repeating words thousands of times."""
    return bool(source) and len(str(text)) > max(4 * len(source), len(source) + 300)


def _quote(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


# The items are numbered in the prompt ("2. Plan a campaign"); the real model sometimes echoes that number
# back at the start of its translation ("2. يخطط ..."), which then reads as the outcome's first "verb".
_ITEM_NUMBER = re.compile(r"^\s*\d{1,2}\s*[.)\-:]\s*")


def _ask(llm: Any, chunk: list[str], target: str, note: str = "") -> list[str]:
    items_block = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(chunk))
    if note:
        items_block += f"\n\n{note}"
    messages = _TRANSLATE.format_messages(
        language_name=LANGUAGE_NAMES[target], count=len(chunk), items_block=items_block)
    answers = [str(a) for a in structured(llm, TranslationBatch, messages, "bilingual translation").translations]
    if not any(_OUTCOME_LABEL.match(text) for text in chunk):
        # The real model also prefixed outcome labels the source does not have ("LO5: يطبق ..."), every time.
        answers = [_OUTCOME_LABEL.sub("", answer, count=1) for answer in answers]
    if any(_ITEM_NUMBER.match(text) for text in chunk):
        return answers   # a source that itself starts with a number keeps it
    return [_ITEM_NUMBER.sub("", answer, count=1) for answer in answers]


_OUTCOME_LABEL = re.compile(r"^\s*(?:LO|L\.O\.)\s*\d{1,2}\s*[:.)\-–]\s*", re.IGNORECASE)


def _translate_chunk(llm: Any, chunk: list[str], target: str) -> list[str]:
    """
    The chunk's translations, in order. If the model miscounts (a real 3B
    model did, even for 5 items), every item is translated on its own
    instead, so a miscount can never shift a translation into the wrong place.
    """
    answer = _ask(llm, chunk, target)
    if len(answer) == len(chunk):
        return answer
    return [_translate_one(llm, text, target) for text in chunk]


def _translate_one(llm: Any, text: str, target: str, note: str = "") -> str:
    for attempt in range(2):
        answer = _ask(llm, [text], target, note if attempt == 0 else
                      (note + " " if note else "") + "(Answer with exactly 1 translation.)")
        if len(answer) == 1:
            return answer[0]
    raise CurriculumModelError(
        f"The translation returned {len(answer)} item(s) for 1 source text; the bilingual companion was not created.")


def generate_companion(
    tools: Any, llm: Any, content: dict[str, Any], curriculum_id: str,
    on_error: Optional[Callable[[str], None]] = None,
) -> Optional[dict[str, Any]]:
    """
    Translates, saves and renders the PDF of the bilingual companion for
    a freshly saved curriculum; links the two together. Returns the save
    result ({"curriculum_id", "version", "review_status", "validation",
    ...}), or None if no faithful translation could be produced -- which
    never undoes or blocks the admin's own, already-saved curriculum.
    on_error, if given, is told why (for the response shown to the admin).

    The companion is saved under exactly the same rule as every other
    curriculum (save_curriculum_draft validates it itself): pending review
    if it passes validation, otherwise a draft that shows its validation
    problems and cannot be approved until it is fixed. It used to be
    discarded outright instead -- so whenever the source itself was a
    draft (a real 3B model's English draft often is), its faithful
    translation failed the same rule and the admin never saw a second
    language at all.
    """
    try:
        translated = translate_curriculum(llm, content)
    except CurriculumModelError as exc:
        if on_error:
            on_error(str(exc))
        return None
    # The same safe, explained fixes the original gets (a translated vague verb, e.g. "فهم").
    fixes = apply_safe_fixes(translated)
    if fixes:
        translated.setdefault("assumptions", []).append("إصلاحات تلقائية آمنة: " + " ".join(fixes)
                                                         if translated["language"] == "ar" else
                                                         "Safe automatic fixes: " + " ".join(fixes))

    saved = tools.save_curriculum_draft(
        curriculum=translated, change_summary="Bilingual companion, generated with the original.",
        origin="request")
    if saved.get("status") != "ok":
        if on_error:
            on_error(saved.get("message") or "The bilingual companion could not be saved.")
        return None

    tools.link_curriculum_translation(curriculum_id, saved["curriculum_id"])
    # No PDF yet: like the original, the companion's PDF is generated when an admin approves it.
    return saved
