"""
Checks for Agent 5: that a campaign says nothing the approved course does not support.

Used twice, with the same rules: on each channel while the model writes it (generator.py, one repair, never saved if
it still fails) and on the whole stored campaign whenever it is saved or edited (marketing_tools.py). A campaign that
fails is never "pending review": the agent saves nothing, an administrator's edit is kept as a draft that cannot be
approved.

What is rejected: numbers and hour figures the curriculum does not state; prices, fees, discounts, deadlines and
"limited" offers; guarantees, job or salary promises and superlatives; accreditation and certificate claims; links,
e-mail addresses and phone numbers; month names (invented dates); text in another writing system, English words glued
to Arabic or plain English words inside Arabic text; repeated sentences; text outside a channel's length limits; and
variations that are not different from one another. A term the curriculum or the administrator's own words contain
is allowed (a course that really does end in a certificate may say so). The checks are heuristics: they block the
common ways of inventing facts, not every possible one, which is why an administrator approves every campaign.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Optional

from agents.curriculum_research.validation import _ACCREDITATION_AR, _ACCREDITATION_EN
from agents.marketing import schemas
from agents.marketing.facts import Allowed, TECHNICAL_TERMS, allowed_for, numbers_in
from agents.text_utils import normalize_for_matching

MAX_AUDIENCE = 300
MAX_TEXT_FIELD = 1500
SIMILAR = 0.8   # two variations of a channel this alike (main text) are not different enough

_HOURS = re.compile(r"(\d+(?:[.,]\d+)?)\s*[- ]?\s*(?:hours?|hrs?|h\b|ساع[ةه]|ساعات)", re.IGNORECASE)
_FOREIGN_SCRIPT = re.compile(r"[Ѐ-ӿ฀-๿ऀ-ॿ぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]+")
_ARABIC_LETTER = "ء-غف-ي"
_GLUED = re.compile(rf"\S*(?:[{_ARABIC_LETTER}][A-Za-z]|[A-Za-z][{_ARABIC_LETTER}])\S*")
_PREFIXED_TERM = re.compile(rf"^[{_ARABIC_LETTER}]{{1,3}}(?=[A-Za-z0-9_-]*[A-Z0-9])[A-Za-z][A-Za-z0-9_-]*$")
_LOWER_WORD = re.compile(r"(?<![A-Za-z0-9_\-#])([a-z]{3,})(?![A-Za-z0-9_\-])")
_PUNCTUATION = ".,;:!?()\"'،؛؟"
_SENTENCE_END = re.compile(r"[.!?؟;؛\n]+")
_ENUMERATOR = re.compile(r"^\s*(?:step\s*|الخطوه\s*)?\d+\s*[.):\-–]\s*", re.IGNORECASE)

_AR = "\u0621-\u064a"


def _ar(alternatives: str) -> str:
    """
    Arabic terms as whole words (with the usual one-letter or "ال" prefixes), so "مارس" (March) is not found inside
    "ممارسة" (practice). The terms are written in normalize_for_matching() spelling (folded alef, ya and ta marbuta).
    """
    return rf"(?<![{_AR}])(?:وال|بال|لل|ال|و|ب|ل|ف|ك)?(?:{alternatives})(?![{_AR}])"


# (code, what is wrong, pattern on normalize_for_matching() text).
_RISKY: tuple[tuple[str, str, re.Pattern], ...] = tuple((code, why, re.compile(pattern, re.IGNORECASE)) for code, why, pattern in (
    ("price_claim", "prices, fees and costs are not in the approved course information",
     r"[$\u20ac\u00a3]|\b(?:usd|eur|egp|sar|aed|dollars?|euros?|pounds?|riyals?|price[sd]?|fees?|cost|costs|tuition|free|cheap|affordable)\b|"
     + _ar(r"دولار|يورو|جنيه|ريال|درهم|سعر|اسعار|رسوم|تكلفه|مجان[" + _AR + r"]{0,2}|رخيص[" + _AR + r"]{0,2}")),
    ("offer_claim", "discounts, limited offers and urgency are not in the approved course information",
     r"\bdiscount\w*|\b\d+\s*%\s*off\b|\bpercent off\b|\bpromo\b|\bpromo[- ]?codes?\b|\bpromotional (?:offers?|prices?|rates?|discounts?|deals?|codes?|pricing)\b|\bpromotion(?:al)? (?:offers?|prices?|discounts?|deals?)\b|\bcoupon\w*|\bearly[- ]bird\b|"
     r"\blimited (?:time|seats|spots|places|offer)\b|\bseats are limited\b|\blast chance\b|\bhurry\b|\bact now\b|\bdon'?t miss\b|"
     + _ar(r"خصم|خصومات|عرض محدود|مقاعد محدوده|اماكن محدوده|اخر فرصه|لفتره محدوده|كوبون|لا تفوت|سارع")),
    ("guarantee_claim", "guarantees and job, salary or promotion promises cannot be supported",
     r"\bguarantee\w*|\bget hired\b|\bland (?:a|your) (?:dream )?job\b|\bjob (?:offer|placement|guarantee)\w*|\bdream job\b|"
     r"\bsalary\b|\bsalaries\b|\bpromotion\b|\bdouble your\b|\bcareer boost\b|"
     + _ar(r"ضمان|مضمون[" + _AR + r"]{0,2}|وظيفه احلامك|راتب|رواتب|ترقيه|فرصه عمل مضمونه")),
    ("superlative_claim", "superlatives such as 'best' or '#1' cannot be supported",
     r"\bbest\b(?!\s+practices?)|#1\b|\bnumber one\b|\btop[- ]rated\b|\bworld[- ]class\b|\bunmatched\b|\bunrivall?ed\b|"
     r"\bmost popular\b|\bonly course\b|\bleading (?:academy|course|provider|institution|program\w*)\b|"
     + _ar(r"افضل(?!\s+الممارسات)|رقم 1|الاول عالميا|لا مثيل|رائد(?:ه)? (?:اكاديميه|دوره|مزود)")),
    ("certificate_claim", "certificates, diplomas and accreditation are not in the approved course information",
     r"\bcertificat\w*|\bcertified\b|\bdiploma\b|\bdegree\b|" + _ar(r"شهاده|شهادات|دبلوم|معتمده?|اعتماد")),
    ("contact_detail", "links, e-mail addresses and phone numbers must not be invented",
     r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.]+\b|\b[\w-]+\.(?:com|org|net|edu|io|co|eg|sa)\b|(?<!\d)\+?\d[\d\s().-]{7,}\d"),
    ("date_claim", "dates and deadlines are not in the approved course information",
     r"\b(?:january|february|march|april|june|july|august|september|october|november|december)\b|"
     r"\bmay\s+\d{1,2}\b|\b(?:in|during|by|until|from|since|next|this|every)\s+may\b|"
     r"\b(?:enrol|enroll|register|apply) by\b|\bdeadline\b|\bstarts? on\b|\bstarting (?:on|from)\b|"
     + _ar(r"يناير|فبراير|مارس|ابريل|مايو|يونيو|يوليو|اغسطس|سبتمبر|اكتوبر|نوفمبر|ديسمبر|الموعد النهائي|يبدا في|يبدا من|التسجيل حتي")),
    ("prerequisite_claim", "a claim about required experience must match the course's stated prerequisites",
     r"\bno (?:prior |previous )?(?:experience|knowledge|background)\b|\banyone can\b|\bno prerequisites\b|"
     + _ar(r"بدون خبره|لا تحتاج (?:الي )?خبره|لا يشترط|دون الحاجه الي خبره")),
))


def issue(code: str, message: str, where: str = "") -> dict[str, str]:
    return {"code": code, "where": where, "message": message}


def _words(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", normalize_for_matching(text)))


# -- one text ---------------------------------------------------------------------------------------------------------

def text_problems(text: str, allowed: Allowed, where: str = "", *, hashtag: bool = False,
                  enumerated: bool = False) -> list[dict[str, str]]:
    """Everything wrong with one piece of campaign text (see the module docstring)."""
    problems: list[dict[str, str]] = []
    plain = _ENUMERATOR.sub("", text) if enumerated else text
    normalized = normalize_for_matching(plain)

    if _FOREIGN_SCRIPT.search(text):
        problems.append(issue("foreign_script", f"The text contains characters from another writing system "
                              f"(\"{' '.join(_FOREIGN_SCRIPT.findall(text))[:30]}\"). Write only in the course's language.", where))
    for code, why, pattern in _RISKY:
        for match in pattern.finditer(normalized):
            term = match.group(0).strip()
            if term and not allowed.mentions(term):
                problems.append(issue(code, f"\"{term[:40]}\": {why}. Remove it; use only facts from the approved course.", where))
                break
    if _ACCREDITATION_EN.search(plain) or _ACCREDITATION_AR.search(normalized):
        problems.append(issue("accreditation_claim", "The text must not claim accreditation, certification or official approval.", where))

    for hours in _HOURS.findall(normalized):
        if allowed.hours and float(hours.replace(",", ".")) not in allowed.hours:
            problems.append(issue("hours_mismatch", f"The text mentions {hours} hours, which the approved course does not state.", where))
    hour_figures = {float(h.replace(",", ".")) for h in _HOURS.findall(normalized)}
    for number in sorted(numbers_in(plain) - allowed.numbers - hour_figures):
        problems.append(issue("number_claim", f"The number {number:g} is not in the approved course information. "
                              "Use only figures the course states (hours, modules, outcomes).", where))

    if allowed.language == "ar" and not hashtag:
        glued = [g for g in _GLUED.findall(text) if not _PREFIXED_TERM.match(g.strip(_PUNCTUATION))]
        if glued:
            shown = glued[0].strip(_PUNCTUATION)[:30]
            problems.append(issue("foreign_word", f"Latin letters are attached to Arabic letters (\"{shown}\"). "
                                  "Write the whole word in Arabic.", where))
        else:
            words = [w for w in _LOWER_WORD.findall(text) if w not in allowed.latin_words and w not in TECHNICAL_TERMS]
            if words:
                problems.append(issue("foreign_word", f"The Arabic text contains the English word \"{words[0]}\". Write it in Arabic; "
                                      "only technical terms and terms from the approved course may stay in English.", where))

    seen: set[str] = set()
    for sentence in _SENTENCE_END.split(text):
        key = _words(sentence)
        if len(key.split()) >= 3:
            if key in seen:
                problems.append(issue("repeated_text", f"A sentence is repeated: \"{sentence.strip()[:50]}\". Say each thing once.", where))
                break
            seen.add(key)
    if re.search(r"[,،;؛:]\s*$", text.rstrip()):
        problems.append(issue("incomplete_text", "The text ends in the middle of a sentence. Finish it.", where))
    return problems


# -- one channel ---------------------------------------------------------------------------------------------------------

def _quote(channel: str, value: str) -> str:
    """For a call to action, the problem message quotes what was written, so a repair can see which text to rewrite."""
    return f' You wrote: "{value.strip()}". Rewrite it as a shorter action phrase.' if channel == "ctas" else ""


def _length_problems(channel: str, variation: dict[str, Any], where: str) -> list[dict[str, str]]:
    problems = []
    for name, (low, high) in schemas.LIMITS[channel].items():
        value = variation.get(name)
        if not isinstance(value, str) or not value.strip():
            problems.append(issue("empty_text", f"The {name.replace('_', ' ')} must be filled in.", f"{where}.{name}"))
        elif not low <= len(value.strip()) <= high:
            problems.append(issue("length", f"The {name.replace('_', ' ')} has {len(value.strip())} characters; "
                                  f"it must have {low} to {high}.{_quote(channel, value)}", f"{where}.{name}"))
    if channel == "ctas" and isinstance(variation.get("text"), str) and len(variation["text"].split()) > schemas.MAX_CTA_WORDS:
        problems.append(issue("word_count", f"The call to action has {len(variation['text'].split())} words; it must have "
                              f"at most {schemas.MAX_CTA_WORDS}.{_quote(channel, variation['text'])}", f"{where}.text"))
    if channel == "social_captions":
        tags = variation.get("hashtags", [])
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            problems.append(issue("hashtags", "The hashtags must be a list of short words.", f"{where}.hashtags"))
        else:
            if len(tags) > schemas.HASHTAG_LIMIT:
                problems.append(issue("hashtags", f"Use at most {schemas.HASHTAG_LIMIT} hashtags.", f"{where}.hashtags"))
            for tag in tags:
                if not tag.strip() or len(tag.strip()) > 30 or " " in tag.strip().lstrip("#"):
                    problems.append(issue("hashtags", f"\"{tag[:30]}\" is not a usable hashtag (one word, at most 30 characters).",
                                          f"{where}.hashtags"))
    return problems


def check_channel(channel: str, variations: Any, allowed: Allowed, count: int) -> list[dict[str, str]]:
    """Problems of one channel's variations (a list of dicts): count, lengths, claims, and that they really differ."""
    where = channel
    label = schemas.CHANNEL_LABELS[channel]["en"]
    if not isinstance(variations, list) or len(variations) != count:
        return [issue("variation_count", f"{label}: write exactly {count} variation(s) "
                      f"(got {len(variations) if isinstance(variations, list) else 'none'}).", where)]
    problems: list[dict[str, str]] = []
    for index, variation in enumerate(variations):
        spot = f"{where}[{index + 1}]"
        if not isinstance(variation, dict):
            problems.append(issue("structure", f"{label}: variation {index + 1} is malformed.", spot))
            continue
        problems += [{**p, "message": f"{label}, variation {index + 1}: {p['message']}"} for p in _length_problems(channel, variation, spot)]
        for name in schemas.LIMITS[channel]:
            value = variation.get(name)
            if isinstance(value, str) and value.strip():
                problems += [{**p, "message": f"{label}, variation {index + 1}, {name.replace('_', ' ')}: {p['message']}"}
                             for p in text_problems(value, allowed, f"{spot}.{name}")]
        if channel == "social_captions" and isinstance(variation.get("hashtags"), list):
            for tag in variation["hashtags"]:
                if isinstance(tag, str):
                    problems += [{**p, "message": f"{label}, variation {index + 1}, hashtag: {p['message']}"}
                                 for p in text_problems(tag, allowed, f"{spot}.hashtags", hashtag=True)]
    main = schemas.MAIN_FIELD[channel]
    texts = [_words(v.get(main, "")) if isinstance(v, dict) else "" for v in variations]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if texts[i] and texts[j] and (texts[i] == texts[j] or SequenceMatcher(None, texts[i], texts[j]).ratio() >= SIMILAR):
                problems.append(issue("variations_similar", f"{label}: variations {i + 1} and {j + 1} are too alike. "
                                      "Write them from different angles, with different wording.", f"{where}[{j + 1}]"))
    return problems


def check_strategy(strategy: Any, allowed: Allowed) -> list[dict[str, str]]:
    if not isinstance(strategy, dict):
        return [issue("structure", "The strategy is missing.", "strategy")]
    problems: list[dict[str, str]] = []
    for key in ("objective", "main_message", "call_to_action"):
        value = strategy.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(issue("empty_text", f"Strategy: the {key.replace('_', ' ')} must be filled in.", f"strategy.{key}"))
        elif len(value) > MAX_TEXT_FIELD:
            problems.append(issue("length", f"Strategy: the {key.replace('_', ' ')} is too long.", f"strategy.{key}"))
        else:
            problems += [{**p, "message": f"Strategy, {key.replace('_', ' ')}: {p['message']}"}
                         for p in text_problems(value, allowed, f"strategy.{key}")]
    steps = strategy.get("content_sequence")
    if not isinstance(steps, list) or not 3 <= len(steps) <= 6 or not all(isinstance(s, str) and s.strip() for s in steps):
        problems.append(issue("structure", "Strategy: the content sequence needs three to six steps.", "strategy.content_sequence"))
    else:
        for index, step in enumerate(steps):
            problems += [{**p, "message": f"Strategy, step {index + 1}: {p['message']}"}
                         for p in text_problems(step, allowed, f"strategy.content_sequence[{index}]", enumerated=True)]
    channels = strategy.get("recommended_channels")
    if not isinstance(channels, list) or not channels or not all(c in schemas.CHANNELS for c in channels):
        problems.append(issue("structure", "Strategy: recommended channels must be from: " + ", ".join(schemas.CHANNELS) + ".",
                              "strategy.recommended_channels"))
    if not isinstance(strategy.get("target_audience"), str) or not strategy["target_audience"].strip():
        problems.append(issue("structure", "Strategy: the target audience is missing.", "strategy.target_audience"))
    return problems


# -- the whole stored campaign ---------------------------------------------------------------------------------------------

def supplied_texts(content: dict[str, Any]) -> list[str]:
    """What the administrator typed for this campaign: its goal and notes (and a confirmed audience of their own)."""
    audience = content.get("audience") if isinstance(content.get("audience"), dict) else {}
    return [str(v) for v in (content.get("goal"), content.get("notes"), audience.get("text") if audience.get("source") == "admin" else None) if v]


def validate_content(content: Any, record: dict[str, Any]) -> dict[str, Any]:
    """
    {"valid", "errors", "warnings"} for one stored campaign against its approved curriculum record (a store ``get``
    result for the approved version). Valid means pending review; it never means approved.
    """
    errors: list[dict[str, str]] = []
    if not isinstance(content, dict):
        return {"valid": False, "errors": [issue("structure", "The campaign content must be an object.")], "warnings": []}
    curriculum = record["content"]
    if content.get("campaign_type") != "marketing_campaign":
        errors.append(issue("structure", "This is not a marketing campaign."))
    if content.get("language") not in ("ar", "en") or content.get("language") != curriculum.get("language"):
        errors.append(issue("language", "The campaign language must be the approved course's language."))
    if content.get("curriculum_id") != record["curriculum_id"] or content.get("curriculum_version") != record["version"]:
        errors.append(issue("traceability", "The campaign does not name the approved course version it is built from."))
    audience = content.get("audience")
    if not isinstance(audience, dict) or not isinstance(audience.get("text"), str) or not audience["text"].strip() \
            or len(audience["text"]) > MAX_AUDIENCE or audience.get("source") not in ("curriculum", "admin"):
        errors.append(issue("audience", "The target audience must be stated (from the course or confirmed by the administrator).", "audience"))
    if content.get("tone") not in schemas.TONES:
        errors.append(issue("structure", "The tone must be one of: " + ", ".join(schemas.TONES) + ".", "tone"))
    count = content.get("variations")
    if not isinstance(count, int) or not schemas.MIN_VARIATIONS <= count <= schemas.MAX_VARIATIONS:
        errors.append(issue("structure", f"The number of variations must be {schemas.MIN_VARIATIONS} to {schemas.MAX_VARIATIONS}.", "variations"))
        count = None
    channels = content.get("channels")
    if not isinstance(channels, dict) or not channels or any(c not in schemas.CHANNELS for c in channels):
        errors.append(issue("structure", "The campaign needs at least one known channel.", "channels"))
        channels = {}
    allowed = allowed_for(curriculum, supplied_texts(content))
    errors += check_strategy(content.get("strategy"), allowed)
    for channel, body in channels.items():
        variations = body.get("variations") if isinstance(body, dict) else None
        if count is not None:
            errors += check_channel(channel, variations, allowed, count)
    return {"valid": not errors, "errors": errors, "warnings": []}


def problem_text(problems: list[dict[str, str]]) -> str:
    return " ".join(p["message"] for p in problems)
