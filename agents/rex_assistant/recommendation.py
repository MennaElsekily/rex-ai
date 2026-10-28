"""
REX Assistant's course-recommendation rules, used by REX's own generate
step (generate_answer() in graph.py) when it writes a recommendation:

- recommendation_task(): the learner profile labels and the per-course
  answer layout, added to the prompt;
- check_recommendation(): deterministic checks applied to the model's
  answer:
    * a course listed twice keeps its first block;
    * a course whose name is not in the retrieved evidence is dropped, and
      so is a block saying the course does not fit or stating no facts;
    * a bracketed course code that does not belong to the course is removed;
    * a course above the learner's level is marked as a step up;
    * a fee above the learner's budget is marked as such;
    * preferences the evidence never mentions (delivery mode, fees) are
      listed as "not stated in the catalogue", so a claimed match the
      evidence does not support is always corrected.
"""

from __future__ import annotations

import re
from typing import Any, Optional

PROFILE_FIELD_LABELS = (
    ("background", "Background"),
    ("career_goal", "Career goal"),
    ("skill_level", "Skill level"),
    ("training_preference", "Training preference (online/offline)"),
    ("budget", "Budget"),
    ("available_time", "Available time"),
    ("interests", "Interests"),
)
_OPTIONAL_FIELDS = ("interests",)

COURSE_RECOMMENDATION_TASK = """
RECOMMENDATION TASK:
The user wants course recommendations for themselves. Speak to them
directly as "you" (never "the learner").
Recommend at most 3 courses or programmes that appear in the document
context, using their names exactly as written there, best fit first.
For each one, use exactly this layout:

**<course name>** [n]
- Why it fits you: link your goal and background to what the context
  says the course teaches or who it is for.
- Level: the level as stated. If it is above your skill level, say it
  may be a step up -- never call it a match.
- Delivery: copy the delivery format as stated. If the context does not
  say online or offline, write "not stated in the catalogue". Never say it
  matches a preference the context does not confirm.
- Fees: as stated, or "not stated in the catalogue". Compare them with
  the budget only when both are known.
- Entry requirements: as stated, and say which ones you may not meet yet.

Then one sentence on which course to start with and why.
Recommend only courses present in the context. If none fits, say so.
""".strip()

# Preferences the evidence must mention before a match can be claimed.
_DELIVERY_WORDS = re.compile(
    r"\b(?:online|offline|in[- ]person|on[- ]?site|remote|virtual|e-?learning|"
    r"self[- ]paced|classroom|hybrid|blended)\b|أونلاين|اونلاين|عن بعد|حضوري|أوفلاين|اوفلاين",
    re.IGNORECASE,
)
_FEE_WORDS = re.compile(
    r"\b(?:fees?|price[sd]?|cost[s]?|tuition|egp|usd|sar)\b|\$|جنيه|ريال|دولار|رسوم|سعر|تكلفة",
    re.IGNORECASE,
)

_UNCONFIRMED = {
    "header": {"en": "**Not stated in the catalogue**", "ar": "**غير مذكور في الكتالوج**"},
    "delivery": {
        "en": "- Whether these courses are online or offline, so your preference "
              "for {value} training could not be checked.",
        "ar": "- هل هذه الدورات أونلاين أم حضورية، لذلك لم أتمكن من التحقق من "
              "تفضيلك ({value}).",
    },
    "fees": {
        "en": "- Course fees, so I could not check them against your budget ({value}).",
        "ar": "- رسوم الدورات، لذلك لم أتمكن من مقارنتها بميزانيتك ({value}).",
    },
    "footer": {
        "en": "Please confirm these with the academy.",
        "ar": "يرجى التأكد منها مع الأكاديمية.",
    },
}

_PREFERENCE_LABELS = {
    "online": {"en": "online", "ar": "أونلاين"},
    "offline": {"en": "offline", "ar": "حضوري"},
    "hybrid": {"en": "hybrid", "ar": "مختلط"},
}

# "**Name** [2]", "**Name** [MOB-101]", "**Name [MOB-101]** [2]", "1. **Name**:"
_COURSE_HEADER = re.compile(
    r"^\s*(?:\d+\.\s*)?\*\*(?P<name>[^*]+?)\*\*\s*(?P<tags>(?:\[[^\]\n]{1,20}\]\s*)*):?\s*$"
)
_TAG = re.compile(r"\[([^\]\n]{1,20})\]")
_CODE_IN_PARENS = re.compile(r"\(([A-Za-z]{2,5}-?\d{2,4})\)")
_COURSE_CODE = re.compile(r"^[A-Za-z]{2,5}-?\d{2,4}$")
_START_WITH_LINE = re.compile(
    r"^\s*-\s*(?:\*\*)?(?:which course to start with|start with)\b", re.IGNORECASE
)
_FEES_LINE = re.compile(r"^\s*-\s*(?:\*\*)?(?:Fees?|الرسوم|السعر)(?:\*\*)?\s*:", re.IGNORECASE)
_NOT_A_FIT = re.compile(
    r"not (?:directly )?relevant|not (?:directly )?related|not suitable|not a (?:good )?fit|"
    r"not applicable|does not fit|غير مناسب|لا يناسب|غير مرتبط",
    re.IGNORECASE,
)
_NOT_STATED = re.compile(r"not stated|غير مذكور", re.IGNORECASE)
_FACT_LINE = re.compile(
    r"^\s*-\s*(?:\*\*)?(?:Level|Delivery|Fees?|المستوى|طريقة|الرسوم)", re.IGNORECASE
)
_MENTIONS_BUDGET = re.compile(r"budget|ميزاني", re.IGNORECASE)
_OVER_BUDGET = {
    "en": " (above your budget of {budget})",
    "ar": " (أعلى من ميزانيتك: {budget})",
}
_LEVEL_LINE = re.compile(r"^\s*-\s*(?:\*\*)?(?:Level|المستوى)(?:\*\*)?\s*:", re.IGNORECASE)
_LEVEL_RANK = (
    (3, ("advanced", "expert", "متقدم")),
    (2, ("intermediate", "متوسط")),
    (1, ("foundation", "beginner", "introductory", "entry", "مبتدئ", "تأسيسي")),
)
_STEP_UP = {
    "en": " (above your {level} level: this may be a step up)",
    "ar": " (أعلى من مستواك الحالي: قد تحتاج إلى تمهيد قبله)",
}


def _lang(language: Optional[str]) -> str:
    return "ar" if language == "ar" else "en"


def profile_block(profile: dict[str, Any], declined: list[str]) -> str:
    lines = []
    for field, label in PROFILE_FIELD_LABELS:
        value = profile.get(field)
        if value:
            lines.append(f"- {label}: {value}")
        elif field in declined:
            lines.append(f"- {label}: not provided (the learner skipped it)")
        elif field not in _OPTIONAL_FIELDS:
            lines.append(f"- {label}: not provided")
    return "LEARNER PROFILE:\n" + "\n".join(lines)


def _level_rank(text: str) -> Optional[int]:
    lowered = text.casefold()
    for rank, words in _LEVEL_RANK:
        if any(word in lowered for word in words):
            return rank
    return None


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("&", "and")).strip().casefold()


def _course_blocks(answer: str) -> tuple[list[str], list[tuple[str, list[str]]], list[str]]:
    """Split a recommendation into (lines before, [(name, block lines)], lines after)."""
    lines = answer.split("\n")
    before, blocks, after = [], [], []
    i = 0
    while i < len(lines) and not _COURSE_HEADER.match(lines[i]):
        before.append(lines[i])
        i += 1
    while i < len(lines):
        header = _COURSE_HEADER.match(lines[i])
        if not header:
            after = lines[i:]
            break
        block = [lines[i]]
        i += 1
        while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("-")):
            block.append(lines[i])
            i += 1
        # "Name [MOB-101]" or "Name (MOB-101)" -> "Name"
        name = re.sub(r"\s*[\[(][^\])]*[\])]\s*$", "", header.group("name")).strip()
        blocks.append((name, block))
    return before, blocks, after


def _first_number(text: str) -> Optional[float]:
    match = re.search(r"\d[\d,]*(?:\.\d+)?", str(text or ""))
    return float(match.group(0).replace(",", "")) if match else None


def _fix_course_ids(header: str, name: str, passages: list[str]) -> str:
    """
    Remove a course code, in [brackets] or (parentheses), that no passage
    about this course contains, and a literal "[n]" layout placeholder.
    """
    header = re.sub(r"\s*\[n\]", "", header)
    about = [p for p in passages if _squash(name) in _squash(p)]
    if not about:
        return header

    def check(match: re.Match) -> str:
        tag = match.group(1).strip()
        if not _COURSE_CODE.match(tag):
            return match.group(0)  # a citation such as [2]
        return match.group(0) if any(tag.casefold() in p.casefold() for p in about) else ""

    header = _TAG.sub(check, header)
    header = _CODE_IN_PARENS.sub(check, header).replace("()", "")
    header = re.sub(r"\s+\*\*(?=\s|$)", "**", header)  # "**Name (X)**" -> "**Name **" -> "**Name**"
    return re.sub(r"\s{2,}", " ", header).rstrip()


def _is_filler(block: list[str]) -> bool:
    """A block that says the course does not fit, or states no facts at all."""
    body = "\n".join(block[1:])
    facts = [line for line in block[1:] if _FACT_LINE.match(line)]
    return bool(_NOT_A_FIT.search(body)) or (bool(facts) and all(_NOT_STATED.search(f) for f in facts))


def tidy_recommendation(
    answer: str, context: str, profile: Optional[dict[str, Any]], language: str
) -> str:
    """
    Deterministic clean-up of a recommendation the model wrote:
    - a course listed twice keeps its first block only;
    - a course whose (Latin-script) name is not in the evidence is dropped,
      and so is a block saying the course does not fit or stating no facts
      (unless that would drop every course);
    - a bracketed course code that does not belong to the course is removed;
    - a course above the learner's skill level gets a "step up" remark;
    - a fee above the learner's budget is marked as such;
    - per-course "which course to start with" lines are removed (the
      closing sentence says it once).
    """
    before, blocks, after = _course_blocks(answer)
    if not blocks:
        return answer

    language = _lang(language)
    evidence = _squash(context)
    learner_rank = _level_rank(str((profile or {}).get("skill_level") or ""))

    budget_text = (profile or {}).get("budget")
    budget = _first_number(budget_text)
    passages = context.split("\n\n---\n\n")

    seen, kept = set(), []
    for name, block in blocks:
        key = _squash(name)
        if key in seen:
            continue
        seen.add(key)
        latin_name = not re.search(r"[؀-ۿ]", name)
        kept.append((name, block, (not latin_name or key in evidence) and not _is_filler(block)))

    verified = [(n, b) for n, b, ok in kept if ok]
    chosen = verified or [(n, b) for n, b, _ in kept]

    out = list(before)
    for name, block in chosen:
        for index, line in enumerate(block):
            if index == 0:
                line = _fix_course_ids(line, name, passages)
            elif _START_WITH_LINE.match(line):
                continue
            elif learner_rank and _LEVEL_LINE.match(line):
                course_rank = _level_rank(line.split(":", 1)[1])
                if course_rank and course_rank > learner_rank:
                    line = line.rstrip() + _STEP_UP[language].format(
                        level=(profile or {}).get("skill_level"))
            elif budget and _FEES_LINE.match(line) and not _MENTIONS_BUDGET.search(line):
                # (A line where the model already compared with the budget is left as is.)
                fee = _first_number(line.split(":", 1)[1])
                if fee and fee > budget:
                    line = line.rstrip() + _OVER_BUDGET[language].format(budget=budget_text)
            out.append(line)
    out.extend(after)
    return "\n".join(out).strip()


def unconfirmed_preferences_note(
    context: str, profile: Optional[dict[str, Any]], language: str
) -> str:
    """
    A fixed note listing learner preferences the evidence does not cover.
    Deterministic, so a model that claims "available online" with no
    evidence is still corrected in the reply.
    """
    profile = profile or {}
    language = _lang(language)
    lines = []

    preference = profile.get("training_preference")
    if preference in _PREFERENCE_LABELS and not _DELIVERY_WORDS.search(context):
        lines.append(_UNCONFIRMED["delivery"][language].format(
            value=_PREFERENCE_LABELS[preference][language]))

    budget = profile.get("budget")
    if budget and not _FEE_WORDS.search(context):
        lines.append(_UNCONFIRMED["fees"][language].format(value=budget))

    if not lines:
        return ""
    return "\n".join([_UNCONFIRMED["header"][language], *lines,
                      _UNCONFIRMED["footer"][language]])


def recommendation_task(
    profile: Optional[dict[str, Any]], declined: Optional[list[str]] = None
) -> str:
    """The task text REX adds to its prompt: the learner profile and the course layout."""
    return f"{profile_block(profile or {}, declined or [])}\n\n{COURSE_RECOMMENDATION_TASK}"


def check_recommendation(
    answer: str, context: str, profile: Optional[dict[str, Any]], language: Optional[str]
) -> str:
    """REX's deterministic checks on the model's recommendation (see module docstring)."""
    language = _lang(language)
    answer = tidy_recommendation(answer, context, profile, language)
    note = unconfirmed_preferences_note(context, profile, language)
    return f"{answer}\n\n{note}" if note else answer
