"""
REX's course recommendations, built from verified catalogue records only.

The records come from the knowledge base (MCP tool get_course_records,
knowledge_base/course_records.py): courses the catalogue identifies by
both an id and a name, with the facts it states. The model never writes
a course name or a fact:

1. It is shown the learner profile and the records, and may only answer
   with record ids, one per line, each with a short reason in the
   learner's language ("<id> :: <why it fits>"), or NONE.
2. An id that is not a record, or a repeated one, is ignored. For a
   learner with a stated level, courses above it are kept only when no
   chosen course is at the learner's level.
3. Each course is written from its record: the exact catalogue name and
   id, level, hours, delivery mode, fee, entry requirements and the
   prerequisite courses the catalogue lists. A fact the catalogue does
   not state is shown as "not stated in the catalogue", and the
   learner's delivery and budget preferences are compared only with
   stated facts. The reason keeps only sentences about fit: a sentence
   stating fees, delivery, durations, requirements or another course is
   removed, so no fact reaches the learner from the model.
4. With no record chosen, REX says no suitable course was found in the
   current catalogue. With no records at all, it says it cannot
   recommend a verified course -- the model is not asked.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.rex_assistant.recommendation import profile_block

logger = logging.getLogger(__name__)

MAX_RECOMMENDATIONS = 3
MAX_RECORDS_IN_PROMPT = 40

_ID = re.compile(r"[A-Za-z]{2,6}(?:-[A-Za-z0-9]{1,8}){1,3}")
_CHOICE_LINE = re.compile(r"^\W*(?P<id>[A-Za-z]{2,6}(?:-[A-Za-z0-9]{1,8}){1,3})\W*?(?:::|:|-|–|—|\|)\s*(?P<reason>.*)$")
_NONE = re.compile(r"^\W*(?:none|no suitable course|لا يوجد)\W*$", re.IGNORECASE)

# Facts only a record may state: a reason sentence mentioning one is dropped.
_FACT_WORDS = re.compile(
    r"prerequisit|pre-requisit|requirement|required|requires|literacy|\bfees?\b|price|cost|tuition|budget|"
    r"\begp\b|\busd\b|\$|\bonline\b|\boffline\b|in[- ]person|on[- ]?campus|hybrid|blended|\bhours?\b|\bweeks?\b|"
    r"متطلب|مطلوب|رسوم|سعر|تكلفة|ميزاني|جنيه|أونلاين|اونلاين|حضوري|عن بعد|ساعة|ساعات|أسبوع|أسابيع",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?؟])\s+")

_LEVEL_WORDS = (
    (1, ("foundation", "beginner", "introductory", "entry", "basic", "مبتدئ", "تأسيسي")),
    (2, ("intermediate", "متوسط")),
    (3, ("advanced", "expert", "متقدم", "خبير")),
)

_PREFERENCE_WORDS = {
    "online": re.compile(r"\b(?:online|virtual|remote|e-?learning)\b", re.IGNORECASE),
    "offline": re.compile(r"\b(?:offline|on[- ]?campus|in[- ]person|on[- ]?site|classroom)\b", re.IGNORECASE),
    "hybrid": re.compile(r"\b(?:hybrid|blended)\b", re.IGNORECASE),
}
_PREFERENCE_NAMES = {"online": {"en": "online", "ar": "أونلاين"}, "offline": {"en": "offline", "ar": "حضوري"},
                     "hybrid": {"en": "hybrid", "ar": "مدمج"}}

TEXT = {
    "intro": {"en": "Based on your profile, these courses from the academy catalogue fit best:",
              "ar": "بناءً على ملفك، هذه أنسب الدورات من كتالوج الأكاديمية:"},
    "why": {"en": "Why it fits you", "ar": "لماذا تناسبك"},
    "for": {"en": "Who it is for (catalogue)", "ar": "لمن هذه الدورة (حسب الكتالوج)"},
    "level": {"en": "Level", "ar": "المستوى"},
    "duration": {"en": "Duration", "ar": "المدة"},
    "hours": {"en": "{n} hours", "ar": "{n} ساعة"},
    "delivery": {"en": "Delivery", "ar": "طريقة الدراسة"},
    "fee": {"en": "Fee", "ar": "الرسوم"},
    "entry": {"en": "Entry requirements", "ar": "متطلبات الالتحاق"},
    "prereq": {"en": "Prerequisite courses", "ar": "الدورات المطلوبة مسبقًا"},
    "source": {"en": "Source", "ar": "المصدر"},
    "not_stated": {"en": "not stated in the catalogue", "ar": "غير مذكور في الكتالوج"},
    "delivery_unknown": {"en": "online or offline is not stated in the catalogue",
                         "ar": "لا يذكر الكتالوج إن كانت أونلاين أم حضورية"},
    "pref_unverified": {"en": " (your preference for {pref} training cannot be verified)",
                        "ar": " (لا يمكن التحقق من تفضيلك: {pref})"},
    "pref_match": {"en": " (matches your preference for {pref} training)", "ar": " (يطابق تفضيلك: {pref})"},
    "pref_mismatch": {"en": " (does not match your preference for {pref} training)",
                      "ar": " (لا يطابق تفضيلك: {pref})"},
    "budget_unverified": {"en": " (so it cannot be checked against your budget of {budget})",
                          "ar": " (لذلك لا يمكن مقارنتها بميزانيتك: {budget})"},
    "within_budget": {"en": " (within your budget of {budget})", "ar": " (ضمن ميزانيتك: {budget})"},
    "over_budget": {"en": " (above your budget of {budget})", "ar": " (أعلى من ميزانيتك: {budget})"},
    "step_up": {"en": " (above your {level} level: check the entry requirements below before enrolling)",
                "ar": " (أعلى من مستواك الحالي: راجع متطلبات الالتحاق أدناه قبل التسجيل)"},
    "start": {"en": "Start with **{name}** ({id}): it is the closest match to your goal at your level.",
              "ar": "ابدأ بـ **{name}** ({id}): فهي الأقرب إلى هدفك ومستواك."},
    "none_suitable": {
        "en": "No suitable course was found in the current academy catalogue for your profile. "
              "I won't suggest a course the catalogue does not offer; please contact the academy for advice.",
        "ar": "لم أجد دورة مناسبة لملفك في كتالوج الأكاديمية الحالي. لن أقترح دورة غير موجودة في الكتالوج؛ "
              "يُرجى التواصل مع الأكاديمية للمشورة.",
    },
    "no_records": {
        "en": "I can't recommend a specific course right now: the knowledge base has no course records with "
              "an ID and a name that I could verify. Please contact the academy for course advice.",
        "ar": "لا أستطيع ترشيح دورة محددة الآن: لا تحتوي قاعدة المعرفة على سجلات دورات بمعرّف واسم يمكنني "
              "التحقق منهما. يُرجى التواصل مع الأكاديمية للمشورة.",
    },
}

SELECTION_TASK = """
RECOMMENDATION TASK:
Choose the courses that best fit this learner, ONLY from the CATALOGUE
COURSES list below (they are the only courses that exist). Best fit
first, at most {limit}. Prefer courses at the learner's level and in the
field of their career goal; use their background (do not assume they lack
skills their background implies).

Answer with one line per chosen course and nothing else, exactly:
<course id> :: <one or two sentences in {language}, speaking to the learner as "you", on why it fits your goal and background>

Rules:
- Use only course ids from the list. Never write a course that is not listed.
- When you mention the learner's background or goal, use the words of the
  LEARNER PROFILE; never add details it does not state.
- In the reason, do not mention fees, delivery, duration, prerequisites or
  entry requirements: the catalogue facts are shown separately.
- If no listed course fits the learner's career goal, answer exactly: NONE
""".strip()

_LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}


def _lang(language: Optional[str]) -> str:
    return "ar" if language == "ar" else "en"


def _t(key: str, language: str, **values: Any) -> str:
    return TEXT[key][_lang(language)].format(**values)


def level_rank(text: Any) -> Optional[int]:
    """1 beginner/foundation, 2 intermediate, 3 advanced; "Foundation to Intermediate" -> 1 (its entry level)."""
    folded = str(text or "").casefold()
    if not folded.strip():
        return None
    ranks = [rank for rank, words in _LEVEL_WORDS if any(w in folded for w in words)]
    if not ranks:
        return None
    if re.search(r"\bto\b|إلى|الى|-", folded) and len(ranks) > 1:
        first = min((folded.find(w), rank) for rank, words in _LEVEL_WORDS for w in words if w in folded)
        return first[1]
    return max(ranks)


def _id_key(value: Any) -> str:
    return re.sub(r"[^0-9a-z]", "", str(value or "").casefold())


def verified_records(records: Any) -> list[dict[str, Any]]:
    """Records with both an id and a name, each once (first wins)."""
    out, seen = [], set()
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        course_id, name = record.get("course_id"), record.get("course_name")
        if not (isinstance(course_id, str) and course_id.strip() and isinstance(name, str) and name.strip()):
            continue
        key = _id_key(course_id)
        if key not in seen:
            seen.add(key)
            out.append(record)
    return out


def _records_in_relevance_order(records: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Records whose passages the profile search retrieved come first, in retrieval order."""
    rank: dict[str, int] = {}
    for position, chunk in enumerate(evidence.get("chunks") or []):
        key = _id_key(chunk.get("course_id"))
        if key and key not in rank:
            rank[key] = position
    return sorted(records, key=lambda r: rank.get(_id_key(r["course_id"]), len(rank) + 1))


def _catalogue_block(records: list[dict[str, Any]]) -> str:
    lines = []
    for r in records[:MAX_RECORDS_IN_PROMPT]:
        parts = [r["course_id"], r["course_name"]]
        for label, key in (("department", "department"), ("level", "level"), ("for", "target_audience"),
                           ("about", "summary")):
            if r.get(key):
                parts.append(f"{label}: {str(r[key])[:220]}")
        lines.append("- " + " | ".join(parts))
    return "CATALOGUE COURSES (id | exact name | facts):\n" + "\n".join(lines)


_NOT_A_FIT = re.compile(
    r"not (?:directly )?relevant|not (?:directly )?related|not suitable|not a (?:good )?fit|"
    r"not applicable|does not fit|doesn't fit|غير مناسب|لا يناسب|غير مرتبط",
    re.IGNORECASE,
)


_BULLET = re.compile(r"^\s*(?:[-•*]\s+|\d+\.\s+(?=[^*]))")
_WHY_LABEL = re.compile(r"^\s*(?:\*\*)?(?:why it fits(?: you)?|reason|لماذا تناسبك)(?:\*\*)?\s*:\s*", re.IGNORECASE)
_CLOSING = re.compile(r"^\s*(?:\*\*)?(?:start with|i recommend starting|which course to start|ابدأ)", re.IGNORECASE)


def _record_in(text: str, by_id: dict[str, dict[str, Any]],
               names: list[tuple[str, dict[str, Any]]]) -> tuple[Optional[dict[str, Any]], bool]:
    """(the record a line names, conflicting): by id or exact name; an id and a name of two
    different records ("**Machine Learning Practitioner** [AI-101]") are a conflict, not a choice."""
    by_code = next((by_id[_id_key(f)] for f in _ID.findall(text) if _id_key(f) in by_id), None)
    folded = re.sub(r"\s+", " ", re.sub(r"[*_`]", "", text)).casefold()
    by_name = next((r for name, r in names if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", folded)), None)
    if by_code and by_name and by_code is not by_name:
        return None, True
    return by_code or by_name, False


def parse_choices(answer: str, records: list[dict[str, Any]]) -> tuple[list[tuple[dict[str, Any], str]], bool]:
    """
    ([(record, reason)], said_none) from the model's reply. A choice is a
    line naming a record by its id ("DA-101 :: ...", "**Excel** [DA-101]")
    or exact catalogue name; bullet lines under it belong to it ("- Why it
    fits you: ..."). Ignored: invented or renamed courses, a line whose id
    and name are of different courses, repeats, closing "start with"
    lines, and a choice whose own text says the course does not fit.
    """
    by_id = {_id_key(r["course_id"]): r for r in records}
    names = sorted(((re.sub(r"\s+", " ", r["course_name"]).casefold(), r) for r in records),
                   key=lambda item: -len(item[0]))
    blocks: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    said_none = False
    for raw in str(answer or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if _NONE.match(line.strip("*")):
            said_none = True
            current = None
            continue
        if _BULLET.match(line) and current is not None:
            body = _BULLET.sub("", line, count=1)
            current["text"].append(body)
            if _WHY_LABEL.match(body) and not current["reason"]:
                current["reason"] = _WHY_LABEL.sub("", body)
            continue
        if _CLOSING.match(line.lstrip("-•* ")):
            current = None
            continue
        head, sep, rest = line.partition("::")
        record, conflict = _record_in(head, by_id, names)
        if record is None:
            current = None   # an invented, renamed or conflicting course: its bullets are ignored too
            continue
        reason = rest.strip()
        if not sep:   # "DA-101: reason" / "DA-101 - reason"
            match = _CHOICE_LINE.match(line.strip("*").strip())
            if match and _id_key(match.group("id")) == _id_key(record["course_id"]):
                reason = match.group("reason").strip()
        current = {"record": record, "reason": reason, "text": [reason]}
        blocks.append(current)

    chosen, seen = [], set()
    for block in blocks:
        key = _id_key(block["record"]["course_id"])
        if key in seen:
            continue
        seen.add(key)
        if _NOT_A_FIT.search(" ".join(block["text"])):
            continue
        chosen.append((block["record"], block["reason"]))
    return chosen, said_none and not chosen


MAX_REASON_SENTENCES = 2
MAX_REASON_CHARS = 320
_NAME_WORD = re.compile(r"[a-z][a-z0-9+#.]*|[ء-ي]{3,}", re.IGNORECASE)
_GENERIC_NAME_WORDS = frozenset("""
and for with the of to in on a an your you introduction intro fundamentals foundations foundation basics basic
advanced intermediate beginner course courses programme program track skills applied practical essentials
""".split())
_REPEATED_WORD = re.compile(r"(\S+(?:\s+\S+){0,3}?)(?:\s+\1){3,}", re.IGNORECASE)   # a phrase looping 4+ times


def _name_terms(name: str) -> set[str]:
    return {w.casefold() for w in _NAME_WORD.findall(name or "") if w.casefold() not in _GENERIC_NAME_WORDS}


def clean_reason(reason: str, record: dict[str, Any], records: list[dict[str, Any]],
                 profile: Optional[dict[str, Any]] = None) -> str:
    """
    Only short sentences about fit. Removed: a sentence stating a catalogue
    fact (fees, delivery, duration, requirements), naming another course or
    its id, or crediting this course with a distinctive term of another
    course's name that neither this course's record nor the learner's
    profile contains (the model said Excel for Business Analysis teaches
    "SQL and Power BI" -- another course's topics). A garbled reply (a word
    repeated over and over) or one too long to be a reason is dropped.
    """
    others = [r for r in records if _id_key(r["course_id"]) != _id_key(record["course_id"])]
    own_text = " ".join(str(record.get(k) or "") for k in
                        ("course_name", "summary", "target_audience", "entry_requirements", "department",
                         "prerequisites_text")).casefold()
    own_text += " " + " ".join(str(v) for v in (profile or {}).values()).casefold()
    borrowed = {term for r in others for term in _name_terms(r["course_name"])
                if not re.search(rf"(?<!\w){re.escape(term)}(?!\w)", own_text)}
    text = re.sub(r"\s*\[\d+\]", "", reason or "").strip()
    if _REPEATED_WORD.search(text):
        return ""
    kept = []
    for sentence in _SENTENCE.split(text):
        folded = sentence.casefold()
        if not sentence.strip() or _FACT_WORDS.search(sentence):
            continue
        if any(r["course_name"].casefold() in folded for r in others):
            continue
        if any(_id_key(found) in {_id_key(r["course_id"]) for r in others} for found in _ID.findall(sentence)):
            continue
        if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", folded) for term in borrowed):
            continue
        kept.append(sentence.strip())
    kept = kept[:MAX_REASON_SENTENCES]
    return " ".join(kept) if len(" ".join(kept)) <= MAX_REASON_CHARS else ""


def apply_level_policy(chosen: list[tuple[dict[str, Any], str]], learner_level: Any) -> list[tuple[dict[str, Any], str]]:
    """A course above the learner's level is kept only when none chosen is at it; at-level courses first."""
    learner = level_rank(learner_level)
    if not learner:
        return chosen
    fitting = [c for c in chosen if (level_rank(c[0].get("level")) or learner) <= learner]
    above = [c for c in chosen if (level_rank(c[0].get("level")) or learner) > learner]
    return fitting if fitting else above


def _number(value: Any) -> Optional[float]:
    match = re.search(r"\d[\d,]*(?:\.\d+)?", str(value or ""))
    return float(match.group(0).replace(",", "")) if match else None


def _fee_text(record: dict[str, Any]) -> Optional[str]:
    if record.get("fee") is None:
        return None
    amount = record["fee"]
    amount = int(amount) if float(amount).is_integer() else amount
    return f"{amount} {record['fee_currency']}" if record.get("fee_currency") else str(amount)


def _delivery_line(record: dict[str, Any], preference: Optional[str], language: str) -> str:
    delivery = record.get("delivery_mode")
    pref = preference if preference in _PREFERENCE_NAMES else None
    pref_name = _PREFERENCE_NAMES[pref][_lang(language)] if pref else ""
    if not delivery:
        line = _t("delivery_unknown", language)
        return line + (_t("pref_unverified", language, pref=pref_name) if pref else "")
    if not pref:
        return delivery
    is_hybrid = bool(_PREFERENCE_WORDS["hybrid"].search(delivery))
    matches = is_hybrid if pref == "hybrid" else (not is_hybrid and bool(_PREFERENCE_WORDS[pref].search(delivery)))
    if not matches and not is_hybrid and not any(p.search(delivery) for p in _PREFERENCE_WORDS.values()):
        return delivery + _t("pref_unverified", language, pref=pref_name)
    return delivery + _t("pref_match" if matches else "pref_mismatch", language, pref=pref_name)


def _fee_line(record: dict[str, Any], budget_text: Any, language: str) -> str:
    fee = _fee_text(record)
    budget = _number(budget_text)
    if fee is None:
        return _t("not_stated", language) + (_t("budget_unverified", language, budget=budget_text) if budget else "")
    if budget is None:
        return fee
    return fee + _t("within_budget" if record["fee"] <= budget else "over_budget", language, budget=budget_text)


def render_course(record: dict[str, Any], reason: str, profile: dict[str, Any], language: str,
                  records: list[dict[str, Any]]) -> str:
    not_stated = _t("not_stated", language)
    lines = [f"**{record['course_name']}** ({record['course_id']})"]
    if reason:
        lines.append(f"- {_t('why', language)}: {reason}")
    if record.get("target_audience"):
        lines.append(f"- {_t('for', language)}: {record['target_audience']}")
    level = record.get("level") or not_stated
    learner = level_rank(profile.get("skill_level"))
    course_level = level_rank(record.get("level"))
    if learner and course_level and course_level > learner:
        level += _t("step_up", language, level=profile.get("skill_level"))
    lines.append(f"- {_t('level', language)}: {level}")
    hours = record.get("total_hours")
    lines.append(f"- {_t('duration', language)}: "
                 + (_t("hours", language, n=hours) if hours else (record.get("duration") or not_stated)))
    lines.append(f"- {_t('delivery', language)}: {_delivery_line(record, profile.get('training_preference'), language)}")
    lines.append(f"- {_t('fee', language)}: {_fee_line(record, profile.get('budget'), language)}")
    lines.append(f"- {_t('entry', language)}: {record.get('entry_requirements') or not_stated}")
    names = {_id_key(r["course_id"]): r for r in records}
    if record.get("prerequisite_courses"):
        listed = [f"{cid} {names[_id_key(cid)]['course_name']}" if _id_key(cid) in names else cid
                  for cid in record["prerequisite_courses"]]
        lines.append(f"- {_t('prereq', language)}: {', '.join(listed)}")
    elif record.get("prerequisites_text"):
        lines.append(f"- {_t('prereq', language)}: {record['prerequisites_text']}")
    if record.get("source_files"):
        lines.append(f"- {_t('source', language)}: {', '.join(record['source_files'])}")
    return "\n".join(lines)


def _sources(chosen: list[dict[str, Any]], evidence_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The retrieved passages of the recommended courses; a course without one cites its catalogue file."""
    out = []
    for record in chosen:
        own = [s for s in evidence_sources if _id_key(s.get("course_id")) == _id_key(record["course_id"])]
        if own:
            out.extend(own)
            continue
        out.append({"reference": None, "course_id": record["course_id"], "course_name": record["course_name"],
                    "source": (record.get("source_files") or [None])[0],
                    "page": (record.get("pages") or [None])[0], "section": None, "section_path": None,
                    "record_id": None, "chunk_id": None})
    return out


MAX_EVIDENCE_PASSAGES = 12
MAX_PASSAGE_CHARS = 400


def _evidence_block(chunks: list[dict[str, Any]], backed: set[str]) -> str:
    """The retrieved passages of the candidate courses, labelled with their course, as the model sees them."""
    lines = []
    for chunk in chunks:
        if _id_key(chunk.get("course_id")) not in backed or len(lines) >= MAX_EVIDENCE_PASSAGES:
            continue
        text = re.sub(r"\s+", " ", str(chunk.get("text") or "")).strip()[:MAX_PASSAGE_CHARS]
        lines.append(f"[{chunk.get('reference')}] ({chunk['course_id']} {chunk.get('course_name') or ''}) {text}")
    return "RETRIEVED CATALOGUE PASSAGES (evidence for the courses above):\n" + "\n".join(lines)


def recommend(llm, invoke, records: Any, evidence: dict[str, Any], evidence_sources: list[dict[str, Any]],
              language: str, profile: Optional[dict[str, Any]], declined: Optional[list[str]],
              system_prompt: str) -> dict[str, Any]:
    """
    {"answer", "sources", "status", "courses"} for a recommendation.
    `invoke(llm, messages) -> str` is REX's model call (graph._invoke_text).

    Candidates are the verified records whose passages the profile search
    retrieved (RAG similarity search): a course no retrieved passage
    supports cannot be recommended. The model sees those records and their
    passages, and picks among them by id.
    """
    language = _lang(language)
    profile = profile or {}
    records = verified_records(records)
    if not records:
        return {"answer": _t("no_records", language), "sources": [], "status": "no_verified_courses", "courses": []}

    chunks = [c for c in evidence.get("chunks") or [] if c.get("course_id")]
    backed_ids = {_id_key(c["course_id"]) for c in chunks}
    candidates = [r for r in _records_in_relevance_order(records, evidence) if _id_key(r["course_id"]) in backed_ids]
    if not candidates:
        return {"answer": _t("none_suitable", language), "sources": [], "status": "no_suitable_course", "courses": []}

    task = SELECTION_TASK.format(limit=MAX_RECOMMENDATIONS, language=_LANGUAGE_NAMES[language])
    prompt = "\n\n".join([profile_block(profile, declined or []), _catalogue_block(candidates),
                           _evidence_block(chunks, backed_ids), task])
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
    answer = invoke(llm, messages)
    chosen, said_none = parse_choices(answer, candidates)

    if not chosen and not said_none:
        # No usable reply (failed call, or no listed id): the courses whose
        # passages the profile search found, in retrieval order, without a
        # model-written reason.
        chosen = [(r, "") for r in candidates][:MAX_RECOMMENDATIONS]
    chosen = apply_level_policy(chosen, profile.get("skill_level"))[:MAX_RECOMMENDATIONS]
    if not chosen:
        return {"answer": _t("none_suitable", language), "sources": [], "status": "no_suitable_course", "courses": []}

    blocks = [render_course(record, clean_reason(reason, record, records, profile), profile, language, records)
              for record, reason in chosen]
    first = chosen[0][0]
    text = "\n\n".join([_t("intro", language), *blocks,
                         _t("start", language, name=first["course_name"], id=first["course_id"])])
    picked = [record for record, _ in chosen]
    return {"answer": text, "sources": _sources(picked, evidence_sources), "status": "answered",
            "courses": [{"course_id": r["course_id"], "course_name": r["course_name"]} for r in picked]}


# ---------------------------------------------------------------------
# Catalogue listing ("What courses do you offer?")
# ---------------------------------------------------------------------

CATALOGUE_TEXT = {
    "complete": {"en": "The academy catalogue lists {n} courses:", "ar": "يضم كتالوج الأكاديمية {n} دورة:"},
    "filtered": {"en": "{k} of the catalogue's {n} courses match your request{what}:",
                 "ar": "{k} من أصل {n} دورة في الكتالوج تطابق طلبك{what}:"},
    "none_match": {"en": "No course in the current academy catalogue matches your request{what}. "
                         "Ask \"What courses does the academy offer?\" to see all {n} courses.",
                   "ar": "لا توجد في كتالوج الأكاديمية الحالي دورة تطابق طلبك{what}. "
                         "اسأل «ما هي الدورات التي تقدمها الأكاديمية؟» لعرض جميع الدورات ({n})."},
    "empty": {"en": "The current catalogue has no courses with an ID and a name I could verify, so I can't list "
                    "any. Please contact the academy.",
              "ar": "لا يحتوي الكتالوج الحالي على دورات بمعرّف واسم يمكنني التحقق منهما، لذلك لا أستطيع عرض أي دورة. "
                    "يُرجى التواصل مع الأكاديمية."},
    "other_department": {"en": "Other", "ar": "أخرى"},
    "similarity_note": {"en": "No course names this topic, so these are the courses whose catalogue pages "
                              "matched it best.",
                        "ar": "لا تحمل أي دورة هذا الموضوع في اسمها، لذا هذه الدورات التي طابقت صفحاتها في الكتالوج "
                              "الموضوع أكثر من غيرها."},
    "footer": {"en": "Ask me about any of these courses (e.g. \"Tell me more about the first one\"), "
                     "or ask for a recommendation that fits your background.",
               "ar": "اسألني عن أي دورة منها (مثلًا: «أخبرني المزيد عن الدورة الأولى»)، "
                     "أو اطلب ترشيحًا يناسب خلفيتك."},
}


def _ct(key: str, language: str, **values: Any) -> str:
    return CATALOGUE_TEXT[key][_lang(language)].format(**values)


def _what(result: dict[str, Any], language: str) -> str:
    """ (topic: data, analytics; level: beginner) -- what narrowed the list."""
    flt = result.get("filter") or {}
    parts = []
    if flt.get("topic_terms"):
        parts.append(("الموضوع: " if language == "ar" else "topic: ") + ", ".join(flt["topic_terms"]))
    if flt.get("level"):
        parts.append(("المستوى: " if language == "ar" else "level: ") + flt["level"])
    return f" ({'; '.join(parts)})" if parts else ""


def render_catalogue(result: dict[str, Any], language: str) -> tuple[str, list[dict[str, Any]]]:
    """
    (answer, listed courses) for a get_course_records result: every course
    once, numbered, grouped by department, each with only the facts the
    catalogue states and "not stated in the catalogue" for the rest.
    """
    language = _lang(language)
    records = verified_records(result.get("records"))
    total = result.get("total_in_catalogue") or len(records)
    if not total:
        return _ct("empty", language), []
    if not records:
        return _ct("none_match", language, what=_what(result, language), n=total), []

    head = (_ct("complete", language, n=total) if result.get("complete")
            else _ct("filtered", language, k=len(records), n=total, what=_what(result, language)))
    lines = [head]
    if result.get("match_method") == "similarity":
        lines.append(_ct("similarity_note", language))
    not_stated = _t("not_stated", language)
    ordered = sorted(records, key=lambda r: (str(r.get("department") or "~"), r["course_id"]))
    department = object()
    listed = []
    for number, record in enumerate(ordered, start=1):
        if record.get("department") != department:
            department = record.get("department")
            lines.append(f"\n**{department or _ct('other_department', language)}**")
        hours = _t("hours", language, n=record["total_hours"]) if record.get("total_hours") else not_stated
        facts = [f"{_t('level', language)}: {record.get('level') or not_stated}",
                 f"{_t('duration', language)}: {hours}",
                 f"{_t('delivery', language)}: {record.get('delivery_mode') or not_stated}",
                 f"{_t('fee', language)}: {_fee_text(record) or not_stated}"]
        lines.append(f"{number}. **{record['course_name']}** ({record['course_id']}) — " + " · ".join(facts))
        listed.append({"course_id": record["course_id"], "course_name": record["course_name"]})
    lines.append("\n" + _ct("footer", language))
    return "\n".join(lines), listed
