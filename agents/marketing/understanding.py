"""
Rule-based reading of a chat request for Agent 5 (English and Arabic): which approved course it names. No language
model is involved. When the Marketing tab already chose the course (and the audience, tone and channels), none of
this is used (see graph.py).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agents.text_utils import normalize_for_matching

CURRICULUM_ID = re.compile(r"\bCUR-[0-9a-f]{12}\b", re.IGNORECASE)

_STOP = frozenset("""a an the for of on in to and or with using use from me us our my please that this it its is are be
approved selected chosen current existing new curriculum curricula course courses campaign campaigns marketing market
promote promotion promotional create generate make write prepare draft build design social media advertisement advertising
ad ads post posts email emails whatsapp linkedin facebook instagram""".split())


def find_curriculum_id(text: str) -> Optional[str]:
    match = CURRICULUM_ID.search(text)
    return "CUR-" + match.group(0)[4:].lower() if match else None


def choose_from_options(text: str, option_ids: list[str]) -> Optional[str]:
    """A reply such as "2" or "number 2" picks the n-th course offered."""
    match = re.fullmatch(r"\s*(?:number\s*|no\.?\s*|#)?(\d{1,2})\s*[.)]?\s*", normalize_for_matching(text))
    if match and 1 <= int(match.group(1)) <= len(option_ids):
        return option_ids[int(match.group(1)) - 1]
    return None


def match_courses(text: str, courses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The approved courses whose title shares the most words with the request (ties kept)."""
    words = {w for w in re.findall(r"[^\W_]+", normalize_for_matching(text)) if len(w) > 1 and w not in _STOP}
    scored = []
    for row in courses:
        title_words = set(re.findall(r"[^\W_]+", normalize_for_matching(str(row.get("title") or ""))))
        score = len(words & title_words)
        if score:
            scored.append((score, row))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    return [row for score, row in scored if score == best]


def drafted_a_curriculum(previous_results: Any) -> bool:
    """Did the Curriculum Agent just answer in this same request (the "curriculum, then marketing" plan)?"""
    return any(isinstance(r, dict) and r.get("agent_name") == "curriculum_research" for r in previous_results or [])


# ---- answers to "who is this campaign for?" ----------------------------------------------------------------------------------------------

MAX_AUDIENCE_WORDS = 14

_CANCEL = {"cancel", "stop", "never mind", "nevermind", "forget it", "abort", "quit", "no", "nope", "skip", "leave it",
           "الغاء", "إلغاء", "الغ", "توقف", "لا", "انسى", "لا شكرا", "لا تكمل"}
_SMALL_TALK = {"ok", "okay", "k", "yes", "yeah", "yep", "sure", "thanks", "thank you", "thx", "hi", "hello", "hey", "good", "great",
               "fine", "cool", "nice", "done", "got it", "alright", "hmm", "please", "حسنا", "حسناً", "تمام", "شكرا", "شكراً",
               "مرحبا", "اهلا", "أهلا", "نعم", "طيب", "ماشي", "اوكي", "أوكي"}
_GENERIC = {"everyone", "everybody", "anyone", "anybody", "all", "people", "public", "general public", "users", "whoever", "no one",
            "الجميع", "الكل", "اي شخص", "أي شخص", "الناس", "كل الناس", "عامة الناس"}
_REQUEST_START = re.compile(
    r"^(?:please\s+)?(?:can|could|would|will)\s+you\b|^(?:please\s+)?(?:create|generate|write|make|prepare|draft|build|design|find|search|"
    r"show|give|tell|explain|approve|reject|send|publish|download|what|which|who|when|where|why|how)\b|"
    r"^(?:اكتب|انشئ|أنشئ|اعمل|جهز|حضر|ابحث|اعرض|اعطني|أعطني|اشرح|اعتمد|ارسل|أرسل|انشر|نزل|ماذا|ما |من |متي|متى|اين|أين|لماذا|كيف|هل)")
_UNSAFE = re.compile(r"https?://|www\.|[\w.+-]+@[\w-]+\.[\w.]+|<[^>]+>|\d{6,}")


def classify_reply(text: str) -> tuple[str, str]:
    """
    What a message sent while the agent waits for the campaign's audience is: (kind, cleaned text) where kind is
      "audience"    a short description of who the campaign is for;
      "cancel"      the user gives up on the campaign;
      "empty"       nothing there;
      "small_talk"  an acknowledgement or greeting ("ok", "thanks"): not an answer;
      "new_request" a question or a request for something else: not an answer;
      "ambiguous"   "everyone": too vague to write for, so the agent asks again;
      "invalid"     too long, or a link, e-mail address, markup or long number: not an audience.
    Only "audience" may start a campaign.
    """
    cleaned = " ".join(str(text or "").split())
    folded = normalize_for_matching(cleaned).strip(" .!?؟,،;:'\"")
    if not cleaned or not re.search(r"[^\W\d_]", cleaned):
        return "empty", cleaned
    if folded in {normalize_for_matching(w) for w in _CANCEL}:
        return "cancel", cleaned
    if folded in {normalize_for_matching(w) for w in _SMALL_TALK}:
        return "small_talk", cleaned
    if cleaned.rstrip().endswith(("?", "؟")) or _REQUEST_START.search(normalize_for_matching(cleaned)) or CURRICULUM_ID.search(cleaned):
        return "new_request", cleaned
    if folded in {normalize_for_matching(w) for w in _GENERIC}:
        return "ambiguous", cleaned
    if len(cleaned.split()) > MAX_AUDIENCE_WORDS or len(cleaned) > 300 or _UNSAFE.search(cleaned):
        return "invalid", cleaned
    return "audience", cleaned


def unsafe_audience(text: str) -> bool:
    """A link, e-mail address, markup or long number is never an audience (checked for the form's field too)."""
    return bool(_UNSAFE.search(str(text or "")))
