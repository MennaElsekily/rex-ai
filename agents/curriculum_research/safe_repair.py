"""
Deterministic repairs for validation problems that have one safe,
explainable fix (no language model). graph.py runs these after the
model's own repair pass; bilingual.py runs them on a translated
companion. Every change is returned as a note shown to the admin, and
nothing here can approve anything: the repaired curriculum is validated
again by the same unchanged rules, and whatever still fails stays a draft.

- A learning outcome that starts with a vague verb ("understand",
  "know", "learn", ...) gets the standard measurable verb for the same
  intent ("explain", "identify", "describe", ...).
- An outcome that no module develops (or no assessment assesses) is
  linked to the module (assessment) whose title/topics/description
  share the most topic words with it. With no shared word there is no
  evidence to link on, so nothing is linked and the problem is reported.
- A module or assessment that develops/assesses no outcome is linked,
  the same way, to the outcome it shares most words with.
"""

from __future__ import annotations

import re
from typing import Any

from agents.curriculum_research.validation import _VAGUE_AR, _VAGUE_EN, first_verb, is_measurable
from agents.text_utils import normalize_for_matching

# Vague -> measurable verb with the same intent (Bloom's taxonomy). Keys are normalized forms.
_VERB_FIX_EN = {
    "understand": "explain", "know": "identify", "learn": "describe", "appreciate": "explain",
    "grasp": "explain", "familiarise": "describe", "familiarize": "describe", "realise": "identify",
    "realize": "identify", "install": "configure", "setup": "configure",
}
_VERB_FIX_AR = {normalize_for_matching(k): v for k, v in {
    "يفهم": "يشرح", "فهم": "شرح", "يعرف": "يحدد", "معرفة": "تحديد", "يتعلم": "يصف", "تعلم": "وصف",
    "يدرك": "يفسر", "إدراك": "تفسير", "يتعرف": "يحدد", "تعرف": "تحديد", "يلم": "يصف", "إلمام": "وصف",
    "يثبت": "يجهز", "تثبيت": "تجهيز", "يخلق": "ينشئ", "خلق": "إنشاء",
}.items()}

# Vague verbs carry no topic. (An outcome's own leading verb is dropped too, in apply_safe_fixes:
# sharing "Apply"/"يطبق" with a module is no evidence the module teaches that outcome.)
_ACTION_WORDS = _VAGUE_EN | _VAGUE_AR | set(_VERB_FIX_EN) | set(_VERB_FIX_AR)

_STOP = frozenset("""
the and for with from into that this these those their your using use used able will learners learner students
basic basics fundamental fundamentals introduction overview simple about through within each how what when
وال في من على الى إلى عن مع هذا هذه ذلك التي الذي باستخدام استخدام اساسيات أساسيات مقدمه مقدمة
""".split())


def _words(text: Any) -> dict[str, str]:
    """{stem: the word as written} for the topic words of a text."""
    words: dict[str, str] = {}
    for original in re.findall(r"[\w-]+", str(text or "")):
        word = normalize_for_matching(original)
        word = re.sub(r"^(?:ال)", "", word) if re.match(r"^ال[ء-ي]{3,}", word) else word
        if len(word) >= 4 and word not in _STOP and word not in _ACTION_WORDS and not word.isdigit():
            # A crude stem, so "evaluate" ~ "evaluating" and "classification" ~ "classifier".
            words.setdefault(word[:7] if re.match(r"[a-z]", word) else word, original)
    return words


def _module_words(module: dict[str, Any]) -> set[str]:
    return _words(" ".join([str(module.get("title") or ""), str(module.get("description") or ""),
                            *[str(t) for t in module.get("topics") or []]]))


def _assessment_words(assessment: dict[str, Any]) -> set[str]:
    return _words(f"{assessment.get('type') or ''} {assessment.get('description') or ''}")


def _fix_verb(statement: str, language: str) -> str | None:
    verb = first_verb(statement, language)
    table = _VERB_FIX_AR if language == "ar" else _VERB_FIX_EN
    replacement = table.get(verb)
    if not replacement:
        return None
    tokens = statement.split(" ")
    for index, token in enumerate(tokens):
        core = re.sub(r"[^\w؀-ۿ-]", "", token)
        key = normalize_for_matching(core) if language == "ar" else core.casefold()
        if key == verb:
            word = replacement.capitalize() if language != "ar" and core[:1].isupper() else replacement
            tokens[index] = token.replace(core, word, 1)
            fixed = " ".join(tokens)
            return fixed if is_measurable(fixed, language) else None
    return None


def _best(target_words: dict[str, str], candidates: list[tuple[str, dict[str, str]]],
          show: str = "target") -> tuple[str | None, list[str]]:
    """The candidate sharing most stems with the target, and the shared words as written in the outcome."""
    best_id, best_shared = None, []
    for candidate_id, words in candidates:
        stems = sorted(set(target_words) & set(words))
        if len(stems) > len(best_shared):
            source = target_words if show == "target" else words
            best_id, best_shared = candidate_id, [source[stem].lower() for stem in stems]
    return best_id, best_shared


def apply_safe_fixes(curriculum: dict[str, Any]) -> list[str]:
    """Repairs `curriculum` in place; returns one note per change made (in the curriculum's language)."""
    language = curriculum.get("language") if curriculum.get("language") in ("ar", "en") else "en"
    ar = language == "ar"
    notes: list[str] = []
    outcomes = [o for o in curriculum.get("learning_outcomes") or [] if isinstance(o, dict) and o.get("id")]
    modules = [m for m in curriculum.get("modules") or [] if isinstance(m, dict)]
    assessments = [a for a in curriculum.get("assessments") or [] if isinstance(a, dict)]

    for outcome in outcomes:
        statement = str(outcome.get("statement") or "")
        if statement and not is_measurable(statement, language):
            fixed = _fix_verb(statement, language)
            if fixed:
                old = first_verb(statement, language)
                outcome["statement"] = fixed
                notes.append(f"{outcome['id']}: استُبدل الفعل «{old}» بالفعل «{first_verb(fixed, language)}»." if ar else
                             f"{outcome['id']}: the vague verb \"{old}\" was replaced with \"{first_verb(fixed, language)}\".")

    def topic_words(outcome: dict[str, Any]) -> dict[str, str]:
        """The outcome's topic words, without its own leading action verb."""
        words = _words(outcome.get("statement"))
        verb = first_verb(str(outcome.get("statement") or ""), language)
        words.pop(verb[:7] if re.match(r"[a-z]", verb) else verb, None)
        return words

    outcome_words = {o["id"]: topic_words(o) for o in outcomes}

    def link(parts: list[dict[str, Any]], words_of, kind_en: str, kind_ar: str, also_linked: set[str]) -> None:
        candidates = [(p.get("id"), words_of(p)) for p in parts if p.get("id")]
        linked = {ref for p in parts for ref in p.get("outcome_ids") or []} | also_linked
        for oid, words in outcome_words.items():
            if oid in linked:
                continue
            target, shared = _best(words, candidates)
            if target:
                part = next(p for p in parts if p.get("id") == target)
                part["outcome_ids"] = [*(part.get("outcome_ids") or []), oid]
                notes.append(f"ربط {oid} بـ{kind_ar} {target} (كلمات مشتركة: {', '.join(shared[:3])})." if ar else
                             f"{oid} was linked to {kind_en} {target} (shared topic words: {', '.join(shared[:3])}).")
        for part in parts:
            if part.get("outcome_ids") or not part.get("id"):
                continue
            oid, shared = _best(words_of(part), list(outcome_words.items()), show="candidate")
            if oid:
                part["outcome_ids"] = [oid]
                notes.append(f"ربط {kind_ar} {part['id']} بـ{oid} (كلمات مشتركة: {', '.join(shared[:3])})." if ar else
                             f"{kind_en.capitalize()} {part['id']} was linked to {oid} "
                             f"(shared topic words: {', '.join(shared[:3])}).")

    capstone = curriculum.get("capstone")
    by_capstone = set(capstone.get("outcome_ids") or []) if isinstance(capstone, dict) else set()
    link(modules, _module_words, "module", "الوحدة", set())
    link(assessments, _assessment_words, "assessment", "التقييم", by_capstone)   # the capstone assesses too
    return notes
