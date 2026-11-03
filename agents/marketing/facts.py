"""
The approved course facts Agent 5 may use, read from one approved curriculum version (Agent 3's content, unchanged).

Nothing here invents anything: ``course_facts`` only selects fields the curriculum records, ``Allowed`` is the set of
numbers, words and hours that a campaign text may therefore mention (plus what the administrator typed into the
request), and ``facts_block`` is the plain-text version of the same facts that the prompts show the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from agents.text_utils import normalize_for_matching

# Lowercase English words that may appear as they are inside Arabic text (tools, libraries, standards). Capitalised,
# upper-case, hyphenated or digit-containing words are never checked, and words the curriculum itself uses are always
# allowed; add a term here to allow it everywhere.
TECHNICAL_TERMS = frozenset("""
python java javascript typescript sql nosql html css json xml csv api rest git github docker linux excel powerpoint
tableau powerbi spark hadoop tensorflow pytorch keras scikit sklearn pandas numpy scipy matplotlib seaborn jupyter
colab aws azure gcp cloud ai ml nlp gpu cpu http https url ide vscode react node django flask streamlit mongodb mysql
postgresql sqlite etl bi kpi roi ui ux devops pdf
""".split())

_NUMBER = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?(?![\w])")


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def numbers_in(text: str) -> set[float]:
    """The numbers written in a text (Arabic-Indic digits included), as floats."""
    found = set()
    for token in _NUMBER.findall(normalize_for_matching(text)):
        try:
            found.add(float(token.replace(",", ".")))
        except ValueError:
            pass
    return found


@dataclass
class Allowed:
    """What a campaign text may mention: only the approved course and what the administrator supplied."""
    language: str
    hours: set[float] = field(default_factory=set)
    numbers: set[float] = field(default_factory=set)
    text: str = ""                          # normalized text of the curriculum and the administrator's notes
    latin_words: set[str] = field(default_factory=set)

    def mentions(self, term: str) -> bool:
        return normalize_for_matching(term).strip() in self.text


def allowed_for(content: dict[str, Any], supplied: Optional[list[str]] = None) -> Allowed:
    """
    content: an approved curriculum's content. supplied: texts the administrator typed (goal, notes), which are a
    legitimate source of facts such as a start date or a link and so allow the numbers and terms they contain.
    """
    texts = [*_strings(content), *(supplied or [])]
    joined = " \n ".join(texts)
    numbers = numbers_in(joined)
    hours = {float(content["total_hours"])} if isinstance(content.get("total_hours"), (int, float)) else set()
    hours |= {float(m["hours"]) for m in content.get("modules") or [] if isinstance(m.get("hours"), (int, float))}
    numbers |= hours
    for key in ("modules", "learning_outcomes", "objectives", "assessments", "prerequisites"):
        if isinstance(content.get(key), list) and content[key]:
            numbers.add(float(len(content[key])))
    return Allowed(language=content.get("language") or "en", hours=hours, numbers=numbers,
                   text=normalize_for_matching(joined),
                   latin_words={w.lower() for w in re.findall(r"[A-Za-z]{3,}", joined)})


def course_facts(content: dict[str, Any]) -> dict[str, Any]:
    """The fields of the approved curriculum a campaign is built from."""
    return {
        "title": content.get("title"), "level": content.get("level"), "total_hours": content.get("total_hours"),
        "target_audience": (content.get("target_audience") or "").strip() or None,
        "overview": content.get("overview"), "rationale": content.get("rationale"),
        "objectives": list(content.get("objectives") or []),
        "prerequisites": list(content.get("prerequisites") or []),
        "outcomes": [o.get("statement") for o in content.get("learning_outcomes") or [] if o.get("statement")],
        "modules": [{"title": m.get("title"), "hours": m.get("hours"), "topics": list(m.get("topics") or [])}
                    for m in content.get("modules") or []],
        "tools": list(content.get("tools_resources") or []),
    }


def cta_facts_block(content: dict[str, Any], audience: str, tone: str) -> str:
    """
    A short fact block for the call-to-action step only: the course's name, level, confirmed audience and tone. A call to
    action does not describe the course, and the full block (overview, objectives, outcomes, modules) pulls a small model
    into writing descriptive sentences. The claims checks still use the whole course (generator.allowed_for).
    """
    f = course_facts(content)
    return "\n".join([f"Course: {f['title']}", f"Level: {f['level']}", f"Target audience (confirmed): {audience}", f"Tone: {tone}"])


def facts_block(content: dict[str, Any], audience: str, tone: str, goal: Optional[str], notes: Optional[str]) -> str:
    """The approved facts as plain text for a prompt, with the confirmed audience, tone, goal and notes."""
    f = course_facts(content)
    lines = [f"Course: {f['title']}", f"Level: {f['level']}", f"Total hours: {f['total_hours']}",
             f"Target audience (confirmed): {audience}"]
    for label, key in (("Overview", "overview"), ("Why the course exists", "rationale")):
        if f[key]:
            lines.append(f"{label}: {f[key]}")
    if f["objectives"]:
        lines += ["Objectives:", *[f"- {o}" for o in f["objectives"]]]
    if f["outcomes"]:
        lines += ["Learning outcomes:", *[f"- {o}" for o in f["outcomes"]]]
    if f["modules"]:
        lines += ["Modules:", *[f"- {m['title']} ({m['hours']} hours): {'; '.join(m['topics'])}" for m in f["modules"]]]
    lines.append("Prerequisites: " + ("; ".join(f["prerequisites"]) if f["prerequisites"] else "none stated"))
    if f["tools"]:
        lines.append("Tools: " + ", ".join(f["tools"]))
    lines.append(f"Tone: {tone}")
    if goal:
        lines.append(f"Campaign goal (from the administrator): {goal}")
    if notes:
        lines.append(f"Extra facts supplied by the administrator: {notes}")
    return "\n".join(lines)
