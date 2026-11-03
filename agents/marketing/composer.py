"""
Builds the stored campaign (plain JSON) and its downloads (Markdown and JSON).

The campaign content::

    {"campaign_type": "marketing_campaign", "title", "course_title", "language", "curriculum_id", "curriculum_version",
     "audience": {"text", "source": "curriculum" | "admin"}, "tone", "goal", "notes", "variations": N,
     "strategy": {"objective", "target_audience", "main_message", "recommended_channels", "content_sequence",
                  "call_to_action"},
     "channels": {channel: {"variations": [{field: text, ...}, ...]}}}

Everything under "strategy" and "channels" is text an administrator may edit; the rest names the approved course version
the campaign was written from and cannot be edited.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agents.marketing import schemas

STRINGS = {
    "en": {"campaign": "Marketing campaign", "course": "Course", "audience": "Target audience", "tone": "Tone", "goal": "Goal",
           "notes": "Notes", "status": "Status", "strategy": "Campaign strategy", "objective": "Objective",
           "main_message": "Main message", "channels": "Recommended channels", "sequence": "Suggested content sequence",
           "cta": "Call to action", "variation": "Variation", "subject": "Subject", "preheader": "Preview text",
           "body": "Body", "headline": "Headline", "primary_text": "Primary text", "text": "Text", "message": "Message",
           "hashtags": "Hashtags", "approved": "Approved", "approved_on": "Approved on", "version": "Version",
           "source": "Built from the approved curriculum", "notice": "Approved marketing content. Nothing has been published or sent."},
    "ar": {"campaign": "حملة تسويقية", "course": "الدورة", "audience": "الجمهور المستهدف", "tone": "النبرة", "goal": "الهدف",
           "notes": "ملاحظات", "status": "الحالة", "strategy": "استراتيجية الحملة", "objective": "الهدف",
           "main_message": "الرسالة الرئيسية", "channels": "القنوات الموصى بها", "sequence": "تسلسل المحتوى المقترح",
           "cta": "الدعوة إلى اتخاذ إجراء", "variation": "الصيغة", "subject": "العنوان", "preheader": "نص المعاينة",
           "body": "النص", "headline": "العنوان الرئيسي", "primary_text": "النص الأساسي", "text": "النص", "message": "الرسالة",
           "hashtags": "الوسوم", "approved": "معتمد", "approved_on": "تاريخ الاعتماد", "version": "النسخة",
           "source": "مبنية على المنهج المعتمد", "notice": "محتوى تسويقي معتمد. لم يُنشر ولم يُرسل شيء."},
}


def compose(record: dict[str, Any], strategy: dict[str, Any], channels: dict[str, list[dict[str, Any]]], *, audience: str,
            audience_source: str, tone: str, goal: Optional[str], notes: Optional[str], variations: int) -> dict[str, Any]:
    """The campaign content for one approved curriculum record, from what the model wrote and the admin confirmed."""
    course = record["content"]
    ordered = {c: {"variations": [dict(v) for v in channels[c]]} for c in schemas.CHANNELS if c in channels}
    return {
        "campaign_type": "marketing_campaign",
        "title": course["title"],
        "course_title": course["title"],
        "language": course["language"],
        "curriculum_id": record["curriculum_id"],
        "curriculum_version": record["version"],
        "audience": {"text": audience, "source": audience_source},
        "tone": tone, "goal": goal or None, "notes": notes or None, "variations": variations,
        "strategy": {**strategy, "target_audience": audience},
        "channels": ordered,
    }


# ---------------------------------------------------------------------------------------------------- editing

def editable_paths(content: dict[str, Any]) -> list[tuple[Any, ...]]:
    """Every path of text an administrator may change: the strategy's text and each variation's fields."""
    paths: list[tuple[Any, ...]] = [("strategy", k) for k in ("objective", "main_message", "call_to_action")]
    paths += [("strategy", "content_sequence", i) for i in range(len(content["strategy"].get("content_sequence") or []))]
    for channel, body in content["channels"].items():
        for index, variation in enumerate(body["variations"]):
            paths += [("channels", channel, "variations", index, name) for name in schemas.LIMITS[channel]]
            if channel == "social_captions":
                paths.append(("channels", channel, "variations", index, "hashtags"))
    return paths


def apply_edit(content: dict[str, Any], path: list[Any], value: Any) -> None:
    """Set one editable text in place. Raises ValueError for any other path (the course facts cannot be edited)."""
    key = tuple(path)
    if key not in set(editable_paths(content)):
        raise ValueError("That part of the campaign cannot be edited: only the strategy text and the written variations can.")
    target = content
    for step in key[:-1]:
        target = target[step]
    if key[-1] == "hashtags":
        lines = value if isinstance(value, list) else str(value or "").replace(",", " ").split()
        target["hashtags"] = [str(t).strip() for t in lines if str(t).strip()]
    else:
        target[key[-1]] = " ".join(str(value or "").split())


# ---------------------------------------------------------------------------------------------------- downloads

def _field_label(s: dict[str, str], name: str) -> str:
    return s.get(name, name.replace("_", " "))


def to_markdown(content: dict[str, Any], *, version: Optional[int] = None, approved_at: Optional[str] = None) -> str:
    """Approved campaign as Markdown, grouped by channel (UTF-8)."""
    language = content["language"] if content["language"] in STRINGS else "en"
    s = STRINGS[language]
    out = [f"# {s['campaign']}: {content['course_title']}", "",
           f"- **{s['course']}:** {content['course_title']} ({content['curriculum_id']} v{content['curriculum_version']})",
           f"- **{s['audience']}:** {content['audience']['text']}",
           f"- **{s['tone']}:** {schemas.TONE_LABELS[content['tone']][language]}"]
    if content.get("goal"):
        out.append(f"- **{s['goal']}:** {content['goal']}")
    if content.get("notes"):
        out.append(f"- **{s['notes']}:** {content['notes']}")
    if version is not None:
        out.append(f"- **{s['version']}:** {version} ({s['approved']}{', ' + approved_at[:10] if approved_at else ''})")
    st = content["strategy"]
    out += ["", f"## {s['strategy']}", "", f"**{s['objective']}:** {st['objective']}", "",
            f"**{s['main_message']}:** {st['main_message']}", "",
            f"**{s['channels']}:** " + ", ".join(schemas.CHANNEL_LABELS[c][language] for c in st["recommended_channels"]), "",
            f"**{s['sequence']}:**"]
    out += [f"{i}. {step}" for i, step in enumerate(st["content_sequence"], start=1)]
    out += ["", f"**{s['cta']}:** {st['call_to_action']}"]
    for channel, body in content["channels"].items():
        out += ["", f"## {schemas.CHANNEL_LABELS[channel][language]}"]
        for number, variation in enumerate(body["variations"], start=1):
            out += ["", f"### {s['variation']} {number}", ""]
            for name in schemas.LIMITS[channel]:
                out.append(f"**{_field_label(s, name)}:** {variation[name]}  ")
            if variation.get("hashtags"):
                out.append(f"**{s['hashtags']}:** " + " ".join("#" + t.lstrip("#") for t in variation["hashtags"]))
    out += ["", "---", s["notice"], ""]
    return "\n".join(out)


def to_json(content: dict[str, Any], *, campaign_id: str, version: int, approved_at: Optional[str] = None) -> str:
    """Approved campaign as JSON for a future distribution step (UTF-8, Arabic kept as text)."""
    return json.dumps({"campaign_id": campaign_id, "version": version, "status": "approved", "approved_at": approved_at,
                       "content": content}, ensure_ascii=False, indent=2)
