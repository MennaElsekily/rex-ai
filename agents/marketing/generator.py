"""
Writes the strategy and each channel's variations with the language model and checks them (validation.py).

One model call per channel (and one for the strategy), each checked as soon as it is written and given ONE repair
attempt with the exact problems listed. A part that still fails ends the run: nothing is saved and the problem names
the channel. A small local model writes better, and fails more visibly, one short part at a time.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from agents.curriculum_research.llm import structured
from agents.marketing import prompts, schemas
from agents.marketing.facts import allowed_for, cta_facts_block, facts_block
from agents.marketing.validation import check_channel, check_strategy, supplied_texts


def _tell(progress: Optional[Callable[[int, int, str], None]], done: int, total: int, label: str) -> None:
    """Tell the caller how far the writing has got. A failing callback never stops the run."""
    if progress is not None:
        try:
            progress(done, total, label)
        except Exception:
            pass


def _problem_block(problems: Optional[list[dict[str, str]]]) -> str:
    if not problems:
        return ""
    return "\nYour previous answer had these problems. Fix them and answer again:\n" + "\n".join(f"- {p['message']}" for p in problems)


def _failed_numbers(problems: list[dict[str, str]]) -> Optional[set[int]]:
    """The 1-based numbers of the variations the problems point at (from their where, e.g. "ctas[2].text"); None when unknown."""
    found = {int(m.group(1)) for p in problems for m in [re.search(r"\[(\d+)\]", p.get("where", ""))] if m}
    return found or None


def generate_campaign(llm: Any, course: dict[str, Any], *, audience: str, audience_source: str, tone: str,
                      goal: Optional[str], notes: Optional[str], channels: list[str], count: int,
                      labels: dict[str, str], progress: Optional[Callable[[int, int, str], None]] = None
                      ) -> tuple[Optional[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    """
    (strategy, {channel: variations}, problems). problems is empty when everything passed (possibly after the one
    repair); otherwise the first part that failed is named and the rest is not written.
    course: the approved curriculum's content.
    """
    language = course["language"]
    supplied = [t for t in (goal, notes, audience if audience_source == "admin" else None) if t]
    allowed = allowed_for(course, supplied)
    facts = facts_block(course, audience, tone, goal, notes)
    base = {"language_name": prompts.LANGUAGE_NAMES[language], "tone": tone, "facts_block": facts}
    steps = ["strategy", *channels]
    written: dict[str, list[dict[str, Any]]] = {}

    def ask(prompt, schema, step: str, problems=None, note: str = "", **extra):
        values = {**base, "facts_block": cta_facts_block(course, audience, tone)} if step == "ctas" else base
        messages = prompt.format_messages(**values, problems_block=_problem_block(problems) + note, **extra)
        return structured(llm, schema, messages, f"marketing {step}")

    # strategy
    _tell(progress, 0, len(steps), labels.get("strategy", "strategy"))
    names = ", ".join(schemas.CHANNEL_LABELS[c][language] for c in channels)   # the names the model must use, in the course's language

    def write_strategy(problems=None) -> dict[str, Any]:
        data = ask(prompts.STRATEGY, schemas.CampaignStrategy, "strategy", problems, channel_names=names).model_dump()
        data["target_audience"] = audience              # the confirmed audience, never the model's
        data["recommended_channels"] = list(channels)   # the channels the administrator chose, set by code (not asked of the model)
        return data

    strategy = write_strategy()
    problems = check_strategy(strategy, allowed)
    if problems:
        strategy = write_strategy(problems)
        problems = check_strategy(strategy, allowed)
        if problems:
            return None, written, [{**p, "where": "strategy"} for p in problems]

    # channels
    for done, channel in enumerate(channels, start=1):
        _tell(progress, done, len(steps), labels.get(channel, channel))
        prompt, schema = prompts.channel_prompt(channel), schemas.CHANNEL_SCHEMAS[channel]

        def write_channel(problems=None, previous=None) -> list[dict[str, Any]]:
            # (a call to action's repair also explains what one is and quotes what was written; see prompts.cta_repair_note)
            note = prompts.cta_repair_note(previous, _failed_numbers(problems)) if problems and channel == "ctas" else ""
            return [v.model_dump() for v in ask(prompt, schema, channel, problems, note, count=count).variations]

        variations = write_channel()
        problems = check_channel(channel, variations, allowed, count)
        if problems:
            variations = write_channel(problems, previous=variations)
            problems = check_channel(channel, variations, allowed, count)
            if problems:
                return strategy, written, [{**p, "where": channel} for p in problems]
        written[channel] = variations
    _tell(progress, len(steps), len(steps), "")
    return strategy, written, []
