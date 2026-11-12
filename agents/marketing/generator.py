"""
Writes the strategy and each channel's variations with the language model and checks them (validation.py).

One model call per channel (and one for the strategy), each checked as soon as it is written and given ONE repair
attempt with the exact problems listed. When the model writes fewer variations than asked, it is asked for the missing
ones only (it is shown the ones already written); the whole set is then checked exactly as before. A part that still fails ends the run: nothing is saved and the problem names
the channel. A small local model writes better, and fails more visibly, one short part at a time.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from agents.curriculum_research.llm import CurriculumModelError, structured
from agents.marketing import prompts, schemas
from agents.marketing.facts import allowed_for, cta_facts_block, facts_block
from agents.marketing.length_repair import repair_channel_lengths, repair_strategy_lengths
from agents.marketing.sentence_repair import repair_channel_sentences, repair_strategy_sentences
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


def _replace_failing_variations(ask: Callable[..., Any], prompt: Any, schema: Any, channel: str,
                                variations: list[dict[str, Any]], problems: list[dict[str, str]], count: int,
                                allowed: Any) -> list[dict[str, Any]]:
    """
    The variations with the failing ones (every problem names its variation, and at least one variation passes)
    written again in one model call; otherwise, or if the model fails, the variations unchanged. Never accepts
    anything by itself: the caller checks the whole set again.
    """
    failing = _failed_numbers(problems)
    if not failing or len(problems) != sum(1 for p in problems if re.search(r"\[\d+\]", p.get("where", ""))) \
            or len(failing) >= len(variations):
        return variations
    good = [v for number, v in enumerate(variations, start=1) if number not in failing]
    try:
        fresh = [v.model_dump() for v in ask(prompt, schema, channel, None, prompts.more_variations_note(good),
                                             count=len(failing)).variations]
    except CurriculumModelError:
        return variations
    if len(fresh) < len(failing):
        return variations
    fresh = repair_channel_sentences(channel, fresh, allowed)
    replaced, new = [], iter(fresh)
    for number, variation in enumerate(variations, start=1):
        replaced.append(next(new) if number in failing else variation)
    return replaced


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
            # A flagged sentence the model's repair kept (an unsupported claim, another script): removed, never
            # reworded; then everything is checked again (sentence_repair.py).
            strategy = repair_strategy_sentences(strategy, allowed)
            problems = check_strategy(strategy, allowed)
        if problems:
            # A length-only problem the model's own repair did not fix (it tends to
            # repeat something close to the same length): a deterministic trim,
            # never inventing text, is tried before giving up on the whole campaign.
            strategy = repair_strategy_lengths(strategy, problems)
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
            written = [v.model_dump() for v in ask(prompt, schema, channel, problems, note, count=count).variations]
            if 0 < len(written) < count:
                # Too few (not a content problem): ask for the missing ones only. check_channel still
                # requires exactly `count` distinct, valid variations.
                more = ask(prompt, schema, channel, None, prompts.more_variations_note(written), count=count - len(written))
                written += [v.model_dump() for v in more.variations][:count - len(written)]
            return written

        def write_one_at_a_time() -> list[dict[str, Any]]:
            # Real model (2026-11-12): asked for two Arabic ad copies in one answer, it looped until Ollama aborted
            # ("token repeat limit reached"), the same way on the retry. One short answer at a time does not.
            out: list[dict[str, Any]] = []
            for _ in range(count):
                note = prompts.more_variations_note(out) if out else ""
                out += [v.model_dump() for v in ask(prompt, schema, channel, None, note, count=1).variations][:1]
            return out

        try:
            variations = write_channel()
        except CurriculumModelError:
            variations = write_one_at_a_time()            # still failing: the error is reported as before
        problems = check_channel(channel, variations, allowed, count)
        if problems:
            try:
                variations = write_channel(problems, previous=variations)
            except CurriculumModelError:
                pass                                      # the repair is a second chance: keep what was written
            problems = check_channel(channel, variations, allowed, count)
            if problems:
                # The flagged sentences removed (never reworded), then everything checked again.
                variations = repair_channel_sentences(channel, variations, allowed)
                problems = check_channel(channel, variations, allowed, count)
            if problems:
                # Same deterministic, never-inventing length fix as the strategy's,
                # only applied when every remaining problem is a length problem
                # (a claim, count or structure problem still fails the campaign).
                variations = repair_channel_lengths(channel, variations, problems, allowed, language)
                problems = check_channel(channel, variations, allowed, count)
            if problems:
                # Last attempt when only SOME variations still fail (a claim in a one-line subject, two variations too
                # alike): those are written again, the model shown the good ones so it takes another angle. The whole
                # set is checked again by the same rules; anything still wrong refuses the campaign as before.
                variations = _replace_failing_variations(ask, prompt, schema, channel, variations, problems, count, allowed)
                problems = check_channel(channel, variations, allowed, count)
            if problems:
                return strategy, written, [{**p, "where": channel} for p in problems]
        written[channel] = variations
    _tell(progress, len(steps), len(steps), "")
    return strategy, written, []
