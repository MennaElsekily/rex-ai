"""
LangChain prompts for Agent 5, kept apart from the schemas (schemas.py).

Every prompt shows the model the approved course facts and the confirmed audience, and tells it to add nothing; the
answer is checked again by validation.py.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

LANGUAGE_NAMES = {"en": "English", "ar": "Arabic (Modern Standard Arabic)"}

_RULES = """You write marketing content for an academy's course, from the approved course facts you are given.
Rules:
- Write every text value in {language_name}. Keep widely used technical terms (for example Python or SQL) as they are,
  and never put English letters inside an Arabic word.
- Use ONLY the approved course facts below. Never invent or imply: prices, fees, costs, "free", discounts, offers,
  urgency ("limited seats", "last chance"), dates, deadlines, links, e-mail addresses, phone numbers, statistics,
  awards, certificates, accreditation, instructors, or job, salary or promotion promises.
- Never guarantee a result and never use "best", "#1" or similar claims. Describe what learners will learn and practise,
  using the course's own outcomes, modules and topics.
- Use only numbers the course states (its hours, its number of modules or outcomes).
- Speak to the confirmed target audience, in a {tone} tone. Be specific to this course; no generic filler.
- If the course states no prerequisites you may say so; otherwise do not say that no experience is needed."""

_VARIATIONS = ("Write exactly {count} variation(s). Each must take a different angle (for example what learners will be "
               "able to do, hands-on practice, who the course is for) and use different wording; never repeat a sentence.")

CHANNEL_TASKS = {
    "social_captions": "Write social media captions: each 30 to 280 characters, with at most 5 short hashtags (one word, "
                       "without the # sign).",
    "ad_copy": "Write Facebook / Instagram ad copy: a headline (at most 60 characters), the primary text (40 to 300 "
               "characters) and the button text (at most 30 characters).",
    "linkedin_posts": "Write LinkedIn posts: professional, 80 to 900 characters, one clear idea about the course, no hashtags spam.",
    "email": "Write marketing e-mails: a subject (at most 70 characters), preview text (at most 100), a body of 80 to 1200 "
             "characters in short paragraphs, and the call-to-action button text (at most 40 characters). No links.",
    "whatsapp": "Write WhatsApp promotional messages: friendly and short (20 to 450 characters), one clear next step, no links.",
    "headlines": "Write campaign headlines: each at most 70 characters.",
    "ctas": "Write calls to action: short, imperative, action-oriented phrases of at most 8 words and at most 60 characters "
            "(aim for 3 to 6 words, well under the limits), each starting with a verb, for example \"Explore the course\", \"Join the first module\" or \"See the full outline\". "
            "Do not write sentences that describe the course or its benefits, and do not end with a full stop. "
            "No urgency or discount.",
}


def _prompt(task: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", _RULES + "\n" + task),
        ("human", "Approved course facts:\n{facts_block}\n{problems_block}"),
    ])


STRATEGY = _prompt(
    "Recommend a simple campaign strategy: the objective (one sentence), the main message (one or two sentences, from the "
    "facts only), a content_sequence of three to six short steps (what to publish first, next and last, for the channels "
    "listed under 'Channels for this campaign'; call them by the names given there, in the campaign's language), and the main "
    "call to action (at most 8 words). No budgets, forecasts, dates or guaranteed results. "
    "Channels for this campaign: {channel_names}")


def channel_prompt(channel: str) -> ChatPromptTemplate:
    return _prompt(CHANNEL_TASKS[channel] + "\n" + _VARIATIONS)


def more_variations_note(written: list[dict]) -> str:
    """
    Added when the model wrote fewer variations than asked (the local model often writes one long e-mail when asked for
    two): it is asked for the missing ones only, and shown what is already written so it takes a different angle.
    """
    def first_line(v: dict) -> str:
        text = next((str(v[k]) for k in ("subject", "headline", "text", "message", "body", "primary_text") if v.get(k)), "")
        return text[:160]

    quoted = "\n".join(f"  - \"{first_line(v)}\"" for v in written)
    return ("\nThese variations are already written; do not repeat them, their wording or their angle:\n" + quoted)


def cta_repair_note(previous: list[dict], invalid: Optional[set[int]] = None) -> str:
    """
    Added to the repair request of the call-to-action suggestions only: explains what a call to action is and quotes what
    the model wrote, so it can see that it wrote descriptive sentences instead of short action phrases. invalid: the
    1-based numbers of the suggestions that failed (None: all); the others are quoted as ones to keep unchanged.
    """
    def line(n: int, v: dict) -> str:
        text = str(v.get("text", ""))
        return f"  {n}. \"{text}\" ({len(text.split())} words, {len(text)} characters)"

    entries = list(enumerate(previous or [], start=1))
    wrong = [(n, v) for n, v in entries if invalid is None or n in invalid] or entries
    right = [(n, v) for n, v in entries if (n, v) not in wrong]
    quoted = "\n".join(line(n, v) for n, v in wrong)
    keep = ("\nThese suggestions are fine; keep them exactly as they are, in the same position:\n"
            + "\n".join(line(n, v) for n, v in right)) if right else ""
    return (
        "\nA call to action is a short, imperative, action-oriented phrase that tells the reader what to do next, such as "
        "\"Explore the course\", \"Join the first module\" or \"See the full outline\". It has at most "
        "8 words and at most 60 characters, starts with a verb and has no full stop. It is not a sentence that describes "
        "the course or what learners will gain.\n"
        "This is what you wrote (these failed the checks):\n" + quoted + keep + "\nRewrite each one as a short action phrase of that kind.")
