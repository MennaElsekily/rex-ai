"""
What a request asks FOR, read clause by clause, for the orchestrator (agents/orchestrator.py).

Keyword routing alone cannot tell the SUBJECT of a deliverable ("a curriculum about digital marketing") from a deliverable
("a marketing campaign"), and cannot see two deliverables in one sentence ("a question bank and a marketing campaign").
This module reads the request in three steps and returns the agents whose deliverable was actually requested, in the order
they should run (curriculum, then documents, then marketing):

1. The topic after a course noun ("curriculum about X", "course on X", "منهج عن X") is a subject, not a request: it is removed.
2. The text is split into clauses at "and", "then", "plus", commas and their Arabic equivalents.
3. Each clause is ONE deliverable, by priority: a supported document (Agent 4), else a marketing deliverable (Agent 5:
   campaign, social post, ad copy, LinkedIn post, WhatsApp message, headline, call to action...), else the creation of a
   curriculum or course (Agent 3). The bare word "marketing" is never a deliverable.

A clause that names a document type but has no verb of its own ("... and a course proposal") inherits the request from the
clause before it. The result is a rule-based reading, not understanding: anything it cannot place is left to the
orchestrator's other rules, and a request that matches nothing stays unrouted.
"""

from __future__ import annotations

import re
from typing import Callable

from agents.document_generation.understanding import find_document_type, is_document_request
from agents.text_utils import normalize_for_matching

CURRICULUM, DOCUMENT, MARKETING = "curriculum_research", "document_generation", "marketing"
ORDER = (CURRICULUM, DOCUMENT, MARKETING)

# "curriculum about X": X is the subject. It runs to the next conjunction or punctuation mark; the course noun is kept.
_TOPIC_EN = re.compile(
    r"\b(curriculum|curricula|syllabus|course outline|course|courses|program|programme|training|workshop)\s+"
    r"(?:about|on|covering|regarding|concerning|titled|called|focused on|focusing on|in)\s+"
    r".*?(?=\s+(?:and|then|plus|also|as well as)\b|\s*[,;.]|$)")
_TOPIC_AR = re.compile(
    r"(?<!\w)(منهج|مناهج|منهاج|دوره|دورات|برنامج)(?:\s+(?:تدريبي|دراسي))?\s+(?:عن|حول|في مجال|في)\s+"
    r".*?(?=\s+ثم\s|\s+و(?:حمله|اعلان|منشور|رساله|محتوي|نص)|\s*[,;.،؛]|$)")

_SPLIT = re.compile(r"\s*(?:,|;|،|؛|\band then\b|\bthen\b|\band\b|\bplus\b|\balso\b|\bas well as\b)\s+|\s+ثم\s+|\s+و(?=(?:انشئ|اكتب|جهز|حضر|اعد|اعمل|ولد|حمله|اعلان))")

_VERB = re.compile(
    r"\b(?:generate|create|make|produce|write|prepare|draft|build|compose|design|develop|need|want|give|get|plan)\b"
    r"|(?<!\w)(?:انشئ|انشاء|اكتب|جهز|حضر|اعد|اعمل|اريد|انتج|ولد|صمم|طور)(?!\w)")

_MARKETING = re.compile(
    r"\b(?:marketing|advertising|advertisement|promotional|social media|e-?mail|whatsapp|facebook|instagram|linkedin|twitter|ad)\s+"
    r"(?:campaigns?|content|copy|materials?|plans?|posts?|messages?|captions?|ads?|newsletters?|texts?|strategy|strategies|"
    r"headlines?|blurbs?)\b"
    r"|\bcampaigns?\b|\badvertisements?\b|\badverts?\b|\bad copy\b|\bcaptions?\b|\bheadlines?\b|\bcall[- ]to[- ]actions?\b"
    r"|\b(?:linkedin|facebook|instagram|twitter|tiktok)\s+(?:posts?|ads?|captions?)\b|\b(?:posts?|ads?|captions?)\s+(?:on|for)\s+(?:linkedin|facebook|instagram|twitter|tiktok|social media)\b|\bpromotional\s+(?:messages?|content|material|texts?|copy|posts?)\b"
    r"|\bsocial media\s+(?:posts?|captions?|content|campaigns?|ads?|copy|plans?|strategy|marketing)\b"
    r"|حمله\s+(?:تسويقيه|اعلانيه)|حملات\s+(?:تسويقيه|اعلانيه)|\bحمله\b|\bحملات\b|محتوي\s+تسويقي|نص(?:وص)?\s+تسويقي(?:ه)?|مواد\s+تسويقيه"
    r"|\bاعلان(?:ات)?\b|منشور(?:ات)?\s+(?:لينكد\w*|فيسبوك|انستغرام|تسويقي\w*|وسائل التواصل)|رسال(?:ه|ات)\s+(?:واتساب|ترويجيه)|بريد\s+تسويقي")

_CREATES_CURRICULUM = re.compile(
    r"\b(?:generate|create|make|build|design|develop|draft|write|prepare|produce|compose)\s+(?:me\s+|us\s+)?(?:an?\s+|the\s+)?"
    r"(?:[\w-]+\s+){0,3}?(?:curriculum|curricula|syllabus|course outline)\b"
    r"|(?<!\w)(?:انشئ|صمم|اعد|اكتب|جهز|اعمل|طور)\s+(?:\S+\s+){0,2}?(?:منهج|مناهج|منهاج)")


def has_marketing_deliverable(clause: str) -> bool:
    """Does this clause ask for marketing content (a campaign, a post, ad copy...)? The word "marketing" alone does not."""
    return bool(_MARKETING.search(normalize_for_matching(clause)))


def _clauses(text: str) -> list[str]:
    normalized = normalize_for_matching(text)
    keep_noun = lambda match: match.group(1)                      # the course noun stays, its subject goes
    normalized = _TOPIC_AR.sub(keep_noun, _TOPIC_EN.sub(keep_noun, normalized))
    return [c.strip() for c in _SPLIT.split(normalized) if c and c.strip()]


def deliverables(text: str, creates_course: Callable[[str], bool]) -> list[str]:
    """
    The agents whose deliverable the request names, in running order, without repeats. creates_course: the orchestrator's own
    "create a course" reader (Orchestrator._is_course_creation).
    """
    found: list[str] = []
    asked = False                                                  # an earlier clause had a request verb
    for clause in _clauses(text):
        verb = bool(_VERB.search(clause))
        document_type = find_document_type(clause)
        if document_type not in (None, "curriculum_pdf") and (is_document_request(clause) or (not verb and asked and
                                                                                           is_document_request("create " + clause))):
            agent = DOCUMENT
        elif has_marketing_deliverable(clause):
            agent = MARKETING
        elif _CREATES_CURRICULUM.search(clause) or creates_course(clause):
            agent = CURRICULUM
        else:
            agent = None
        if agent and agent not in found:
            found.append(agent)
        asked = asked or verb
    return [a for a in ORDER if a in found]
