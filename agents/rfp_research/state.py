"""
State for the RFP/Tender Research Agent's internal LangGraph workflow.

Separate from, and invisible to, the top-level REX AI orchestrator's
OrchestratorState (agents/orchestrator.py). The orchestrator only ever
sees this agent's final normalized response (see agent.py).
"""

from typing import Any, Optional, TypedDict

# Brief Section "Geographic Scope": the only two countries this agent
# is allowed to return opportunities for.
ALLOWED_COUNTRIES = ("Egypt", "Saudi Arabia")

# A tender is a training tender only if its TITLE (what is being bought)
# names training. The description is not enough: it often names the
# buyer, and a training organisation also buys archiving or IT services
# (a live search found "Documents and Archives Center ... archiving" from
# the "General Corporation for Technical and Vocational Training").
# Topic words alone ("software", "digital") are not training either.
TRAINING_TITLE_TERMS = (
    "training", "trainer", "course", "curriculum", "curricula", "capacity building",
    "capacity-building", "workshop", "learning", "upskilling", "reskilling", "bootcamp",
    "teaching", "teacher", "education", "educational", "tvet", "vocational",
    "skills development",
    "تدريب", "التدريب", "تدريبية", "تدريبي", "دورة", "دورات", "تعليم", "التعليم",
    "تعليمية", "تعلم", "التعلم", "مناهج", "بناء القدرات", "بناء قدرات", "ورشة", "ورش عمل",
    "تأهيل",
)


def names_training(title: str | None) -> bool:
    """Whether a tender's title says it is about training or education."""
    lowered = str(title or "").casefold()
    return any(term in lowered for term in TRAINING_TITLE_TERMS)


# Brief requirement 10: the training categories an opportunity is
# labelled with. Checked in this order, most specific first, so a
# software training tender is not labelled only "Education and Training".
TRAINING_CATEGORIES: dict[str, tuple[str, ...]] = {
    "Software and IT Training": (
        "software", "it training", "ict", "programming", "coding", "data analytics",
        "cyber", "cloud", "information technology", "برمجيات", "برمجة", "تقنية المعلومات",
        "الحاسب", "الأمن السيبراني",
    ),
    "Digital Transformation": (
        "digital transformation", "digitalization", "digitisation", "digitization",
        "e-government", "التحول الرقمي", "تحول رقمي", "رقمنة",
    ),
    "Technical Training": (
        "technical training", "vocational", "tvet", "technical and vocational",
        "التدريب الفني", "التدريب المهني", "تدريب فني", "تدريب مهني",
    ),
    "Corporate Learning": (
        "corporate learning", "corporate training", "e-learning", "elearning", "lms",
        "learning management", "staff training", "employee training", "تعلم إلكتروني",
        "التعليم الإلكتروني", "تدريب الموظفين",
    ),
    "Professional Development": (
        "professional development", "capacity building", "leadership", "upskilling",
        "skills development", "تطوير مهني", "بناء القدرات", "بناء قدرات", "تنمية المهارات",
    ),
    "Education and Training": (
        "education", "training", "teacher", "school", "university", "curriculum",
        "تعليم", "تدريب", "معلم", "مدارس", "جامعة", "مناهج",
    ),
}


def training_category(text: str) -> str | None:
    """The first (most specific) training category the text mentions."""
    lowered = str(text or "").casefold()
    for category, keywords in TRAINING_CATEGORIES.items():
        if any(keyword.casefold() in lowered for keyword in keywords):
            return category
    return None


class RFPResearchState(TypedDict, total=False):
    # Set by the agent before invoking the graph.
    request_text: str
    conversation_id: Optional[str]
    # "ar" or "en": the language user-facing messages are written in.
    language: str

    # Optional search details from the admin's search form (brief
    # section 10 questions): countries, categories, priority
    # (organisations/sectors), eligibility_notes, deadline_from,
    # deadline_to (ISO dates). Missing keys use the defaults.
    search: dict[str, Any]

    # Populated by web_research: live web results queued for admin
    # review (never used as approved knowledge until approved).
    web_research: dict[str, Any]

    # Populated by validate_request.
    requested_countries: list[str]
    out_of_scope_country: Optional[str]

    # Populated by query_apis: one APIQueryResult (as a dict) per
    # requested country's connector, so the UI can show exactly which
    # source was used and whether it actually succeeded.
    api_results: list[dict[str, Any]]
    candidate_records_raw: list[dict[str, Any]]

    # Populated by filter_relevant: NormalizedTender dicts that pass a
    # lightweight keyword pre-check (saves LLM calls on obviously
    # unrelated API records before the real relevance judgment), and how
    # many were dropped at each check (shown in the reply).
    candidate_records: list[dict[str, Any]]
    filter_report: dict[str, int]
    # Populated by assess_and_filter: tenders the relevance check rejected.
    not_relevant_count: int

    # Populated by detect_duplicates (against this conversation's
    # previously seen source_ids, via ConversationStore).
    new_records: list[dict[str, Any]]
    duplicate_count: int

    # Populated by assess_and_filter (LLM relevance + qualification).
    assessed_results: list[dict[str, Any]]

    # Populated by apply_business_rule and store_opportunities.
    final_results: list[dict[str, Any]]
    newly_discovered_count: int
    # Set when save_discovered_tenders fails: the reply must then not say
    # the tenders are waiting for review.
    storage_error: str

    # Populated by compose_response.
    message: str
    response: dict[str, Any]
