"""
LangGraph workflow for the RFP/Tender Research Agent (Agent 2).

    validate_request    countries (search form or request text); refuse
                        countries outside Egypt / Saudi Arabia
      -> query_apis     World Bank API (Egypt); Etimad (Saudi: no public API,
                        reported as unavailable, never as "no tenders")
      -> web_research   web search for both countries: official portals,
                        tender sites, open web (web_tenders.py); the tender
                        fields are extracted from each page
      -> filter_relevant  category, expired, deadline range
      -> detect_duplicates  this conversation + the stored catalogue
      -> assess_and_filter  relevance, then qualification (two LLM chains)
      -> apply_business_rule  Egypt: flagged for immediate review;
                        Saudi Arabia: monitored pipeline
      -> store_opportunities  saved as PENDING for admin approval
      -> compose_response

Every tender, from the API or the web, waits for an admin decision
(agents/admin_service.py); approved ones form the searchable dashboard.
A source that fails or is unavailable is reported as such, never as
"zero tenders found".

Sources and the tender catalogue are reached only through the tender
tools (agents/tender_mcp_server.py, called with TenderMcpClient; tests
use the in-process tender_tools.LocalTenderTools): query_apis,
web_research, detect_duplicates and store_opportunities each call one
tool. The rest -- the LLM chains, filters, business rule and reply --
is this agent's own reasoning and stays here.

This graph is internal to the RFP agent. The top-level REX AI
orchestrator (agents/orchestrator.py) never sees these nodes or this
state -- it only calls RFPResearchAgent.process_request() (agent.py).
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any, Optional, Protocol
from urllib.parse import urlparse

from langgraph.graph import END, StateGraph

from agents.conversation_store import ConversationStore
from agents.rfp_research import web_tenders
from agents.rfp_research.api_clients.tavily import WebPage
from agents.rfp_research.opportunity_store import (
    FLAGGED,
    MONITORED,
    dedupe_key,
)
from agents.rfp_research.state import (
    ALLOWED_COUNTRIES,
    TRAINING_CATEGORIES,
    RFPResearchState,
    names_training,
    training_category,
)
from agents.rfp_research.tools import (
    build_extraction_chain,
    build_qualification_chain,
    build_relevance_chain,
)
from agents.text_utils import normalize_for_matching

logger = logging.getLogger(__name__)

RFP_AGENT_NAME = "rfp_research"


class TenderTools(Protocol):
    """
    The four tender tools the graph uses: TenderMcpClient (over MCP) or
    tender_tools.LocalTenderTools (in-process). See tender_tools.py.
    """

    def search_procurement_notices(self, country: str, query: str) -> dict[str, Any]: ...
    def search_tender_web_pages(self, countries, categories=None, priority=None) -> dict[str, Any]: ...
    def find_known_tenders(self, tenders: list[dict[str, Any]]) -> dict[str, Any]: ...
    def save_discovered_tenders(self, tenders: list[dict[str, Any]]) -> dict[str, Any]: ...


def _tool_failure(name: str, exc: Exception) -> str:
    logger.warning("Tender tool %s failed: %s", name, exc)
    return f"The {name} tool could not be reached ({type(exc).__name__})."

CATEGORY_QUERY = "training"
# Confirmed via a direct API check during development: the World Bank
# API's qterm appears to match ALL space-separated terms together
# (AND semantics), so a long compound query like "education training
# technical IT software digital skills professional development"
# returns zero results even though each term alone matches real
# records. A short, broad term plus client-side keyword relevance
# filtering (_is_training_tender below) and the LLM relevance
# judgment in tools.py together do the real category narrowing.


def _whole_word(pattern: str) -> re.Pattern[str]:
    # Latin-script names must match whole words: "usa" is not in
    # "causal"/"thousand" and "uk" is not in "Duke".
    return re.compile(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])")


_DISALLOWED_COUNTRY_PATTERNS = (
    ("usa", _whole_word(r"usa|united states|u\.s\.(?:a\.?)?")),
    ("europe", _whole_word(r"europe(?:an)?|eu")),
    ("uk", _whole_word(r"uk|united kingdom|britain")),
    ("germany", _whole_word("germany")),
    ("france", _whole_word("france")),
    ("canada", _whole_word("canada")),
    ("australia", _whole_word("australia")),
    ("spain", _whole_word("spain")),
    ("italy", _whole_word("italy")),
)

# Arabic names are matched as substrings of the normalized text, since
# prefixes attach to the word ("بألمانيا" = "in Germany").
_DISALLOWED_COUNTRY_NAMES_AR = (
    ("usa", "امريكا"), ("usa", "الولايات المتحده"), ("europe", "اوروبا"),
    ("uk", "بريطانيا"), ("germany", "المانيا"), ("france", "فرنسا"),
    ("canada", "كندا"), ("australia", "استراليا"), ("spain", "اسبانيا"),
    ("italy", "ايطاليا"),
)


def _name_or_domain(names: str, tld: str) -> re.Pattern[str]:
    # The top-level domain is preceded by a letter ("gov.eg"), so it
    # cannot go through _whole_word's left boundary.
    return re.compile(rf"{_whole_word(names).pattern}|\.{tld}(?![a-z0-9.])")


_COUNTRY_PATTERNS = {
    "Egypt": (_name_or_domain(r"egypt\w*", "eg"), ("مصر",)),
    "Saudi Arabia": (
        _name_or_domain(r"saudi\w*|ksa", "sa"), ("السعودية", "سعودية"),
    ),
}


def _detect_requested_countries(text: str) -> list[str]:
    normalized = text.casefold()

    return [
        country
        for country, (latin, arabic) in _COUNTRY_PATTERNS.items()
        if latin.search(normalized) or any(name in normalized for name in arabic)
    ]


def _detect_out_of_scope_country(text: str) -> Optional[str]:
    normalized = text.casefold()

    for label, pattern in _DISALLOWED_COUNTRY_PATTERNS:
        if pattern.search(normalized):
            return label

    arabic = normalize_for_matching(text)
    for label, name in _DISALLOWED_COUNTRY_NAMES_AR:
        if name in arabic:
            return label

    return None


def _is_training_tender(record: dict[str, Any]) -> bool:
    """
    The title must name training (state.names_training). Never the
    category label: for web tenders that label is our own (e.g. "Software
    and IT Training"), so checking it would let every tender through.
    """
    return names_training(record.get("title"))


# Works and supply contracts for a school or university are not training
# tenders. The 3B relevance model sometimes accepts them because the title
# names a university, so code drops them unless the title itself names
# training. "teaching" alone is not enough ("housing for teaching staff").
_WORKS_TERMS = (
    "construction", "revamp", "renovation", "rehabilitation", "civil work", "building work",
    "housing", "maintenance", "maintaining", "cleaning", "equipping", "furniture", "supply of",
    "electrical", "plumbing", "facilities management", "facility management",
    "إنشاء", "تشييد", "صيانة", "توريد", "ترميم", "نظافة",
)
_TRAINING_TERMS = (
    "training", "course", "learning", "capacity building", "curriculum", "workshop",
    "تدريب", "دورة", "دورات", "تعلم",
)

# The name of an organisation or place inside a title ("... at the
# Education and Training Evaluation Authority", "... للمؤسسة العامة للتدريب
# التقني"). A training word there names the buyer or the venue, not the
# work, so it must not turn a works contract into a training tender. A
# live search found "Operating, Maintaining And Cleaning The Computerized
# Testing Facilities At The Education And Training Evaluation Authority".
_PREPOSITION = r"(?:at|for|of|in|by|to|from|on)"
_INSTITUTION_EN = re.compile(
    rf"\b{_PREPOSITION}\s+(?:the\s+)?(?:(?!{_PREPOSITION}\b)[\w'’&.-]+\s+){{0,8}}?"
    r"(?:authority|cent(?:er|re)|institute|corporation|university|college|academy|ministry|directorate|"
    r"agency|organi[sz]ation|council|commission|foundation|hospital|school|faculty|department|bureau|"
    r"company|establishment)s?\b",
    re.IGNORECASE,
)
_INSTITUTION_AR = re.compile(
    r"(?:^|\s)و?(?:ب|ل|لل)?(?:ال)?(?:مؤسسة|هيئة|مركز|معهد|جامعة|وزارة|كلية|أكاديمية|اكاديمية|مديرية|إدارة|ادارة)"
    r"\S*(?:\s+\S+){0,6}"
)


def _without_institution_names(title: str) -> str:
    return _INSTITUTION_AR.sub(" ", _INSTITUTION_EN.sub(" ", title))


# Renting or leasing is never a training service, even when the title
# names a training centre (a live search found "Renting 3 rooms ... in the
# Technical Training and Skills Development Center").
_RENTAL_TERMS = ("renting", "rental", "lease of", "leasing", "rent of", "إيجار", "تأجير", "استئجار")


def _is_works_contract(record: dict[str, Any]) -> bool:
    """
    A works, supply or rental contract. A training word in the title
    overrides a works word ("Supply of medical training courses") -- but
    not when the training word is only part of an organisation's or a
    place's name ("Cleaning ... at the Education and Training ... Authority").
    """
    title = str(record.get("title") or "").casefold()
    if any(term in title for term in _RENTAL_TERMS):
        return True
    work = _without_institution_names(title)
    return any(term in title for term in _WORKS_TERMS) and not any(
        term in work for term in _TRAINING_TERMS
    )


def _is_expired(deadline: Optional[str]) -> bool:
    if not deadline:
        return False

    try:
        deadline_date = datetime.fromisoformat(str(deadline)).date()
    except ValueError:
        return web_tenders.is_expired(deadline)

    return deadline_date < date.today()


def _search_options(state: RFPResearchState) -> dict[str, Any]:
    return dict(state.get("search") or {})


def _selected_categories(options: dict[str, Any]) -> list[str]:
    return [c for c in options.get("categories") or [] if c in TRAINING_CATEGORIES]


def _priority_terms(options: dict[str, Any]) -> list[str]:
    raw = str(options.get("priority") or "")
    return [term.strip().casefold() for term in re.split(r"[,;\n]+", raw) if len(term.strip()) >= 3]


# ---------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------

def make_validate_request_node():
    def validate_request_node(state: RFPResearchState) -> dict[str, Any]:
        text = state.get("request_text", "")
        chosen = [c for c in _search_options(state).get("countries") or [] if c in ALLOWED_COUNTRIES]

        return {
            "requested_countries": (
                chosen or _detect_requested_countries(text) or list(ALLOWED_COUNTRIES)
            ),
            "out_of_scope_country": _detect_out_of_scope_country(text),
        }

    return validate_request_node


def route_after_validation(state: RFPResearchState) -> str:
    return "out_of_scope" if state.get("out_of_scope_country") else "query"


def make_query_apis_node(tools: TenderTools):
    def query_apis_node(state: RFPResearchState) -> dict[str, Any]:
        countries = state.get("requested_countries", list(ALLOWED_COUNTRIES))
        api_results: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []

        # Tool search_procurement_notices: World Bank (Egypt) and Etimad
        # (Saudi Arabia: status "unavailable", no records, by design --
        # never reported as "zero tenders found"; Saudi tenders come from
        # web research instead).
        for country in ("Egypt", "Saudi Arabia"):
            if country not in countries:
                continue
            try:
                result = tools.search_procurement_notices(country, CATEGORY_QUERY)
            except Exception as exc:
                result = {"source_name": f"Procurement source ({country})", "status": "error",
                          "message": _tool_failure("search_procurement_notices", exc), "records": []}
            api_results.append(result)
            if result.get("status") == "success":
                candidates.extend(result.get("records") or [])

        return {"api_results": api_results, "candidate_records_raw": candidates}

    return query_apis_node


def make_web_research_node(model: str, tools: TenderTools, enabled: bool = True):
    def web_research_node(state: RFPResearchState) -> dict[str, Any]:
        """
        Web research for every requested country (web_tenders.py): pages
        from official portals, tender sites and the open web; the brief's
        tender fields are extracted from each page. The records join the
        API records; each search is reported as a source.
        """
        if not enabled:
            return {"web_research": {"status": "disabled", "pages": 0, "records": 0}}

        options = _search_options(state)
        countries = state.get("requested_countries", list(ALLOWED_COUNTRIES))
        # Tool search_tender_web_pages returns the pages; reading the
        # tenders out of them (the extraction chain) is this agent's job.
        try:
            found = tools.search_tender_web_pages(
                countries, _selected_categories(options), options.get("priority") or None,
            )
        except Exception as exc:
            found = {"status": "error", "pages": [], "sources": [{
                "source_name": "Web search", "status": "error",
                "message": _tool_failure("search_tender_web_pages", exc)}]}
        reports = found.get("sources") or [
            {"source_name": "Web search", "status": found.get("status", "error"),
             "message": found.get("message", "")}]
        pages = found.get("pages") or []

        records: list[dict[str, Any]] = []
        # Pages without a readable tender name (a URL set: each page once).
        # Kept apart from filter_report: these never reached the filters.
        unreadable: set[str] = set()
        if pages:
            extraction = build_extraction_chain(model)
            for page in pages:
                web_page = WebPage.model_validate(
                    {k: v for k, v in page.items() if k in WebPage.model_fields})
                records.extend(web_tenders.page_to_records(
                    web_page, page["country"], extraction, _detect_requested_countries,
                    unreadable_pages=unreadable,
                ))

        return {
            "api_results": [*state.get("api_results", []), *reports],
            "candidate_records_raw": [*state.get("candidate_records_raw", []), *records],
            "web_research": {
                "status": found.get("status", "error"),
                "pages": len(pages),
                "records": len(records),
                "unreadable_pages": len(unreadable),
                "unreadable_from_etimad": sum(1 for url in unreadable if "etimad.sa" in urlparse(url).netloc),
            },
        }

    return web_research_node


def make_filter_relevant_node():
    def filter_relevant_node(state: RFPResearchState) -> dict[str, Any]:
        """
        Keeps current education/training tenders: drops expired ones
        (and old notices with no deadline, web_tenders.is_stale), works
        and supply contracts, those outside the admin's deadline range,
        and those outside the chosen
        categories. Each kept record gets its training category.
        """
        options = _search_options(state)
        categories = _selected_categories(options)
        candidates = []
        # Why each tender was dropped (first failing check), for the reply.
        report = {"read": 0, "expired": 0, "not_training": 0, "works": 0,
                  "outside_dates": 0, "outside_categories": 0}

        for record in state.get("candidate_records_raw", []):
            report["read"] += 1
            if _is_expired(record.get("deadline")) or web_tenders.is_stale(
                record.get("deadline"), record.get("publication_date")
            ):
                report["expired"] += 1
                continue
            if not _is_training_tender(record):
                report["not_training"] += 1
                continue
            if _is_works_contract(record):
                report["works"] += 1
                continue
            if not web_tenders.within_deadline_range(
                record.get("deadline"), options.get("deadline_from"), options.get("deadline_to")
            ):
                report["outside_dates"] += 1
                continue
            category = record.get("training_category") or training_category(
                f"{record.get('title') or ''} {record.get('scope_of_work') or ''}"
            )
            if categories and category not in categories:
                report["outside_categories"] += 1
                continue
            candidates.append({**record, "training_category": category})

        return {"candidate_records": candidates, "filter_report": report}

    return filter_relevant_node


def route_after_filter(state: RFPResearchState) -> str:
    any_source_succeeded = any(
        result.get("status") == "success"
        for result in state.get("api_results", [])
    )

    if not any_source_succeeded:
        return "no_api_data"

    return "dedup" if state.get("candidate_records") else "no_results"


def make_detect_duplicates_node(
    conversation_store: Optional[ConversationStore],
    tools: TenderTools,
):
    def detect_duplicates_node(state: RFPResearchState) -> dict[str, Any]:
        """
        Duplicates (requirement 13): seen earlier in this conversation,
        the same URL twice in this run, or already in the catalogue under
        any URL (same tender on another site: tool find_known_tenders).
        If the catalogue check fails, nothing is dropped here; saving
        still refuses duplicates (OpportunityStore.save).
        """
        candidates = state.get("candidate_records", [])
        conversation_id = state.get("conversation_id")

        try:
            known = tools.find_known_tenders(candidates)
            in_catalogue = {item["index"] for item in known.get("known") or []}
        except Exception as exc:
            _tool_failure("find_known_tenders", exc)
            in_catalogue = set()

        seen_ids: set[str] = set()

        if conversation_store is not None and conversation_id:
            previous = conversation_store.get(conversation_id, RFP_AGENT_NAME)

            if previous and isinstance(previous.get("seen_source_ids"), list):
                seen_ids = set(previous["seen_source_ids"])

        new_records = []
        duplicate_count = 0
        batch_seen: set[str] = set()
        batch_titles: set[str] = set()

        for index, record in enumerate(candidates):
            key = f"{record.get('source_name')}::{record.get('source_id')}"
            title_key = dedupe_key(record.get("title"), record.get("country")) or key

            if (
                key in seen_ids or key in batch_seen or title_key in batch_titles
                or index in in_catalogue
            ):
                duplicate_count += 1
                continue

            batch_seen.add(key)
            batch_titles.add(title_key)
            new_records.append(record)

        if conversation_store is not None and conversation_id and new_records:
            conversation_store.set(
                conversation_id,
                RFP_AGENT_NAME,
                {"seen_source_ids": sorted(seen_ids | batch_seen)},
            )

        return {"new_records": new_records, "duplicate_count": duplicate_count}

    return detect_duplicates_node


def route_after_dedup(state: RFPResearchState) -> str:
    return "assess" if state.get("new_records") else "no_new_results"


def make_assess_and_filter_node(model: str):
    def assess_and_filter_node(state: RFPResearchState) -> dict[str, Any]:
        # Relevance and qualification are deliberately TWO separate LLM
        # calls/chains (see tools.py module docstring): missing
        # eligibility information must never be able to make a
        # genuinely relevant tender look irrelevant, and giving the
        # relevance call no eligibility context at all enforces that
        # architecturally rather than by prompt instruction alone.
        relevance_chain = build_relevance_chain(model)
        qualification_chain = build_qualification_chain(model)
        academy_notes = str(_search_options(state).get("eligibility_notes") or "").strip() or "None given."
        assessed: list[dict[str, Any]] = []
        not_relevant = 0

        for record in state.get("new_records", []):
            try:
                relevance = relevance_chain.invoke({
                    "title": record.get("title") or "",
                    "organization": record.get("organization") or "",
                    "country": record.get("country") or "",
                    "category": record.get("category") or "",
                    "scope_of_work": record.get("scope_of_work") or "",
                })
            except Exception as exc:
                # Fail open: a relevance-assessment failure must not
                # silently discard a candidate the keyword filter
                # already judged topically plausible.
                assessed.append({
                    **record,
                    "ai_summary": None,
                    "qualification_status": "unknown",
                    "qualification_reason": f"Relevance assessment unavailable: {exc}",
                })
                continue

            if not relevance.is_relevant:
                not_relevant += 1
                continue

            if not record.get("eligibility"):
                # Whether eligibility was stated is a plain fact, not a
                # judgment call -- deciding "unknown" here in code,
                # without an LLM call, is both cheaper and more
                # reliable. A live run showed qwen2.5:3b asked to
                # assess qualification with no stated eligibility
                # sometimes answering "not_qualified" ("no information
                # provided...") despite being explicitly instructed
                # that absence of eligibility means "unknown" -- the
                # same kind of entanglement the relevance/qualification
                # split above was meant to prevent, just one level
                # deeper. See tools.py's QualificationAssessment
                # docstring and the brief's explicit example (a
                # "Training and Capacity Building Consultancy Services"
                # tender with no stated eligibility must stay "unknown").
                qualification_status = "unknown"
                qualification_reason = (
                    "Eligibility requirements were not stated in the "
                    "source record."
                )
            else:
                try:
                    qualification = qualification_chain.invoke({
                        "title": record.get("title") or "",
                        "scope_of_work": record.get("scope_of_work") or "",
                        "eligibility": "; ".join(record.get("eligibility") or []),
                        "deadline": record.get("deadline") or "Not specified",
                        "academy_notes": academy_notes,
                    })
                    qualification_status = qualification.qualification_status
                    qualification_reason = qualification.qualification_reason
                except Exception as exc:
                    qualification_status = "unknown"
                    qualification_reason = f"Qualification assessment unavailable: {exc}"

            assessed.append({
                **record,
                "ai_summary": relevance.summary,
                "qualification_status": qualification_status,
                "qualification_reason": qualification_reason,
            })

        return {"assessed_results": assessed, "not_relevant_count": not_relevant}

    return assess_and_filter_node


def make_apply_business_rule_node():
    def apply_business_rule_node(state: RFPResearchState) -> dict[str, Any]:
        priority = _priority_terms(_search_options(state))
        results = []

        for record in state.get("assessed_results", []):
            # "Important Business Rule" (brief, Agent 2): Egypt
            # opportunities are flagged immediately for review; Saudi
            # Arabian opportunities enter the monitored pipeline. This
            # label describes REVIEW PRIORITY only -- it does not mean
            # the tender has been verified as still open or that the
            # academy qualifies (see qualification_status for that).
            review_status = FLAGGED if record.get("country") == "Egypt" else MONITORED
            text = " ".join(str(record.get(k) or "") for k in ("title", "organization", "scope_of_work")).casefold()
            results.append({
                **record,
                "review_status": review_status,
                "priority_match": bool(priority) and any(term in text for term in priority),
            })

        return {"final_results": results}

    return apply_business_rule_node


def make_store_opportunities_node(tools: TenderTools):
    def store_opportunities_node(state: RFPResearchState) -> dict[str, Any]:
        # Tool save_discovered_tenders: new tenders become PENDING for the
        # admin; known ones keep their decision.
        try:
            saved = tools.save_discovered_tenders(state.get("final_results", []))
        except Exception as exc:
            return {"newly_discovered_count": 0,
                    "storage_error": _tool_failure("save_discovered_tenders", exc)}
        if saved.get("status") != "ok":
            return {"newly_discovered_count": 0,
                    "storage_error": saved.get("message") or "The tenders could not be saved."}
        return {"newly_discovered_count": saved.get("new", 0)}

    return store_opportunities_node


_MESSAGES = {
    "out_of_scope": {
        "en": "This agent only covers tenders in Egypt and Saudi Arabia. "
              "'{country}' is outside the supported scope.",
        "ar": "هذا الوكيل يغطي المناقصات في مصر والسعودية فقط. "
              "'{country}' خارج النطاق المدعوم.",
    },
    "no_data": {
        "en": "No tender data could be retrieved -- this is NOT the same as "
              "finding zero relevant tenders.",
        "ar": "تعذّر الحصول على بيانات المناقصات -- وهذا لا يعني عدم وجود "
              "مناقصات مناسبة.",
    },
    "all_duplicates": {
        "en": "No new tenders: all {count} relevant match(es) are already in "
              "the review queue or dashboard.",
        "ar": "لا توجد مناقصات جديدة: كل النتائج المناسبة ({count}) موجودة بالفعل "
              "في قائمة المراجعة أو لوحة المناقصات.",
    },
    "none_relevant": {
        "en": "The sources responded, but no relevant, current "
              "education/training tenders were found for {countries}.",
        "ar": "استجابت المصادر، لكن لم يتم العثور على مناقصات تعليمية أو تدريبية "
              "مناسبة وسارية في {countries}.",
    },
    "found": {
        "en": "{new} new tender(s) waiting for admin review.",
        "ar": "{new} مناقصة جديدة بانتظار مراجعة المسؤول.",
    },
    "not_saved": {
        "en": "Found {total} relevant tender(s), but they could NOT be saved for admin "
              "review: {error}",
        "ar": "تم العثور على {total} مناقصة مناسبة، لكن تعذّر حفظها لمراجعة المسؤول: {error}",
    },
    "country_egypt": {"en": "Egypt: {n} (flagged for immediate review)", "ar": "مصر: {n} (مراجعة فورية)"},
    "country_saudi": {"en": "Saudi Arabia: {n} (monitored)", "ar": "السعودية: {n} (متابعة)"},
    "checked": {"en": "What was checked: {read} tender(s) read", "ar": "ما تم فحصه: {read} مناقصة"},
    "no_deadline": {
        "en": "⚠ {n} of these tender(s) state no deadline, so your date range could not "
              "be checked: check before approving.",
        "ar": "⚠ {n} منها بلا موعد نهائي مذكور، فلم يمكن التحقق من نطاق التاريخ: "
              "تحقّق قبل القبول.",
    },
    "unreadable": {
        "en": "ℹ {n} web page(s) had no readable tender name and were skipped{why}.",
        "ar": "ℹ تم تخطي {n} صفحة ويب لا تذكر اسم المنافسة بشكل مقروء{why}.",
    },
    "unreadable_etimad": {
        "en": " ({n} from Etimad, which loads its tender details with JavaScript and protects them)",
        "ar": " (منها {n} من اعتماد، الذي يعرض تفاصيل المنافسات بالجافاسكربت ويحميها)",
    },
    "sources": {"en": "Sources: {items}.", "ar": "المصادر: {items}."},
}

# Why tenders were left out, in the order the checks run.
_DROPPED = (
    ("expired", {"en": "{n} expired or old", "ar": "{n} منتهية أو قديمة"}),
    ("not_training", {"en": "{n} not training (by title)", "ar": "{n} ليست تدريبًا (حسب العنوان)"}),
    ("works", {"en": "{n} works or supply contracts", "ar": "{n} عقود أعمال أو توريد"}),
    ("outside_dates", {"en": "{n} outside your dates", "ar": "{n} خارج التواريخ المختارة"}),
    ("outside_categories", {"en": "{n} outside the chosen categories", "ar": "{n} خارج المجالات المختارة"}),
    ("duplicates", {"en": "{n} already known", "ar": "{n} معروفة مسبقًا"}),
    ("not_relevant", {"en": "{n} judged not relevant by the AI check", "ar": "{n} غير مناسبة حسب تقييم الذكاء الاصطناعي"}),
)

_SOURCE_TEXT = {
    "world_bank_ok": {"en": "World Bank: {n} notice(s)", "ar": "البنك الدولي: {n} إعلان"},
    "etimad": {"en": "Etimad: no public API (Saudi tenders come from web research)",
               "ar": "اعتماد: لا توجد واجهة برمجية عامة (مناقصات السعودية من البحث على الويب)"},
    "web_ok": {"en": "Web search: {ok} of {total} searches answered, {pages} page(s) read",
               "ar": "البحث على الويب: استجاب {ok} من {total} بحثًا، وتمت قراءة {pages} صفحة"},
    "failed": {"en": "{name}: failed ({why})", "ar": "{name}: فشل ({why})"},
    "not_configured": {"en": "{name}: not configured ({why})", "ar": "{name}: غير مُعدّ ({why})"},
}

_COUNTRY_AR = {"Egypt": "مصر", "Saudi Arabia": "السعودية"}


def _msg(key: str, language: str, **values: Any) -> str:
    return _MESSAGES[key]["ar" if language == "ar" else "en"].format(**values)


def _text(table: dict[str, str], language: str, **values: Any) -> str:
    return table["ar" if language == "ar" else "en"].format(**values)


def _source_items(api_results: list[dict[str, Any]], web: dict[str, Any], language: str) -> list[str]:
    """One readable item per source; failures say why."""
    items, web_reports = [], []
    for result in api_results:
        name = str(result.get("source_name") or "")
        status = result.get("status")
        if name.startswith("Web search"):
            web_reports.append(result)
            continue
        short = "World Bank" if name.startswith("World Bank") else name.split(" -- ")[0].split(" (")[0]
        if status == "success" and short == "World Bank":
            items.append(_text(_SOURCE_TEXT["world_bank_ok"], language, n=len(result.get("records") or [])))
        elif status == "unavailable" and short.startswith("Etimad"):
            items.append(_text(_SOURCE_TEXT["etimad"], language))
        elif status == "awaiting_credentials":
            items.append(_text(_SOURCE_TEXT["not_configured"], language, name=short,
                               why=str(result.get("message") or "")[:120]))
        elif status != "success":
            items.append(_text(_SOURCE_TEXT["failed"], language, name=short,
                               why=str(result.get("message") or status)[:120]))
        else:
            items.append(short)
    if web_reports:
        ok = sum(1 for r in web_reports if r.get("status") == "success")
        if ok:
            items.append(_text(_SOURCE_TEXT["web_ok"], language, ok=ok, total=len(web_reports),
                               pages=web.get("pages", 0)))
        else:
            status = web_reports[0].get("status")
            key = "not_configured" if status == "awaiting_credentials" else "failed"
            items.append(_text(_SOURCE_TEXT[key], language, name="Web search",
                               why=str(web_reports[0].get("message") or status)[:120]))
    return items


def _checked_line(state: RFPResearchState, language: str) -> Optional[str]:
    report = dict(state.get("filter_report") or {})
    if not report.get("read"):
        return None
    report["duplicates"] = state.get("duplicate_count", 0)
    report["not_relevant"] = state.get("not_relevant_count", 0)
    parts = [_text(table, language, n=report[key]) for key, table in _DROPPED if report.get(key)]
    line = _msg("checked", language, read=report["read"])
    return line + (" → " + " · ".join(parts) if parts else "")


def make_compose_response_node():
    def compose_response_node(state: RFPResearchState) -> dict[str, Any]:
        """
        The reply, one line per point: the result, per country, what was
        checked and why tenders were left out, a warning for tenders whose
        deadline could not be checked, and each source's status.
        """
        api_results = state.get("api_results", [])
        language = state.get("language", "en")
        web = state.get("web_research") or {}
        countries = state.get("requested_countries", list(ALLOWED_COUNTRIES))

        def respond(headline, opportunities=(), duplicates=0, new=0, results=api_results, extra=()):
            lines = [headline, *extra]
            if results:
                checked = _checked_line(state, language)
                if checked:
                    lines.append(checked)
                if web.get("unreadable_pages"):
                    etimad = web.get("unreadable_from_etimad", 0)
                    lines.append(_msg("unreadable", language, n=web["unreadable_pages"], why=(
                        _msg("unreadable_etimad", language, n=etimad) if etimad else "")))
                lines.append(_msg("sources", language, items=" · ".join(_source_items(results, web, language))))
            message = "\n".join(lines)
            return {
                "message": message,
                "response": {
                    "message": message,
                    "opportunities": list(opportunities),
                    "api_results": results,
                    "duplicate_count": duplicates,
                    "newly_discovered_count": new,
                    "web_research": web,
                    "filter_report": state.get("filter_report") or {},
                    "not_relevant_count": state.get("not_relevant_count", 0),
                },
            }

        if state.get("out_of_scope_country"):
            return respond(
                _msg("out_of_scope", language, country=state["out_of_scope_country"]),
                results=[],
            )

        any_source_succeeded = any(
            result.get("status") == "success" for result in api_results
        )

        if not any_source_succeeded:
            # An API error/unavailability must never read like "no
            # tenders found" -- state explicitly what happened (the
            # sources line says why each source failed).
            return respond(_msg("no_data", language))

        if "final_results" not in state:
            duplicates = state.get("duplicate_count", 0)
            if "new_records" in state:
                return respond(
                    _msg("all_duplicates", language, count=duplicates),
                    duplicates=duplicates,
                )
            names = [(_COUNTRY_AR.get(c, c) if language == "ar" else c) for c in countries]
            return respond(_msg("none_relevant", language, countries=", ".join(names)),
                           duplicates=duplicates)

        results = state["final_results"]
        new_count = state.get("newly_discovered_count", len(results))
        duplicates = state.get("duplicate_count", 0)

        if state.get("storage_error"):
            return respond(
                _msg("not_saved", language, total=len(results), error=state["storage_error"]),
                opportunities=results, duplicates=duplicates,
            )

        per_country = []
        if "Egypt" in countries:
            per_country.append(_msg("country_egypt", language,
                                    n=sum(1 for r in results if r.get("country") == "Egypt")))
        if "Saudi Arabia" in countries:
            per_country.append(_msg("country_saudi", language,
                                    n=sum(1 for r in results if r.get("country") == "Saudi Arabia")))
        extra = ["• " + " · ".join(per_country)] if per_country else []
        options = _search_options(state)
        no_deadline = sum(1 for r in results if not r.get("deadline"))
        if no_deadline and (options.get("deadline_from") or options.get("deadline_to")):
            extra.append(_msg("no_deadline", language, n=no_deadline))

        return respond(
            _msg("found", language, new=new_count),
            opportunities=results, duplicates=duplicates, new=new_count, extra=extra,
        )

    return compose_response_node


def build_graph(
    tools: TenderTools,
    conversation_store: Optional[ConversationStore] = None,
    model: str = "qwen2.5:3b",
    web_search: bool = True,
):
    """
    Build and compile the RFP Research Agent's internal LangGraph
    workflow. tools: the tender tools (TenderMcpClient or LocalTenderTools).
    """
    graph = StateGraph(RFPResearchState)

    graph.add_node("validate_request", make_validate_request_node())
    graph.add_node("query_apis", make_query_apis_node(tools))
    graph.add_node("web_research", make_web_research_node(model, tools, enabled=web_search))
    graph.add_node("filter_relevant", make_filter_relevant_node())
    graph.add_node(
        "detect_duplicates",
        make_detect_duplicates_node(conversation_store, tools),
    )
    graph.add_node("assess_and_filter", make_assess_and_filter_node(model))
    graph.add_node("apply_business_rule", make_apply_business_rule_node())
    graph.add_node("store_opportunities", make_store_opportunities_node(tools))
    graph.add_node("compose_response", make_compose_response_node())

    graph.set_entry_point("validate_request")

    graph.add_conditional_edges(
        "validate_request",
        route_after_validation,
        {"out_of_scope": "compose_response", "query": "query_apis"},
    )

    graph.add_edge("query_apis", "web_research")
    graph.add_edge("web_research", "filter_relevant")

    graph.add_conditional_edges(
        "filter_relevant",
        route_after_filter,
        {
            "no_api_data": "compose_response",
            "no_results": "compose_response",
            "dedup": "detect_duplicates",
        },
    )

    graph.add_conditional_edges(
        "detect_duplicates",
        route_after_dedup,
        {"no_new_results": "compose_response", "assess": "assess_and_filter"},
    )

    graph.add_edge("assess_and_filter", "apply_business_rule")
    graph.add_edge("apply_business_rule", "store_opportunities")
    graph.add_edge("store_opportunities", "compose_response")
    graph.add_edge("compose_response", END)

    return graph.compile()
