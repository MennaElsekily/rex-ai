"""
Web research for tenders in Egypt and Saudi Arabia (brief Agent 2,
requirements 9-11).

1. Searches (Tavily, which also returns each page's text, so no fragile
   per-site HTML scraper is needed), per requested country:
     - the official government tender portal (Arabic query);
     - established tender aggregators (English query);
     - the open web, in English and Arabic.
   The admin's chosen categories and priority organisations/sectors are
   added to the queries.
2. Each page's text goes through the LLM extraction chain (tools.py) for
   the brief's fields -- several tenders when the page is a listing.
3. Code then checks and completes every record (nothing is guessed):
   country, ISO dates, document links found in the page, the training
   category, the submission link, and the list of missing fields.

The records then join the API records in the normal pipeline (filter,
duplicates, relevance, qualification, business rule) and are stored for
admin review. Web text is untrusted: it is capped, cleaned, and only
ever treated as data.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from agents.rfp_research.api_clients.tavily import WebPage, search_tavily
from agents.rfp_research.state import TRAINING_CATEGORIES, training_category
from agents.text_utils import normalize_for_matching
from knowledge_base.language import contains_arabic

logger = logging.getLogger(__name__)

# "Approved sources" (requirement 9). Official portals render most of
# their content with JavaScript, so they often return little text; the
# aggregators publish full tender pages.
OFFICIAL_PORTALS = {
    "Egypt": ("etenders.gov.eg",),
    "Saudi Arabia": ("tenders.etimad.sa",),
}
PORTAL_QUERIES_AR = {
    "Egypt": "مناقصة تدريب",
    "Saudi Arabia": "منافسة تدريب",
}
TENDER_AGGREGATORS = (
    "tenderimpulse.com", "tendersinfo.com", "globaltenders.com", "developmentaid.org",
    "dgmarket.com", "ungm.org", "bidsinfo.com", "tendersontime.com",
)
COUNTRY_NAMES_AR = {"Egypt": "مصر", "Saudi Arabia": "السعودية"}

RESULTS_PER_SEARCH = 5
MAX_PAGES_PER_COUNTRY = 8
MAX_PAGE_TEXT_CHARS = 5000

_DOCUMENT_LINK = re.compile(r"https?://[^\s\"'<>)\]]+?\.(?:pdf|docx?|xlsx?|zip|rar)\b", re.IGNORECASE)
_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}
REQUIRED_FIELDS = (
    "title", "organization", "country", "deadline", "eligibility", "training_category",
    "submission_url", "publication_date", "available_documents",
)


@dataclass
class WebSearch:
    label: str
    country: str
    query: str
    domains: Optional[tuple[str, ...]] = None


def build_searches(
    country: str,
    categories: Optional[list[str]] = None,
    priority: Optional[str] = None,
) -> list[WebSearch]:
    """The searches for one country, official portal first."""
    topic = " ".join(c.lower() for c in categories) if categories else "education training"
    extra = f" {priority.strip()}" if priority and priority.strip() else ""
    searches = []
    if country in OFFICIAL_PORTALS:
        searches.append(WebSearch("official portal", country, PORTAL_QUERIES_AR[country] + extra,
                                  OFFICIAL_PORTALS[country]))
    searches += [
        WebSearch("tender sites", country, f"{country} {topic} tender{extra}", TENDER_AGGREGATORS),
        WebSearch("open web", country, f"{country} {topic} tender RFP{extra}"),
        WebSearch("open web (Arabic)", country,
                  f"مناقصة تدريب {COUNTRY_NAMES_AR.get(country, country)}{extra}"),
    ]
    return searches


def collect_pages(
    countries: list[str],
    categories: Optional[list[str]] = None,
    priority: Optional[str] = None,
    search: Optional[Callable[..., Any]] = None,
) -> tuple[list[tuple[WebPage, str, str]], list[dict[str, Any]]]:
    """
    Run the searches. Returns ([(page, country, search label)], source
    reports for the response); each URL appears once.
    """
    # Looked up at call time, so tests can replace the module's search_tavily.
    search = search or search_tavily
    pages: list[tuple[WebPage, str, str]] = []
    reports: list[dict[str, Any]] = []
    seen: set[str] = set()

    for country in countries:
        kept = 0
        for web_search in build_searches(country, categories, priority):
            result = search(web_search.query, max_results=RESULTS_PER_SEARCH,
                            include_domains=list(web_search.domains) if web_search.domains else None,
                            include_raw_content=True)
            reports.append({
                "source_name": f"Web search: {web_search.label} ({country})",
                "status": result.status,
                "message": result.message,
            })
            for page in result.pages:
                if page.url in seen or kept >= MAX_PAGES_PER_COUNTRY:
                    continue
                seen.add(page.url)
                pages.append((page, country, web_search.label))
                kept += 1
    return pages, reports


def iso_date(value: Any) -> Optional[str]:
    """'2026-11-22', '22/11/2026', '22 November 2026', 'Nov 22, 2026' -> '2026-11-22'."""
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if match:
        y, m, d = map(int, match.groups())
    else:
        match = re.search(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", text)
        if match:
            d, m, y = map(int, match.groups())
        else:
            match = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,})\.?,?\s+(\d{4})\b", text) or None
            if match:
                d, month, y = int(match.group(1)), match.group(2)[:3].lower(), int(match.group(3))
            else:
                match = re.search(r"\b([A-Za-z]{3,})\.?\s+(\d{1,2}),?\s+(\d{4})\b", text)
                if not match:
                    return None
                month, d, y = match.group(1)[:3].lower(), int(match.group(2)), int(match.group(3))
            m = _MONTHS.get(month)
            if not m:
                return None
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def document_links(text: str, limit: int = 10) -> list[str]:
    """Links to tender documents written in the page itself."""
    return list(dict.fromkeys(_DOCUMENT_LINK.findall(text or "")))[:limit]


def _slug(text: str) -> str:
    return re.sub(r"[^0-9a-z؀-ۿ]+", "-", text.casefold()).strip("-")[:60]


def _web_url(value: Optional[str], fallback: str) -> str:
    parsed = urlparse(str(value or ""))
    return value if parsed.scheme in ("http", "https") and parsed.netloc else fallback


def _supported_category(label: Optional[str], title: str, own_text: str) -> Optional[str]:
    """
    The AI's category label, only if the tender's own text contains that
    category's words; otherwise the category the words show (title first).
    A live search showed the label copying the listing site's sector
    ("IT-Software") onto an archiving tender.
    """
    lowered = own_text.casefold()
    if label in TRAINING_CATEGORIES and any(k.casefold() in lowered for k in TRAINING_CATEGORIES[label]):
        return label
    return training_category(title) or training_category(own_text)


# Page headings, site names and menu labels -- never a tender's name. A
# live search found Etimad pages whose only text is the heading
# "تفاصيل المنافسة" (the details load later with JavaScript), and the
# extraction used that heading as the tender title.
GENERIC_TITLES = tuple(normalize_for_matching(t) for t in (
    "تفاصيل المنافسة", "تفاصيل المنافسات", "المنافسات", "المنافسات المطروحة", "منافسة", "منافسات",
    "المنافسات الحالية", "قائمة المنافسات", "جميع المنافسات", "تفاصيل المناقصة", "المناقصات",
    "مناقصة", "مناقصات", "الإعلانات", "دعوات التأهيل", "اعلانات قوائم الموردين",
    "مشاريع المنافسات المستقبلية", "الصفحة الرئيسية", "تسجيل الدخول", "اعتماد", "منصة اعتماد",
    "tender details", "tenders", "tender", "tender notice", "tender notices", "rfp", "rfq",
    "details", "notice details", "view tender", "all tenders", "tender list", "search results",
    "home", "login", "etimad", "etimad tenders", "government tenders", "latest tenders",
))
# Explicit title fields a page may state (Arabic and English).
_TITLE_LABELS = (
    "اسم المنافسة", "عنوان المنافسة", "اسم المناقصة", "عنوان المناقصة",
    "tender name", "tender title", "title of tender", "title of the tender", "name of tender",
)
_LABEL_LINE = re.compile(
    r"^[\s#*>|_-]*(" + "|".join(re.escape(label) for label in _TITLE_LABELS) + r")[\s*_]*[:：|\-–]?\s*(.*)$",
    re.IGNORECASE,
)
_MARKDOWN_NOISE = re.compile(r"[*_#>|`]+")


# A heading that names a group of tenders, not one tender: English ending
# in a plural ("Saudi Arabia Education - Educational Training Tenders",
# seen live as a listing page's heading), Arabic starting with one.
_GROUP_HEADING = re.compile(
    r"(?:\b(?:tenders|tender notices|tender opportunities|rfps|bids)$)|"
    r"^(?:" + "|".join(normalize_for_matching(w) for w in ("مناقصات", "المناقصات", "منافسات", "المنافسات")) + r")\b"
)


def is_generic_title(title: Optional[str]) -> bool:
    """
    A page heading or menu label ("تفاصيل المنافسة", "Tender Details"...),
    a heading naming a group of tenders ("... Training Tenders"), or nothing.
    """
    normalized = " ".join(_MARKDOWN_NOISE.sub(" ", normalize_for_matching(str(title or ""))).split())
    normalized = normalized.strip(" :.-")
    return not normalized or normalized in GENERIC_TITLES or bool(_GROUP_HEADING.search(normalized))


def is_link_label_only(title: str, text: str) -> bool:
    """
    Whether the title appears in the page only as a link's text (a menu
    item such as "* [Training Courses](/courses)"), never as content.
    """
    occurrences = [m.start() for m in re.finditer(re.escape(title), text or "")]
    if not occurrences:
        return False
    in_link = re.compile(r"\[[^\]\n]*" + re.escape(title) + r"[^\[\]\n]*\]\(")
    return all(any(m.start() <= pos < m.end() for m in in_link.finditer(text)) for pos in occurrences)


def title_from_labels(text: str) -> Optional[str]:
    """
    The tender's name from an explicit title field in the page text
    ("اسم المنافسة: ...", "Tender Name" then the value on the next line).
    Only labelled values: never a heading, menu item or free text. None
    when the page states no usable name.
    """
    lines = (text or "").splitlines()
    for index, line in enumerate(lines):
        match = _LABEL_LINE.match(line.strip())
        if not match:
            continue
        value = match.group(2)
        if not value.strip():  # "Label" alone, the value on the next non-empty line
            value = next((l for l in lines[index + 1:index + 4] if l.strip()), "")
        value = " ".join(_MARKDOWN_NOISE.sub(" ", value).split()).strip(" :-–")
        if 8 <= len(value) <= 250 and not is_generic_title(value) and not _LABEL_LINE.match(value):
            return value
    return None


def page_to_records(
    page: WebPage,
    country: str,
    extraction_chain,
    country_check: Callable[[str], list[str]],
    unreadable_pages: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """
    The tenders one page announces, as NormalizedTender-shaped dicts.
    country_check(text) returns the allowed countries the text names; a
    tender is kept only when it names the requested country.

    A tender needs its own name. A missing or generic title (a page
    heading such as "تفاصيل المنافسة") is replaced only by an explicit
    title field on a single-tender page (title_from_labels); otherwise the
    tender is skipped -- nothing is guessed. A page that gives no record
    because of that (or whose own title is only a generic heading) is
    added to unreadable_pages (a set of URLs, so a page counts once).
    """
    text = (page.raw_content or page.content or "")[:MAX_PAGE_TEXT_CHARS]
    try:
        extracted = extraction_chain.invoke({
            "page_title": page.title, "page_url": page.url, "page_text": text,
        }).tenders
    except Exception as exc:
        logger.warning("Tender extraction failed for %s: %s", page.source_domain, type(exc).__name__)
        return []

    records = []
    single = len(extracted) == 1
    links = document_links(page.raw_content or "")
    untitled = False
    for tender in extracted[:5]:
        title = (tender.title or "").strip()
        # On a single-tender page, a title found only as link text is a
        # menu item. (Listing pages are exempt: their items are links.)
        if is_generic_title(title) or (single and is_link_label_only(title, text)):
            # Only a single-tender page's labelled title field can stand in.
            title = (title_from_labels(text) if single else None) or ""
            if not title:
                untitled = True
                continue
        if len(title) < 8:
            continue
        # A listing page's title names one country for all its items, so
        # it is evidence only for a single-tender page.
        evidence = " ".join(filter(None, [tender.country, title, tender.issuing_organization,
                                          *((page.title, page.source_domain) if single else ())]))
        if country not in country_check(evidence):
            continue

        # A listing page's text describes many tenders, so a tender from
        # a listing is judged by its own title only; otherwise every item
        # on a "training tenders" page would look like a training tender.
        own_text = f"{title} {text[:1500]}" if single else title
        category = _supported_category(tender.training_category, title, own_text)
        key = page.url if single else f"{page.url}#{_slug(title)}"
        record = {
            "url": key,
            "source_url": page.url,
            "source_id": key,
            "source_name": f"Web: {page.source_domain}",
            "source_domain": page.source_domain,
            "title": title,
            "reference_number": tender.reference_number,
            "organization": tender.issuing_organization,
            "country": country,
            "deadline": iso_date(tender.deadline),
            "publication_date": iso_date(tender.publication_date) or iso_date(page.published_date),
            "eligibility": [e for e in tender.eligibility if e and e.strip()],
            "category": category,
            "training_category": category,
            "scope_of_work": (page.content or text)[:1500] if single else title,
            "submission_url": _web_url(tender.submission_link, page.url),
            "available_documents": list(dict.fromkeys(
                [d for d in tender.available_documents if d and d.strip()] + links))[:10],
            "verification_status": "extracted_from_web_page",
            "language": "ar" if contains_arabic(title + text[:500]) else "en",
            "retrieved_at": page.retrieved_at,
        }
        record["missing_information"] = [f for f in REQUIRED_FIELDS if not record.get(f)]
        records.append(record)

    # Unreadable: a tender without a usable name, or a page that is only a
    # generic heading with nothing extracted. A tender that was read but
    # dropped for another reason (e.g. another country) is not counted.
    if unreadable_pages is not None and not records and (
        untitled or (not extracted and is_generic_title(page.title))
    ):
        unreadable_pages.add(page.url)
    return records


def within_deadline_range(deadline: Optional[str], start: Optional[str], end: Optional[str]) -> bool:
    """True unless the deadline is known and outside the admin's range."""
    when = iso_date(deadline)
    if not when:
        return True
    earliest, latest = iso_date(start), iso_date(end)
    if earliest and when < earliest:
        return False
    if latest and when > latest:
        return False
    return True


def is_expired(deadline: Optional[str], today: Optional[date] = None) -> bool:
    when = iso_date(deadline)
    return bool(when) and datetime.fromisoformat(when).date() < (today or date.today())


# A notice with no stated deadline that was published this long ago is
# treated as closed (e.g. World Bank notices from 2006).
MAX_AGE_WITHOUT_DEADLINE_DAYS = 365


def is_stale(deadline: Optional[str], publication_date: Optional[str],
             today: Optional[date] = None) -> bool:
    if iso_date(deadline):
        return False
    published = iso_date(publication_date)
    return bool(published) and (
        (today or date.today()) - datetime.fromisoformat(published).date()
    ).days > MAX_AGE_WITHOUT_DEADLINE_DAYS
