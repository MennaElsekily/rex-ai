"""
RFP Research web research (agents/rfp_research/web_tenders.py and the
graph's web_research node): web search for tenders in Egypt and Saudi
Arabia -> tender fields extracted from each page -> the normal pipeline
(filter, duplicates, relevance, qualification, business rule) -> stored
as pending for admin approval.

Mocked: the web search (web_tenders.search_tavily), the tender APIs and
the three LLM chains. Temp SQLite files.
"""

from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from agents.conversation_store import ConversationStore
from agents.rfp_research import RFPResearchAgent, tender_tools, web_tenders
from agents.rfp_research import graph as rfp_graph
from agents.rfp_research.api_clients.base import APIQueryResult
from agents.rfp_research.api_clients.tavily import WebPage, WebSearchResult
from agents.rfp_research.opportunity_store import FLAGGED, MONITORED, PENDING, OpportunityStore
from agents.rfp_research.tools import ExtractedTender

FUTURE = (date.today() + timedelta(days=40)).isoformat()
LATER = (date.today() + timedelta(days=200)).isoformat()


def web_page(url, title, content="", raw=None):
    return WebPage(title=title, url=url, content=content or title, raw_content=raw,
                   source_domain=url.split("/")[2])


EGYPT_PAGE = web_page(
    "https://etenders.gov.eg/t/1", "Teacher training services - Ministry of Education",
    raw="Deadline 22/11/2030. Documents: https://etenders.gov.eg/docs/tor.pdf",
)
SAUDI_PAGE = web_page("https://tenders.etimad.sa/t/7", "منافسة تدريب الموظفين على التحول الرقمي")

TENDERS = {
    EGYPT_PAGE.url: [ExtractedTender(
        title="Teacher Training Services for Public Schools", issuing_organization="Ministry of Education",
        country="Egypt", deadline=FUTURE, eligibility=["Registered training provider"],
        training_category="Education and Training", available_documents=["Terms of reference"],
    )],
    SAUDI_PAGE.url: [ExtractedTender(
        title="Digital transformation training for government staff",
        issuing_organization="Digital Government Authority", country="Saudi Arabia",
        deadline=LATER, submission_link="https://tenders.etimad.sa/submit/7",
    )],
}


def web_search_returns(monkeypatch, pages_by_country, status="success"):
    queries = []

    def fake(query, max_results=5, include_domains=None, include_raw_content=False):
        queries.append((query, include_domains))
        saudi = "السعودية" in query or "Saudi" in query or include_domains == ["tenders.etimad.sa"]
        pages = pages_by_country.get("Saudi Arabia" if saudi else "Egypt", [])
        return WebSearchResult(status=status, message="ok", query=query,
                               pages=pages if status == "success" else [])

    monkeypatch.setattr(web_tenders, "search_tavily", fake)
    return queries


class Extraction:
    def __init__(self, tenders=None, error=None):
        self.tenders, self.error, self.calls = tenders or TENDERS, error, []

    def invoke(self, values):
        self.calls.append(values)
        if self.error:
            raise self.error
        return SimpleNamespace(tenders=self.tenders.get(values["page_url"], []))


class Chain:
    def __init__(self, result):
        self.result, self.calls = result, []

    def invoke(self, values):
        self.calls.append(values)
        return self.result


@pytest.fixture
def llm(monkeypatch):
    chains = SimpleNamespace(
        extraction=Extraction(),
        relevance=Chain(SimpleNamespace(is_relevant=True, summary="Training tender.")),
        qualification=Chain(SimpleNamespace(qualification_status="unknown",
                                            qualification_reason="Not enough information.")),
    )
    monkeypatch.setattr(rfp_graph, "build_extraction_chain", lambda model: chains.extraction)
    monkeypatch.setattr(rfp_graph, "build_relevance_chain", lambda model: chains.relevance)
    monkeypatch.setattr(rfp_graph, "build_qualification_chain", lambda model: chains.qualification)
    return chains


@pytest.fixture
def apis(monkeypatch):
    calls = []

    def world_bank(query):
        calls.append("world_bank")
        return APIQueryResult(source_name="World Bank", status="success", message="0 records", records=[])

    def etimad(query):
        calls.append("etimad")
        return APIQueryResult(source_name="Etimad", status="unavailable", message="No public API.")

    monkeypatch.setattr(tender_tools, "query_world_bank_egypt", world_bank)
    monkeypatch.setattr(tender_tools, "query_etimad_saudi_arabia", etimad)
    return calls


@pytest.fixture
def store(tmp_path):
    s = OpportunityStore(db_path=tmp_path / "rfp.sqlite3")
    yield s
    s.close()


def ask(store, text="Find education and training tenders", search=None, language="en", tmp=None):
    agent = RFPResearchAgent(
        conversation_store=ConversationStore(db_path=tmp / "conv.sqlite3") if tmp else None,
        opportunity_store=store, model="test",
    )
    response = agent.process_request({"request_id": "r1", "user_input": text, "context": {
        "conversation_id": "c1", "language": language, "search": search or {}}})
    assert response["status"] == "success", response
    return response["result"]


# --------------------------------------------------------------------
# Searches and page extraction
# --------------------------------------------------------------------

def test_each_country_searches_its_official_portal_tender_sites_and_the_web():
    searches = web_tenders.build_searches("Saudi Arabia", ["Corporate Learning"], "Ministry of Health")

    assert [s.label for s in searches] == ["official portal", "tender sites", "open web", "open web (Arabic)"]
    assert searches[0].domains == ("tenders.etimad.sa",) and "منافسة" in searches[0].query
    assert all("Ministry of Health" in s.query for s in searches)
    assert "corporate learning" in searches[1].query
    assert web_tenders.build_searches("Egypt")[0].domains == ("etenders.gov.eg",)


def test_collect_pages_reports_every_search_and_keeps_each_url_once(monkeypatch):
    web_search_returns(monkeypatch, {"Egypt": [EGYPT_PAGE]})

    pages, reports = web_tenders.collect_pages(["Egypt"])

    assert [page.url for page, _, _ in pages] == [EGYPT_PAGE.url]
    assert len(reports) == 4 and {r["status"] for r in reports} == {"success"}


def test_a_page_becomes_a_complete_tender_record():
    records = web_tenders.page_to_records(EGYPT_PAGE, "Egypt", Extraction(),
                                          rfp_graph._detect_requested_countries)

    [record] = records
    assert record["url"] == EGYPT_PAGE.url and record["source_name"] == "Web: etenders.gov.eg"
    assert record["deadline"] == FUTURE
    assert record["organization"] == "Ministry of Education"
    assert record["training_category"] == "Education and Training"
    assert record["available_documents"] == ["Terms of reference", "https://etenders.gov.eg/docs/tor.pdf"]
    assert record["submission_url"] == EGYPT_PAGE.url  # no stated link: the source page
    assert "publication_date" in record["missing_information"]
    assert record["verification_status"] == "extracted_from_web_page"


def test_a_listing_page_gives_one_record_per_tender():
    page = web_page("https://tendersinfo.com/egypt", "Egypt training tenders")
    tenders = {page.url: [ExtractedTender(title="E-learning platform for universities", country="Egypt"),
                          ExtractedTender(title="Vocational training for youth", country="Egypt"),
                          ExtractedTender(title="RFP", country="Egypt")]}

    records = web_tenders.page_to_records(page, "Egypt", Extraction(tenders),
                                          rfp_graph._detect_requested_countries)

    assert [r["url"] for r in records] == [
        "https://tendersinfo.com/egypt#e-learning-platform-for-universities",
        "https://tendersinfo.com/egypt#vocational-training-for-youth",
    ]


def test_a_listing_item_is_judged_by_its_own_title(monkeypatch, store, llm, apis):
    # The page is about training tenders; one of its items is not.
    page = web_page("https://globaltenders.com/egypt-training", "Egypt education and training tenders",
                    content="Latest education and training tenders in Egypt.")
    llm.extraction.tenders = {page.url: [
        ExtractedTender(title="Equipping the seminar hall with electrical work", country="Egypt"),
        ExtractedTender(title="Vocational training for youth in Alexandria", country="Egypt"),
    ]}
    web_search_returns(monkeypatch, {"Egypt": [page]})

    result = ask(store, search={"countries": ["Egypt"]})

    assert [o["title"] for o in result["opportunities"]] == ["Vocational training for youth in Alexandria"]
    assert result["opportunities"][0]["training_category"] == "Technical Training"


def test_works_contracts_are_dropped_unless_the_title_names_training(monkeypatch, store, llm, apis):
    page = web_page("https://tendersinfo.com/egypt-edu", "Egypt education tenders")
    llm.extraction.tenders = {page.url: [
        ExtractedTender(title="Revamping of the university housing for teaching staff", country="Egypt"),
        ExtractedTender(title="Supply of specialized medical training courses", country="Egypt"),
    ]}
    web_search_returns(monkeypatch, {"Egypt": [page]})

    result = ask(store, search={"countries": ["Egypt"]})

    assert [o["title"] for o in result["opportunities"]] == ["Supply of specialized medical training courses"]


def test_the_same_title_with_site_filler_words_is_one_tender(monkeypatch, store, llm, apis):
    page = web_page("https://globaltenders.com/egypt", "Egypt training tenders")
    llm.extraction.tenders = {page.url: [
        ExtractedTender(title="RFP - Emergency Care Training Programs In Egypt", country="Egypt"),
        ExtractedTender(title="Emergency Care Training Programs", country="Egypt"),
    ]}
    web_search_returns(monkeypatch, {"Egypt": [page]})

    result = ask(store, search={"countries": ["Egypt"]})

    assert len(result["opportunities"]) == 1 and result["duplicate_count"] == 1


def test_old_notices_without_a_deadline_are_dropped():
    today = date(2026, 10, 27)
    assert web_tenders.is_stale(None, "2012-12-26", today)
    assert not web_tenders.is_stale(None, "2026-06-01", today)
    assert not web_tenders.is_stale("2026-12-01", "2012-12-26", today)  # a stated deadline decides
    assert not web_tenders.is_stale(None, None, today)


def test_a_tender_for_another_country_is_dropped():
    page = web_page("https://example.org/kenya", "Kenya training tender")
    tenders = {page.url: [ExtractedTender(title="Teacher training in Nairobi schools", country="Kenya")]}

    assert web_tenders.page_to_records(page, "Egypt", Extraction(tenders),
                                       rfp_graph._detect_requested_countries) == []


def test_an_extraction_failure_skips_the_page():
    assert web_tenders.page_to_records(EGYPT_PAGE, "Egypt", Extraction(error=RuntimeError("down")),
                                       rfp_graph._detect_requested_countries) == []


@pytest.mark.parametrize("text, expected", [
    ("2030-11-22", "2030-11-22"), ("22/11/2030", "2030-11-22"),
    ("22 November 2030", "2030-11-22"), ("Nov 22, 2030", "2030-11-22"),
    ("soon", None), ("31/02/2030", None), (None, None),
])
def test_dates_are_normalised(text, expected):
    assert web_tenders.iso_date(text) == expected


def test_deadline_range_and_expiry():
    assert web_tenders.within_deadline_range("2030-05-01", "2030-01-01", "2030-12-31")
    assert not web_tenders.within_deadline_range("2031-05-01", None, "2030-12-31")
    assert web_tenders.within_deadline_range(None, "2030-01-01", None)  # unknown deadline is kept
    assert web_tenders.is_expired("01/01/2020") and not web_tenders.is_expired(FUTURE)


# --------------------------------------------------------------------
# The whole agent
# --------------------------------------------------------------------

def test_both_countries_are_searched_and_queued_for_admin_approval(monkeypatch, tmp_path, store, llm, apis):
    web_search_returns(monkeypatch, {"Egypt": [EGYPT_PAGE], "Saudi Arabia": [SAUDI_PAGE]})

    result = ask(store, tmp=tmp_path)

    rows = {r["country"]: r for r in store.search()}
    assert set(rows) == {"Egypt", "Saudi Arabia"}
    assert {r["approval_status"] for r in rows.values()} == {PENDING}
    assert rows["Egypt"]["review_status"] == FLAGGED
    assert rows["Saudi Arabia"]["review_status"] == MONITORED
    assert rows["Saudi Arabia"]["submission_url"] == "https://tenders.etimad.sa/submit/7"
    assert "waiting for admin review" in result["message"]
    assert "Egypt: 1 (flagged for immediate review)" in result["message"]
    assert "Saudi Arabia: 1 (monitored)" in result["message"]
    assert "web research" in result["message"]  # why Saudi tenders come from the web
    assert result["web_research"] == {"status": "success", "pages": 2, "records": 2,
                                      "unreadable_pages": 0, "unreadable_from_etimad": 0}
    assert store.counts()["flagged_pending"] == 1


def test_saudi_tenders_are_found_although_etimad_has_no_api(monkeypatch, store, llm, apis):
    web_search_returns(monkeypatch, {"Saudi Arabia": [SAUDI_PAGE]})

    result = ask(store, "Find training tenders in Saudi Arabia")

    assert apis == ["etimad"]
    assert [o["country"] for o in result["opportunities"]] == ["Saudi Arabia"]
    assert "NOT the same as finding zero" not in result["message"]


def test_search_form_choices_are_applied(monkeypatch, store, llm, apis):
    web_search_returns(monkeypatch, {"Egypt": [EGYPT_PAGE], "Saudi Arabia": [SAUDI_PAGE]})

    result = ask(store, search={
        "countries": ["Saudi Arabia"], "categories": ["Digital Transformation"],
        "priority": "Digital Government", "eligibility_notes": "ISO 9001 certified",
        "deadline_from": None, "deadline_to": LATER,
    })

    assert apis == ["etimad"]  # Egypt was not chosen
    [tender] = result["opportunities"]
    assert tender["training_category"] == "Digital Transformation"
    assert tender["priority_match"] is True
    assert store.search()[0]["priority_match"] is True


def test_category_and_deadline_choices_filter_tenders(monkeypatch, store, llm, apis):
    web_search_returns(monkeypatch, {"Egypt": [EGYPT_PAGE], "Saudi Arabia": [SAUDI_PAGE]})

    by_category = ask(store, search={"categories": ["Corporate Learning"]})
    by_deadline = ask(store, search={"deadline_to": (date.today() + timedelta(days=10)).isoformat()})

    assert by_category["opportunities"] == [] and by_deadline["opportunities"] == []
    assert store.search() == []


def test_admin_eligibility_notes_reach_the_qualification_check(monkeypatch, store, llm, apis):
    web_search_returns(monkeypatch, {"Egypt": [EGYPT_PAGE]})

    ask(store, search={"countries": ["Egypt"], "eligibility_notes": "Registered training provider in Egypt"})

    assert llm.qualification.calls[0]["academy_notes"] == "Registered training provider in Egypt"
    assert "eligibility" not in llm.relevance.calls[0]


def test_the_same_tender_on_another_site_is_a_duplicate(monkeypatch, store, llm, apis):
    store.save({"url": "https://tendersinfo.com/eg/55", "title": "Teacher Training Services for Public Schools",
                "organization": "Ministry of Education", "country": "Egypt"})
    web_search_returns(monkeypatch, {"Egypt": [EGYPT_PAGE]})

    result = ask(store, search={"countries": ["Egypt"]})

    assert result["opportunities"] == [] and result["duplicate_count"] == 1
    assert "already" in result["message"]
    assert len(store.search()) == 1


def test_a_known_tender_keeps_the_admin_decision(monkeypatch, store, llm, apis):
    web_search_returns(monkeypatch, {"Egypt": [EGYPT_PAGE]})
    ask(store, search={"countries": ["Egypt"]})
    opp_id = store.search()[0]["opportunity_id"]
    store.decide(opp_id, "rejected", "admin-session-1")

    ask(store, search={"countries": ["Egypt"]})

    assert store.get(opp_id)["approval_status"] == "rejected"


def test_arabic_response(monkeypatch, store, llm, apis):
    web_search_returns(monkeypatch, {"Egypt": [EGYPT_PAGE]})

    result = ask(store, "ابحث عن مناقصات تدريب في مصر", language="ar")

    assert "بانتظار مراجعة" in result["message"]


def test_when_every_source_fails_it_says_so(monkeypatch, store, llm):
    monkeypatch.setattr(tender_tools, "query_world_bank_egypt", lambda q: APIQueryResult(
        source_name="World Bank", status="error", message="timed out"))
    monkeypatch.setattr(tender_tools, "query_etimad_saudi_arabia", lambda q: APIQueryResult(
        source_name="Etimad", status="unavailable", message="No public API."))
    web_search_returns(monkeypatch, {}, status="error")

    result = ask(store)

    assert "NOT the same as finding zero" in result["message"]
    assert result["web_research"]["status"] == "error"
    assert llm.extraction.calls == []


def test_unconfigured_web_search_is_reported(store, llm, apis):
    # conftest.py: the web search reports awaiting_credentials.
    result = ask(store, "Find training tenders in Saudi Arabia")

    assert result["web_research"]["status"] == "awaiting_credentials"
    assert "Web search: not configured" in result["message"]


def test_web_search_can_be_turned_off(monkeypatch, store, llm, apis):
    queries = web_search_returns(monkeypatch, {"Egypt": [EGYPT_PAGE]})
    agent = RFPResearchAgent(opportunity_store=store, model="test", web_search=False)

    result = agent.process_request({"request_id": "r", "user_input": "Egypt training tenders"})["result"]

    assert queries == [] and result["web_research"]["status"] == "disabled"


# --------------------------------------------------------------------
# The graph reaches sources and the catalogue only through the tools
# --------------------------------------------------------------------

class RecordingTools(tender_tools.LocalTenderTools):
    """The in-process tools, recording which tool each node called."""

    def __init__(self, store, fail=()):
        super().__init__(store)
        self.calls, self.fail = [], set(fail)

    def _record(self, name):
        self.calls.append(name)
        if name in self.fail:
            from agents.mcp_tool_client import McpToolError
            raise McpToolError("The tender MCP server is not available.")

    def search_procurement_notices(self, country, query):
        self._record("search_procurement_notices")
        return super().search_procurement_notices(country, query)

    def search_tender_web_pages(self, countries, categories=None, priority=None):
        self._record("search_tender_web_pages")
        return super().search_tender_web_pages(countries, categories, priority)

    def find_known_tenders(self, tenders):
        self._record("find_known_tenders")
        return super().find_known_tenders(tenders)

    def save_discovered_tenders(self, tenders):
        self._record("save_discovered_tenders")
        return super().save_discovered_tenders(tenders)


def run_with(tools, store):
    agent = RFPResearchAgent(opportunity_store=store, model="test", tools=tools)
    return agent.process_request({"request_id": "r", "user_input": "Find training tenders",
                                  "context": {"search": {"countries": ["Egypt"]}}})["result"]


def test_every_source_and_catalogue_step_is_a_tool_call(monkeypatch, store, llm, apis):
    web_search_returns(monkeypatch, {"Egypt": [EGYPT_PAGE]})
    tools = RecordingTools(store)

    result = run_with(tools, store)

    assert tools.calls == ["search_procurement_notices", "search_tender_web_pages",
                           "find_known_tenders", "save_discovered_tenders"]
    assert result["newly_discovered_count"] == 1 and store.search()[0]["approval_status"] == "pending"


def test_a_failed_save_is_never_reported_as_waiting_for_review(monkeypatch, store, llm, apis):
    web_search_returns(monkeypatch, {"Egypt": [EGYPT_PAGE]})

    result = run_with(RecordingTools(store, fail={"save_discovered_tenders"}), store)

    assert "could NOT be saved" in result["message"]
    assert "waiting for admin review" not in result["message"]
    assert store.search() == []


def test_unreachable_tools_are_reported_as_failed_sources(monkeypatch, store, llm, apis):
    tools = RecordingTools(store, fail={"search_procurement_notices", "search_tender_web_pages"})

    result = run_with(tools, store)

    assert "NOT the same as finding zero" in result["message"]
    assert {r["status"] for r in result["api_results"]} == {"error"}


def test_a_failed_catalogue_check_drops_nothing_and_saving_still_dedupes(monkeypatch, store, llm, apis):
    store.save({"url": "https://tendersinfo.com/eg/55", "title": "Teacher Training Services for Public Schools",
                "organization": "Ministry of Education", "country": "Egypt"})
    web_search_returns(monkeypatch, {"Egypt": [EGYPT_PAGE]})

    result = run_with(RecordingTools(store, fail={"find_known_tenders"}), store)

    assert result["newly_discovered_count"] == 0  # OpportunityStore.save refused the duplicate
    assert len(store.search()) == 1


def test_the_agent_uses_the_tender_mcp_server_by_default(monkeypatch, store, real_default_tender_tools):
    from agents.rfp_research import agent as agent_module
    from agents.rfp_research.mcp_client import TenderMcpClient

    monkeypatch.setattr(agent_module, "default_tender_tools", real_default_tender_tools)
    agent = agent_module.RFPResearchAgent(opportunity_store=store, model="test")

    assert isinstance(agent.tools, TenderMcpClient)
    # The server is told to use the agent's own database file.
    assert agent.tools._server_params.env == {"REX_TENDER_DB_PATH": str(store.db_path)}
    agent.close()  # never started: nothing to stop


# --------------------------------------------------------------------
# Live-search regression: a training ORGANISATION buying archiving
# services is not a training tender; readable, honest reply
# --------------------------------------------------------------------

ARCHIVING_PAGE = web_page(
    "https://tenderimpulse.com/government-tenders/saudi-arab/archives-12461035",
    "Implementing the requirements of the Documents and Archives Center",
    content="The tender has been issued by General Corporation for Technical and Vocational Training "
            "in Saudi Arabia. Such a type of tender falls under Document management system, within the "
            "IT-Software sector.",
)
ARCHIVING = ExtractedTender(
    title="Implementing the requirements of the Documents and Archives Center, organizing, archiving, and destroying",
    country="Saudi Arabia", training_category="Software and IT Training")


def test_a_training_organisation_buying_archiving_is_not_a_training_tender(monkeypatch, store, llm, apis):
    llm.extraction.tenders = {ARCHIVING_PAGE.url: [ARCHIVING]}
    web_search_returns(monkeypatch, {"Saudi Arabia": [ARCHIVING_PAGE]})

    result = ask(store, search={"countries": ["Saudi Arabia"], "categories": ["Software and IT Training"]})

    assert result["opportunities"] == [] and store.search() == []
    assert result["filter_report"]["not_training"] == 1
    assert "1 not training (by title)" in result["message"]
    assert llm.relevance.calls == []  # dropped by code, before the AI check


@pytest.mark.parametrize("title, training", [
    ("Supply of ERP software licences for the Ministry", False),
    ("Cloud migration of government data centres", False),
    ("Cloud computing training for government IT staff", True),
    ("E-learning platform for universities", True),
    ("برنامج تدريب المعلمين على المهارات الرقمية", True),
])
def test_topic_words_alone_are_not_training(title, training):
    from agents.rfp_research.state import names_training

    assert names_training(title) is training


def test_the_ai_category_label_must_be_supported_by_the_tender_text():
    page = web_page("https://etenders.gov.eg/t/42", "Python programming training for teachers")
    tenders = {page.url: [ExtractedTender(title="Python programming training for teachers",
                                          country="Egypt", training_category="Corporate Learning")]}

    [record] = web_tenders.page_to_records(page, "Egypt", Extraction(tenders), rfp_graph._detect_requested_countries)

    assert record["training_category"] == "Software and IT Training"  # from "programming", not the AI label


def test_the_reply_explains_what_was_checked_and_warns_about_missing_deadlines(monkeypatch, store, llm, apis):
    page = web_page("https://tendersinfo.com/sa/it", "Saudi Arabia IT training tenders")
    llm.extraction.tenders = {page.url: [
        ExtractedTender(title="Cybersecurity training for government staff", country="Saudi Arabia"),  # no deadline
        ExtractedTender(title="Cloud computing training programme", country="Saudi Arabia",
                        deadline=(date.today() + timedelta(days=300)).isoformat()),                  # outside dates
        ExtractedTender(title="Leadership training for managers", country="Saudi Arabia"),          # other category
    ]}
    web_search_returns(monkeypatch, {"Saudi Arabia": [page]})

    result = ask(store, search={"countries": ["Saudi Arabia"], "categories": ["Software and IT Training"],
                                "deadline_to": (date.today() + timedelta(days=60)).isoformat()})

    lines = result["message"].split("\n")
    assert lines[0] == "1 new tender(s) waiting for admin review."
    assert lines[1] == "• Saudi Arabia: 1 (monitored)"
    assert lines[2].startswith("⚠ 1 of these tender(s) state no deadline")
    assert lines[3] == ("What was checked: 3 tender(s) read → 1 outside your dates · "
                        "1 outside the chosen categories")
    assert lines[4].startswith("Sources: Etimad: no public API")
    assert "Web search: 4 of 4 searches answered, 1 page(s) read" in lines[4]


def test_the_reply_lines_in_arabic(monkeypatch, store, llm, apis):
    web_search_returns(monkeypatch, {"Egypt": [EGYPT_PAGE]})

    result = ask(store, "ابحث عن مناقصات تدريب في مصر", language="ar")

    lines = result["message"].split("\n")
    assert lines[0] == "1 مناقصة جديدة بانتظار مراجعة المسؤول."
    assert lines[1] == "• مصر: 1 (مراجعة فورية)"
    assert lines[-1].startswith("المصادر: البنك الدولي: 0 إعلان")


@pytest.mark.parametrize("title, works", [
    ("Renting 3 rooms on the second floor of the Technical Training and Skills Development Center", True),
    ("تأجير قاعة تدريب في مركز التدريب", True),
    ("Supply of specialized medical training courses", False),
    ("Leadership training for managers", False),
])
def test_rentals_are_never_training_services(title, works):
    assert rfp_graph._is_works_contract({"title": title}) is works


# --------------------------------------------------------------------
# Tender titles: generic page headings and menu labels are never titles
# --------------------------------------------------------------------

# What Tavily returned for an Etimad tender page (live, 2026-10-27): only
# the site's menu and the heading -- the details load with JavaScript.
ETIMAD_SHELL_TEXT = (
    "![loader](/Etimad-UI/assets/imgs/loader.gif)\n\n##### معلومات المورد\n\n*تحميل البيانات*\n\n"
    "* menu\n* close\n\n* [person تسجيل الدخول](/Dashboard/Index)\n\n"
    "* [المنافسات](/Tender/AllTendersForVisitor)\n* [دعوات التأهيل](/Qualification/QualificationsForVisitor)\n"
    "* [الإعلانات](/Announcement/AllVisitorAnnouncements)\n\n## تفاصيل المنافسة\n\n"
    "[*assignment* تفاصيل المنافسة](OpenTenderDetailsReportForVisitor?tenderIdString=x2Ag== \"تفاصيل المنافسة\")\n\n---"
)
ETIMAD_URL = "https://tenders.etimad.sa/Tender/DetailsForVisitor?STenderId=x2AgGsr"


def etimad_page(url=ETIMAD_URL, text=ETIMAD_SHELL_TEXT):
    return web_page(url, "تفاصيل المنافسة", content=text[:300], raw=text)


def records_for(page, tenders, country="Saudi Arabia", unreadable=None):
    return web_tenders.page_to_records(page, country, Extraction({page.url: tenders}),
                                       rfp_graph._detect_requested_countries, unreadable_pages=unreadable)


@pytest.mark.parametrize("title", ["تفاصيل المنافسة", "المنافسات المطروحة", "Tender Details", "## TENDERS",
                                   "تفاصيل المنافسه", None, "  ",
                                   # Listing-page headings naming a group (live, 2026-10-27):
                                   "Saudi Arabia Education - Educational Training Tenders",
                                   "Egypt IT Training Tender Notices", "مناقصات التدريب والتعليم في السعودية"])
def test_generic_headings_are_not_titles(title):
    assert web_tenders.is_generic_title(title)


@pytest.mark.parametrize("title", ["Cybersecurity training for government staff", "تدريب منسوبي الجهة",
                                   "مناقصة تدريب الموظفين على الأمن السيبراني",
                                   "Training services for tender evaluation committees"])
def test_real_names_are_titles(title):
    assert not web_tenders.is_generic_title(title)


def test_a_listing_page_heading_is_not_saved_as_a_tender(monkeypatch, store, llm, apis):
    page = web_page("https://www.globaltenders.com/saudi-arabia-education-training-tenders",
                    "Saudi Arabia Education - Educational Training Tenders")
    llm.extraction.tenders = {page.url: [
        ExtractedTender(title="Saudi Arabia Education - Educational Training Tenders", country="Saudi Arabia")]}
    web_search_returns(monkeypatch, {"Saudi Arabia": [page]})

    result = ask(store, search={"countries": ["Saudi Arabia"]})

    assert result["opportunities"] == [] and store.search() == []
    assert result["web_research"]["unreadable_pages"] == 1


def test_an_arabic_generic_heading_is_replaced_by_the_labelled_tender_name():
    text = ("## تفاصيل المنافسة\n\nاسم المنافسة: تدريب منسوبي الجهة على الأمن السيبراني\n\n"
            "الرقم المرجعي: 261039001234\n\nالجهة الحكومية: وزارة الموارد البشرية")
    page = etimad_page("https://tenders.etimad.sa/Tender/DetailsForVisitor?STenderId=AR1", text)
    unreadable = set()

    [record] = records_for(page, [ExtractedTender(title="تفاصيل المنافسة", country="Saudi Arabia",
                                                  reference_number="261039001234",
                                                  issuing_organization="وزارة الموارد البشرية")],
                           unreadable=unreadable)

    assert record["title"] == "تدريب منسوبي الجهة على الأمن السيبراني"
    assert (record["reference_number"], record["organization"]) == ("261039001234", "وزارة الموارد البشرية")
    assert record["url"] == page.url and record["source_url"] == page.url
    assert record["language"] == "ar" and unreadable == set()


def test_an_english_generic_heading_is_replaced_from_a_metadata_block():
    text = ("# Tender Details\n\n| Field | Value |\n\n**Tender Name**\n\n"
            "Cloud Computing Training for Government IT Staff\n\n**Reference Number**\n\n2610-77\n\n"
            "**Closing Date**\n\n2030-12-01")
    page = web_page("https://tenderimpulse.com/government-tenders/saudi-arab/cloud-77", "Tender Details",
                    content=text, raw=text)

    [record] = records_for(page, [ExtractedTender(title="Tender Details", country="Saudi Arabia",
                                                  reference_number="2610-77", deadline="2030-12-01")])

    assert record["title"] == "Cloud Computing Training for Government IT Staff"
    assert record["deadline"] == "2030-12-01" and record["reference_number"] == "2610-77"


@pytest.mark.parametrize("extracted", [
    [ExtractedTender(title="تفاصيل المنافسة", country="Saudi Arabia")],  # the heading copied as a title
    [ExtractedTender(title=None, country="Saudi Arabia")],              # no name stated
    [],                                                                  # nothing extracted
], ids=["heading-as-title", "no-title", "nothing-extracted"])
def test_the_real_etimad_page_gives_no_record_and_counts_as_unreadable_once(extracted):
    unreadable = set()

    assert records_for(etimad_page(), extracted, unreadable=unreadable) == []
    assert records_for(etimad_page(), extracted, unreadable=unreadable) == []  # the same URL again

    assert unreadable == {ETIMAD_URL}


def test_misleading_navigation_text_never_becomes_the_title():
    text = ("# Tenders\n\n* [Training Courses](/courses)\n* [Tenders](/tenders)\n* [Login](/login)\n\n"
            "Loading data...")
    page = web_page("https://tenders.example.sa/t/9", "Tenders", content=text, raw=text)
    unreadable = set()

    # The AI picks a menu item as the title: it is only link text, so no record.
    assert records_for(page, [ExtractedTender(title="Training Courses", country="Saudi Arabia")],
                       unreadable=unreadable) == []
    assert unreadable == {page.url}


def test_listing_page_items_that_are_links_are_kept():
    text = ("## Saudi Arabia training tenders\n\n* [Leadership Training for Managers](/t/1)\n"
            "* [Cyber Training Programme](/t/2)")
    page = web_page("https://www.globaltenders.com/saudi-training", "Saudi training tenders", content=text, raw=text)

    records = records_for(page, [ExtractedTender(title="Leadership Training for Managers", country="Saudi Arabia"),
                                 ExtractedTender(title="Cyber Training Programme", country="Saudi Arabia")])

    assert [r["title"] for r in records] == ["Leadership Training for Managers", "Cyber Training Programme"]


def test_unreadable_pages_are_reported_apart_from_filtered_tenders(monkeypatch, store, llm, apis):
    """
    Through the whole agent: an Etimad shell (unreadable); recovered
    titles that are a software supply and a rental (read, then filtered);
    a maintenance contract (filtered); a genuine Saudi IT training tender
    (kept).
    """
    supply_text = "## تفاصيل المنافسة\n\nاسم المنافسة: Supply of ERP software licences for the Ministry"
    supply = web_page("https://tenders.etimad.sa/Tender/DetailsForVisitor?STenderId=SUP", "تفاصيل المنافسة",
                      content=supply_text, raw=supply_text)
    rental_text = "## تفاصيل المنافسة\n\nاسم المنافسة: تأجير قاعات في مركز التدريب التقني"
    rental = web_page("https://tenders.etimad.sa/Tender/DetailsForVisitor?STenderId=RNT", "تفاصيل المنافسة",
                      content=rental_text, raw=rental_text)
    maintenance = web_page("https://tendersinfo.com/sa/ac-1", "Preventive maintenance of air conditioning systems")
    training = web_page("https://tendersinfo.com/sa/cyber-1", "Cybersecurity training programme for ministry staff")
    llm.extraction.tenders = {
        ETIMAD_URL: [ExtractedTender(title="تفاصيل المنافسة", country="Saudi Arabia")],
        supply.url: [ExtractedTender(title="تفاصيل المنافسة", country="Saudi Arabia",
                                     training_category="Software and IT Training")],
        rental.url: [ExtractedTender(title="تفاصيل المنافسة", country="Saudi Arabia")],
        maintenance.url: [ExtractedTender(title="Preventive maintenance of air conditioning systems",
                                          country="Saudi Arabia")],
        training.url: [ExtractedTender(title="Cybersecurity training programme for ministry staff",
                                       country="Saudi Arabia", deadline=FUTURE)],
    }
    web_search_returns(monkeypatch, {"Saudi Arabia": [etimad_page(), supply, rental, maintenance, training]})

    result = ask(store, search={"countries": ["Saudi Arabia"]})

    assert [o["title"] for o in result["opportunities"]] == ["Cybersecurity training programme for ministry staff"]
    assert result["web_research"]["unreadable_pages"] == 1
    assert result["web_research"]["unreadable_from_etimad"] == 1
    # Read and filtered -- not "unreadable".
    assert result["filter_report"]["not_training"] == 2  # the software supply, the maintenance contract
    assert result["filter_report"]["works"] == 1          # the rental, though its title names a training centre
    lines = result["message"].split("\n")
    assert any(line.startswith("ℹ 1 web page(s) had no readable tender name and were skipped (1 from Etimad")
               for line in lines)
    assert ("What was checked: 4 tender(s) read → 2 not training (by title) · "
            "1 works or supply contracts") in lines
    assert [r["title"] for r in store.search()] == ["Cybersecurity training programme for ministry staff"]


def test_the_extraction_schema_asks_for_the_official_name():
    description = ExtractedTender.model_fields["title"].description

    assert "official name" in description and "تفاصيل المنافسة" in description and "Tender Details" in description
    assert ExtractedTender(country="Saudi Arabia").title is None  # an unstated title stays empty


# --------------------------------------------------------------------
# A training word in an organisation's or place's name does not make a
# works contract a training tender
# --------------------------------------------------------------------

@pytest.mark.parametrize("title, works", [
    # Live search, 2026-10-27: the buyer's name contains "Training".
    ("Operating, Maintaining And Cleaning The Computerized Testing Facilities At The "
     "Education And Training Evaluation Authority", True),
    ("Construction of the Vocational Training Institute", True),
    ("Maintenance of training centre air conditioning", True),
    ("Integrated facilities management for the Technical Training Corporation", True),
    ("صيانة مباني المؤسسة العامة للتدريب التقني والمهني", True),
    ("إنشاء مركز تدريب", True),
    # The training word describes the work itself: still training.
    ("Supply of specialized medical training courses for the Ministry", False),
    ("Supply Of Consumer Requirements For Specialized Medical Training Courses For The Ongoing "
     "Professional Development Programme", False),
    ("Leadership Training At The Institute Of Public Administration", False),
    ("Training of trainers at the Training Center", False),
    ("IT training for the Ministry of Housing", False),
    ("Operation, Maintenance And Development Of The National E-Learning Platform", False),
    ("توريد دورات تدريبية لمنسوبي وزارة الصحة", False),
])
def test_an_institution_name_cannot_override_a_works_contract(title, works):
    assert rfp_graph._is_works_contract({"title": title}) is works


def test_a_works_contract_for_a_training_body_is_dropped_but_its_training_tender_kept(monkeypatch, store, llm, apis):
    page = web_page("https://www.globaltenders.com/saudi-education", "Saudi Arabia education tenders")
    llm.extraction.tenders = {page.url: [
        ExtractedTender(title="Operating, Maintaining And Cleaning The Computerized Testing Facilities At The "
                              "Education And Training Evaluation Authority", country="Saudi Arabia"),
        ExtractedTender(title="Assessor Training Programme For The Education And Training Evaluation Authority",
                        country="Saudi Arabia"),
    ]}
    web_search_returns(monkeypatch, {"Saudi Arabia": [page]})

    result = ask(store, search={"countries": ["Saudi Arabia"]})

    assert [o["title"] for o in result["opportunities"]] == [
        "Assessor Training Programme For The Education And Training Evaluation Authority"]
    assert result["filter_report"]["works"] == 1
    assert len(llm.relevance.calls) == 1  # the works contract never reached the AI check
