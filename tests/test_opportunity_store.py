"""
Tests for agents.rfp_research.opportunity_store.OpportunityStore --
the searchable, persistent catalog of discovered tender opportunities
(brief requirement 7).

qualification_status is stored as a real 3-state string
("qualified"/"not_qualified"/"unknown"). An earlier version of this
store used `1 if qualifies else 0`, which silently collapsed an
"unknown" assessment into the same stored value as "not qualified" --
test_unknown_qualification_round_trips_correctly guards against that
regression.
"""

import pytest

from agents.rfp_research.opportunity_store import OpportunityStore


@pytest.fixture
def store(tmp_path):
    return OpportunityStore(db_path=tmp_path / "opportunities.sqlite3")


def _opportunity(url, country="Egypt", title="Test Opportunity", qualification_status="qualified"):
    return {
        "url": url,
        "title": title,
        "reference_number": "REF-001",
        "organization": "Example Org",
        "country": country,
        "category": "Education and training",
        "scope_of_work": "Deliver training.",
        "deadline": None,
        "eligibility": ["Relevant training capability"],
        "publication_date": None,
        "submission_instructions": None,
        "submission_url": None,
        "available_documents": [],
        "ai_summary": "A summary.",
        "qualification_status": qualification_status,
        "qualification_reason": "Matches eligibility.",
        "missing_information": [],
        "verification_status": "verified_from_source_page",
        "review_status": "flagged_for_immediate_review",
    }


def test_save_reports_new_opportunity(store):
    is_new = store.save(_opportunity("https://example.com/t1"))
    assert is_new is True


def test_save_reports_not_new_on_repeat(store):
    store.save(_opportunity("https://example.com/t1"))
    is_new = store.save(_opportunity("https://example.com/t1"))
    assert is_new is False


def test_save_requires_url(store):
    with pytest.raises(ValueError):
        store.save(_opportunity(url=None))


def test_search_by_country(store):
    store.save(_opportunity("https://example.com/eg", country="Egypt"))
    store.save(_opportunity("https://example.com/sa", country="Saudi Arabia"))

    results = store.search(country="Egypt")
    assert len(results) == 1
    assert results[0]["country"] == "Egypt"


def test_search_by_keyword(store):
    store.save(_opportunity("https://example.com/t1", title="Digital Skills Tender"))
    store.save(_opportunity("https://example.com/t2", title="Bridge Construction"))

    results = store.search(keyword="Digital")
    assert len(results) == 1
    assert "Digital" in results[0]["title"]


def test_stored_record_round_trips_lists(store):
    opportunity = _opportunity("https://example.com/t1")
    opportunity["eligibility"] = ["A", "B"]
    opportunity["available_documents"] = ["Annex 1"]
    opportunity["missing_information"] = ["deadline"]

    store.save(opportunity)
    results = store.search()

    assert results[0]["eligibility"] == ["A", "B"]
    assert results[0]["available_documents"] == ["Annex 1"]
    assert results[0]["missing_information"] == ["deadline"]


def test_unknown_qualification_round_trips_correctly(store):
    """
    Regression test: qualification_status="unknown" must come back as
    "unknown", not collapse into the same stored value as
    "not_qualified" (the bug in the previous boolean-based schema).
    """
    store.save(
        _opportunity("https://example.com/t1", qualification_status="unknown")
    )
    store.save(
        _opportunity(
            "https://example.com/t2", title="A Different Training Opportunity",
            qualification_status="not_qualified",
        )
    )

    results = {r["url"]: r for r in store.search()}
    assert results["https://example.com/t1"]["qualification_status"] == "unknown"
    assert results["https://example.com/t2"]["qualification_status"] == "not_qualified"


def test_saves_api_sourced_record_without_literal_url(store):
    """
    API-sourced records (world_bank.py) always carry source_id +
    source_name, and usually a source_url, but not necessarily a
    bare "url" key. The store must key on source_url, or synthesize a
    stable key from source_name+source_id, rather than require "url".
    """
    opportunity = {
        "title": "Digital Skills Tender",
        "source_id": "OP001",
        "source_name": "World Bank Procurement Notices API",
        "source_url": "https://search.worldbank.org/notice/OP001",
        "country": "Egypt",
        "qualification_status": "qualified",
        "review_status": "flagged_for_immediate_review",
    }

    is_new = store.save(opportunity)
    assert is_new is True

    is_new_again = store.save(opportunity)
    assert is_new_again is False


def test_tests_never_open_the_real_tender_database(tmp_path):
    from agents.rfp_research import opportunity_store

    real = opportunity_store.Path(opportunity_store.__file__).resolve().parent / "rfp_opportunities.sqlite3"
    store = OpportunityStore()  # no path: the default

    assert store.db_path != real
    assert store.db_path.parent == tmp_path  # conftest.py's temporary file
    store.close()


def test_persists_across_instances_on_same_file(tmp_path):
    db_path = tmp_path / "shared.sqlite3"

    store1 = OpportunityStore(db_path=db_path)
    store1.save(_opportunity("https://example.com/t1"))
    store1.close()

    store2 = OpportunityStore(db_path=db_path)
    results = store2.search()

    assert len(results) == 1


# --------------------------------------------------------------------
# Admin approval workflow, cross-source duplicates and alerts
# --------------------------------------------------------------------

def test_new_tenders_are_pending(store):
    store.save(_opportunity("https://example.com/t1"))

    row = store.search()[0]
    assert row["approval_status"] == "pending"
    assert row["opportunity_id"]


def test_decisions_are_audited_and_kept_on_rediscovery(store):
    store.save(_opportunity("https://example.com/t1"))
    opp_id = store.search()[0]["opportunity_id"]

    store.decide(opp_id, "approved", "admin-session-1")
    store.save(_opportunity("https://example.com/t1"))  # found again

    assert store.get(opp_id)["approval_status"] == "approved"
    trail = store.audit_trail(opp_id)
    assert [(a["previous_status"], a["decision"], a["reviewer"]) for a in trail] == [
        ("pending", "approved", "admin-session-1")
    ]


@pytest.mark.parametrize("decision, reviewer, error", [
    ("approved", "", ValueError), ("published", "admin", ValueError),
])
def test_invalid_decisions_are_rejected(store, decision, reviewer, error):
    store.save(_opportunity("https://example.com/t1"))
    opp_id = store.search()[0]["opportunity_id"]

    with pytest.raises(error):
        store.decide(opp_id, decision, reviewer)


def test_unknown_tender_cannot_be_decided(store):
    with pytest.raises(KeyError):
        store.decide("nope", "approved", "admin")


def test_dashboard_search_by_approval_status(store):
    store.save(_opportunity("https://example.com/t1", title="Teacher Training Programme Tender"))
    store.save(_opportunity("https://example.com/t2", title="Corporate Learning Platform Tender"))
    first = store.search(keyword="Teacher")[0]["opportunity_id"]
    store.decide(first, "approved", "admin")

    assert [r["title"] for r in store.search(approval_status="approved")] == ["Teacher Training Programme Tender"]
    assert len(store.search(approval_status="pending")) == 1


def test_same_tender_on_another_site_is_a_duplicate(store):
    assert store.save(_opportunity("https://etenders.gov.eg/t/1", title="Teacher Training Services Tender")) is True

    again = _opportunity("https://tendersinfo.com/x/99", title="Teacher Training Services Tender")
    assert store.save(again) is False
    assert len(store.search()) == 1


def test_same_generic_title_from_different_issuers_is_not_a_duplicate(store):
    store.save({**_opportunity("https://a.example/1", title="Consultancy Services for Training"),
                "organization": "Ministry of Education"})
    store.save({**_opportunity("https://b.example/2", title="Consultancy Services for Training"),
                "organization": "Ministry of Health"})

    assert len(store.search()) == 2


def test_egypt_flagged_tenders_are_listed_first_and_counted(store):
    store.save({**_opportunity("https://a.example/sa", country="Saudi Arabia", title="Saudi Digital Skills Tender"),
                "review_status": "monitored_pipeline"})
    store.save({**_opportunity("https://a.example/eg", title="Egypt Teacher Training Tender"),
                "review_status": "flagged_for_immediate_review"})

    assert store.search()[0]["country"] == "Egypt"
    assert store.counts()["flagged_pending"] == 1


def test_alerts_show_new_tenders_until_marked_seen(store):
    store.save(_opportunity("https://example.com/t1", title="Teacher Training Programme Tender"))

    assert len(store.unseen_alerts()) == 1
    store.mark_alerts_seen()
    assert store.unseen_alerts() == []

    store.save(_opportunity("https://example.com/t2", title="Corporate Learning Platform Tender"))
    assert [a["url"] for a in store.unseen_alerts()] == ["https://example.com/t2"]


def test_old_database_rows_are_migrated_to_pending(tmp_path):
    import sqlite3

    db = tmp_path / "old.sqlite3"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE opportunities (url TEXT PRIMARY KEY, title TEXT, country TEXT)")
    connection.execute("INSERT INTO opportunities VALUES ('https://old.example/1', 'Old Training Tender', 'Egypt')")
    connection.commit()
    connection.close()

    from agents.rfp_research.opportunity_store import OpportunityStore
    rows = OpportunityStore(db_path=db).search()

    assert rows[0]["url"] == "https://old.example/1"
    assert rows[0]["approval_status"] == "pending" and rows[0]["opportunity_id"]
