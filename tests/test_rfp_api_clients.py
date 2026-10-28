"""
Tests for agents.rfp_research.api_clients -- the real API connectors
that replaced web search/scraping as Agent 2's discovery mechanism.

query_world_bank_egypt makes a real HTTP call to the live World Bank
API, so these tests mock urllib.request.urlopen for determinism. A
separate, manually-run live check against the real API is reported in
the final report, not here.

query_etimad_saudi_arabia makes no network call at all -- it always
reports "unavailable" (see that module's docstring for the
investigation behind that), so its tests need no mocking.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.rfp_research.api_clients.etimad import query_etimad_saudi_arabia
from agents.rfp_research.api_clients.world_bank import query_world_bank_egypt


def _mock_response(payload: dict):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    return mock_resp


def _raw_record(
    id_="OP001",
    country="Egypt, Arab Republic of",
    notice_type="Invitation for Bids",
    deadline="2027-01-01T00:00:00Z",
    noticedate="15-Jan-2024",
):
    record = {
        "id": id_,
        "notice_type": notice_type,
        "notice_status": "Published",
        "project_ctry_name": country,
        "project_name": "Example Ministry Project",
        "bid_reference_no": "REF-001",
        "bid_description": "Digital skills training for public-sector staff.",
        "submission_deadline_date": deadline,
    }
    if noticedate:
        record["noticedate"] = noticedate
    return record


def test_world_bank_success_returns_normalized_records():
    payload = {"total": "1", "procnotices": [_raw_record()]}

    with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
        result = query_world_bank_egypt("training education")

    assert result.status == "success"
    assert len(result.records) == 1
    record = result.records[0]
    assert record.country == "Egypt"
    assert record.source_id == "OP001"
    assert record.reference_number == "REF-001"
    assert record.deadline == "2027-01-01"
    assert record.publication_date == "2024-01-15"
    assert record.source_name.startswith("World Bank")
    # Not provided by this API for any record -- always flagged, never
    # silently left as an empty list that looks like "confirmed none."
    assert "eligibility" in record.missing_information
    assert "available_documents" in record.missing_information


def test_world_bank_prefers_contact_organization_over_project_name():
    raw = _raw_record()
    raw["contact_organization"] = "Egyptian Environmental Affairs Agency"

    with patch(
        "urllib.request.urlopen",
        return_value=_mock_response({"total": "1", "procnotices": [raw]}),
    ):
        result = query_world_bank_egypt("training")

    assert result.records[0].organization == "Egyptian Environmental Affairs Agency"


def test_world_bank_extracts_submission_url_and_instructions_when_present():
    raw = _raw_record()
    raw["contact_web_url"] = "www.example-ministry.gov.eg"
    raw["contact_name"] = "Jane Doe"
    raw["contact_email"] = "jane@example-ministry.gov.eg"

    with patch(
        "urllib.request.urlopen",
        return_value=_mock_response({"total": "1", "procnotices": [raw]}),
    ):
        result = query_world_bank_egypt("training")

    record = result.records[0]
    assert record.submission_url == "www.example-ministry.gov.eg"
    assert "Jane Doe" in record.submission_instructions
    assert "jane@example-ministry.gov.eg" in record.submission_instructions
    assert "submission_url" not in record.missing_information
    assert "submission_instructions" not in record.missing_information


def test_world_bank_drops_non_egypt_records_despite_text_match():
    """
    Regression-style guard: the free-text qterm search can return
    loose matches from other countries; the client must verify
    project_ctry_name itself, not trust the query alone.
    """
    payload = {
        "total": "1",
        "procnotices": [_raw_record(country="Yemen, Republic of")],
    }

    with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
        result = query_world_bank_egypt("training")

    assert result.status == "success"
    assert result.records == []
    assert result.raw_record_count == 1


def test_world_bank_excludes_historical_award_notices():
    payload = {
        "total": "1",
        "procnotices": [_raw_record(notice_type="Contract Award")],
    }

    with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
        result = query_world_bank_egypt("training")

    assert result.records == []


def test_world_bank_flags_missing_fields_without_inventing_them():
    raw = _raw_record(deadline=None, noticedate=None)
    raw["bid_reference_no"] = None
    raw["project_name"] = None
    payload = {"total": "1", "procnotices": [raw]}

    with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
        result = query_world_bank_egypt("training")

    record = result.records[0]
    assert record.deadline is None
    assert record.organization is None
    assert record.reference_number is None
    assert record.publication_date is None
    assert record.submission_url is None
    assert record.submission_instructions is None
    # Every one of the nine required fields this API cannot supply is
    # explicitly marked -- never silently invented or left ambiguous.
    assert set(record.missing_information) == {
        "deadline", "organization", "reference_number",
        "publication_date", "submission_url", "submission_instructions",
        "eligibility", "available_documents",
    }


def test_world_bank_network_failure_returns_error_status_not_empty_success():
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        result = query_world_bank_egypt("training")

    assert result.status == "error"
    assert "timed out" in result.message
    assert result.records == []


def test_world_bank_malformed_response_returns_error_status():
    with patch(
        "urllib.request.urlopen",
        return_value=_mock_response({"procnotices": "not a list"}),
    ):
        result = query_world_bank_egypt("training")

    assert result.status == "error"


def test_etimad_always_reports_unavailable_not_empty_success():
    result = query_etimad_saudi_arabia("training")

    assert result.status == "unavailable"
    assert "No documented, publicly accessible API" in result.message


def test_etimad_notes_when_api_key_is_configured(monkeypatch):
    monkeypatch.setenv("ETIMAD_API_KEY", "fake-key-for-test")

    result = query_etimad_saudi_arabia("training")

    assert result.status == "unavailable"
    assert "ETIMAD_API_KEY is set" in result.message


def test_etimad_makes_no_network_call():
    with patch("urllib.request.urlopen") as mock_urlopen:
        query_etimad_saudi_arabia("training")

    mock_urlopen.assert_not_called()
