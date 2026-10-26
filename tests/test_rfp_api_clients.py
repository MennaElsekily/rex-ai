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
):
    return {
        "id": id_,
        "notice_type": notice_type,
        "notice_status": "Published",
        "project_ctry_name": country,
        "project_name": "Example Ministry Project",
        "bid_reference_no": "REF-001",
        "bid_description": "Digital skills training for public-sector staff.",
        "submission_deadline_date": deadline,
    }


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
    assert record.source_name.startswith("World Bank")


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
    raw = _raw_record(deadline=None)
    raw["bid_reference_no"] = None
    raw["project_name"] = None
    payload = {"total": "1", "procnotices": [raw]}

    with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
        result = query_world_bank_egypt("training")

    record = result.records[0]
    assert record.deadline is None
    assert record.organization is None
    assert record.reference_number is None
    assert set(record.missing_information) == {
        "deadline", "organization", "reference_number",
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
