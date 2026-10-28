"""
Tavily web search client (agents/rfp_research/api_clients/tavily.py).

Mocked: Tavily HTTP (httpx.MockTransport). The key is a fake value.
"""

import json

import httpx
import pytest

from agents.rfp_research.api_clients import tavily
from agents.rfp_research.api_clients.tavily import (
    MAX_CONTENT_CHARS,
    MAX_RAW_CONTENT_CHARS,
    search_tavily,
)


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setattr(tavily, "get_secret", lambda name: "tvly-test-key")


def client_returning(status=200, body=None, raise_exc=None, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        if raise_exc:
            raise raise_exc
        return httpx.Response(status, json=body if body is not None else {})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_missing_key_is_reported_not_treated_as_no_results(monkeypatch):
    monkeypatch.setattr(tavily, "get_secret", lambda name: None)

    result = search_tavily("education tenders Egypt")

    assert result.status == "awaiting_credentials"
    assert result.pages == []


def test_successful_search_returns_clean_pages(with_key):
    seen = []
    body = {"results": [
        {"title": "Training tender", "url": "https://etenders.gov.eg/t/1",
         "content": "Ministry seeks training\x07 provider.", "score": 0.8,
         "published_date": "2026-09-01"},
        {"title": "Bad scheme", "url": "javascript:alert(1)", "content": "x"},
        {"title": "No content", "url": "https://example.com/empty", "content": ""},
        {"title": "Long", "url": "https://example.com/long", "content": "a" * 10000},
    ]}

    result = search_tavily("education tenders Egypt", http_client=client_returning(body=body, seen=seen))

    assert result.status == "success"
    assert [p.url for p in result.pages] == ["https://etenders.gov.eg/t/1", "https://example.com/long"]
    assert "\x07" not in result.pages[0].content
    assert result.pages[0].source_domain == "etenders.gov.eg"
    assert len(result.pages[1].content) == MAX_CONTENT_CHARS
    # The key travels only in the Authorization header.
    assert seen[0].headers["Authorization"] == "Bearer tvly-test-key"
    assert "tvly-test-key" not in seen[0].content.decode()


def test_site_search_and_full_page_text(with_key):
    seen = []
    body = {"results": [{"title": "Tender", "url": "https://tenders.etimad.sa/t/9",
                         "content": "Short.", "raw_content": "Full text " + "b" * 20000}]}

    result = search_tavily("منافسة تدريب", include_domains=["tenders.etimad.sa"],
                           include_raw_content=True,
                           http_client=client_returning(body=body, seen=seen))

    sent = json.loads(seen[0].content)
    assert sent["include_domains"] == ["tenders.etimad.sa"]
    assert sent["include_raw_content"] is True
    assert len(result.pages[0].raw_content) == MAX_RAW_CONTENT_CHARS


@pytest.mark.parametrize("status, phrase", [
    (401, "rejected the API key"),
    (403, "rejected the API key"),
    (429, "rate limit"),
    (500, "failed"),
])
def test_http_failures_are_errors(with_key, status, phrase):
    result = search_tavily("q", http_client=client_returning(status=status))

    assert result.status == "error"
    assert phrase in result.message
    assert "tvly-test-key" not in result.message


def test_timeout_is_an_error(with_key):
    result = search_tavily("q", http_client=client_returning(raise_exc=httpx.ReadTimeout("slow")))

    assert result.status == "error"
    assert "timed out" in result.message


def test_malformed_response_is_an_error(with_key):
    result = search_tavily("q", http_client=client_returning(body={"unexpected": True}))

    assert result.status == "error"
