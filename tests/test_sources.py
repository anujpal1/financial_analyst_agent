from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from financial_analyst.market import YFinanceClient, peer_comparison_unavailable
from financial_analyst.models import Availability
from financial_analyst.sec import SECClient
from financial_analyst.transcripts import FMPTranscriptClient

FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self.payload


class FakeSECSession:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.mapping = json.loads(
            (FIXTURES / "sec_company_tickers.json").read_text(encoding="utf-8")
        )
        self.facts = json.loads((FIXTURES / "sec_companyfacts.json").read_text(encoding="utf-8"))

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(url)
        return FakeResponse(self.mapping if "company_tickers" in url else self.facts)


def test_missing_sec_user_agent_is_handled_without_network() -> None:
    session = FakeSECSession()
    result = SECClient(
        user_agent=None,
        timeout=5,
        retry_count=0,
        session=session,
        minimum_interval_seconds=0,
    ).company_facts("MSFT")
    assert result.status is Availability.UNAVAILABLE
    assert "SEC_USER_AGENT" in result.message
    assert session.calls == []


def test_invalid_sec_user_agent_is_handled_without_network() -> None:
    session = FakeSECSession()
    result = SECClient(
        user_agent="anonymous-client",
        timeout=5,
        retry_count=0,
        session=session,
        minimum_interval_seconds=0,
    ).company_facts("MSFT")
    assert result.status is Availability.UNAVAILABLE
    assert "contact email" in result.message
    assert session.calls == []


def test_sec_fixture_preserves_annual_and_quarterly_context() -> None:
    session = FakeSECSession()
    client = SECClient(
        user_agent="TestApp test@example.com",
        timeout=5,
        retry_count=0,
        session=session,
        minimum_interval_seconds=0,
    )
    result = client.company_facts("MSFT")
    revenue = result.values["facts"]["revenue"]
    assert result.status is Availability.PARTIAL
    assert revenue["annual"]["form"] == "10-K"
    assert revenue["annual"]["period_end"] == "2025-06-30"
    assert revenue["quarterly"]["form"] == "10-Q"
    assert revenue["quarterly"]["filed"] == "2025-10-29"
    assert any(item.accession_number for item in result.evidence)
    assert any(item.fiscal_period == "Q1" for item in result.evidence)


def test_sec_response_cache_avoids_duplicate_requests() -> None:
    session = FakeSECSession()
    client = SECClient(
        user_agent="TestApp test@example.com",
        timeout=5,
        retry_count=0,
        session=session,
        minimum_interval_seconds=0,
    )
    client.company_facts("MSFT")
    client.company_facts("MSFT")
    assert len(session.calls) == 2


def test_missing_market_data_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyTicker:
        def __init__(self) -> None:
            self.fast_info = {}

    monkeypatch.setattr("financial_analyst.market.yf.Ticker", lambda ticker: EmptyTicker())
    result = YFinanceClient().market_snapshot("NONE")
    assert result.status is Availability.UNAVAILABLE
    assert result.values == {}


def test_peer_comparison_never_uses_etf_fallback() -> None:
    result = peer_comparison_unavailable("UNKNOWN")
    assert result.status is Availability.UNAVAILABLE
    assert "ETF" in result.message
    assert "SPY" not in result.model_dump_json()


def test_transcript_unavailable_is_labelled_as_transcript() -> None:
    result = FMPTranscriptClient(api_key=None, timeout=5).fetch("MSFT", 2025, 2)
    assert result.status is Availability.UNAVAILABLE
    assert result.content_type == "transcript"
    assert "Transcript unavailable" in result.message
    assert "news" not in result.model_dump_json().lower()


def test_optional_transcript_provider_fails_gracefully() -> None:
    class EmptySession:
        def get(self, url: str, **kwargs: Any) -> FakeResponse:
            return FakeResponse([])

    result = FMPTranscriptClient(
        api_key=SecretStr("optional-secret"),
        timeout=5,
        session=EmptySession(),
    ).fetch("MSFT", 2025, 2)
    assert result.status is Availability.UNAVAILABLE
    assert result.content_type == "transcript"
