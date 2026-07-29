from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest
from pydantic import SecretStr

from financial_analyst.market import YFinanceClient
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

        def history(self, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr("financial_analyst.market.yf.Ticker", lambda ticker: EmptyTicker())
    result = YFinanceClient().market_snapshot("NONE")
    assert result.status is Availability.UNAVAILABLE
    assert result.values["price"] is None
    assert result.values["status"] == Availability.UNAVAILABLE.value


def test_market_snapshot_uses_latest_daily_bar_when_fast_quote_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HistoryTicker:
        def __init__(self) -> None:
            self.fast_info: dict[str, Any] = {}

        def history(self, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "Open": [98.0],
                    "High": [102.0],
                    "Low": [97.0],
                    "Close": [101.0],
                    "Volume": [1_250_000],
                },
                index=pd.to_datetime(["2026-07-28"]),
            )

    monkeypatch.setattr("financial_analyst.market.yf.Ticker", lambda ticker: HistoryTicker())
    result = YFinanceClient().market_snapshot("MSFT")
    assert result.status is Availability.PARTIAL
    assert result.values["price"] == 101.0
    assert result.values["price_basis"] == "latest daily close"
    assert result.values["trading_date"] == "2026-07-28"


def test_financial_statements_align_annual_periods_and_derive_fcf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = pd.to_datetime(["2024-12-31", "2023-12-31", "2022-12-31"])

    class StatementTicker:
        info: ClassVar[dict[str, str]] = {
            "financialCurrency": "USD",
            "longName": "Fixture Company",
        }
        income_stmt: ClassVar[pd.DataFrame] = pd.DataFrame(
            {
                columns[0]: [121.0, 16.0, 10.4],
                columns[1]: [110.0, 13.0, 10.2],
                columns[2]: [100.0, 10.0, 10.0],
            },
            index=["Total Revenue", "Net Income", "Diluted Average Shares"],
        )
        cashflow: ClassVar[pd.DataFrame] = pd.DataFrame(
            {
                columns[0]: [25.0, -5.0],
                columns[1]: [21.0, -4.0],
                columns[2]: [18.0, -3.0],
            },
            index=["Operating Cash Flow", "Capital Expenditure"],
        )
        balance_sheet: ClassVar[pd.DataFrame] = pd.DataFrame(
            {
                columns[0]: [32.0, 22.0],
                columns[1]: [24.0, 27.0],
                columns[2]: [20.0, 30.0],
            },
            index=["Cash And Cash Equivalents", "Total Debt"],
        )

    monkeypatch.setattr("financial_analyst.market.yf.Ticker", lambda ticker: StatementTicker())
    result = YFinanceClient().financial_statements("MSFT")
    periods = result.values["annual_periods"]
    assert [period["period_end"] for period in periods] == [
        "2024-12-31",
        "2023-12-31",
        "2022-12-31",
    ]
    assert periods[0]["free_cash_flow"] == 20.0
    assert periods[2]["cash"] == 20.0


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
