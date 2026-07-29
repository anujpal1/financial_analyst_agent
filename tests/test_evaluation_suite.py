"""Compact offline evaluation suite for the stage-two research pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from financial_analyst.analytics import (
    assess_evidence_quality,
    build_dashboard,
    build_historical_analysis,
    build_scorecard,
)
from financial_analyst.evidence import (
    build_claims,
    build_evidence_catalog,
    build_source_records,
    validate_report,
    verify_claims,
)
from financial_analyst.market import YFinanceClient
from financial_analyst.models import (
    Availability,
    Claim,
    ConfidenceCategory,
    DataResult,
    EvidenceRef,
    ResearchRequest,
    SourceRecord,
    SupportStatus,
)
from financial_analyst.reporting import DISCLAIMER, build_validated_report
from financial_analyst.security import redact_text
from financial_analyst.tickers import resolve_ticker
from financial_analyst.tools import run_dcf_tool
from financial_analyst.valuation import DCFInputs, calculate_dcf


def _market(*, price: float | None = 100.0) -> DataResult:
    status = Availability.AVAILABLE if price is not None else Availability.UNAVAILABLE
    return DataResult(
        name="market_snapshot",
        status=status,
        source="Yahoo Finance via yfinance",
        values={
            "ticker": "MSFT",
            "price": price,
            "currency": "USD",
            "trading_date": "2026-07-28",
            "retrieval_timestamp": "2026-07-29T00:00:00Z",
            "market_cap": 2_500_000_000_000.0,
            "history": [],
        },
        evidence=[
            EvidenceRef(
                source="Yahoo Finance via yfinance",
                source_type="market_data",
                provider="yfinance",
                title="MSFT market data",
                url="https://finance.yahoo.com/quote/MSFT",
                period_end="2026-07-28",
                metric="market_price",
                value=price,
                unit="USD",
                evidence_status=status,
            )
        ],
        message="Market price unavailable." if price is None else None,
    )


def _statements() -> DataResult:
    periods = [
        {
            "period_end": "2022-12-31",
            "fiscal_year": 2022,
            "currency": "USD",
            "revenue": 100.0,
            "net_income": 10.0,
            "operating_cash_flow": 18.0,
            "capital_expenditure": -3.0,
            "free_cash_flow": 15.0,
            "cash": 20.0,
            "debt": 30.0,
            "diluted_shares": 10.0,
        },
        {
            "period_end": "2023-12-31",
            "fiscal_year": 2023,
            "currency": "USD",
            "revenue": 110.0,
            "net_income": 13.2,
            "operating_cash_flow": 21.0,
            "capital_expenditure": -4.0,
            "free_cash_flow": 17.0,
            "cash": 24.0,
            "debt": 27.0,
            "diluted_shares": 10.2,
        },
        {
            "period_end": "2024-12-31",
            "fiscal_year": 2024,
            "currency": "USD",
            "revenue": 121.0,
            "net_income": 16.94,
            "operating_cash_flow": 25.0,
            "capital_expenditure": -5.0,
            "free_cash_flow": 20.0,
            "cash": 32.0,
            "debt": 22.0,
            "diluted_shares": 10.4,
        },
    ]
    latest = periods[-1]
    return DataResult(
        name="financial_statements",
        status=Availability.AVAILABLE,
        source="Yahoo Finance via yfinance",
        values={
            **latest,
            "income_period_end": latest["period_end"],
            "cash_flow_period_end": latest["period_end"],
            "balance_sheet_period_end": latest["period_end"],
            "statement_frequency": "annual",
            "annual_periods": list(reversed(periods)),
            "dataset_statuses": {
                "annual_income_statement": "available",
                "annual_cash_flow_statement": "available",
                "annual_balance_sheet": "available",
            },
        },
        evidence=[
            EvidenceRef(
                source="Yahoo Finance via yfinance",
                source_type="annual_financial_statement",
                provider="yfinance",
                title="MSFT annual statements",
                url="https://finance.yahoo.com/quote/MSFT/financials",
                period_end=latest["period_end"],
            )
        ],
    )


def _sec() -> DataResult:
    return DataResult(
        name="sec_company_facts",
        status=Availability.AVAILABLE,
        source="SEC EDGAR Company Facts",
        values={
            "facts": {
                "revenue": {
                    "unit": "USD",
                    "annual": {
                        "value": 121.0,
                        "period_end": "2024-12-31",
                        "form": "10-K",
                    },
                    "quarterly": {
                        "value": 32.0,
                        "period_end": "2025-03-31",
                        "form": "10-Q",
                    },
                }
            }
        },
        evidence=[
            EvidenceRef(
                source="SEC EDGAR Company Facts",
                source_type="official_filing_fact",
                provider="SEC EDGAR",
                title="SEC revenue fact",
                url="https://data.sec.gov/api/xbrl/companyfacts/CIK.json",
                period_end="2024-12-31",
                form="10-K",
                accession_number="0001",
                metric="revenue",
                value=121.0,
                unit="USD",
            )
        ],
    )


def _dcf() -> DataResult:
    return run_dcf_tool(
        base_free_cash_flow=20.0,
        growth_rate=0.05,
        discount_rate=0.10,
        terminal_growth_rate=0.025,
        cash=32.0,
        debt=22.0,
        diluted_shares=10.4,
        currency="USD",
        period_end="2024-12-31",
    )


def _analysis_data() -> list[DataResult]:
    return [_market(), _statements(), _sec(), _dcf()]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Analyze MSFT annual cash flow", "MSFT"),
        ("What is the valuation of AAPL?", "AAPL"),
        ("Research NYSE:KO profitability", "KO"),
    ],
)
def test_eval_ticker_extraction(query: str, expected: str) -> None:
    assert resolve_ticker(query, None) == expected


def test_eval_annual_periods_are_aligned_oldest_to_newest() -> None:
    history = build_historical_analysis(_statements())
    assert [period.fiscal_year for period in history.periods] == [2022, 2023, 2024]


def test_eval_annual_and_quarterly_revenue_remain_distinct() -> None:
    facts = _sec().values["facts"]["revenue"]
    assert facts["annual"]["form"] == "10-K"
    assert facts["quarterly"]["form"] == "10-Q"
    assert facts["annual"]["value"] != facts["quarterly"]["value"]


def test_eval_revenue_growth_is_calculated_from_adjacent_annual_periods() -> None:
    history = build_historical_analysis(_statements())
    assert history.revenue_growth[-1]["value"] == pytest.approx(0.10)


def test_eval_free_cash_flow_margin_uses_same_period_values() -> None:
    history = build_historical_analysis(_statements())
    quality = assess_evidence_quality(_analysis_data(), history)
    dashboard = build_dashboard(_analysis_data(), history, quality)
    metric = next(item for item in dashboard.metrics if item.key == "fcf_margin")
    assert metric.value == pytest.approx(20.0 / 121.0)


def test_eval_net_cash_is_cash_less_debt() -> None:
    history = build_historical_analysis(_statements())
    quality = assess_evidence_quality(_analysis_data(), history)
    dashboard = build_dashboard(_analysis_data(), history, quality)
    metric = next(item for item in dashboard.metrics if item.key == "net_cash")
    assert metric.value == pytest.approx(10.0)


def test_eval_scorecard_exposes_metric_contributions() -> None:
    history = build_historical_analysis(_statements())
    scorecard = build_scorecard(_analysis_data(), history)
    profitability = next(item for item in scorecard.components if item.name == "Profitability")
    assert profitability.score is not None
    assert {item.metric for item in profitability.contributions} == {
        "Net margin",
        "Free-cash-flow margin",
    }


def test_eval_missing_score_inputs_are_not_neutralized() -> None:
    scorecard = build_scorecard([_market()], build_historical_analysis(None))
    profitability = next(item for item in scorecard.components if item.name == "Profitability")
    assert profitability.score is None
    assert profitability.explanation == "Not scored — insufficient data"


def test_eval_dcf_sensitivity_has_explicit_axes_and_base_cell() -> None:
    result = calculate_dcf(
        DCFInputs(
            base_free_cash_flow=20.0,
            growth_rate=0.05,
            discount_rate=0.10,
            terminal_growth_rate=0.025,
            cash=32.0,
            debt=22.0,
            diluted_shares=10.4,
        )
    )
    assert result.sensitivity is not None
    assert 0.10 in result.sensitivity.discount_rates
    assert 0.025 in result.sensitivity.terminal_growth_rates


def test_eval_dcf_invalid_sensitivity_combinations_are_blank() -> None:
    result = calculate_dcf(
        DCFInputs(
            base_free_cash_flow=20.0,
            growth_rate=0.05,
            discount_rate=0.05,
            terminal_growth_rate=0.025,
            cash=32.0,
            debt=22.0,
            diluted_shares=10.4,
        )
    )
    assert result.sensitivity is not None
    assert result.sensitivity.invalid_cells
    assert any(cell is None for row in result.sensitivity.values for cell in row)


def test_eval_public_dcf_stops_when_diluted_shares_are_missing() -> None:
    result = run_dcf_tool(20.0, 0.05, 0.10, 0.025, cash=32.0, debt=22.0)
    assert result.status is Availability.UNAVAILABLE
    assert "diluted shares" in result.message.lower()


def test_eval_claims_link_to_evidence_or_calculation() -> None:
    data = _analysis_data()
    history = build_historical_analysis(_statements())
    quality = assess_evidence_quality(data, history)
    dashboard = build_dashboard(data, history, quality)
    evidence = build_evidence_catalog(data)
    claims = verify_claims(build_claims(dashboard, history, evidence), evidence)
    assert claims
    assert all(
        claim.evidence_ids or claim.calculation_id
        for claim in claims
        if claim.support_status is SupportStatus.VERIFIED
    )


def test_eval_unsupported_claim_is_rejected() -> None:
    claim = Claim(
        claim_id="unsupported",
        text="Unsupported factual assertion",
        category="test",
        support_status=SupportStatus.VERIFIED,
        confidence_category=ConfidenceCategory.HIGH,
    )
    verified = verify_claims([claim], [])
    assert verified[0].support_status is SupportStatus.UNSUPPORTED


def test_eval_conflicting_evidence_marks_claim_conflicting() -> None:
    evidence = EvidenceRef(
        evidence_id="ev-conflict",
        source="Cross-source check",
        metric="revenue",
        evidence_status=Availability.CONFLICT,
    )
    claim = Claim(
        claim_id="conflict",
        text="Revenue fact",
        category="test",
        evidence_ids=["ev-conflict"],
        support_status=SupportStatus.VERIFIED,
        confidence_category=ConfidenceCategory.HIGH,
    )
    verified = verify_claims([claim], [evidence])
    assert verified[0].support_status is SupportStatus.CONFLICTING
    assert verified[0].conflict_status


def test_eval_missing_market_price_omits_upside() -> None:
    data = [_market(price=None), _statements(), _dcf()]
    history = build_historical_analysis(_statements())
    quality = assess_evidence_quality(data, history)
    dashboard = build_dashboard(data, history, quality)
    upside = next(item for item in dashboard.metrics if item.key == "upside")
    assert upside.value is None
    assert upside.formatted_value == "Unavailable"


def test_eval_failed_market_retrieval_does_not_create_unsupported_timestamp_claim() -> None:
    result = build_validated_report(
        llm=FakeListChatModel(
            responses=[
                (
                    "## Research Conclusion\nEvidence is insufficient.\n"
                    "## Risk Factors\n- Core data is missing.\n"
                    "## Assumptions\n- No quantitative assumptions."
                )
            ]
        ),
        request=ResearchRequest(query="Analyze MSFT.", ticker="MSFT"),
        ticker="MSFT",
        data=[_market(price=None)],
        analysis_date=datetime(2026, 7, 29, tzinfo=UTC),
    )
    claims, validation = result[5], result[-1]
    assert all(claim.claim_id != "claim-market_timestamp" for claim in claims)
    assert validation.report_complete


def test_eval_sources_are_deduplicated_by_dataset_and_provenance() -> None:
    sources = build_source_records([_market(), _market()])
    assert len(sources) == 1


def test_eval_statement_sources_have_specific_dataset_labels() -> None:
    sources = build_source_records([_statements()])
    assert {item.dataset for item in sources} == {
        "Annual income statement",
        "Annual cash-flow statement",
        "Annual balance sheet",
    }


def test_eval_news_filtering_and_deduplication(monkeypatch: pytest.MonkeyPatch) -> None:
    class NewsTicker:
        def __init__(self) -> None:
            self.info = {"shortName": "Microsoft"}
            self.news = [
                {
                    "content": {
                        "title": "Microsoft reports cloud strategy update",
                        "summary": "Microsoft outlined material product and financial changes.",
                        "canonicalUrl": {"url": "https://example.test/msft?utm_source=a"},
                        "provider": {"displayName": "Reuters"},
                        "pubDate": "2026-07-28T00:00:00Z",
                    }
                },
                {
                    "content": {
                        "title": "Microsoft reports cloud strategy update",
                        "summary": "Duplicate wire copy about Microsoft.",
                        "canonicalUrl": {"url": "https://duplicate.test/story"},
                        "provider": {"displayName": "Reuters"},
                        "pubDate": "2026-07-28T00:00:00Z",
                    }
                },
                {
                    "content": {
                        "title": "Unrelated airline traffic report",
                        "summary": "Aviation demand was discussed.",
                        "canonicalUrl": {"url": "https://example.test/airline"},
                        "provider": {"displayName": "Reuters"},
                        "pubDate": "2026-07-28T00:00:00Z",
                    }
                },
            ]

    monkeypatch.setattr("financial_analyst.market.yf.Ticker", lambda ticker: NewsTicker())
    result = YFinanceClient().recent_news("MSFT")
    assert result.status is Availability.AVAILABLE
    assert len(result.values["articles"]) == 1
    assert "company identified in title" in result.values["articles"][0]["relevance_reason"]


def test_eval_market_query_is_reused_for_history(monkeypatch: pytest.MonkeyPatch) -> None:
    class MarketTicker:
        def __init__(self) -> None:
            self.fast_info = {"last_price": 101.0, "currency": "USD"}
            self.calls = 0

        def history(self, **kwargs: Any) -> pd.DataFrame:
            self.calls += 1
            return pd.DataFrame(
                {"Close": [99.0, 101.0]},
                index=pd.to_datetime(["2026-07-27", "2026-07-28"]),
            )

    ticker = MarketTicker()
    monkeypatch.setattr("financial_analyst.market.yf.Ticker", lambda symbol: ticker)
    client = YFinanceClient()
    snapshot = client.market_snapshot("MSFT")
    history = client.price_history("MSFT")
    assert snapshot.values["history"] == history.values["points"]
    assert ticker.calls == 1


def test_eval_evidence_quality_is_deterministic() -> None:
    data = _analysis_data()
    history = build_historical_analysis(_statements())
    first = assess_evidence_quality(data, history)
    second = assess_evidence_quality(data, history)
    assert first == second
    assert first.label in {
        ConfidenceCategory.MODERATE,
        ConfidenceCategory.HIGH,
    }


def test_eval_validator_blocks_price_without_canonical_market() -> None:
    validation = validate_report(
        report=(
            "# Report\n\n**Analysis date:** 2026-07-29\n\nThe current market price is 123 USD."
        ),
        data=[_market(price=None)],
        claims=[],
        sources=build_source_records([_market(price=None)]),
        analysis_date=datetime(2026, 7, 29, tzinfo=UTC),
    )
    assert any(item.code == "market_price_without_source" for item in validation.blocking_errors)


def test_eval_validator_blocks_news_transcript_confusion() -> None:
    news = DataResult(
        name="recent_news",
        status=Availability.AVAILABLE,
        source="Yahoo Finance news via yfinance",
        values={"articles": [{"title": "Microsoft update"}]},
        content_type="news",
    )
    validation = validate_report(
        report=(
            "# Report\n\n**Analysis date:** 2026-07-29\n\n"
            "Recent developments from the earnings-call transcript discuss strategy."
        ),
        data=[news],
        claims=[],
        sources=build_source_records([news]),
        analysis_date=datetime(2026, 7, 29, tzinfo=UTC),
    )
    assert any(item.code == "news_transcript_confusion" for item in validation.blocking_errors)


def test_eval_report_regenerates_once_after_blocking_validation() -> None:
    llm = FakeListChatModel(
        responses=[
            (
                "## Research Conclusion\nProfile is mixed.\n"
                "## Research Conclusion\nRepeated.\n"
                "## Risk Factors\n- Evidence gaps.\n"
                "## Assumptions\n- Structured inputs."
            ),
            (
                "## Research Conclusion\nProfile is mixed.\n"
                "## Risk Factors\n- Evidence gaps.\n"
                "## Assumptions\n- Structured inputs."
            ),
        ]
    )
    result = build_validated_report(
        llm=llm,
        request=ResearchRequest(query="Analyze MSFT.", ticker="MSFT"),
        ticker="MSFT",
        data=_analysis_data(),
        analysis_date=datetime(2026, 7, 29, tzinfo=UTC),
    )
    report, validation = result[0], result[-1]
    assert validation.regeneration_attempted
    assert validation.report_complete
    assert report.count("## Research Conclusion") == 1


def test_eval_persistent_validation_failure_returns_partial_report() -> None:
    duplicate = (
        "## Research Conclusion\nProfile is mixed.\n"
        "## Research Conclusion\nRepeated.\n"
        "## Risk Factors\n- Evidence gaps.\n"
        "## Assumptions\n- Structured inputs."
    )
    result = build_validated_report(
        llm=FakeListChatModel(responses=[duplicate, duplicate]),
        request=ResearchRequest(query="Analyze MSFT.", ticker="MSFT"),
        ticker="MSFT",
        data=_analysis_data(),
        analysis_date=datetime(2026, 7, 29, tzinfo=UTC),
    )
    report, validation = result[0], result[-1]
    assert not validation.report_complete
    assert report.startswith("> **Partial report")
    assert DISCLAIMER in report


def test_eval_secret_redaction_never_returns_api_key() -> None:
    secret = "sk-stage-two-secret-123456"
    assert secret not in redact_text(f"token={secret}", [secret])


def test_eval_source_record_model_preserves_filing_accession() -> None:
    source = SourceRecord(
        source_id="src-sec",
        dataset="SEC company facts",
        provider="SEC EDGAR",
        title="Official filing",
        status=Availability.AVAILABLE,
        accession_number="0001",
    )
    assert source.accession_number == "0001"
