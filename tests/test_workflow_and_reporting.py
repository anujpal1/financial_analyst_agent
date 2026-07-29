from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import StructuredTool

from financial_analyst.config import AppSettings
from financial_analyst.models import (
    Availability,
    DataResult,
    EvidenceRef,
    ResearchRequest,
    utc_now,
)
from financial_analyst.reporting import (
    DISCLAIMER,
    build_report,
    detect_conflicts,
)
from financial_analyst.security import new_session_id
from financial_analyst.tools import build_tool_registry
from financial_analyst.workflow import build_research_graph, run_research


def _available(name: str, values: dict[str, Any] | None = None) -> DataResult:
    return DataResult(
        name=name,
        status=Availability.AVAILABLE,
        source=f"Mock {name}",
        values=values or {},
        evidence=[EvidenceRef(source=f"Mock {name}", url=f"https://example.test/{name}")],
    )


def _mock_tools() -> tuple[StructuredTool, ...]:
    def market_snapshot(ticker: str) -> DataResult:
        return _available(
            "market_snapshot",
            {
                "price": 100.0,
                "day_high": 102.0,
                "day_low": 99.0,
                "volume": 1_000_000,
                "currency": "USD",
                "history": [
                    {"date": "2026-01-02", "close": 99.0},
                    {"date": "2026-01-03", "close": 100.0},
                ],
            },
        )

    def financial_statements(ticker: str) -> DataResult:
        return _available(
            "financial_statements",
            {
                "revenue": 200_000_000.0,
                "net_income": 20_000_000.0,
                "operating_cash_flow": 30_000_000.0,
                "capital_expenditure": -5_000_000.0,
                "free_cash_flow": 25_000_000.0,
                "cash": 10_000_000.0,
                "debt": 2_000_000.0,
                "diluted_shares": 10_000_000.0,
                "currency": "USD",
                "income_period_end": "2025-12-31",
                "cash_flow_period_end": "2025-12-31",
                "balance_sheet_period_end": "2025-12-31",
                "statement_frequency": "annual",
            },
        )

    def recent_news(ticker: str) -> DataResult:
        return DataResult.unavailable(
            name="recent_news",
            source="Mock news",
            message="No recent news in fixture.",
            content_type="news",
        )

    def sec_company_facts(ticker: str) -> DataResult:
        return DataResult.unavailable(
            name="sec_company_facts",
            source="Mock SEC",
            message="SEC fixture intentionally unavailable.",
        )

    def earnings_transcript(ticker: str, year: int, quarter: int) -> DataResult:
        return DataResult.unavailable(
            name="earnings_transcript",
            source="Mock transcript",
            message="Transcript unavailable.",
            content_type="transcript",
        )

    from financial_analyst.tools import run_dcf_tool

    functions = (
        (market_snapshot, "market_snapshot"),
        (financial_statements, "financial_statements"),
        (recent_news, "recent_news"),
        (sec_company_facts, "sec_company_facts"),
        (earnings_transcript, "earnings_transcript"),
        (run_dcf_tool, "discounted_cash_flow"),
    )
    return tuple(
        StructuredTool.from_function(
            func=function,
            name=name,
            description=f"Mock {name}",
        )
        for function, name in functions
    )


def _fake_llm() -> FakeListChatModel:
    return FakeListChatModel(
        responses=[
            "## Research Conclusion\nEvidence is partial.\n"
            "## Risk Factors\n- Missing official facts.\n"
            "## Assumptions\n- Provider observations are used as labelled."
        ]
    )


def test_tool_registry_has_no_duplicate_names() -> None:
    names = [tool.name for tool in _mock_tools()]
    assert len(names) == len(set(names))


def test_real_tool_registry_has_no_duplicate_names() -> None:
    names = [tool.name for tool in build_tool_registry(AppSettings(_env_file=None))]
    assert len(names) == len(set(names))


def test_agent_graph_accepts_common_chat_model_interface() -> None:
    graph = build_research_graph(
        llm=_fake_llm(),
        settings=AppSettings(_env_file=None),
        tools=_mock_tools(),
        require_tool_calling=False,
    )
    result = run_research(
        graph,
        ResearchRequest(query="Analyze MSFT cash flow.", ticker="MSFT"),
    )
    assert result.ticker == "MSFT"
    assert "Financial Research Report" in result.report_markdown


def test_mocked_end_to_end_flow_includes_dcf_and_disclaimer() -> None:
    graph = build_research_graph(
        llm=_fake_llm(),
        settings=AppSettings(_env_file=None),
        tools=_mock_tools(),
        require_tool_calling=False,
    )
    result = run_research(
        graph,
        ResearchRequest(query="Run a DCF valuation for MSFT.", ticker="MSFT"),
        session_id="isolated-test-session",
    )
    assert result.session_id == "isolated-test-session"
    assert any(item.name == "discounted_cash_flow" for item in result.data)
    assert "Bear scenario" in result.report_markdown
    assert DISCLAIMER in result.report_markdown
    assert "BUY" not in result.report_markdown


def test_session_ids_are_unique() -> None:
    assert new_session_id() != new_session_id()


def test_current_date_is_dynamic() -> None:
    before = datetime.now(UTC)
    observed = utc_now()
    after = datetime.now(UTC)
    assert before <= observed <= after


def test_report_generation_with_partial_data() -> None:
    unavailable = DataResult.unavailable(
        name="market_snapshot",
        source="Mock source",
        message="Market data unavailable.",
    )
    report, quality = build_report(
        llm=_fake_llm(),
        request=ResearchRequest(query="Analyze MSFT.", ticker="MSFT"),
        ticker="MSFT",
        data=[unavailable],
        analysis_date=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert "Market data unavailable" in report
    assert "Insufficient" in quality
    assert DISCLAIMER in report


def test_like_period_conflicts_are_explicit() -> None:
    statements = _available(
        "financial_statements",
        {
            "revenue": 100.0,
            "net_income": 10.0,
            "income_period_end": "2025-12-31",
        },
    )
    sec = _available(
        "sec_company_facts",
        {
            "facts": {
                "revenue": {"annual": {"value": 120.0, "period_end": "2025-12-31"}},
                "net_income": {"annual": {"value": 10.0, "period_end": "2025-12-31"}},
            }
        },
    )
    conflict = detect_conflicts([statements, sec])
    assert conflict is not None
    assert conflict.status is Availability.CONFLICT
    assert conflict.values["conflicts"][0]["metric"] == "revenue"
