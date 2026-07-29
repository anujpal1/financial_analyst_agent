"""Single canonical registry of deterministic financial tools."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from financial_analyst.config import AppSettings
from financial_analyst.market import YFinanceClient
from financial_analyst.models import Availability, DataResult
from financial_analyst.sec import SECClient
from financial_analyst.transcripts import FMPTranscriptClient
from financial_analyst.valuation import DCFInputs, calculate_dcf


def build_tool_registry(
    settings: AppSettings,
    *,
    market_client: YFinanceClient | None = None,
    sec_client: SECClient | None = None,
    transcript_client: FMPTranscriptClient | None = None,
) -> tuple[BaseTool, ...]:
    """Build the only tool registry used by the workflow."""

    market = market_client or YFinanceClient()
    sec = sec_client or SECClient(
        user_agent=settings.sec_user_agent,
        timeout=settings.request_timeout_seconds,
        retry_count=settings.retry_count,
    )
    transcripts = transcript_client or FMPTranscriptClient(
        api_key=settings.fmp_api_key,
        timeout=settings.request_timeout_seconds,
        retry_count=settings.retry_count,
    )

    tools = (
        StructuredTool.from_function(
            func=market.market_snapshot,
            name="market_snapshot",
            description="Retrieve a current market snapshot for one validated ticker.",
        ),
        StructuredTool.from_function(
            func=market.financial_statements,
            name="financial_statements",
            description="Retrieve annual income, cash-flow, and balance-sheet observations.",
        ),
        StructuredTool.from_function(
            func=market.recent_news,
            name="recent_news",
            description="Retrieve relevant, deduplicated company news labelled explicitly as news.",
        ),
        StructuredTool.from_function(
            func=sec.company_facts,
            name="sec_company_facts",
            description="Retrieve official SEC Company Facts with filing metadata.",
        ),
        StructuredTool.from_function(
            func=transcripts.fetch,
            name="earnings_transcript",
            description="Retrieve an actual optional-provider earnings-call transcript.",
        ),
        StructuredTool.from_function(
            func=run_dcf_tool,
            name="discounted_cash_flow",
            description="Run transparent bear, base, and bull DCF scenarios from supplied facts.",
        ),
    )
    assert_unique_tool_names(tools)
    return tools


def assert_unique_tool_names(tools: Sequence[BaseTool]) -> None:
    names = [tool.name for tool in tools]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate tool names: {', '.join(duplicates)}")


def tool_mapping(tools: Sequence[BaseTool]) -> dict[str, BaseTool]:
    assert_unique_tool_names(tools)
    return {tool.name: tool for tool in tools}


def run_dcf_tool(
    base_free_cash_flow: float | None,
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    cash: float | None = None,
    debt: float | None = None,
    diluted_shares: float | None = None,
    currency: str = "USD",
    period_end: str | None = None,
    projection_years: int = 5,
) -> DataResult:
    """Calculate an FCFE DCF without adding cash or subtracting debt."""

    required = {
        "base_free_cash_flow": base_free_cash_flow,
        "diluted_shares": diluted_shares,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        readable_missing = ", ".join(name.replace("_", " ") for name in missing)
        reason = (
            f"base free cash flow is missing; essential inputs missing: {readable_missing}"
            if base_free_cash_flow is None
            else f"essential inputs are missing: {readable_missing}"
        )
        return DataResult.unavailable(
            name="discounted_cash_flow",
            source="Deterministic DCF calculation",
            message=f"DCF unavailable because {reason}.",
            missing_fields=missing,
            content_type="valuation",
        )
    try:
        inputs = DCFInputs(
            base_free_cash_flow=base_free_cash_flow,
            growth_rate=growth_rate,
            discount_rate=discount_rate,
            terminal_growth_rate=terminal_growth_rate,
            projection_years=projection_years,
            cash=cash,
            debt=debt,
            diluted_shares=diluted_shares,
            currency=currency,
            period_end=period_end,
        )
        result = calculate_dcf(inputs)
        status = Availability.PARTIAL if result.warnings else Availability.AVAILABLE
        values = result.model_dump(mode="json")
        values["inputs"] = {
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_growth_rate": terminal_growth_rate,
            "projection_years": inputs.projection_years,
            "method": "FCFE",
            "cash_flow_definition": result.cash_flow_definition,
            "cash": cash,
            "debt": debt,
            "diluted_shares": diluted_shares,
        }
        return DataResult(
            name="discounted_cash_flow",
            status=status,
            source="Deterministic FCFE DCF calculation",
            values=values,
            message="; ".join(result.warnings) if result.warnings else None,
            missing_fields=_valuation_missing_fields(
                diluted_shares=diluted_shares,
            ),
            content_type="valuation",
        )
    except ValueError as error:
        return DataResult.unavailable(
            name="discounted_cash_flow",
            source="Deterministic DCF calculation",
            message=f"DCF assumptions are invalid: {error}",
            content_type="valuation",
        )


def _valuation_missing_fields(**values: Any) -> list[str]:
    return [name for name, value in values.items() if value is None]
