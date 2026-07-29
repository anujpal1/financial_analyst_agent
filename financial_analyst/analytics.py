"""Deterministic financial analytics, dashboard metrics, and transparent scoring rules."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from itertools import pairwise
from numbers import Real
from typing import Any

from financial_analyst.models import (
    AnnualFinancialPeriod,
    Availability,
    ConfidenceCategory,
    DashboardMetric,
    DataQualityRow,
    DataResult,
    EvidenceQualityAssessment,
    ExecutiveDashboard,
    FinancialScorecard,
    HistoricalAnalysis,
    ScoreComponent,
    ScoreContribution,
    TrendMetric,
)

_USABLE = {Availability.AVAILABLE, Availability.PARTIAL, Availability.STALE}
_NOT_SCORED = "Not scored — insufficient data"


def build_historical_analysis(statements: DataResult | None) -> HistoricalAnalysis:
    """Calculate like-frequency annual trends without filling missing observations."""

    if not statements or statements.status not in _USABLE:
        return HistoricalAnalysis()
    periods = [
        AnnualFinancialPeriod.model_validate(item)
        for item in statements.values.get("annual_periods", [])
    ]
    if not periods and statements.values.get("income_period_end"):
        periods = [
            AnnualFinancialPeriod(
                period_end=statements.values["income_period_end"],
                **{
                    field: statements.values.get(field)
                    for field in AnnualFinancialPeriod.model_fields
                    if field not in {"period_end", "fiscal_year"}
                },
            )
        ]
    periods = sorted(periods, key=lambda period: period.period_end)[-5:]
    growth = _growth_series(periods, "revenue")
    metrics = _trend_metrics(periods)
    observations = _observations(periods, growth)
    return HistoricalAnalysis(
        periods=periods,
        revenue_growth=growth,
        observations=observations,
        metrics=metrics,
    )


def revenue_cagr(history: HistoricalAnalysis) -> float | None:
    """Return annualized revenue growth over the observed endpoints."""

    values = [
        (period.period_end, period.revenue)
        for period in history.periods
        if _positive(period.revenue)
    ]
    if len(values) < 2:
        return None
    first, last = values[0][1], values[-1][1]
    years = len(values) - 1
    if first is None or last is None or years <= 0:
        return None
    return (last / first) ** (1 / years) - 1


def build_scorecard(
    data: list[DataResult],
    history: HistoricalAnalysis,
) -> FinancialScorecard:
    """Score six documented components; missing inputs never receive neutral points."""

    by_name = {item.name: item for item in data}
    statements = by_name.get("financial_statements")
    market = by_name.get("market_snapshot")
    dcf = by_name.get("discounted_cash_flow")
    latest = history.periods[-1] if history.periods else None
    components = [
        _profitability_score(latest),
        _growth_score(history),
        _cash_flow_score(history),
        _balance_sheet_score(latest),
        _valuation_score(market, dcf),
        _evidence_completeness_score(data, history),
    ]
    scored = [component.score for component in components if component.score is not None]
    if len(scored) < 4:
        overall = None
        explanation = (
            f"{_NOT_SCORED}; at least four of six components are required for an overall score."
        )
    else:
        overall = round(sum(scored) / len(scored), 1)
        explanation = (
            "Equal-weight average of the scored components only; missing components are excluded, "
            "not assigned a neutral score."
        )
    if not statements or statements.status not in _USABLE:
        explanation = f"{explanation} Annual financial statements are unavailable."
    return FinancialScorecard(
        components=components,
        overall_score=overall,
        overall_explanation=explanation,
    )


def assess_evidence_quality(
    data: list[DataResult],
    history: HistoricalAnalysis,
    *,
    unsupported_claims: int = 0,
) -> EvidenceQualityAssessment:
    """Assess source coverage with fixed observable criteria, not LLM confidence."""

    by_name = {item.name: item for item in data}
    conflicts = [item.message or item.name for item in data if item.status is Availability.CONFLICT]
    checks = {
        "Official filing data": _status_text(by_name.get("sec_company_facts")),
        "Canonical market price": _status_text(by_name.get("market_snapshot")),
        "Annual statements": _status_text(by_name.get("financial_statements")),
        "Historical depth": f"{len(history.periods)} annual periods",
        "Relevant company news": _status_text(by_name.get("recent_news")),
        "Valuation inputs": _status_text(by_name.get("discounted_cash_flow")),
        "Unsupported claims": str(unsupported_claims),
        "Source conflicts": str(len(conflicts)),
    }
    weights = {
        "sec_company_facts": 20,
        "market_snapshot": 20,
        "financial_statements": 25,
        "recent_news": 5,
        "discounted_cash_flow": 10,
    }
    score = sum(
        weight * _availability_fraction(by_name.get(name)) for name, weight in weights.items()
    )
    score += min(len(history.periods), 4) / 4 * 15
    score += 5 if _data_is_fresh(by_name.get("market_snapshot")) else 0
    score -= min(len(conflicts) * 10, 20)
    score -= min(unsupported_claims * 10, 30)
    score = round(max(0.0, min(100.0, score)), 1)
    core_count = sum(
        by_name.get(name) is not None and by_name[name].status in _USABLE
        for name in ("market_snapshot", "financial_statements", "sec_company_facts")
    )
    if core_count == 3 and score >= 80 and not conflicts and unsupported_claims == 0:
        label = ConfidenceCategory.HIGH
    elif core_count >= 2 and score >= 55:
        label = ConfidenceCategory.MODERATE
    elif core_count >= 1 and score >= 30:
        label = ConfidenceCategory.LOW
    else:
        label = ConfidenceCategory.INSUFFICIENT
    missing = [
        label
        for name, label in (
            ("market_snapshot", "Canonical market price"),
            ("financial_statements", "Annual financial statements"),
            ("sec_company_facts", "Official SEC filing facts"),
        )
        if name not in by_name or by_name[name].status not in _USABLE
    ]
    freshness = _freshness_summary(data)
    return EvidenceQualityAssessment(
        label=label,
        coverage_score=score,
        components=checks,
        missing_evidence=missing,
        conflicts=conflicts,
        freshness_summary=freshness,
    )


def build_data_quality(data: list[DataResult]) -> list[DataQualityRow]:
    """Return one source-specific row per meaningful dataset."""

    by_name = {item.name: item for item in data}
    rows: list[DataQualityRow] = []
    market = by_name.get("market_snapshot")
    rows.append(_quality_row("Market data", market, period_key="trading_date"))

    statements = by_name.get("financial_statements")
    dataset_statuses = statements.values.get("dataset_statuses", {}) if statements else {}
    for dataset, label, period_key in (
        ("annual_income_statement", "Annual financials", "income_period_end"),
        ("annual_cash_flow_statement", "Cash flow", "cash_flow_period_end"),
        ("annual_balance_sheet", "Balance sheet", "balance_sheet_period_end"),
    ):
        rows.append(
            _quality_row(
                label,
                statements,
                period_key=period_key,
                status_override=_availability(dataset_statuses.get(dataset)),
            )
        )
    rows.extend(
        [
            _quality_row("SEC filing facts", by_name.get("sec_company_facts")),
            _quality_row("Company news", by_name.get("recent_news")),
            _quality_row("Uploaded PDF", by_name.get("uploaded_documents")),
            _quality_row("Valuation inputs", by_name.get("discounted_cash_flow")),
        ]
    )
    return rows


def build_dashboard(
    data: list[DataResult],
    history: HistoricalAnalysis,
    quality: EvidenceQualityAssessment,
) -> ExecutiveDashboard:
    """Build presentation-ready cards only from deterministic structured values."""

    by_name = {item.name: item for item in data}
    market = by_name.get("market_snapshot")
    statements = by_name.get("financial_statements")
    dcf = by_name.get("discounted_cash_flow")
    mv = market.values if market else {}
    sv = statements.values if statements else {}
    latest = history.periods[-1] if history.periods else None
    currency = (
        (latest.currency if latest else None) or sv.get("currency") or mv.get("currency") or "USD"
    )
    period = latest.period_end if latest else sv.get("income_period_end")
    growth = history.revenue_growth[-1].get("value") if history.revenue_growth else None
    fcf_margin = _ratio(
        latest.free_cash_flow if latest else sv.get("free_cash_flow"),
        latest.revenue if latest else sv.get("revenue"),
    )
    cash = latest.cash if latest else sv.get("cash")
    debt = latest.debt if latest else sv.get("debt")
    net_cash = cash - debt if _real(cash) and _real(debt) else None
    scenarios = dcf.values.get("scenarios", []) if dcf and dcf.status in _USABLE else []
    per_share = {
        item.get("name"): item.get("per_share_value")
        for item in scenarios
        if _real(item.get("per_share_value"))
    }
    base_value = per_share.get("Base")
    range_values = [value for key, value in per_share.items() if key in {"Bear", "Base", "Bull"}]
    valuation_range = (
        f"{_compact_money(min(range_values), currency)} to "
        f"{_compact_money(max(range_values), currency)}"
        if range_values
        else None
    )
    price = mv.get("price") if market and market.status in _USABLE else None
    upside = _ratio(base_value - price, price) if _real(base_value) and _positive(price) else None
    complete = sum(item.status in _USABLE for item in data)
    total = max(len(data), 1)

    metric_specs = [
        ("price", "Latest price", price, _compact_money, mv.get("trading_date"), market),
        (
            "market_timestamp",
            "Market-data timestamp",
            mv.get("retrieval_timestamp"),
            _text,
            mv.get("trading_date"),
            market,
        ),
        (
            "market_cap",
            "Market capitalisation",
            mv.get("market_cap"),
            _compact_money,
            mv.get("trading_date"),
            market,
        ),
        (
            "revenue",
            "Annual revenue",
            _value(latest, sv, "revenue"),
            _compact_money,
            period,
            statements,
        ),
        ("revenue_growth", "Revenue growth", growth, _format_percent, period, statements),
        (
            "net_income",
            "Net income",
            _value(latest, sv, "net_income"),
            _compact_money,
            period,
            statements,
        ),
        (
            "operating_cash_flow",
            "Operating cash flow",
            _value(latest, sv, "operating_cash_flow"),
            _compact_money,
            period,
            statements,
        ),
        (
            "free_cash_flow",
            "Free cash flow",
            _value(latest, sv, "free_cash_flow"),
            _compact_money,
            period,
            statements,
        ),
        ("fcf_margin", "Free-cash-flow margin", fcf_margin, _format_percent, period, statements),
        ("cash", "Cash", cash, _compact_money, period, statements),
        ("debt", "Debt", debt, _compact_money, period, statements),
        (
            "net_cash",
            "Net cash / (debt)",
            net_cash,
            _compact_money,
            period,
            statements,
        ),
        (
            "diluted_shares",
            "Diluted shares",
            _value(latest, sv, "diluted_shares"),
            _compact_quantity,
            period,
            statements,
        ),
        (
            "dcf_base",
            "DCF base case",
            base_value,
            _compact_money,
            dcf.values.get("period_end") if dcf else None,
            dcf,
        ),
        (
            "dcf_range",
            "DCF modelled range",
            valuation_range,
            _text,
            dcf.values.get("period_end") if dcf else None,
            dcf,
        ),
        (
            "upside",
            "Base-case upside / (downside)",
            upside,
            _format_percent,
            mv.get("trading_date"),
            dcf if upside is not None else None,
        ),
        ("evidence", "Evidence quality", quality.label.value, _text, None, None),
        ("completeness", "Data completeness", complete / total, _format_percent, None, None),
    ]
    metrics = [
        DashboardMetric(
            key=key,
            label=label,
            value=value,
            formatted_value=formatter(value, currency),
            period=metric_period,
            source=result.source if result else "Deterministic calculation",
            status=(
                result.status
                if result
                else Availability.AVAILABLE
                if value is not None
                else Availability.UNAVAILABLE
            ),
            detail=_dashboard_detail(key, mv),
        )
        for key, label, value, formatter, metric_period, result in metric_specs
    ]
    return ExecutiveDashboard(metrics=metrics)


def _trend_metrics(periods: list[AnnualFinancialPeriod]) -> list[TrendMetric]:
    currency = next(
        (period.currency for period in reversed(periods) if period.currency), "currency"
    )
    specs: tuple[tuple[str, str, Callable[[AnnualFinancialPeriod], float | None]], ...] = (
        ("Revenue", currency, lambda period: period.revenue),
        ("Net income", currency, lambda period: period.net_income),
        ("Net margin", "%", lambda period: _ratio(period.net_income, period.revenue)),
        ("Operating cash flow", currency, lambda period: period.operating_cash_flow),
        ("Capital expenditure", currency, lambda period: period.capital_expenditure),
        ("Free cash flow", currency, lambda period: period.free_cash_flow),
        (
            "Free-cash-flow margin",
            "%",
            lambda period: _ratio(period.free_cash_flow, period.revenue),
        ),
        ("Cash", currency, lambda period: period.cash),
        ("Debt", currency, lambda period: period.debt),
        ("Diluted shares", "shares", lambda period: period.diluted_shares),
    )
    metrics = [
        TrendMetric(
            name=name,
            unit=unit,
            values=[
                {"period": period.period_end, "value": value}
                for period in periods
                if (value := getter(period)) is not None
            ],
        )
        for name, unit, getter in specs
    ]
    metrics.append(
        TrendMetric(
            name="Revenue growth",
            unit="%",
            values=[
                {"period": item["period"], "value": item["value"]}
                for item in _growth_series(periods, "revenue")
                if item["value"] is not None
            ],
        )
    )
    return metrics


def _growth_series(
    periods: list[AnnualFinancialPeriod],
    field: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for previous, current in pairwise(periods):
        old_value = getattr(previous, field)
        new_value = getattr(current, field)
        value = (
            _ratio(new_value - old_value, old_value)
            if _positive(old_value) and _real(new_value)
            else None
        )
        output.append(
            {
                "period": current.period_end,
                "prior_period": previous.period_end,
                "value": value,
            }
        )
    return output


def _observations(
    periods: list[AnnualFinancialPeriod],
    growth: list[dict[str, Any]],
) -> list[str]:
    observations: list[str] = []
    revenues = [period.revenue for period in periods if _real(period.revenue)]
    if len(revenues) >= 3 and len(revenues) == len(periods):
        if all(current > previous for previous, current in pairwise(revenues)):
            observations.append(
                f"Revenue increased in each of the {len(revenues)} observed annual periods."
            )
        elif all(current < previous for previous, current in pairwise(revenues)):
            observations.append(
                f"Revenue declined in each of the {len(revenues)} observed annual periods."
            )
    if growth and growth[-1]["value"] is not None:
        observations.append(
            f"Latest annual revenue growth was {_format_percent(growth[-1]['value'], '')}."
        )
    _append_margin_observation(observations, periods)
    _append_direction_observation(
        observations,
        periods,
        "diluted_shares",
        "Share count",
    )
    if len(periods) >= 2:
        first, last = periods[0], periods[-1]
        if all(_real(value) for value in (first.debt, last.debt, first.cash, last.cash)) and (
            last.debt < first.debt and last.cash > first.cash
        ):
            observations.append("Debt decreased while cash increased over the observed period.")
    return observations or [
        "Insufficient complete annual observations for a directional trend statement."
    ]


def _append_margin_observation(
    observations: list[str],
    periods: list[AnnualFinancialPeriod],
) -> None:
    margins = [
        (period.period_end, _ratio(period.free_cash_flow, period.revenue)) for period in periods
    ]
    margins = [(period, value) for period, value in margins if value is not None]
    if len(margins) < 2:
        return
    change = margins[-1][1] - margins[0][1]
    verb = "increased" if change >= 0 else "declined"
    observations.append(
        f"Free-cash-flow margin {verb} by {abs(change) * 100:.1f} percentage points "
        f"from {margins[0][0]} to {margins[-1][0]}."
    )


def _append_direction_observation(
    observations: list[str],
    periods: list[AnnualFinancialPeriod],
    field: str,
    label: str,
) -> None:
    values = [
        (period.period_end, getattr(period, field))
        for period in periods
        if _real(getattr(period, field))
    ]
    if len(values) < 2 or values[0][1] == values[-1][1]:
        return
    verb = "increased" if values[-1][1] > values[0][1] else "decreased"
    observations.append(f"{label} {verb} from {values[0][0]} to {values[-1][0]}.")


def _profitability_score(period: AnnualFinancialPeriod | None) -> ScoreComponent:
    if not period:
        return _missing_component("Profitability", ["net margin", "free-cash-flow margin"])
    net_margin = _ratio(period.net_income, period.revenue)
    fcf_margin = _ratio(period.free_cash_flow, period.revenue)
    if net_margin is None or fcf_margin is None:
        return _missing_component(
            "Profitability",
            [
                name
                for value, name in (
                    (net_margin, "net margin"),
                    (fcf_margin, "free-cash-flow margin"),
                )
                if value is None
            ],
        )
    contributions = [
        _band_contribution(
            "Net margin", net_margin, ((0.20, 100), (0.10, 75), (0.0, 50), (-math.inf, 10))
        ),
        _band_contribution(
            "Free-cash-flow margin",
            fcf_margin,
            ((0.15, 100), (0.08, 75), (0.0, 50), (-math.inf, 10)),
        ),
    ]
    return _scored_component("Profitability", contributions)


def _growth_score(history: HistoricalAnalysis) -> ScoreComponent:
    cagr = revenue_cagr(history)
    valid_growth = [item["value"] for item in history.revenue_growth if item["value"] is not None]
    if cagr is None or len(valid_growth) < 2:
        return _missing_component("Growth", ["revenue CAGR", "two annual growth intervals"])
    positive_share = sum(value > 0 for value in valid_growth) / len(valid_growth)
    contributions = [
        _band_contribution(
            "Revenue CAGR", cagr, ((0.15, 100), (0.08, 80), (0.03, 60), (0.0, 40), (-math.inf, 10))
        ),
        ScoreContribution(
            metric="Positive revenue-growth periods",
            value=positive_share,
            points=round(positive_share * 100, 1),
            rule="100 times the share of annual intervals with positive revenue growth.",
        ),
    ]
    return _scored_component("Growth", contributions)


def _cash_flow_score(history: HistoricalAnalysis) -> ScoreComponent:
    periods = [
        period
        for period in history.periods
        if _real(period.operating_cash_flow)
        and _real(period.net_income)
        and _real(period.free_cash_flow)
    ]
    if len(periods) < 2:
        return _missing_component(
            "Cash-flow quality",
            ["two annual periods with operating cash flow, net income, and free cash flow"],
        )
    latest = periods[-1]
    conversion = _ratio(latest.operating_cash_flow, abs(latest.net_income))
    positive_share = sum(period.free_cash_flow > 0 for period in periods) / len(periods)
    contributions = [
        _band_contribution(
            "Operating cash flow / net income",
            conversion,
            ((1.2, 100), (1.0, 80), (0.7, 55), (0.0, 25), (-math.inf, 0)),
        ),
        ScoreContribution(
            metric="Positive free-cash-flow periods",
            value=positive_share,
            points=round(positive_share * 100, 1),
            rule="100 times the share of complete annual periods with positive free cash flow.",
        ),
    ]
    return _scored_component("Cash-flow quality", contributions)


def _balance_sheet_score(period: AnnualFinancialPeriod | None) -> ScoreComponent:
    if not period or not all(
        _real(value) for value in (period.cash, period.debt, period.free_cash_flow)
    ):
        return _missing_component("Balance-sheet strength", ["cash", "debt", "free cash flow"])
    net_cash = period.cash - period.debt
    debt_to_fcf = (
        _ratio(period.debt, period.free_cash_flow) if _positive(period.free_cash_flow) else None
    )
    if debt_to_fcf is None:
        debt_points = 10.0
        debt_rule = (
            "10 points when free cash flow is non-positive and leverage cannot be supported."
        )
    else:
        debt_contribution = _band_contribution(
            "Debt / free cash flow",
            -debt_to_fcf,
            ((-1.0, 100), (-2.0, 75), (-4.0, 50), (-math.inf, 15)),
        )
        debt_points = debt_contribution.points
        debt_rule = (
            "100 at <=1x debt/FCF, 75 at <=2x, 50 at <=4x, otherwise 15; "
            "implemented on the negated ratio."
        )
    contributions = [
        ScoreContribution(
            metric="Net cash",
            value=net_cash,
            points=100.0 if net_cash >= 0 else 25.0,
            rule="100 for net cash; 25 for net debt.",
        ),
        ScoreContribution(
            metric="Debt / free cash flow",
            value=debt_to_fcf if debt_to_fcf is not None else "Unavailable",
            points=debt_points,
            rule=debt_rule,
        ),
    ]
    return _scored_component("Balance-sheet strength", contributions)


def _valuation_score(
    market: DataResult | None,
    dcf: DataResult | None,
) -> ScoreComponent:
    price = market.values.get("price") if market and market.status in _USABLE else None
    scenarios = dcf.values.get("scenarios", []) if dcf and dcf.status in _USABLE else []
    per_share = {
        item.get("name"): item.get("per_share_value")
        for item in scenarios
        if _positive(item.get("per_share_value"))
    }
    if not _positive(price) or not all(name in per_share for name in ("Bear", "Base", "Bull")):
        return _missing_component(
            "Valuation attractiveness",
            ["canonical market price", "bear/base/bull DCF per-share values"],
        )
    base_ratio = per_share["Base"] / price - 1
    range_position = (
        100.0 if price < per_share["Bear"] else 65.0 if price <= per_share["Bull"] else 20.0
    )
    contributions = [
        _band_contribution(
            "DCF base-case upside",
            base_ratio,
            ((0.25, 100), (0.10, 80), (0.0, 60), (-0.15, 35), (-math.inf, 10)),
        ),
        ScoreContribution(
            metric="Price position versus DCF range",
            value=price,
            points=range_position,
            rule="100 below bear value, 65 inside the modelled range, 20 above bull value.",
        ),
    ]
    return _scored_component("Valuation attractiveness", contributions)


def _evidence_completeness_score(
    data: list[DataResult],
    history: HistoricalAnalysis,
) -> ScoreComponent:
    by_name = {item.name: item for item in data}
    contributions: list[ScoreContribution] = []
    for name, label, points in (
        ("market_snapshot", "Canonical market data", 20),
        ("financial_statements", "Annual financial statements", 25),
        ("sec_company_facts", "Official SEC facts", 25),
        ("recent_news", "Relevant company news", 5),
        ("discounted_cash_flow", "Valuation inputs", 10),
    ):
        fraction = _availability_fraction(by_name.get(name))
        contributions.append(
            ScoreContribution(
                metric=label,
                value=_status_text(by_name.get(name)),
                points=points * fraction,
                rule=(
                    f"{points} available points times status coverage "
                    "(available 1, partial 0.5, unavailable 0)."
                ),
            )
        )
    history_points = min(len(history.periods), 3) / 3 * 15
    contributions.append(
        ScoreContribution(
            metric="Historical annual depth",
            value=len(history.periods),
            points=history_points,
            rule="15 points for at least three annual periods, proportional below that.",
        )
    )
    score = round(sum(item.points for item in contributions), 1)
    missing = [
        label
        for name, label in (
            ("market_snapshot", "canonical market data"),
            ("financial_statements", "annual financial statements"),
            ("sec_company_facts", "official SEC facts"),
        )
        if name not in by_name or by_name[name].status not in _USABLE
    ]
    return ScoreComponent(
        name="Evidence completeness",
        score=score,
        contributions=contributions,
        missing_metrics=missing,
        explanation=(
            "Fixed 100-point source-coverage rubric: market 20, statements 25, SEC 25, "
            "news 5, valuation 10, and three-year history 15."
        ),
    )


def _band_contribution(
    name: str,
    value: float,
    bands: tuple[tuple[float, float], ...],
) -> ScoreContribution:
    points = next(points for threshold, points in bands if value >= threshold)
    readable = ", ".join(
        f"{threshold:.1%}→{points:g}" if math.isfinite(threshold) else f"below→{points:g}"
        for threshold, points in bands
    )
    return ScoreContribution(
        metric=name,
        value=value,
        points=float(points),
        rule=f"Threshold rubric: {readable}.",
    )


def _scored_component(
    name: str,
    contributions: list[ScoreContribution],
) -> ScoreComponent:
    score = round(sum(item.points for item in contributions) / len(contributions), 1)
    return ScoreComponent(
        name=name,
        score=score,
        contributions=contributions,
        explanation="Arithmetic mean of the listed deterministic contributions.",
    )


def _missing_component(name: str, missing: list[str]) -> ScoreComponent:
    return ScoreComponent(
        name=name,
        missing_metrics=missing,
        explanation=_NOT_SCORED,
    )


def _quality_row(
    dataset: str,
    result: DataResult | None,
    *,
    period_key: str | None = None,
    status_override: Availability | None = None,
) -> DataQualityRow:
    if result is None:
        return DataQualityRow(
            dataset=dataset,
            status=Availability.UNAVAILABLE,
            source="Not requested or not returned",
            warning="Dataset was not collected for this analysis.",
        )
    retrieved = next(
        (evidence.retrieved_at for evidence in result.evidence if evidence.retrieved_at),
        None,
    )
    return DataQualityRow(
        dataset=dataset,
        status=status_override or result.status,
        source=result.source,
        period=result.values.get(period_key) if period_key else _result_period(result),
        retrieved_at=retrieved,
        warning=result.message,
    )


def _result_period(result: DataResult) -> str | None:
    for evidence in result.evidence:
        if evidence.period_end:
            return evidence.period_end
    return (
        result.values.get("period_end")
        or result.values.get("income_period_end")
        or result.values.get("trading_date")
    )


def _availability(value: Any) -> Availability | None:
    try:
        return Availability(value) if value else None
    except ValueError:
        return Availability.INVALID


def _availability_fraction(result: DataResult | None) -> float:
    if not result:
        return 0.0
    if result.status is Availability.AVAILABLE:
        return 1.0
    if result.status in {Availability.PARTIAL, Availability.STALE}:
        return 0.5
    return 0.0


def _status_text(result: DataResult | None) -> str:
    return result.status.value if result else Availability.UNAVAILABLE.value


def _data_is_fresh(result: DataResult | None) -> bool:
    if not result or result.status not in _USABLE:
        return False
    timestamps = [evidence.retrieved_at for evidence in result.evidence]
    if not timestamps:
        return False
    now = datetime.now(UTC)
    return any((now - timestamp).total_seconds() <= 2 * 24 * 60 * 60 for timestamp in timestamps)


def _freshness_summary(data: Iterable[DataResult]) -> str:
    timestamps = [
        evidence.retrieved_at
        for result in data
        for evidence in result.evidence
        if evidence.retrieved_at
    ]
    if not timestamps:
        return "No retrieval timestamps were available."
    latest = max(timestamps)
    return f"Latest source retrieval: {latest.strftime('%Y-%m-%d %H:%M UTC')}."


def _dashboard_detail(key: str, market: dict[str, Any]) -> str | None:
    if key == "price" and market.get("is_delayed"):
        return "Latest available daily close; delayed rather than a live quote."
    if key == "upside":
        return "Calculated from the canonical market price and DCF base-case per-share value."
    return None


def _value(
    period: AnnualFinancialPeriod | None,
    fallback: dict[str, Any],
    field: str,
) -> Any:
    return getattr(period, field) if period else fallback.get(field)


def _compact_money(value: Any, currency: str) -> str:
    if not _real(value):
        return "Unavailable"
    return f"{currency} {_compact_number(float(value))}"


def _compact_quantity(value: Any, currency: str = "") -> str:
    if not _real(value):
        return "Unavailable"
    return _compact_number(float(value))


def _compact_number(value: float) -> str:
    absolute = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if absolute >= threshold:
            return f"{value / threshold:,.1f}{suffix}"
    return f"{value:,.2f}"


def _format_percent(value: Any, currency: str = "") -> str:
    if not _real(value):
        return "Unavailable"
    return f"{float(value) * 100:.1f}%"


def _text(value: Any, currency: str = "") -> str:
    return str(value) if value not in (None, "") else "Unavailable"


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if not _real(numerator) or not _real(denominator) or float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def _real(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive(value: Any) -> bool:
    return _real(value) and float(value) > 0
