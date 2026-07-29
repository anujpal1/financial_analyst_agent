"""Transparent discounted-cash-flow calculations with explicit missing-data behavior."""

from __future__ import annotations

from numbers import Real
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from financial_analyst.models import (
    CalculationRecord,
    EvidenceRef,
    ExecutiveDashboard,
    HistoricalAnalysis,
    SupportStatus,
)


def build_calculations(
    dashboard: ExecutiveDashboard,
    history: HistoricalAnalysis,
    evidence: list[EvidenceRef],
) -> list[CalculationRecord]:
    """Create and verify auditable records for deterministic dashboard values."""

    metrics = {item.key: item for item in dashboard.metrics}
    evidence_by_metric: dict[str, list[str]] = {}
    for item in evidence:
        if item.metric:
            evidence_by_metric.setdefault(item.metric, []).append(item.evidence_id)
    records: list[CalculationRecord] = []
    periods = history.periods
    if len(periods) >= 2 and _real(periods[-2].revenue) and _real(periods[-1].revenue):
        records.append(
            _calculation(
                "calc-revenue_growth",
                "growth",
                "(current / prior) - 1",
                {"current": periods[-1].revenue, "prior": periods[-2].revenue},
                metrics.get("revenue_growth"),
                evidence_by_metric.get("revenue", []),
            )
        )
    latest = periods[-1] if periods else None
    if latest and _real(latest.free_cash_flow) and _real(latest.revenue):
        records.append(
            _calculation(
                "calc-fcf_margin",
                "ratio",
                "numerator / denominator",
                {"numerator": latest.free_cash_flow, "denominator": latest.revenue},
                metrics.get("fcf_margin"),
                [
                    *evidence_by_metric.get("free_cash_flow", []),
                    *evidence_by_metric.get("revenue", []),
                ],
            )
        )
    if latest and _real(latest.cash) and _real(latest.debt):
        records.append(
            _calculation(
                "calc-net_cash",
                "difference",
                "left - right",
                {"left": latest.cash, "right": latest.debt},
                metrics.get("net_cash"),
                [
                    *evidence_by_metric.get("cash", []),
                    *evidence_by_metric.get("debt", []),
                ],
            )
        )
    price = metrics.get("price")
    dcf_base = metrics.get("dcf_base")
    if price and dcf_base and _real(price.value) and _real(dcf_base.value):
        records.append(
            _calculation(
                "calc-upside",
                "upside",
                "(model_value - price) / price",
                {"model_value": dcf_base.value, "price": price.value},
                metrics.get("upside"),
                [
                    *evidence_by_metric.get("market_price", []),
                    *evidence_by_metric.get("dcf_base", []),
                ],
            )
        )
    return verify_calculations(records)


def verify_calculations(records: list[CalculationRecord]) -> list[CalculationRecord]:
    """Recompute allowlisted formulas rather than trusting a calculation ID."""

    output = []
    for record in records:
        recomputed = _recompute(record)
        valid = (
            recomputed is not None
            and abs(recomputed - record.output) <= max(1e-9, abs(record.output) * 1e-8)
            and bool(record.input_source_ids)
        )
        output.append(
            record.model_copy(
                update={
                    "recomputed_value": recomputed,
                    "status": SupportStatus.VERIFIED if valid else SupportStatus.UNSUPPORTED,
                    "validation_checks": [
                        "Formula is allowlisted.",
                        "Output matches recomputation within relative tolerance."
                        if recomputed is not None
                        else "Formula or inputs could not be recomputed.",
                        "Input evidence identifiers are present."
                        if record.input_source_ids
                        else "Input evidence identifiers are missing.",
                    ],
                }
            )
        )
    return output


def _calculation(
    calculation_id: str,
    calculation_type: str,
    formula: str,
    inputs: dict[str, float],
    metric: Any,
    input_source_ids: list[str],
) -> CalculationRecord:
    return CalculationRecord(
        calculation_id=calculation_id,
        calculation_type=calculation_type,
        formula=formula,
        inputs={key: float(value) for key, value in inputs.items()},
        input_source_ids=list(dict.fromkeys(input_source_ids)),
        output=float(metric.value),
        unit="ratio" if calculation_type in {"growth", "ratio", "upside"} else None,
        period=metric.period,
    )


def _recompute(record: CalculationRecord) -> float | None:
    inputs = record.inputs
    if record.calculation_type == "growth" and inputs.get("prior") not in {None, 0}:
        return inputs["current"] / inputs["prior"] - 1
    if record.calculation_type == "ratio" and inputs.get("denominator") not in {None, 0}:
        return inputs["numerator"] / inputs["denominator"]
    if record.calculation_type == "difference":
        return inputs["left"] - inputs["right"]
    if record.calculation_type == "upside" and inputs.get("price") not in {None, 0}:
        return (inputs["model_value"] - inputs["price"]) / inputs["price"]
    return None


def _real(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


class DCFInputs(BaseModel):
    """Levered free-cash-flow inputs for an explicit FCFE valuation."""

    model_config = ConfigDict(allow_inf_nan=False)

    base_free_cash_flow: float
    growth_rate: float = Field(default=0.05, gt=-1.0, lt=1.0)
    discount_rate: float = Field(default=0.10, gt=0.0, lt=1.0)
    terminal_growth_rate: float = Field(default=0.025, gt=-1.0, lt=1.0)
    projection_years: int = Field(default=5, ge=1, le=10)
    method: Literal["FCFE"] = "FCFE"
    cash: float | None = Field(default=None, ge=0.0)
    debt: float | None = Field(default=None, ge=0.0)
    diluted_shares: float | None = Field(default=None, gt=0.0)
    currency: str = Field(default="USD", min_length=3, max_length=8)
    period_end: str | None = None

    @model_validator(mode="after")
    def validate_terminal_rate(self) -> DCFInputs:
        if self.terminal_growth_rate >= self.discount_rate:
            raise ValueError("Terminal growth must be lower than the discount rate.")
        return self


class DCFScenario(BaseModel):
    name: str
    growth_rate: float
    discount_rate: float
    terminal_growth_rate: float
    projected_cash_flows: list[float]
    terminal_value: float
    discounted_terminal_value: float
    terminal_value_percentage: float
    enterprise_value: None = None
    equity_value: float
    per_share_value: float | None


class DCFSensitivityTable(BaseModel):
    """Per-share values over explicit discount and terminal-growth assumptions."""

    discount_rates: list[float]
    terminal_growth_rates: list[float]
    values: list[list[float | None]]
    invalid_cells: list[str]


class DCFResult(BaseModel):
    method: Literal["FCFE"]
    cash_flow_definition: str
    discount_rate_label: str
    currency: str
    base_free_cash_flow: float
    period_end: str | None
    projection_years: int
    scenarios: list[DCFScenario]
    sensitivity: DCFSensitivityTable | None = None
    warnings: list[str]


def calculate_dcf(inputs: DCFInputs) -> DCFResult:
    """Calculate FCFE scenarios without applying an enterprise-value debt bridge."""

    warnings: list[str] = []
    if inputs.base_free_cash_flow <= 0:
        warnings.append(
            "Base free cash flow is non-positive; conventional terminal-value interpretation "
            "may not be meaningful."
        )
    if inputs.diluted_shares is None:
        warnings.append("Diluted shares are unavailable, so per-share value is not calculated.")
    if inputs.cash is not None or inputs.debt is not None:
        warnings.append(
            "Cash and debt are disclosed for context but are not added or subtracted in the "
            "FCFE method."
        )

    definitions = (
        (
            "Bear",
            inputs.growth_rate - 0.03,
            inputs.discount_rate + 0.01,
            inputs.terminal_growth_rate - 0.005,
        ),
        ("Base", inputs.growth_rate, inputs.discount_rate, inputs.terminal_growth_rate),
        (
            "Bull",
            inputs.growth_rate + 0.03,
            inputs.discount_rate - 0.01,
            inputs.terminal_growth_rate + 0.005,
        ),
    )
    scenarios = [
        _calculate_scenario(inputs, name, growth, discount, terminal)
        for name, growth, discount, terminal in definitions
    ]
    sensitivity = _calculate_sensitivity(inputs)
    if any(scenario.terminal_value_percentage > 0.75 for scenario in scenarios):
        warnings.append(
            "Terminal value exceeds 75% of modelled equity value in at least one scenario; "
            "the result is highly assumption-sensitive."
        )
    warnings.append(
        "Values are educational estimates; display precision does not imply valuation certainty."
    )
    return DCFResult(
        method="FCFE",
        cash_flow_definition=(
            "Levered free cash flow to equity: provider free cash flow or operating cash "
            "flow less capital expenditure."
        ),
        discount_rate_label="Cost of equity",
        currency=inputs.currency,
        base_free_cash_flow=inputs.base_free_cash_flow,
        period_end=inputs.period_end,
        projection_years=inputs.projection_years,
        scenarios=scenarios,
        sensitivity=sensitivity,
        warnings=warnings,
    )


def _calculate_sensitivity(inputs: DCFInputs) -> DCFSensitivityTable | None:
    if inputs.diluted_shares is None:
        return None
    discount_rates = sorted(
        {
            round(inputs.discount_rate + delta, 4)
            for delta in (-0.02, -0.01, 0.0, 0.01, 0.02)
            if inputs.discount_rate + delta > 0
        }
    )
    terminal_rates = sorted(
        {
            round(inputs.terminal_growth_rate + delta, 4)
            for delta in (-0.01, -0.005, 0.0, 0.005, 0.01)
            if inputs.terminal_growth_rate + delta > -1
        }
    )
    values: list[list[float | None]] = []
    invalid: list[str] = []
    for terminal_rate in terminal_rates:
        row: list[float | None] = []
        for discount_rate in discount_rates:
            if terminal_rate >= discount_rate:
                row.append(None)
                invalid.append(
                    f"Terminal growth {terminal_rate:.2%} must be below "
                    f"discount rate {discount_rate:.2%}."
                )
                continue
            scenario = _calculate_scenario(
                inputs,
                "Sensitivity",
                inputs.growth_rate,
                discount_rate,
                terminal_rate,
            )
            row.append(scenario.per_share_value)
        values.append(row)
    return DCFSensitivityTable(
        discount_rates=discount_rates,
        terminal_growth_rates=terminal_rates,
        values=values,
        invalid_cells=invalid,
    )


def _calculate_scenario(
    inputs: DCFInputs,
    name: str,
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
) -> DCFScenario:
    if growth_rate <= -1:
        raise ValueError(f"{name} scenario growth must be greater than -100%.")
    if discount_rate <= 0:
        raise ValueError(f"{name} scenario discount rate must be positive.")
    if terminal_growth_rate <= -1:
        raise ValueError(f"{name} scenario terminal growth must be greater than -100%.")
    if terminal_growth_rate >= discount_rate:
        raise ValueError(f"{name} scenario terminal growth must be below its discount rate.")

    projected: list[float] = []
    cash_flow = inputs.base_free_cash_flow
    for year in range(1, inputs.projection_years + 1):
        cash_flow *= 1 + growth_rate
        projected.append(cash_flow / ((1 + discount_rate) ** year))

    terminal_value = cash_flow * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    discounted_terminal = terminal_value / ((1 + discount_rate) ** inputs.projection_years)
    equity_value = sum(projected) + discounted_terminal
    terminal_percentage = discounted_terminal / equity_value if equity_value != 0 else 0.0

    per_share_value = None
    if inputs.diluted_shares is not None:
        per_share_value = equity_value / inputs.diluted_shares

    return DCFScenario(
        name=name,
        growth_rate=growth_rate,
        discount_rate=discount_rate,
        terminal_growth_rate=terminal_growth_rate,
        projected_cash_flows=projected,
        terminal_value=terminal_value,
        discounted_terminal_value=discounted_terminal,
        terminal_value_percentage=terminal_percentage,
        enterprise_value=None,
        equity_value=equity_value,
        per_share_value=per_share_value,
    )
