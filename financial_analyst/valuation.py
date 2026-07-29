"""Transparent discounted-cash-flow calculations with explicit missing-data behavior."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DCFInputs(BaseModel):
    """Required observations and user-visible valuation assumptions."""

    model_config = ConfigDict(allow_inf_nan=False)

    base_free_cash_flow: float
    growth_rate: float = Field(default=0.05, gt=-1.0, lt=1.0)
    discount_rate: float = Field(default=0.10, gt=0.0, lt=1.0)
    terminal_growth_rate: float = Field(default=0.025, gt=-1.0, lt=1.0)
    projection_years: int = Field(default=5, ge=1, le=10)
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
    enterprise_value: float
    equity_value: float | None
    per_share_value: float | None


class DCFResult(BaseModel):
    currency: str
    base_free_cash_flow: float
    period_end: str | None
    scenarios: list[DCFScenario]
    warnings: list[str]


def calculate_dcf(inputs: DCFInputs) -> DCFResult:
    """Calculate bear, base, and bull cases without inventing absent balance-sheet facts."""

    warnings: list[str] = []
    if inputs.base_free_cash_flow <= 0:
        warnings.append(
            "Base free cash flow is non-positive; conventional terminal-value interpretation "
            "may not be meaningful."
        )
    if inputs.cash is None or inputs.debt is None:
        warnings.append(
            "Cash or debt is unavailable, so equity value and per-share value are not calculated."
        )
    if inputs.diluted_shares is None:
        warnings.append("Diluted shares are unavailable, so per-share value is not calculated.")

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
    return DCFResult(
        currency=inputs.currency,
        base_free_cash_flow=inputs.base_free_cash_flow,
        period_end=inputs.period_end,
        scenarios=scenarios,
        warnings=warnings,
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
    enterprise_value = sum(projected) + discounted_terminal

    equity_value = None
    if inputs.cash is not None and inputs.debt is not None:
        equity_value = enterprise_value + inputs.cash - inputs.debt

    per_share_value = None
    if equity_value is not None and inputs.diluted_shares is not None:
        per_share_value = equity_value / inputs.diluted_shares

    return DCFScenario(
        name=name,
        growth_rate=growth_rate,
        discount_rate=discount_rate,
        terminal_growth_rate=terminal_growth_rate,
        projected_cash_flows=projected,
        terminal_value=terminal_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        per_share_value=per_share_value,
    )
