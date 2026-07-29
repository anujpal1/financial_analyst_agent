from __future__ import annotations

import pytest
from pydantic import ValidationError

from financial_analyst.models import Availability
from financial_analyst.tools import run_dcf_tool
from financial_analyst.valuation import DCFInputs, calculate_dcf


def _valid_inputs(**updates: object) -> DCFInputs:
    values = {
        "base_free_cash_flow": 10_000_000.0,
        "growth_rate": 0.05,
        "discount_rate": 0.10,
        "terminal_growth_rate": 0.025,
        "cash": 2_000_000.0,
        "debt": 1_000_000.0,
        "diluted_shares": 1_000_000.0,
        "currency": "USD",
    }
    values.update(updates)
    return DCFInputs(**values)


def test_valid_dcf_inputs_produce_three_scenarios() -> None:
    result = calculate_dcf(_valid_inputs())
    assert [scenario.name for scenario in result.scenarios] == ["Bear", "Base", "Bull"]
    assert result.method == "FCFE"
    assert all(scenario.enterprise_value is None for scenario in result.scenarios)
    assert all(scenario.equity_value > 0 for scenario in result.scenarios)
    assert all(scenario.per_share_value is not None for scenario in result.scenarios)


def test_missing_free_cash_flow_refuses_to_run() -> None:
    result = run_dcf_tool(None, 0.05, 0.10, 0.025)
    assert result.status is Availability.UNAVAILABLE
    assert "base free cash flow is missing" in result.message.lower()


def test_missing_shares_omits_only_per_share_value() -> None:
    result = calculate_dcf(_valid_inputs(diluted_shares=None))
    assert all(scenario.equity_value is not None for scenario in result.scenarios)
    assert all(scenario.per_share_value is None for scenario in result.scenarios)
    assert any("Diluted shares" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("cash", "debt"),
    [(None, 1_000_000.0), (2_000_000.0, None), (None, None)],
)
def test_fcfe_does_not_apply_cash_or_debt_bridge(
    cash: float | None,
    debt: float | None,
) -> None:
    result = calculate_dcf(_valid_inputs(cash=cash, debt=debt))
    baseline = calculate_dcf(_valid_inputs(cash=None, debt=None))
    assert [item.equity_value for item in result.scenarios] == [
        item.equity_value for item in baseline.scenarios
    ]
    assert all(scenario.per_share_value is not None for scenario in result.scenarios)


@pytest.mark.parametrize("discount_rate", [0.0, -0.1, 1.0])
def test_invalid_discount_rate_is_rejected(discount_rate: float) -> None:
    with pytest.raises(ValidationError):
        _valid_inputs(discount_rate=discount_rate)


def test_terminal_growth_must_be_below_discount_rate() -> None:
    with pytest.raises(ValidationError, match="Terminal growth"):
        _valid_inputs(discount_rate=0.05, terminal_growth_rate=0.05)


def test_negative_fcf_is_allowed_but_warned() -> None:
    result = calculate_dcf(_valid_inputs(base_free_cash_flow=-1_000_000.0))
    assert any("non-positive" in warning for warning in result.warnings)


def test_negative_debt_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _valid_inputs(debt=-1.0)


def test_scenario_output_is_consistent_for_positive_fcf() -> None:
    result = calculate_dcf(_valid_inputs())
    equity_values = [scenario.equity_value for scenario in result.scenarios]
    assert equity_values == sorted(equity_values)
    assert all(0 < scenario.terminal_value_percentage < 1 for scenario in result.scenarios)
