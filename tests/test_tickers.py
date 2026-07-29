from __future__ import annotations

import pytest

from financial_analyst.workflow import (
    TickerValidationError,
    normalize_ticker,
    resolve_ticker,
)


def test_explicit_ticker_is_normalized() -> None:
    assert normalize_ticker(" msft ") == "MSFT"


def test_lowercase_known_ticker_resolves_from_query() -> None:
    assert resolve_ticker("analyze msft free cash flow") == "MSFT"


def test_company_name_resolves_without_model_call() -> None:
    assert resolve_ticker("Review Microsoft valuation") == "MSFT"


def test_invalid_ticker_is_rejected() -> None:
    with pytest.raises(TickerValidationError):
        normalize_ticker("../../secret")
