"""Free-core market and statement access through yfinance."""

from __future__ import annotations

import math
from datetime import datetime
from numbers import Real
from typing import Any

import yfinance as yf

from financial_analyst.models import Availability, DataResult, EvidenceRef
from financial_analyst.security import safe_error_message


def _number(value: Any) -> float | int | None:
    if isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _lookup(container: Any, key: str) -> Any:
    try:
        if hasattr(container, "get"):
            return container.get(key)
        return container[key]
    except (KeyError, TypeError, AttributeError):
        return None


def _period_text(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)


class YFinanceClient:
    """Boundary around yfinance that returns typed failures instead of guessed numbers."""

    def market_snapshot(self, ticker: str) -> DataResult:
        source = "Yahoo Finance via yfinance"
        try:
            stock = yf.Ticker(ticker)
            fast = stock.fast_info
            values = {
                "price": _number(_lookup(fast, "last_price")),
                "day_high": _number(_lookup(fast, "day_high")),
                "day_low": _number(_lookup(fast, "day_low")),
                "volume": _number(_lookup(fast, "last_volume")),
                "currency": _lookup(fast, "currency"),
            }
            missing = [name for name, value in values.items() if value is None]
            if values["price"] is None or values["price"] <= 0:
                return DataResult.unavailable(
                    name="market_snapshot",
                    source=source,
                    message=f"No active market snapshot was returned for {ticker}.",
                    missing_fields=missing,
                )
            return DataResult(
                name="market_snapshot",
                status=Availability.PARTIAL if missing else Availability.AVAILABLE,
                source=source,
                values=values,
                missing_fields=missing,
                evidence=[
                    EvidenceRef(
                        source=source,
                        url=f"https://finance.yahoo.com/quote/{ticker}",
                    )
                ],
            )
        except Exception as error:
            return DataResult.unavailable(
                name="market_snapshot",
                source=source,
                message=safe_error_message(error, context=f"Market data unavailable for {ticker}"),
            )

    def financial_statements(self, ticker: str) -> DataResult:
        source = "Yahoo Finance via yfinance"
        try:
            stock = yf.Ticker(ticker)
            income = stock.income_stmt
            cash_flow = stock.cashflow
            balance = stock.balance_sheet
            info = stock.info or {}

            income_period, income_values = _latest_column(income)
            cash_period, cash_values = _latest_column(cash_flow)
            balance_period, balance_values = _latest_column(balance)

            operating_cash_flow = _number(_lookup(cash_values, "Operating Cash Flow"))
            capex = _number(_lookup(cash_values, "Capital Expenditure"))
            reported_fcf = _number(_lookup(cash_values, "Free Cash Flow"))
            if reported_fcf is None and operating_cash_flow is not None and capex is not None:
                reported_fcf = (
                    operating_cash_flow + capex if capex < 0 else operating_cash_flow - capex
                )

            diluted_shares = _number(_lookup(income_values, "Diluted Average Shares"))
            values = {
                "revenue": _number(_lookup(income_values, "Total Revenue")),
                "net_income": _number(_lookup(income_values, "Net Income")),
                "operating_cash_flow": operating_cash_flow,
                "capital_expenditure": capex,
                "free_cash_flow": reported_fcf,
                "cash": _first_number(
                    balance_values,
                    "Cash Cash Equivalents And Short Term Investments",
                    "Cash And Cash Equivalents",
                ),
                "debt": _first_number(balance_values, "Total Debt"),
                "diluted_shares": diluted_shares,
                "currency": info.get("financialCurrency") or info.get("currency"),
                "income_period_end": _period_text(income_period),
                "cash_flow_period_end": _period_text(cash_period),
                "balance_sheet_period_end": _period_text(balance_period),
                "statement_frequency": "annual",
            }
            financial_fields = (
                "revenue",
                "net_income",
                "operating_cash_flow",
                "capital_expenditure",
                "free_cash_flow",
                "cash",
                "debt",
                "diluted_shares",
            )
            missing = [name for name in financial_fields if values[name] is None]
            if len(missing) == len(financial_fields):
                return DataResult.unavailable(
                    name="financial_statements",
                    source=source,
                    message=f"No statement data was returned for {ticker}.",
                    missing_fields=missing,
                )
            period_end = (
                values["cash_flow_period_end"]
                or values["income_period_end"]
                or values["balance_sheet_period_end"]
            )
            return DataResult(
                name="financial_statements",
                status=Availability.PARTIAL if missing else Availability.AVAILABLE,
                source=source,
                values=values,
                missing_fields=missing,
                evidence=[
                    EvidenceRef(
                        source=source,
                        url=f"https://finance.yahoo.com/quote/{ticker}/financials",
                        period_end=period_end,
                    )
                ],
            )
        except Exception as error:
            return DataResult.unavailable(
                name="financial_statements",
                source=source,
                message=safe_error_message(
                    error, context=f"Financial statements unavailable for {ticker}"
                ),
            )

    def price_history(self, ticker: str) -> DataResult:
        source = "Yahoo Finance via yfinance"
        try:
            history = yf.Ticker(ticker).history(period="6mo", interval="1d")
            if history is None or history.empty or "Close" not in history:
                return DataResult.unavailable(
                    name="price_history",
                    source=source,
                    message=f"No six-month price history was returned for {ticker}.",
                )
            points = [
                {
                    "date": _history_date(index),
                    "close": close,
                }
                for index, value in history["Close"].items()
                if (close := _number(value)) is not None
            ]
            if not points:
                return DataResult.unavailable(
                    name="price_history",
                    source=source,
                    message=f"Price history for {ticker} contained no usable close values.",
                )
            return DataResult(
                name="price_history",
                status=Availability.AVAILABLE,
                source=source,
                values={"period": "6 months", "interval": "daily", "points": points},
                evidence=[
                    EvidenceRef(
                        source=source,
                        url=f"https://finance.yahoo.com/quote/{ticker}/history",
                    )
                ],
            )
        except Exception as error:
            return DataResult.unavailable(
                name="price_history",
                source=source,
                message=safe_error_message(
                    error, context=f"Price history unavailable for {ticker}"
                ),
            )

    def recent_news(self, ticker: str) -> DataResult:
        source = "Yahoo Finance news via yfinance"
        try:
            raw_items = yf.Ticker(ticker).news or []
            articles = []
            for item in raw_items[:8]:
                content = item.get("content", item)
                title = content.get("title")
                canonical = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
                url = canonical.get("url") if isinstance(canonical, dict) else canonical
                provider = content.get("provider", {})
                provider_name = (
                    provider.get("displayName") if isinstance(provider, dict) else str(provider)
                )
                if title:
                    articles.append(
                        {
                            "title": title,
                            "publisher": provider_name or None,
                            "url": url,
                            "published_at": content.get("pubDate"),
                            "content_type": "news",
                        }
                    )
            if not articles:
                return DataResult.unavailable(
                    name="recent_news",
                    source=source,
                    message=f"No recent news was returned for {ticker}.",
                    content_type="news",
                )
            return DataResult(
                name="recent_news",
                status=Availability.AVAILABLE,
                source=source,
                values={"articles": articles},
                evidence=[
                    EvidenceRef(source=source, url=article.get("url")) for article in articles
                ],
                content_type="news",
            )
        except Exception as error:
            return DataResult.unavailable(
                name="recent_news",
                source=source,
                message=safe_error_message(error, context=f"Recent news unavailable for {ticker}"),
                content_type="news",
            )


def peer_comparison_unavailable(ticker: str) -> DataResult:
    """Refuse to substitute a broad-market ETF for unknown company peers."""

    return DataResult.unavailable(
        name="peer_comparison",
        source="No reliable peer source configured",
        message=(
            f"Reliable peers for {ticker} could not be identified from the enabled sources. "
            "No ETF or broad-market benchmark was substituted."
        ),
        content_type="peer_comparison",
    )


def _latest_column(frame: Any) -> tuple[Any, Any]:
    if frame is None or getattr(frame, "empty", True):
        return None, {}
    period = frame.columns[0]
    return period, frame[period]


def _first_number(container: Any, *keys: str) -> float | int | None:
    for key in keys:
        value = _number(_lookup(container, key))
        if value is not None:
            return value
    return None


def _history_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)
