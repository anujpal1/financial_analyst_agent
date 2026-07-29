"""Deterministic ticker normalization without a model call."""

from __future__ import annotations

import re

_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
_QUERY_TICKER_PATTERN = re.compile(r"(?<![A-Za-z0-9])\$?([A-Z]{1,5}(?:[.-][A-Z])?)(?![A-Za-z0-9])")
_EXCHANGE_TICKER_PATTERN = re.compile(
    r"\b(?:NYSE|NASDAQ|AMEX|LSE|TSX)\s*:\s*\$?([A-Z][A-Z0-9.-]{0,9})\b",
    re.IGNORECASE,
)
_COMMON_COMPANIES = {
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "amd": "AMD",
    "apple": "AAPL",
    "intel": "INTC",
    "meta": "META",
    "microsoft": "MSFT",
    "netflix": "NFLX",
    "nvidia": "NVDA",
    "tesla": "TSLA",
}
_STOP_WORDS = {"A", "AI", "AND", "DCF", "ETF", "FOR", "IN", "NEWS", "OF", "ON", "SEC", "THE"}


class TickerValidationError(ValueError):
    """Raised when no safe ticker can be resolved."""


def normalize_ticker(value: str) -> str:
    normalized = value.strip().upper().lstrip("$")
    if not _TICKER_PATTERN.fullmatch(normalized):
        raise TickerValidationError(
            "Ticker must be 1-10 letters/numbers and may include '.' or '-'."
        )
    return normalized


def resolve_ticker(query: str, explicit_ticker: str | None = None) -> str:
    if explicit_ticker and explicit_ticker.strip():
        return normalize_ticker(explicit_ticker)

    lower_query = query.lower()
    exchange_match = _EXCHANGE_TICKER_PATTERN.search(query)
    if exchange_match:
        return normalize_ticker(exchange_match.group(1))
    known_ticker_matches = [
        ticker
        for ticker in dict.fromkeys(_COMMON_COMPANIES.values())
        if re.search(rf"\b{re.escape(ticker)}\b", query, re.IGNORECASE)
    ]
    if known_ticker_matches:
        return known_ticker_matches[0]

    company_matches = [ticker for name, ticker in _COMMON_COMPANIES.items() if name in lower_query]
    if company_matches:
        return company_matches[0]

    symbols = [match for match in _QUERY_TICKER_PATTERN.findall(query) if match not in _STOP_WORDS]
    if symbols:
        return normalize_ticker(symbols[0])
    raise TickerValidationError(
        "Enter a ticker symbol or mention a supported company name in the question."
    )
