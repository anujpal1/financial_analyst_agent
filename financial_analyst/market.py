"""Canonical yfinance market, annual-statement, and relevant-news access."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from numbers import Real
from typing import Any

import yfinance as yf

from financial_analyst.models import (
    AnnualFinancialPeriod,
    Availability,
    CanonicalMarketData,
    DataResult,
    EvidenceRef,
    PricePoint,
    utc_now,
)
from financial_analyst.security import safe_error_message

_NEWS_PUBLISHER_QUALITY = {
    "reuters",
    "associated press",
    "bloomberg",
    "financial times",
    "the wall street journal",
    "cnbc",
    "sec",
}
_LOW_INFORMATION_NEWS = re.compile(
    r"(?i)\b(price prediction|stock prediction|could .* stock|why .* stock (?:rose|fell)|video)\b"
)


def _number(value: Any) -> float | None:
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
    """Per-analysis yfinance boundary with bounded in-memory request reuse."""

    def __init__(self) -> None:
        self._tickers: dict[str, Any] = {}
        self._market_results: dict[str, DataResult] = {}
        self._statement_results: dict[str, DataResult] = {}
        self._news_results: dict[str, DataResult] = {}
        self._info_results: dict[str, dict[str, Any]] = {}

    def _ticker(self, ticker: str) -> Any:
        normalized = ticker.upper()
        if normalized not in self._tickers:
            self._tickers[normalized] = yf.Ticker(normalized)
        return self._tickers[normalized]

    def market_snapshot(self, ticker: str) -> DataResult:
        """Return the one canonical price object used throughout a research run."""

        ticker = ticker.upper()
        if ticker in self._market_results:
            return self._market_results[ticker].model_copy(deep=True)

        source = "Yahoo Finance via yfinance"
        retrieval_time = utc_now()
        try:
            stock = self._ticker(ticker)
            fast = _safe_fast_info(stock)
            history = _safe_history(stock, period="6mo", interval="1d")
            points = _history_points(history)

            price = _number(_lookup(fast, "last_price"))
            price_basis = "fast quote"
            trading_date = None
            delayed = False
            if price is None or price <= 0:
                price = points[-1].close if points else None
                price_basis = "latest daily close" if price is not None else None
                trading_date = points[-1].date if points else None
                delayed = price is not None

            previous_close = _number(_lookup(fast, "previous_close"))
            if previous_close is None and len(points) >= 2:
                previous_close = points[-2].close
            if trading_date is None and points:
                trading_date = points[-1].date

            market_data = CanonicalMarketData(
                ticker=ticker,
                price=price,
                currency=_lookup(fast, "currency"),
                trading_date=trading_date,
                retrieval_timestamp=retrieval_time,
                market_state=_lookup(fast, "market_state"),
                previous_close=previous_close,
                day_high=_number(_lookup(fast, "day_high")),
                day_low=_number(_lookup(fast, "day_low")),
                volume=_number(_lookup(fast, "last_volume")),
                market_cap=_number(_lookup(fast, "market_cap")),
                price_basis=price_basis,
                history=points,
                source=source,
                status=Availability.AVAILABLE,
                is_delayed=delayed,
            )
            if price is None:
                market_data.status = Availability.UNAVAILABLE
                market_data.error_reason = (
                    f"Unable to retrieve a fast quote or recent daily close for {ticker}."
                )
            elif delayed or not market_data.currency or not points:
                market_data.status = Availability.PARTIAL

            missing = [
                field
                for field in ("price", "currency", "previous_close", "market_cap")
                if getattr(market_data, field) is None
            ]
            result = DataResult(
                name="market_snapshot",
                status=market_data.status,
                source=source,
                values=market_data.model_dump(mode="json"),
                missing_fields=missing,
                message=(
                    "The fast quote was unavailable; the canonical price uses the latest "
                    "available daily close."
                    if delayed
                    else market_data.error_reason
                ),
                evidence=[
                    EvidenceRef(
                        source=source,
                        source_type="market_data",
                        provider="yfinance",
                        title=f"{ticker} market data",
                        url=f"https://finance.yahoo.com/quote/{ticker}",
                        period_end=trading_date,
                        metric="market_price",
                        value=price,
                        unit=market_data.currency,
                        evidence_status=market_data.status,
                    )
                ],
            )
        except Exception as error:
            result = DataResult.unavailable(
                name="market_snapshot",
                source=source,
                message=safe_error_message(
                    error,
                    context=f"Unable to retrieve market data for {ticker}",
                ),
            )
        self._market_results[ticker] = result
        return result.model_copy(deep=True)

    def price_history(self, ticker: str) -> DataResult:
        """Compatibility view derived from the canonical market object without a new query."""

        market = self.market_snapshot(ticker)
        points = market.values.get("history", [])
        if not points:
            return DataResult.unavailable(
                name="price_history",
                source=market.source,
                message=f"No six-month price history was returned for {ticker.upper()}.",
            )
        return DataResult(
            name="price_history",
            status=Availability.AVAILABLE,
            source=market.source,
            values={"period": "6 months", "interval": "daily", "points": points},
            evidence=market.evidence,
        )

    def financial_statements(self, ticker: str) -> DataResult:
        """Return up to five aligned annual periods from yfinance statements."""

        ticker = ticker.upper()
        if ticker in self._statement_results:
            return self._statement_results[ticker].model_copy(deep=True)

        source = "Yahoo Finance via yfinance"
        try:
            stock = self._ticker(ticker)
            income = stock.income_stmt
            cash_flow = stock.cashflow
            balance = stock.balance_sheet
            info = self._info(ticker)
            currency = info.get("financialCurrency") or info.get("currency")
            annual_periods = _build_annual_periods(
                income=income,
                cash_flow=cash_flow,
                balance=balance,
                currency=currency,
            )
            if not annual_periods:
                result = DataResult.unavailable(
                    name="financial_statements",
                    source=source,
                    message=f"No annual statement data was returned for {ticker}.",
                )
            else:
                latest = annual_periods[0]
                dataset_statuses = {
                    "annual_income_statement": _dataset_status(
                        latest.revenue,
                        latest.net_income,
                    ),
                    "annual_cash_flow_statement": _dataset_status(
                        latest.operating_cash_flow,
                        latest.capital_expenditure,
                        latest.free_cash_flow,
                    ),
                    "annual_balance_sheet": _dataset_status(
                        latest.cash,
                        latest.debt,
                    ),
                }
                tracked_fields = (
                    "revenue",
                    "net_income",
                    "operating_cash_flow",
                    "capital_expenditure",
                    "free_cash_flow",
                    "cash",
                    "debt",
                    "diluted_shares",
                )
                missing = [name for name in tracked_fields if getattr(latest, name) is None]
                values = latest.model_dump(mode="json")
                values.update(
                    {
                        "income_period_end": latest.period_end,
                        "cash_flow_period_end": latest.period_end,
                        "balance_sheet_period_end": latest.period_end,
                        "statement_frequency": "annual",
                        "company_name": info.get("longName") or info.get("shortName"),
                        "annual_periods": [
                            period.model_dump(mode="json") for period in annual_periods
                        ],
                        "dataset_statuses": {
                            name: status.value for name, status in dataset_statuses.items()
                        },
                    }
                )
                status = (
                    Availability.PARTIAL
                    if missing
                    or any(item is not Availability.AVAILABLE for item in dataset_statuses.values())
                    else Availability.AVAILABLE
                )
                result = DataResult(
                    name="financial_statements",
                    status=status,
                    source=source,
                    values=values,
                    missing_fields=missing,
                    evidence=[
                        EvidenceRef(
                            source=source,
                            source_type="annual_financial_statement",
                            provider="yfinance",
                            title=f"{ticker} annual financial statements",
                            url=f"https://finance.yahoo.com/quote/{ticker}/financials",
                            period_end=latest.period_end,
                            fiscal_year=latest.fiscal_year,
                            evidence_status=status,
                        )
                    ],
                )
        except Exception as error:
            result = DataResult.unavailable(
                name="financial_statements",
                source=source,
                message=safe_error_message(
                    error,
                    context=f"Financial statements unavailable for {ticker}",
                ),
            )
        self._statement_results[ticker] = result
        return result.model_copy(deep=True)

    def recent_news(self, ticker: str) -> DataResult:
        """Return deduplicated company-relevant news with transparent scoring reasons."""

        ticker = ticker.upper()
        if ticker in self._news_results:
            return self._news_results[ticker].model_copy(deep=True)

        source = "Yahoo Finance news via yfinance"
        retrieval_time = utc_now()
        try:
            stock = self._ticker(ticker)
            raw_items = stock.news or []
            info = self._info(ticker)
            company_name = str(info.get("shortName") or info.get("longName") or "").strip()
            articles: list[dict[str, Any]] = []
            seen_urls: set[str] = set()
            seen_titles: set[str] = set()
            for item in raw_items:
                article = _parse_news_item(
                    item=item,
                    ticker=ticker,
                    company_name=company_name,
                    retrieval_time=retrieval_time,
                )
                if (
                    not article
                    or article["relevance_score"] < 2
                    or "company identified" not in article["relevance_reason"]
                ):
                    continue
                canonical_url = _canonical_url(article.get("url"))
                normalized_title = _normalize_title(article["title"])
                if (canonical_url and canonical_url in seen_urls) or (
                    normalized_title and normalized_title in seen_titles
                ):
                    continue
                if canonical_url:
                    seen_urls.add(canonical_url)
                if normalized_title:
                    seen_titles.add(normalized_title)
                articles.append(article)
            articles.sort(
                key=lambda article: (
                    article["relevance_score"],
                    article.get("published_at") or "",
                ),
                reverse=True,
            )
            articles = articles[:6]
            if not articles:
                result = DataResult.unavailable(
                    name="recent_news",
                    source=source,
                    message=f"No sufficiently relevant recent company news was found for {ticker}.",
                    content_type="news",
                )
            else:
                result = DataResult(
                    name="recent_news",
                    status=Availability.AVAILABLE,
                    source=source,
                    values={
                        "company_name": company_name or None,
                        "articles": articles,
                    },
                    evidence=[
                        EvidenceRef(
                            source=source,
                            source_type="company_news",
                            provider=article.get("publisher") or "Yahoo Finance",
                            title=article["title"],
                            url=article.get("url"),
                            retrieved_at=retrieval_time,
                            evidence_status=Availability.AVAILABLE,
                            excerpt=article.get("description"),
                        )
                        for article in articles
                    ],
                    content_type="news",
                )
        except Exception as error:
            result = DataResult.unavailable(
                name="recent_news",
                source=source,
                message=safe_error_message(
                    error,
                    context=f"Recent company news unavailable for {ticker}",
                ),
                content_type="news",
            )
        self._news_results[ticker] = result
        return result.model_copy(deep=True)

    def _info(self, ticker: str) -> dict[str, Any]:
        normalized = ticker.upper()
        if normalized not in self._info_results:
            try:
                self._info_results[normalized] = self._ticker(normalized).info or {}
            except Exception:
                self._info_results[normalized] = {}
        return self._info_results[normalized]


def _safe_fast_info(stock: Any) -> Any:
    try:
        return stock.fast_info
    except Exception:
        return {}


def _safe_history(stock: Any, **kwargs: Any) -> Any:
    try:
        return stock.history(auto_adjust=False, **kwargs)
    except Exception:
        return None


def _history_points(history: Any) -> list[PricePoint]:
    if history is None or getattr(history, "empty", True) or "Close" not in history:
        return []
    return [
        PricePoint(date=_period_text(index) or str(index), close=close)
        for index, value in history["Close"].items()
        if (close := _number(value)) is not None
    ]


def _build_annual_periods(
    *,
    income: Any,
    cash_flow: Any,
    balance: Any,
    currency: str | None,
) -> list[AnnualFinancialPeriod]:
    columns: dict[str, Any] = {}
    for frame in (income, cash_flow, balance):
        if frame is None or getattr(frame, "empty", True):
            continue
        for column in frame.columns:
            period = _period_text(column)
            if period:
                columns[period] = column

    periods: list[AnnualFinancialPeriod] = []
    for period_end in sorted(columns, reverse=True)[:5]:
        column = columns[period_end]
        income_values = _column_values(income, column)
        cash_values = _column_values(cash_flow, column)
        balance_values = _column_values(balance, column)
        operating_cash_flow = _number(_lookup(cash_values, "Operating Cash Flow"))
        capital_expenditure = _number(_lookup(cash_values, "Capital Expenditure"))
        free_cash_flow = _number(_lookup(cash_values, "Free Cash Flow"))
        if (
            free_cash_flow is None
            and operating_cash_flow is not None
            and capital_expenditure is not None
        ):
            free_cash_flow = (
                operating_cash_flow + capital_expenditure
                if capital_expenditure < 0
                else operating_cash_flow - capital_expenditure
            )
        periods.append(
            AnnualFinancialPeriod(
                period_end=period_end,
                fiscal_year=_fiscal_year(period_end),
                currency=currency,
                revenue=_number(_lookup(income_values, "Total Revenue")),
                net_income=_number(_lookup(income_values, "Net Income")),
                operating_cash_flow=operating_cash_flow,
                capital_expenditure=capital_expenditure,
                free_cash_flow=free_cash_flow,
                cash=_first_number(
                    balance_values,
                    "Cash Cash Equivalents And Short Term Investments",
                    "Cash And Cash Equivalents",
                ),
                debt=_first_number(balance_values, "Total Debt"),
                diluted_shares=_number(_lookup(income_values, "Diluted Average Shares")),
            )
        )
    return [
        period
        for period in periods
        if any(
            getattr(period, field) is not None
            for field in (
                "revenue",
                "net_income",
                "operating_cash_flow",
                "free_cash_flow",
                "cash",
                "debt",
            )
        )
    ]


def _column_values(frame: Any, column: Any) -> Any:
    if frame is None or getattr(frame, "empty", True):
        return {}
    try:
        if column in frame.columns:
            return frame[column]
        target = _period_text(column)
        for candidate in frame.columns:
            if _period_text(candidate) == target:
                return frame[candidate]
    except (KeyError, TypeError, AttributeError):
        return {}
    return {}


def _first_number(container: Any, *keys: str) -> float | None:
    for key in keys:
        value = _number(_lookup(container, key))
        if value is not None:
            return value
    return None


def _dataset_status(*values: float | None) -> Availability:
    available = sum(value is not None for value in values)
    if available == 0:
        return Availability.UNAVAILABLE
    if available < len(values):
        return Availability.PARTIAL
    return Availability.AVAILABLE


def _fiscal_year(period_end: str) -> int | None:
    try:
        return int(period_end[:4])
    except (TypeError, ValueError):
        return None


def _parse_news_item(
    *,
    item: dict[str, Any],
    ticker: str,
    company_name: str,
    retrieval_time: datetime,
) -> dict[str, Any] | None:
    content = item.get("content", item)
    title = str(content.get("title") or "").strip()
    if not title:
        return None
    description = str(
        content.get("summary") or content.get("description") or content.get("snippet") or ""
    ).strip()
    canonical = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
    url = canonical.get("url") if isinstance(canonical, dict) else canonical
    provider = content.get("provider", {})
    publisher = provider.get("displayName") if isinstance(provider, dict) else str(provider or "")
    published_at = content.get("pubDate") or content.get("providerPublishTime")
    score, reasons = _news_relevance(
        ticker=ticker,
        company_name=company_name,
        title=title,
        description=description,
        url=str(url or ""),
        publisher=str(publisher or ""),
        published_at=published_at,
        retrieval_time=retrieval_time,
    )
    return {
        "title": title,
        "description": description or None,
        "publisher": publisher or None,
        "url": url,
        "published_at": str(published_at) if published_at is not None else None,
        "content_type": "news",
        "relevance_score": score,
        "relevance_reason": "; ".join(reasons),
        "retrieved_at": retrieval_time.isoformat(),
    }


def _news_relevance(
    *,
    ticker: str,
    company_name: str,
    title: str,
    description: str,
    url: str,
    publisher: str,
    published_at: Any,
    retrieval_time: datetime,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    identifiers = [ticker.lower()]
    if company_name:
        identifiers.extend(_company_tokens(company_name))
    title_lower = title.lower()
    description_lower = description.lower()
    url_lower = url.lower()
    if any(identifier in title_lower for identifier in identifiers):
        score += 4
        reasons.append("company identified in title")
    if any(identifier in description_lower for identifier in identifiers):
        score += 2
        reasons.append("company identified in description")
    if any(identifier in url_lower for identifier in identifiers):
        score += 1
        reasons.append("company identified in URL")
    age_days = _publication_age_days(published_at, retrieval_time)
    if age_days is not None and age_days <= 7:
        score += 2
        reasons.append("published within seven days")
    elif age_days is not None and age_days <= 30:
        score += 1
        reasons.append("published within thirty days")
    if publisher.lower() in _NEWS_PUBLISHER_QUALITY:
        score += 1
        reasons.append("established financial-news publisher")
    if _LOW_INFORMATION_NEWS.search(title):
        score -= 2
        reasons.append("low-information format deprioritized")
    return max(score, 0), reasons


def _company_tokens(company_name: str) -> list[str]:
    stop = {"inc", "inc.", "corp", "corporation", "company", "plc", "ltd"}
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", company_name.lower())
        if len(token) >= 3 and token not in stop
    ]
    return tokens[:3]


def _publication_age_days(value: Any, retrieval_time: datetime) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            published = datetime.fromtimestamp(value, tz=UTC)
        else:
            published = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if published.tzinfo is None:
                published = published.replace(tzinfo=UTC)
        return max((retrieval_time - published).days, 0)
    except (OSError, TypeError, ValueError):
        return None


def _canonical_url(url: str | None) -> str:
    if not url:
        return ""
    return str(url).split("?", 1)[0].rstrip("/").lower()


def _normalize_title(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))
