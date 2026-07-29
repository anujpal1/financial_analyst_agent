"""Respectful SEC EDGAR Company Facts access with retries, caching, and metadata."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from financial_analyst.models import Availability, DataResult, EvidenceRef
from financial_analyst.security import safe_error_message

_MAPPING_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


class SECClient:
    """Small SEC client that preserves filing context for selected facts."""

    def __init__(
        self,
        *,
        user_agent: str | None,
        timeout: float,
        retry_count: int,
        session: requests.Session | None = None,
        minimum_interval_seconds: float = 0.12,
        cache_ttl_seconds: float = 900.0,
    ) -> None:
        self.user_agent = (user_agent or "").strip()
        self.timeout = timeout
        self.minimum_interval_seconds = minimum_interval_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self._session = session or _retrying_session(retry_count)
        self._cache: dict[str, tuple[float, Any]] = {}
        self._last_request = 0.0
        self._lock = threading.Lock()

    def company_facts(self, ticker: str) -> DataResult:
        source = "SEC EDGAR Company Facts"
        if not self.user_agent:
            return DataResult.unavailable(
                name="sec_company_facts",
                source=source,
                message=(
                    "SEC data is disabled until SEC_USER_AGENT is configured with an "
                    "application name and contact email."
                ),
            )
        if "@" not in self.user_agent or len(self.user_agent) < 10:
            return DataResult.unavailable(
                name="sec_company_facts",
                source=source,
                message=(
                    "SEC_USER_AGENT must include a descriptive application name and contact email."
                ),
            )

        ticker = ticker.upper()
        try:
            mapping = self._get_json(_MAPPING_URL)
            cik = _find_cik(mapping, ticker)
            if cik is None:
                return DataResult.unavailable(
                    name="sec_company_facts",
                    source=source,
                    message=f"SEC ticker mapping did not contain {ticker}.",
                )

            facts_url = _FACTS_URL.format(cik=cik)
            payload = self._get_json(facts_url)
            parsed_values, evidence, missing = _parse_company_facts(payload, facts_url)
            if not parsed_values:
                return DataResult.unavailable(
                    name="sec_company_facts",
                    source=source,
                    message=f"No supported SEC facts were found for {ticker}.",
                    missing_fields=missing,
                )
            return DataResult(
                name="sec_company_facts",
                status=Availability.PARTIAL if missing else Availability.AVAILABLE,
                source=source,
                values={
                    "ticker": ticker,
                    "cik": cik,
                    "entity_name": payload.get("entityName"),
                    "facts": parsed_values,
                },
                evidence=evidence,
                missing_fields=missing,
            )
        except (requests.RequestException, ValueError, TypeError, KeyError) as error:
            return DataResult.unavailable(
                name="sec_company_facts",
                source=source,
                message=safe_error_message(error, context="SEC data unavailable"),
            )

    def _get_json(self, url: str) -> Any:
        now = time.monotonic()
        cached = self._cache.get(url)
        if cached and now - cached[0] < self.cache_ttl_seconds:
            return cached[1]

        with self._lock:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.minimum_interval_seconds:
                time.sleep(self.minimum_interval_seconds - elapsed)
            response = self._session.get(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept-Encoding": "gzip, deflate",
                },
                timeout=self.timeout,
            )
            self._last_request = time.monotonic()
        response.raise_for_status()
        payload = response.json()
        self._cache[url] = (time.monotonic(), payload)
        return payload


def _retrying_session(retry_count: int) -> requests.Session:
    retry = Retry(
        total=retry_count,
        connect=retry_count,
        read=retry_count,
        status=retry_count,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    return session


def _find_cik(mapping: Mapping[str, Any], ticker: str) -> str | None:
    for item in mapping.values():
        if str(item.get("ticker", "")).upper() == ticker:
            return f"{int(item['cik_str']):010d}"
    return None


def _parse_company_facts(
    payload: Mapping[str, Any], source_url: str
) -> tuple[dict[str, Any], list[EvidenceRef], list[str]]:
    us_gaap = payload.get("facts", {}).get("us-gaap", {})
    concepts = {
        "revenue": (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
        ),
        "net_income": ("NetIncomeLoss", "ProfitLoss"),
        "assets": ("Assets",),
        "cash": (
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
        "debt": (
            "LongTermDebtAndFinanceLeaseObligations",
            "LongTermDebt",
        ),
    }
    values: dict[str, Any] = {}
    evidence: list[EvidenceRef] = []
    missing: list[str] = []

    for display_name, tags in concepts.items():
        entries: list[Mapping[str, Any]] = []
        selected_tag = None
        unit = None
        for tag in tags:
            units = us_gaap.get(tag, {}).get("units", {})
            for unit_name in ("USD", "shares", "USD/shares"):
                candidate = units.get(unit_name, [])
                if candidate:
                    entries = candidate
                    selected_tag = tag
                    unit = unit_name
                    break
            if entries:
                break
        if not entries:
            missing.append(display_name)
            continue

        annual = _latest_filing(entries, {"10-K", "10-K/A"})
        quarterly = _latest_filing(entries, {"10-Q", "10-Q/A"})
        fact_value: dict[str, Any] = {"concept": selected_tag, "unit": unit}
        for period_type, entry in (("annual", annual), ("quarterly", quarterly)):
            if entry is None:
                continue
            fact_value[period_type] = _entry_value(entry)
            evidence.append(_entry_evidence(entry, source_url, unit))
        if len(fact_value) == 2:
            missing.append(f"{display_name}_filing_context")
        values[display_name] = fact_value

    return values, evidence, missing


def _latest_filing(
    entries: list[Mapping[str, Any]], allowed_forms: set[str]
) -> Mapping[str, Any] | None:
    candidates = [
        entry
        for entry in entries
        if entry.get("form") in allowed_forms and entry.get("val") is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda entry: (
            str(entry.get("filed", "")),
            str(entry.get("end", "")),
            str(entry.get("accn", "")),
        ),
    )


def _entry_value(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "value": entry.get("val"),
        "form": entry.get("form"),
        "filed": entry.get("filed"),
        "period_start": entry.get("start"),
        "period_end": entry.get("end"),
        "fiscal_year": entry.get("fy"),
        "fiscal_period": entry.get("fp"),
        "accession_number": entry.get("accn"),
        "frame": entry.get("frame"),
    }


def _entry_evidence(entry: Mapping[str, Any], source_url: str, unit: str | None) -> EvidenceRef:
    return EvidenceRef(
        source="SEC EDGAR Company Facts",
        url=source_url,
        form=entry.get("form"),
        filing_date=entry.get("filed"),
        period_start=entry.get("start"),
        period_end=entry.get("end"),
        fiscal_year=entry.get("fy"),
        fiscal_period=entry.get("fp"),
        accession_number=entry.get("accn"),
        unit=unit,
    )
