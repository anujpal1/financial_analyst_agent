"""Respectful SEC EDGAR Company Facts access with retries, caching, and metadata."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from datetime import date
from numbers import Real
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from financial_analyst.models import (
    AnnualFinancialPeriod,
    Availability,
    DataResult,
    EvidenceRef,
    ReconciliationAlternative,
    ReconciliationRecord,
)
from financial_analyst.security import safe_error_message

_MAPPING_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_SEC_COMPARABLE_CONCEPTS = {
    "revenue": {
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
    },
    "net_income": {"NetIncomeLoss", "ProfitLoss"},
}


def reconcile_financial_sources(
    statements: DataResult | None,
    sec_facts: DataResult | None,
) -> tuple[DataResult, list[ReconciliationRecord]]:
    """Build canonical annual facts using SEC only when contexts are comparable."""

    raw_periods = statements.values.get("annual_periods", []) if statements else []
    provider_periods = [AnnualFinancialPeriod.model_validate(item) for item in raw_periods]
    if not provider_periods and statements and statements.values.get("income_period_end"):
        provider_periods = [
            AnnualFinancialPeriod(
                period_end=statements.values["income_period_end"],
                **{
                    field: statements.values.get(field)
                    for field in AnnualFinancialPeriod.model_fields
                    if field
                    not in {
                        "period_end",
                        "fiscal_year",
                        "definitions",
                        "source_by_metric",
                        "evidence_by_metric",
                    }
                },
            )
        ]
    sec_values = sec_facts.values.get("facts", {}) if sec_facts else {}
    records: list[ReconciliationRecord] = []
    evidence = [
        *(statements.evidence if statements else []),
        *(sec_facts.evidence if sec_facts else []),
    ]
    by_period = {period.period_end: period for period in provider_periods}
    latest_provider = max(provider_periods, key=lambda item: item.period_end, default=None)

    for metric in ("revenue", "net_income"):
        annual = sec_values.get(metric, {}).get("annual", {})
        sec_value = annual.get("value")
        period_end = annual.get("period_end")
        unit = sec_values.get(metric, {}).get("unit")
        concept = sec_values.get(metric, {}).get("concept")
        duration = annual.get("duration_days")
        provider_period = by_period.get(period_end) if period_end else latest_provider
        if period_end is None and provider_period:
            period_end = provider_period.period_end
        provider_value = getattr(provider_period, metric, None) if provider_period else None
        provider_currency = provider_period.currency if provider_period else None
        comparable = bool(
            isinstance(sec_value, Real)
            and period_end
            and concept in _SEC_COMPARABLE_CONCEPTS[metric]
            and duration is not None
            and 300 <= duration <= 430
            and (not provider_currency or not unit or provider_currency == unit)
        )
        sec_ids = [
            item.evidence_id
            for item in (sec_facts.evidence if sec_facts else [])
            if item.metric == metric and item.period_type == "annual"
        ]
        alternatives = []
        if isinstance(provider_value, Real):
            alternatives.append(
                ReconciliationAlternative(
                    value=float(provider_value),
                    source=statements.source if statements else "yfinance",
                    period=period_end,
                    unit=provider_currency,
                    definition=provider_period.definitions.get(metric) if provider_period else None,
                    evidence_ids=[
                        item.evidence_id for item in (statements.evidence if statements else [])
                    ],
                )
            )
        if comparable:
            difference = (
                float(sec_value) - float(provider_value)
                if isinstance(provider_value, Real)
                else None
            )
            relative = (
                abs(difference) / max(abs(float(sec_value)), abs(float(provider_value)), 1.0)
                if difference is not None
                else None
            )
            conflict = relative is not None and relative > 0.01
            records.append(
                ReconciliationRecord(
                    metric=metric,
                    canonical_value=float(sec_value),
                    canonical_source="SEC EDGAR Company Facts",
                    canonical_period=period_end,
                    canonical_unit=unit,
                    canonical_definition=f"US-GAAP {concept}",
                    canonical_evidence_ids=sec_ids,
                    alternatives=alternatives,
                    difference_amount=difference,
                    difference_percentage=relative,
                    comparable_definitions=True,
                    conflict_status=Availability.CONFLICT if conflict else Availability.AVAILABLE,
                    resolution_reason=(
                        "Official annual SEC fact selected because taxonomy, duration, period, "
                        "unit, and concept are comparable."
                    ),
                    unresolved_warning=(
                        "Comparable provider value differs by more than 1%; the official SEC "
                        "value remains canonical and the alternative is preserved."
                        if conflict
                        else None
                    ),
                )
            )
            if provider_period:
                by_period[period_end] = provider_period.model_copy(
                    update={
                        metric: float(sec_value),
                        "source_by_metric": {
                            **provider_period.source_by_metric,
                            metric: "SEC EDGAR Company Facts",
                        },
                        "evidence_by_metric": {
                            **provider_period.evidence_by_metric,
                            metric: sec_ids,
                        },
                        "definitions": {
                            **provider_period.definitions,
                            metric: f"US-GAAP {concept}",
                        },
                    }
                )
            else:
                by_period[period_end] = AnnualFinancialPeriod(
                    period_end=period_end,
                    fiscal_year=annual.get("fiscal_year"),
                    currency=unit,
                    **{metric: float(sec_value)},
                    source_by_metric={metric: "SEC EDGAR Company Facts"},
                    evidence_by_metric={metric: sec_ids},
                    definitions={metric: f"US-GAAP {concept}"},
                )
        elif isinstance(provider_value, Real):
            records.append(
                ReconciliationRecord(
                    metric=metric,
                    canonical_value=float(provider_value),
                    canonical_source=statements.source if statements else "yfinance",
                    canonical_period=period_end,
                    canonical_unit=provider_currency,
                    canonical_definition=(
                        provider_period.definitions.get(metric) if provider_period else None
                    ),
                    canonical_evidence_ids=[
                        item.evidence_id for item in (statements.evidence if statements else [])
                    ],
                    comparable_definitions=False,
                    resolution_reason=(
                        "Provider annual statement retained because no reliable comparable "
                        "SEC annual context was available."
                    ),
                )
            )

    if latest_provider:
        for metric in (
            "operating_cash_flow",
            "capital_expenditure",
            "free_cash_flow",
            "cash",
            "debt",
            "diluted_shares",
        ):
            value = getattr(latest_provider, metric)
            if not isinstance(value, Real):
                continue
            records.append(
                ReconciliationRecord(
                    metric=metric,
                    canonical_value=float(value),
                    canonical_source=statements.source if statements else "yfinance",
                    canonical_period=latest_provider.period_end,
                    canonical_unit=latest_provider.currency,
                    canonical_definition=latest_provider.definitions.get(metric),
                    canonical_evidence_ids=[
                        item.evidence_id for item in (statements.evidence if statements else [])
                    ],
                    comparable_definitions=False,
                    resolution_reason=(
                        "Provider value retained because no definition-compatible official "
                        "SEC fact is available for this metric."
                    ),
                )
            )

    canonical_periods = sorted(
        by_period.values(),
        key=lambda period: period.period_end,
        reverse=True,
    )
    if not canonical_periods:
        return (
            DataResult.unavailable(
                name="canonical_financials",
                source="Canonical source reconciliation",
                message="No reliable annual financial facts were available for reconciliation.",
            ),
            records,
        )
    latest = canonical_periods[0]
    values = latest.model_dump(mode="json")
    values.update(
        {
            "annual_periods": [period.model_dump(mode="json") for period in canonical_periods],
            "income_period_end": latest.period_end,
            "cash_flow_period_end": latest.period_end,
            "balance_sheet_period_end": latest.period_end,
            "statement_frequency": "annual",
            "reconciliations": [record.model_dump(mode="json") for record in records],
        }
    )
    unresolved = [record for record in records if record.unresolved_warning]
    return (
        DataResult(
            name="canonical_financials",
            status=Availability.PARTIAL if unresolved else Availability.AVAILABLE,
            source="Canonical SEC-first reconciliation with yfinance fallback",
            values=values,
            evidence=evidence,
            message="; ".join(
                record.unresolved_warning for record in unresolved if record.unresolved_warning
            )
            or None,
        ),
        records,
    )


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
                    "taxonomy": "us-gaap",
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

    entity = str(payload.get("entityName") or "")
    cik = str(payload.get("cik") or "")
    for display_name, tags in concepts.items():
        entries: list[Mapping[str, Any]] = []
        selected_tag = None
        unit = None
        concept_payload: Mapping[str, Any] = {}
        for tag in tags:
            concept_payload = us_gaap.get(tag, {})
            units = concept_payload.get("units", {})
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

        is_duration = any(entry.get("start") for entry in entries)
        selections = (
            {
                "annual": _select_fact(entries, period_type="annual"),
                "quarterly": _select_fact(entries, period_type="quarter_only"),
                "year_to_date": _select_fact(entries, period_type="year_to_date"),
            }
            if is_duration
            else {
                "annual": _select_fact(entries, period_type="annual_instant"),
                "quarterly": _select_fact(entries, period_type="quarterly_instant"),
            }
        )
        fact_value: dict[str, Any] = {
            "taxonomy": "us-gaap",
            "concept": selected_tag,
            "label": concept_payload.get("label"),
            "description": concept_payload.get("description"),
            "unit": unit,
            "fact_type": "duration" if is_duration else "instant",
        }
        for period_type, selection in selections.items():
            if selection is None:
                continue
            entry, reason = selection
            if entry is None:
                continue
            fact_value[period_type] = _entry_value(entry, period_type, reason)
            evidence.append(
                _entry_evidence(
                    entry,
                    source_url,
                    unit,
                    metric=display_name,
                    taxonomy="us-gaap",
                    concept=selected_tag,
                    definition=concept_payload.get("label"),
                    period_type=period_type,
                    selection_reason=reason,
                )
            )
        if not any(key in fact_value for key in ("annual", "quarterly", "year_to_date")):
            missing.append(f"{display_name}_filing_context")
        fact_value["entity"] = entity
        fact_value["cik"] = cik
        values[display_name] = fact_value

    return values, evidence, missing


def _select_fact(
    entries: list[Mapping[str, Any]],
    *,
    period_type: str,
) -> tuple[Mapping[str, Any], str] | None:
    """Select a comparable SEC context without deriving quarter-only values."""

    candidates = []
    for entry in entries:
        if entry.get("val") is None:
            continue
        form = str(entry.get("form", ""))
        duration = _duration_days(entry)
        if period_type == "annual":
            valid = form in {"10-K", "10-K/A"} and duration is not None and 300 <= duration <= 430
        elif period_type == "quarter_only":
            valid = form in {"10-Q", "10-Q/A"} and duration is not None and 70 <= duration <= 120
        elif period_type == "year_to_date":
            valid = form in {"10-Q", "10-Q/A"} and duration is not None and 121 <= duration <= 300
        elif period_type == "annual_instant":
            valid = form in {"10-K", "10-K/A"} and duration is None
        elif period_type == "quarterly_instant":
            valid = form in {"10-Q", "10-Q/A"} and duration is None
        else:
            valid = False
        if valid:
            candidates.append(entry)
    if not candidates:
        return None
    deduplicated: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for entry in candidates:
        context = (
            entry.get("start"),
            entry.get("end"),
            entry.get("fy"),
            entry.get("fp"),
            entry.get("unit"),
        )
        existing = deduplicated.get(context)
        if existing is None or _filing_sort_key(entry) > _filing_sort_key(existing):
            deduplicated[context] = entry
    selected = max(
        deduplicated.values(),
        key=_filing_sort_key,
    )
    duration = _duration_days(selected)
    reason = (
        f"Most recently filed valid {period_type.replace('_', ' ')} context"
        f"; form={selected.get('form')}; fiscal_period={selected.get('fp') or 'unavailable'}"
        f"; duration_days={duration if duration is not None else 'instant'}."
    )
    return selected, reason


def _filing_sort_key(entry: Mapping[str, Any]) -> tuple[str, int, str, str]:
    form = str(entry.get("form", ""))
    amended = 1 if form.endswith("/A") else 0
    return (
        str(entry.get("filed", "")),
        amended,
        str(entry.get("end", "")),
        str(entry.get("accn", "")),
    )


def _duration_days(entry: Mapping[str, Any]) -> int | None:
    start = entry.get("start")
    end = entry.get("end")
    if not start or not end:
        return None
    try:
        return (date.fromisoformat(str(end)) - date.fromisoformat(str(start))).days + 1
    except ValueError:
        return None


def _latest_filing(
    entries: list[Mapping[str, Any]], allowed_forms: set[str]
) -> Mapping[str, Any] | None:
    """Compatibility helper retained for callers; selection is filing-aware."""

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


def _entry_value(
    entry: Mapping[str, Any],
    period_type: str,
    selection_reason: str,
) -> dict[str, Any]:
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
        "duration_days": _duration_days(entry),
        "period_type": period_type,
        "selection_reason": selection_reason,
    }


def _entry_evidence(
    entry: Mapping[str, Any],
    source_url: str,
    unit: str | None,
    *,
    metric: str,
    taxonomy: str,
    concept: str | None,
    definition: str | None,
    period_type: str,
    selection_reason: str,
) -> EvidenceRef:
    return EvidenceRef(
        source="SEC EDGAR Company Facts",
        source_type="official_filing_fact",
        provider="SEC EDGAR",
        title=f"SEC Company Fact: {metric.replace('_', ' ').title()}",
        url=source_url,
        form=entry.get("form"),
        filing_date=entry.get("filed"),
        period_start=entry.get("start"),
        period_end=entry.get("end"),
        fiscal_year=entry.get("fy"),
        fiscal_period=entry.get("fp"),
        accession_number=entry.get("accn"),
        unit=unit,
        metric=metric,
        value=entry.get("val"),
        taxonomy=taxonomy,
        concept=concept,
        definition=definition,
        duration_days=_duration_days(entry),
        period_type=period_type,
        selection_reason=selection_reason,
    )
