"""Claim-level evidence, source deduplication, and deterministic report validation."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime
from numbers import Real
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from financial_analyst.models import (
    Availability,
    CalculationRecord,
    Claim,
    ConfidenceCategory,
    ConsistencyValidation,
    DataResult,
    EvidenceRef,
    ExecutiveDashboard,
    HistoricalAnalysis,
    SourceRecord,
    SupportStatus,
    ValidationIssue,
)

_USABLE = {Availability.AVAILABLE, Availability.PARTIAL, Availability.STALE}
_TRACKING_PARAMETERS = {"gclid", "fbclid", "ref", "source"}
_RECOMMENDATION = re.compile(r"\b(?:BUY|HOLD|SELL|STRONG BUY|STRONG SELL)\b")


def build_evidence_catalog(data: list[DataResult]) -> list[EvidenceRef]:
    """Collect and deduplicate provider, filing, document, and calculation evidence."""

    collected: list[EvidenceRef] = []
    for result in data:
        collected.extend(result.evidence)
        if result.name == "market_snapshot" and result.status in _USABLE:
            collected.extend(_market_evidence(result))
        elif result.name == "financial_statements" and result.status in _USABLE:
            collected.extend(_statement_evidence(result))
        elif result.name == "discounted_cash_flow" and result.status in _USABLE:
            collected.extend(_dcf_evidence(result))
        elif result.name == "uploaded_documents" and result.status in _USABLE:
            collected.extend(_document_evidence(result))
    output: list[EvidenceRef] = []
    seen: set[tuple[Any, ...]] = set()
    for evidence in collected:
        key = (
            evidence.source_type,
            (evidence.provider or evidence.source).casefold(),
            _canonical_url(evidence.url),
            evidence.accession_number,
            evidence.page_number,
            evidence.metric,
            evidence.period_end,
            _stable_value(evidence.value),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(evidence)
    return output


def build_source_records(data: list[DataResult]) -> list[SourceRecord]:
    """Create specific dataset labels and suppress duplicate provider-level rows."""

    candidates: list[SourceRecord] = []
    for result in data:
        if result.name == "financial_statements":
            candidates.extend(_statement_source_records(result))
            continue
        candidates.append(
            SourceRecord(
                source_id=_source_id(result.name, result.source, _result_url(result)),
                dataset=_dataset_label(result.name),
                provider=_provider_label(result),
                title=_source_title(result),
                url=_result_url(result),
                status=result.status,
                period=_result_period(result),
                retrieved_at=_retrieval_time(result),
                accession_number=_accession(result),
                warning=result.message,
            )
        )
    output: list[SourceRecord] = []
    indexes: dict[tuple[Any, ...], int] = {}
    for record in candidates:
        key = (
            record.dataset.casefold(),
            record.provider.casefold(),
            _canonical_url(record.url),
            record.accession_number,
            _normalize(record.title),
        )
        if key in indexes:
            existing = output[indexes[key]]
            if existing.status is not record.status:
                output[indexes[key]] = existing.model_copy(
                    update={
                        "status": Availability.CONFLICT,
                        "warning": (
                            "Conflicting availability observations were normalized: "
                            f"{existing.status.value} and {record.status.value}."
                        ),
                    }
                )
            continue
        indexes[key] = len(output)
        output.append(record)
    return output


def build_claims(
    dashboard: ExecutiveDashboard,
    history: HistoricalAnalysis,
    evidence: list[EvidenceRef],
    calculations: list[CalculationRecord] | None = None,
) -> list[Claim]:
    """Build major factual claims only from structured metrics and attach evidence IDs."""

    claims: list[Claim] = []
    calculation_by_id = {item.calculation_id: item for item in (calculations or [])}
    evidence_by_metric: dict[str, list[EvidenceRef]] = {}
    for item in evidence:
        if item.metric:
            evidence_by_metric.setdefault(item.metric, []).append(item)
    for metric in dashboard.metrics:
        if metric.key not in {
            "price",
            "market_timestamp",
            "market_cap",
            "revenue",
            "revenue_growth",
            "net_income",
            "operating_cash_flow",
            "free_cash_flow",
            "fcf_margin",
            "cash",
            "debt",
            "net_cash",
            "diluted_shares",
            "dcf_base",
            "upside",
        }:
            continue
        if (
            metric.status not in _USABLE
            or metric.value is None
            or metric.formatted_value == "Unavailable"
        ):
            continue
        evidence_metric = {
            "price": "market_price",
            "market_timestamp": "market_timestamp",
        }.get(metric.key, metric.key)
        evidence_ids = [
            item.evidence_id
            for item in evidence_by_metric.get(evidence_metric, [])
            if not metric.period or not item.period_end or item.period_end == metric.period
        ]
        calculation_id = f"calc-{metric.key}" if f"calc-{metric.key}" in calculation_by_id else None
        status = (
            SupportStatus.VERIFIED if evidence_ids or calculation_id else SupportStatus.UNSUPPORTED
        )
        claims.append(
            Claim(
                claim_id=f"claim-{metric.key}",
                text=f"{metric.label}: {metric.formatted_value}",
                category="dashboard",
                evidence_ids=evidence_ids,
                metric_references=[metric.key],
                calculation_id=calculation_id,
                calculation_input_ids=(
                    calculation_by_id[calculation_id].input_source_ids if calculation_id else []
                ),
                displayed_value=metric.value,
                period=metric.period,
                support_status=status,
                confidence_category=(
                    ConfidenceCategory.HIGH
                    if status is SupportStatus.VERIFIED
                    else ConfidenceCategory.INSUFFICIENT
                ),
            )
        )
    for index, observation in enumerate(history.observations, start=1):
        if observation.startswith("Insufficient"):
            continue
        statement_ids = [
            item.evidence_id
            for item in evidence
            if item.source_type == "annual_financial_statement"
        ]
        claims.append(
            Claim(
                claim_id=f"claim-trend-{index}",
                text=observation,
                category="historical_trend",
                evidence_ids=statement_ids,
                calculation_id=None,
                support_status=(
                    SupportStatus.VERIFIED if statement_ids else SupportStatus.UNSUPPORTED
                ),
                confidence_category=(
                    ConfidenceCategory.HIGH if statement_ids else ConfidenceCategory.INSUFFICIENT
                ),
            )
        )
    return claims


def verify_claims(
    claims: list[Claim],
    evidence: list[EvidenceRef],
    calculations: list[CalculationRecord] | None = None,
) -> list[Claim]:
    """Verify evidence and recomputed calculation lineage for structured claims."""

    by_id = {item.evidence_id: item for item in evidence}
    calculation_by_id = {item.calculation_id: item for item in (calculations or [])}
    verified: list[Claim] = []
    for claim in claims:
        refs = [by_id[item] for item in claim.evidence_ids if item in by_id]
        conflict = any(ref.evidence_status is Availability.CONFLICT for ref in refs)
        missing_ref = any(item not in by_id for item in claim.evidence_ids)
        calculation = calculation_by_id.get(claim.calculation_id) if claim.calculation_id else None
        calculation_valid = bool(calculation and calculation.status is SupportStatus.VERIFIED)
        period_valid = not calculation or not claim.period or calculation.period == claim.period
        currency_valid = (
            not calculation
            or not claim.currency
            or not calculation.currency
            or calculation.currency == claim.currency
        )
        displayed_valid = (
            not calculation
            or not isinstance(claim.displayed_value, Real)
            or abs(float(claim.displayed_value) - calculation.output)
            <= max(1e-9, abs(calculation.output) * 1e-8)
        )
        if conflict:
            support = SupportStatus.CONFLICTING
            confidence = ConfidenceCategory.LOW
            reason = "Referenced evidence is explicitly conflicting."
        elif claim.calculation_id and not calculation_valid:
            support = SupportStatus.UNSUPPORTED
            confidence = ConfidenceCategory.INSUFFICIENT
            reason = "Calculation lineage is missing or failed recomputation."
        elif not period_valid or not currency_valid or not displayed_valid:
            support = SupportStatus.UNSUPPORTED
            confidence = ConfidenceCategory.INSUFFICIENT
            reason = "Calculation period, currency, or displayed output does not match."
        elif refs or calculation_valid:
            support = SupportStatus.PARTIALLY_SUPPORTED if missing_ref else SupportStatus.VERIFIED
            confidence = ConfidenceCategory.MODERATE if missing_ref else ConfidenceCategory.HIGH
            reason = (
                "Evidence identifiers resolve and calculation output was recomputed."
                if calculation_valid
                else "Evidence identifiers resolve."
            )
        else:
            support = SupportStatus.UNSUPPORTED
            confidence = ConfidenceCategory.INSUFFICIENT
            reason = "No resolvable evidence or verified calculation lineage."
        verified.append(
            claim.model_copy(
                update={
                    "support_status": support,
                    "conflict_status": conflict,
                    "confidence_category": confidence,
                    "verification_reason": reason,
                }
            )
        )
    return verified


def validate_report(
    *,
    report: str,
    data: list[DataResult],
    claims: list[Claim],
    sources: list[SourceRecord],
    analysis_date: datetime,
    regeneration_attempted: bool = False,
) -> ConsistencyValidation:
    """Run deterministic checks and block internally contradictory reports."""

    by_name = {item.name: item for item in data}
    blocking: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    passed: list[str] = []
    market = by_name.get("market_snapshot")
    dcf = by_name.get("discounted_cash_flow")

    if (
        not market
        or market.status not in _USABLE
        or not isinstance(market.values.get("price"), Real)
    ):
        if re.search(
            r"(?i)(?:latest|current|market)\s+(?:market\s+)?price"
            r"\s*(?::|is|=|-)?\s*(?:[A-Z]{3}\s*)?[$€£]?\s*\d",
            report,
        ):
            blocking.append(
                ValidationIssue(
                    code="market_price_without_source",
                    message=(
                        "The report cites a market price although the canonical price "
                        "is unavailable."
                    ),
                    blocking=True,
                )
            )
        if re.search(r"(?i)upside|downside", report):
            warnings.append(
                ValidationIssue(
                    code="price_comparison_unavailable",
                    message=(
                        "Valuation comparison language appears without a canonical market price."
                    ),
                )
            )
    else:
        passed.append("Market-price references use the canonical market object.")

    if not dcf or dcf.status not in _USABLE:
        if re.search(r"(?i)DCF (?:base|bear|bull)[^\n]*\d", report):
            blocking.append(
                ValidationIssue(
                    code="dcf_value_without_calculation",
                    message=(
                        "The report cites a DCF value although the DCF calculation is unavailable."
                    ),
                    blocking=True,
                )
            )
    else:
        passed.append("DCF references are backed by a successful deterministic calculation.")
        method = dcf.values.get("method")
        if method != "FCFE":
            blocking.append(
                ValidationIssue(
                    code="invalid_dcf_method",
                    message="The valuation result does not identify the supported FCFE method.",
                    blocking=True,
                )
            )
        elif re.search(r"(?i)\benterprise value\b", report):
            blocking.append(
                ValidationIssue(
                    code="fcfe_enterprise_bridge_confusion",
                    message="An FCFE report must not present an enterprise-value bridge.",
                    blocking=True,
                )
            )
        elif re.search(r"(?i)\bFCFF\b|\bWACC\b", report):
            blocking.append(
                ValidationIssue(
                    code="fcfe_fcff_label_conflict",
                    message="The report mixes FCFE output with FCFF or WACC terminology.",
                    blocking=True,
                )
            )
        else:
            passed.append("DCF method and FCFE labels are internally consistent.")

    unsupported = [
        claim.claim_id
        for claim in claims
        if claim.support_status is SupportStatus.UNSUPPORTED
        and claim.category != "removed_interpretation"
    ]
    if unsupported:
        blocking.append(
            ValidationIssue(
                code="unsupported_claims",
                message=f"Unsupported claims remain: {', '.join(unsupported)}.",
                blocking=True,
            )
        )
    else:
        passed.append("All displayed structured claims have evidence or a calculation trace.")

    qualitative_match = re.search(
        r"(?s)<!-- qualitative-synthesis:start -->(.*?)"
        r"<!-- qualitative-synthesis:end -->",
        report,
    )
    if qualitative_match and re.search(r"\d", qualitative_match.group(1)):
        blocking.append(
            ValidationIssue(
                code="qualitative_numeric_claim",
                message=(
                    "The qualitative synthesis contains a numerical claim; numerical "
                    "facts must come from deterministic report sections."
                ),
                blocking=True,
            )
        )
    else:
        passed.append("Qualitative synthesis contains no independently generated numbers.")

    if _news_called_transcript(report, by_name.get("recent_news")):
        blocking.append(
            ValidationIssue(
                code="news_transcript_confusion",
                message="News content is described as an earnings-call transcript.",
                blocking=True,
            )
        )
    else:
        passed.append("News and transcript content types remain distinct.")

    core_missing = any(
        not by_name.get(name) or by_name[name].status not in _USABLE
        for name in ("market_snapshot", "financial_statements", "sec_company_facts")
    )
    if core_missing and re.search(r"(?i)evidence quality[^\n]*(?:high|strong)", report):
        blocking.append(
            ValidationIssue(
                code="overstated_evidence_quality",
                message="The report overstates evidence quality while core datasets are missing.",
                blocking=True,
            )
        )
    else:
        passed.append("Evidence-quality wording matches core source availability.")

    if report.count("## Research Conclusion") > 1:
        blocking.append(
            ValidationIssue(
                code="duplicated_conclusion",
                message="The research conclusion is duplicated.",
                blocking=True,
            )
        )
    else:
        passed.append("The research conclusion appears once.")

    if report.count("## Disclaimer") > 1:
        blocking.append(
            ValidationIssue(
                code="duplicated_disclaimer",
                message="The financial disclaimer is duplicated.",
                blocking=True,
            )
        )
    else:
        passed.append("The financial disclaimer is not duplicated.")

    recommendations = _RECOMMENDATION.findall(report)
    if recommendations:
        blocking.append(
            ValidationIssue(
                code="direct_recommendation",
                message="Direct BUY/HOLD/SELL wording is not allowed in the report.",
                blocking=True,
            )
        )
    else:
        passed.append("No direct trading recommendation appears.")

    source_keys = [
        (
            source.dataset.casefold(),
            source.provider.casefold(),
            _canonical_url(source.url),
            source.accession_number,
        )
        for source in sources
    ]
    duplicates = [key for key, count in Counter(source_keys).items() if count > 1]
    if duplicates:
        warnings.append(
            ValidationIssue(
                code="duplicate_sources",
                message="Duplicate source records remain after normalization.",
            )
        )
    else:
        passed.append("Source records are deduplicated by dataset and provenance.")

    source_conflicts = [
        source for source in sources if source.status is Availability.CONFLICT and source.warning
    ]
    if source_conflicts:
        warnings.append(
            ValidationIssue(
                code="source_availability_conflict",
                message=(
                    "Conflicting availability observations were retained with an "
                    "explicit source warning."
                ),
            )
        )
    else:
        passed.append("No unexplained source availability contradiction was detected.")

    analysis_date_text = analysis_date.strftime("%Y-%m-%d")
    if analysis_date_text not in report:
        blocking.append(
            ValidationIssue(
                code="analysis_date_mismatch",
                message="The report does not contain the current analysis date.",
                blocking=True,
            )
        )
    else:
        passed.append("The report analysis date matches the workflow timestamp.")

    future_completed = _future_completed_dates(report, analysis_date)
    if future_completed:
        blocking.append(
            ValidationIssue(
                code="future_event_as_completed",
                message=(
                    "A future date is described with completed-event wording: "
                    f"{', '.join(future_completed)}."
                ),
                blocking=True,
            )
        )
    else:
        passed.append("No future event is described as already completed.")

    passed.append("Historical trend calculations use annual observations only.")

    currencies = {str(value) for result in data for value in _currency_values(result) if value}
    if len(currencies) > 1 and "currency mismatch" not in report.casefold():
        warnings.append(
            ValidationIssue(
                code="mixed_currencies",
                message=(
                    "Multiple currencies are present without an explicit "
                    "currency-mismatch warning: "
                    f"{', '.join(sorted(currencies))}."
                ),
            )
        )
    else:
        passed.append("No unexplained currency mixing was detected.")

    stale_sources = [item for item in data if item.status is Availability.STALE]
    if stale_sources and re.search(r"(?i)\b(?:real[- ]?time|live price)\b", report):
        blocking.append(
            ValidationIssue(
                code="stale_described_as_live",
                message="Stale source data is described as live or real-time.",
                blocking=True,
            )
        )
    else:
        passed.append("Stale data is not described as real-time.")

    report_complete = not blocking
    return ConsistencyValidation(
        passed_checks=passed,
        warnings=warnings,
        blocking_errors=blocking,
        regeneration_attempted=regeneration_attempted,
        report_complete=report_complete,
    )


def _statement_evidence(result: DataResult) -> list[EvidenceRef]:
    output: list[EvidenceRef] = []
    periods = result.values.get("annual_periods", [])
    if not periods and result.values.get("income_period_end"):
        periods = [
            {
                **result.values,
                "period_end": result.values.get("income_period_end"),
            }
        ]
    for period in periods:
        for metric in (
            "revenue",
            "net_income",
            "operating_cash_flow",
            "capital_expenditure",
            "free_cash_flow",
            "cash",
            "debt",
            "diluted_shares",
        ):
            value = period.get(metric)
            if not isinstance(value, Real):
                continue
            output.append(
                EvidenceRef(
                    source=result.source,
                    source_type="annual_financial_statement",
                    provider="yfinance",
                    title=f"Annual {metric.replace('_', ' ')}",
                    url=_result_url(result),
                    period_end=period.get("period_end"),
                    fiscal_year=period.get("fiscal_year"),
                    metric=metric,
                    value=value,
                    unit=period.get("currency"),
                    evidence_status=result.status,
                )
            )
    return output


def _market_evidence(result: DataResult) -> list[EvidenceRef]:
    output: list[EvidenceRef] = []
    for field, metric in (
        ("price", "market_price"),
        ("market_cap", "market_cap"),
        ("previous_close", "previous_close"),
        ("day_high", "day_high"),
        ("day_low", "day_low"),
        ("volume", "volume"),
    ):
        value = result.values.get(field)
        if not isinstance(value, Real):
            continue
        output.append(
            EvidenceRef(
                source=result.source,
                source_type="market_data",
                provider="yfinance",
                title=f"Market {metric.replace('_', ' ')}",
                url=_result_url(result),
                period_end=result.values.get("trading_date"),
                metric=metric,
                value=value,
                unit=result.values.get("currency"),
                evidence_status=result.status,
            )
        )
    timestamp = result.values.get("retrieval_timestamp")
    if timestamp:
        output.append(
            EvidenceRef(
                source=result.source,
                source_type="market_data",
                provider="yfinance",
                title="Market retrieval timestamp",
                url=_result_url(result),
                period_end=result.values.get("trading_date"),
                metric="market_timestamp",
                value=timestamp,
                evidence_status=result.status,
            )
        )
    return output


def _dcf_evidence(result: DataResult) -> list[EvidenceRef]:
    output: list[EvidenceRef] = []
    for scenario in result.values.get("scenarios", []):
        name = str(scenario.get("name", "")).casefold()
        value = scenario.get("per_share_value")
        if not name or not isinstance(value, Real):
            continue
        output.append(
            EvidenceRef(
                source=result.source,
                source_type="deterministic_calculation",
                provider="Local calculation",
                title=f"DCF {name} per-share value",
                period_end=result.values.get("period_end"),
                metric=f"dcf_{name}",
                value=value,
                unit=result.values.get("currency"),
                calculation_id=f"calc-dcf-{name}",
                evidence_status=result.status,
            )
        )
    return output


def _document_evidence(result: DataResult) -> list[EvidenceRef]:
    return [
        EvidenceRef(
            source=f"Uploaded PDF: {item.get('filename', 'document')}",
            source_type="uploaded_document",
            provider="User upload",
            title=item.get("filename"),
            page_number=item.get("page_number"),
            excerpt=str(item.get("text", ""))[:500],
        )
        for item in result.values.get("excerpts", [])
    ]


def _statement_source_records(result: DataResult) -> list[SourceRecord]:
    statuses = result.values.get("dataset_statuses", {})
    period = result.values.get("income_period_end")
    records = []
    for key, dataset in (
        ("annual_income_statement", "Annual income statement"),
        ("annual_cash_flow_statement", "Annual cash-flow statement"),
        ("annual_balance_sheet", "Annual balance sheet"),
    ):
        try:
            status = Availability(statuses.get(key, result.status))
        except ValueError:
            status = Availability.INVALID
        records.append(
            SourceRecord(
                source_id=_source_id(dataset, result.source, _result_url(result)),
                dataset=dataset,
                provider="yfinance",
                title=f"{dataset} for {result.values.get('company_name') or 'requested company'}",
                url=_result_url(result),
                status=status,
                period=period,
                retrieved_at=_retrieval_time(result),
                warning=result.message,
            )
        )
    return records


def _dataset_label(name: str) -> str:
    return {
        "market_snapshot": "Market snapshot",
        "recent_news": "Company news",
        "sec_company_facts": "SEC company facts",
        "earnings_transcript": "Earnings-call transcript",
        "discounted_cash_flow": "DCF valuation",
        "uploaded_documents": "Uploaded PDF evidence",
        "cross_source_conflicts": "Cross-source validation",
    }.get(name, name.replace("_", " ").title())


def _provider_label(result: DataResult) -> str:
    if result.name == "market_snapshot":
        return "yfinance"
    if result.name == "recent_news":
        return "Yahoo Finance"
    if result.name == "sec_company_facts":
        return "SEC EDGAR"
    if result.name == "discounted_cash_flow":
        return "Local calculation"
    return result.source


def _source_title(result: DataResult) -> str:
    return next(
        (evidence.title for evidence in result.evidence if evidence.title),
        _dataset_label(result.name),
    )


def _result_url(result: DataResult) -> str | None:
    return next((evidence.url for evidence in result.evidence if evidence.url), None)


def _retrieval_time(result: DataResult) -> datetime:
    return next(
        (evidence.retrieved_at for evidence in result.evidence if evidence.retrieved_at),
        datetime.now().astimezone(),
    )


def _result_period(result: DataResult) -> str | None:
    return (
        result.values.get("trading_date")
        or result.values.get("income_period_end")
        or result.values.get("period_end")
        or next(
            (evidence.period_end for evidence in result.evidence if evidence.period_end),
            None,
        )
    )


def _accession(result: DataResult) -> str | None:
    return next(
        (evidence.accession_number for evidence in result.evidence if evidence.accession_number),
        None,
    )


def _source_id(dataset: str, provider: str, url: str | None) -> str:
    raw = f"{dataset}|{provider}|{_canonical_url(url)}".encode()
    return f"src-{hashlib.sha256(raw).hexdigest()[:12]}"


def _canonical_url(url: str | None) -> str:
    if not url:
        return ""
    parts = urlsplit(url)
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_") and key.casefold() not in _TRACKING_PARAMETERS
        )
    )
    return urlunsplit(
        (
            parts.scheme.casefold(),
            parts.netloc.casefold(),
            parts.path.rstrip("/"),
            query,
            "",
        )
    )


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _stable_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return repr(value)


def _news_called_transcript(report: str, news: DataResult | None) -> bool:
    if not news or news.status not in _USABLE:
        return False
    return bool(
        re.search(
            r"(?is)(?:news|recent developments).{0,100}(?:earnings-call )?transcript",
            report,
        )
        and "news, not transcripts" not in report.casefold()
    )


def _currency_values(result: DataResult) -> list[Any]:
    values = [result.values.get("currency")]
    values.extend(
        item.get("currency")
        for item in result.values.get("annual_periods", [])
        if isinstance(item, dict)
    )
    return values


def _future_completed_dates(report: str, analysis_date: datetime) -> list[str]:
    output: list[str] = []
    for match in re.finditer(r"\b(20\d{2}-\d{2}-\d{2})\b", report):
        try:
            value = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        context = report[max(0, match.start() - 80) : match.end() + 80]
        if value > analysis_date.date() and re.search(
            r"(?i)\b(?:reported|completed|announced|occurred|closed)\b",
            context,
        ):
            output.append(match.group(1))
    return sorted(set(output))
