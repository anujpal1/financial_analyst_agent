"""Structured data exchanged between financial sources and the reporting layer."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return a timezone-aware retrieval timestamp."""

    return datetime.now(UTC)


class Availability(StrEnum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    CONFLICT = "conflict"
    INVALID = "invalid"


class AnalysisDepth(StrEnum):
    QUICK = "Quick"
    STANDARD = "Standard"
    DETAILED = "Detailed"


class SupportStatus(StrEnum):
    VERIFIED = "Verified"
    PARTIALLY_SUPPORTED = "Partially supported"
    UNSUPPORTED = "Unsupported"
    CONFLICTING = "Conflicting"
    NOT_APPLICABLE = "Not applicable"


class ConfidenceCategory(StrEnum):
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    INSUFFICIENT = "Insufficient"


class EvidenceRef(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f"ev-{uuid4().hex[:12]}")
    source: str
    source_type: str = "provider_data"
    provider: str | None = None
    title: str | None = None
    url: str | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    form: str | None = None
    filing_date: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    accession_number: str | None = None
    unit: str | None = None
    page_number: int | None = None
    metric: str | None = None
    value: Any = None
    evidence_status: Availability = Availability.AVAILABLE
    excerpt: str | None = None
    calculation_id: str | None = None


class DataResult(BaseModel):
    """A typed source result that never replaces absent facts with invented values."""

    name: str
    status: Availability
    source: str
    values: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    message: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    content_type: str = "financial_data"

    @classmethod
    def unavailable(
        cls,
        *,
        name: str,
        source: str,
        message: str,
        missing_fields: list[str] | None = None,
        content_type: str = "financial_data",
    ) -> DataResult:
        return cls(
            name=name,
            status=Availability.UNAVAILABLE,
            source=source,
            message=message,
            missing_fields=missing_fields or [],
            content_type=content_type,
        )


class DocumentChunk(BaseModel):
    document_id: str
    safe_filename: str
    page_number: int
    chunk_index: int
    text: str


class UploadedDocument(BaseModel):
    document_id: str
    safe_filename: str
    page_count: int
    chunks: list[DocumentChunk]
    extraction_errors: list[str] = Field(default_factory=list)


class PricePoint(BaseModel):
    date: str
    close: float


class CanonicalMarketData(BaseModel):
    ticker: str
    price: float | None = None
    currency: str | None = None
    trading_date: str | None = None
    retrieval_timestamp: datetime = Field(default_factory=utc_now)
    market_state: str | None = None
    previous_close: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    volume: float | None = None
    market_cap: float | None = None
    price_basis: str | None = None
    history: list[PricePoint] = Field(default_factory=list)
    source: str = "Yahoo Finance via yfinance"
    status: Availability
    error_reason: str | None = None
    is_delayed: bool = False


class AnnualFinancialPeriod(BaseModel):
    period_end: str
    fiscal_year: int | None = None
    currency: str | None = None
    revenue: float | None = None
    net_income: float | None = None
    operating_cash_flow: float | None = None
    capital_expenditure: float | None = None
    free_cash_flow: float | None = None
    cash: float | None = None
    debt: float | None = None
    diluted_shares: float | None = None


class TrendMetric(BaseModel):
    name: str
    unit: str
    values: list[dict[str, Any]]


class HistoricalAnalysis(BaseModel):
    periods: list[AnnualFinancialPeriod] = Field(default_factory=list)
    revenue_growth: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    metrics: list[TrendMetric] = Field(default_factory=list)


class DashboardMetric(BaseModel):
    key: str
    label: str
    value: float | str | None = None
    formatted_value: str = "Unavailable"
    period: str | None = None
    source: str | None = None
    status: Availability = Availability.UNAVAILABLE
    detail: str | None = None


class ExecutiveDashboard(BaseModel):
    metrics: list[DashboardMetric] = Field(default_factory=list)


class ScoreContribution(BaseModel):
    metric: str
    value: float | str
    points: float
    rule: str


class ScoreComponent(BaseModel):
    name: str
    score: float | None = Field(default=None, ge=0, le=100)
    contributions: list[ScoreContribution] = Field(default_factory=list)
    missing_metrics: list[str] = Field(default_factory=list)
    explanation: str


class FinancialScorecard(BaseModel):
    components: list[ScoreComponent] = Field(default_factory=list)
    overall_score: float | None = Field(default=None, ge=0, le=100)
    overall_explanation: str


class DataQualityRow(BaseModel):
    dataset: str
    status: Availability
    source: str
    period: str | None = None
    retrieved_at: datetime | None = None
    warning: str | None = None


class EvidenceQualityAssessment(BaseModel):
    label: ConfidenceCategory
    coverage_score: float = Field(ge=0, le=100)
    components: dict[str, str] = Field(default_factory=dict)
    missing_evidence: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    freshness_summary: str


class SourceRecord(BaseModel):
    source_id: str
    dataset: str
    provider: str
    title: str
    url: str | None = None
    status: Availability
    period: str | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    accession_number: str | None = None
    warning: str | None = None


class Claim(BaseModel):
    claim_id: str
    text: str
    category: str
    evidence_ids: list[str] = Field(default_factory=list)
    calculation_id: str | None = None
    support_status: SupportStatus
    conflict_status: bool = False
    confidence_category: ConfidenceCategory


class ValidationIssue(BaseModel):
    code: str
    message: str
    blocking: bool = False


class ConsistencyValidation(BaseModel):
    passed_checks: list[str] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    blocking_errors: list[ValidationIssue] = Field(default_factory=list)
    regeneration_attempted: bool = False
    report_complete: bool = True


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4000)
    ticker: str | None = None
    documents: list[UploadedDocument] = Field(default_factory=list)
    dcf_growth_rate: float = Field(default=0.05, gt=-1.0, lt=1.0)
    dcf_discount_rate: float = Field(default=0.10, gt=0.0, lt=1.0)
    dcf_terminal_growth_rate: float = Field(default=0.025, gt=-1.0, lt=1.0)
    analysis_depth: AnalysisDepth = AnalysisDepth.STANDARD


class ResearchResult(BaseModel):
    session_id: str
    ticker: str
    analysis_date: datetime
    report_markdown: str
    data: list[DataResult]
    evidence_quality: str
    evidence_quality_detail: EvidenceQualityAssessment | None = None
    dashboard: ExecutiveDashboard | None = None
    historical_analysis: HistoricalAnalysis | None = None
    scorecard: FinancialScorecard | None = None
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    sources: list[SourceRecord] = Field(default_factory=list)
    data_quality: list[DataQualityRow] = Field(default_factory=list)
    validation: ConsistencyValidation | None = None
