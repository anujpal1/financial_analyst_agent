"""Structured data exchanged between financial sources and the reporting layer."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return a timezone-aware retrieval timestamp."""

    return datetime.now(UTC)


class Availability(StrEnum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    CONFLICT = "conflict"


class EvidenceRef(BaseModel):
    source: str
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


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4000)
    ticker: str | None = None
    documents: list[UploadedDocument] = Field(default_factory=list)
    dcf_growth_rate: float = Field(default=0.05, gt=-1.0, lt=1.0)
    dcf_discount_rate: float = Field(default=0.10, gt=0.0, lt=1.0)
    dcf_terminal_growth_rate: float = Field(default=0.025, gt=-1.0, lt=1.0)


class ResearchResult(BaseModel):
    session_id: str
    ticker: str
    analysis_date: datetime
    report_markdown: str
    data: list[DataResult]
    evidence_quality: str
