"""Small deterministic offline benchmark for core financial-research behavior."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import StructuredTool

from financial_analyst.documents import hybrid_retrieve
from financial_analyst.evidence import validate_report, verify_claims
from financial_analyst.llm import create_research_plan
from financial_analyst.models import (
    AnalysisDepth,
    Availability,
    CalculationRecord,
    Claim,
    ConfidenceCategory,
    DataResult,
    DocumentChunk,
    EvidenceRef,
    ResearchRequest,
    SupportStatus,
    UploadedDocument,
)
from financial_analyst.sec import _select_fact, reconcile_financial_sources
from financial_analyst.valuation import verify_calculations


@dataclass(frozen=True)
class EvaluationSummary:
    metrics: dict[str, float]
    task_count: int
    passed: bool
    failed_thresholds: tuple[str, ...]


THRESHOLDS = {
    "numeric_accuracy": 0.95,
    "sec_selection_accuracy": 0.90,
    "tool_plan_precision": 0.85,
    "tool_plan_recall": 0.85,
    "retrieval_recall_at_k": 0.80,
    "citation_precision": 0.80,
    "citation_recall": 0.90,
    "unsupported_claim_rate": 0.05,
    "consistency_pass_rate": 0.90,
    "source_conflict_detection_rate": 0.90,
    "provider_format_success_rate": 0.90,
}


def run_evaluation() -> EvaluationSummary:
    """Run 53 offline fixture tasks and evaluate measured deterministic outcomes."""

    metrics: dict[str, float] = {}
    numeric_score, numeric_count = _numeric_accuracy()
    sec_score, sec_count = _sec_accuracy()
    plan_precision, plan_recall, plan_count = _tool_plan_quality()
    retrieval_score, retrieval_count = _retrieval_quality()
    citation_precision, citation_recall, unsupported_rate, citation_count = _citation_quality()
    consistency_score, consistency_count = _consistency_quality()
    conflict_score, conflict_count = _conflict_quality()
    provider_score, provider_count = _provider_format_quality()
    metrics.update(
        {
            "numeric_accuracy": numeric_score,
            "sec_selection_accuracy": sec_score,
            "tool_plan_precision": plan_precision,
            "tool_plan_recall": plan_recall,
            "retrieval_recall_at_k": retrieval_score,
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "unsupported_claim_rate": unsupported_rate,
            "consistency_pass_rate": consistency_score,
            "source_conflict_detection_rate": conflict_score,
            "provider_format_success_rate": provider_score,
        }
    )
    failed = tuple(
        name
        for name, threshold in THRESHOLDS.items()
        if (
            metrics[name] > threshold
            if name == "unsupported_claim_rate"
            else metrics[name] < threshold
        )
    )
    return EvaluationSummary(
        metrics=metrics,
        task_count=(
            numeric_count
            + sec_count
            + plan_count
            + retrieval_count
            + citation_count
            + consistency_count
            + conflict_count
            + provider_count
        ),
        passed=not failed,
        failed_thresholds=failed,
    )


def _numeric_accuracy() -> tuple[float, int]:
    definitions = [
        ("growth", {"current": 110.0, "prior": 100.0}, 0.1),
        ("growth", {"current": 90.0, "prior": 100.0}, -0.1),
        ("growth", {"current": 121.0, "prior": 100.0}, 0.21),
        ("ratio", {"numerator": 20.0, "denominator": 100.0}, 0.2),
        ("ratio", {"numerator": -5.0, "denominator": 100.0}, -0.05),
        ("ratio", {"numerator": 30.0, "denominator": 25.0}, 1.2),
        ("difference", {"left": 20.0, "right": 5.0}, 15.0),
        ("difference", {"left": 5.0, "right": 20.0}, -15.0),
        ("difference", {"left": 0.0, "right": 0.0}, 0.0),
        ("upside", {"model_value": 120.0, "price": 100.0}, 0.2),
        ("upside", {"model_value": 80.0, "price": 100.0}, -0.2),
        ("upside", {"model_value": 100.0, "price": 100.0}, 0.0),
    ]
    records = [
        CalculationRecord(
            calculation_id=f"calc-{index}",
            calculation_type=kind,
            formula=kind,
            inputs=inputs,
            input_source_ids=[f"ev-{index}"],
            output=expected,
        )
        for index, (kind, inputs, expected) in enumerate(definitions)
    ]
    verified = verify_calculations(records)
    correct = sum(item.status is SupportStatus.VERIFIED for item in verified)
    return correct / len(records), len(records)


def _sec_accuracy() -> tuple[float, int]:
    entries = [
        _sec_entry("annual", "2024-01-01", "2024-12-31", "10-K", "2025-02-01"),
        _sec_entry("annual-amended", "2024-01-01", "2024-12-31", "10-K/A", "2025-03-01"),
        _sec_entry("quarter", "2025-04-01", "2025-06-30", "10-Q", "2025-08-01"),
        _sec_entry("ytd", "2025-01-01", "2025-06-30", "10-Q", "2025-08-01"),
        _sec_entry("instant-k", None, "2024-12-31", "10-K", "2025-02-01"),
        _sec_entry("instant-q", None, "2025-06-30", "10-Q", "2025-08-01"),
    ]
    cases = [
        (entries[:2], "annual", "annual-amended"),
        (entries[2:4], "quarter_only", "quarter"),
        (entries[2:4], "year_to_date", "ytd"),
        ([entries[4]], "annual_instant", "instant-k"),
        ([entries[5]], "quarterly_instant", "instant-q"),
        ([entries[3]], "quarter_only", None),
        ([entries[2]], "year_to_date", None),
        ([], "annual", None),
    ]
    correct = 0
    for candidates, period_type, expected in cases:
        selected = _select_fact(candidates, period_type=period_type)
        observed = selected[0]["accn"] if selected else None
        correct += observed == expected
    return correct / len(cases), len(cases)


def _tool_plan_quality() -> tuple[float, float, int]:
    available = [
        "market_snapshot",
        "financial_statements",
        "sec_company_facts",
        "recent_news",
        "earnings_transcript",
        "discounted_cash_flow",
    ]
    cases = [
        ("Analyze MSFT.", AnalysisDepth.QUICK, {"market_snapshot", "financial_statements"}),
        (
            "Run a DCF valuation for MSFT.",
            AnalysisDepth.QUICK,
            {"market_snapshot", "financial_statements", "discounted_cash_flow"},
        ),
        (
            "Analyze MSFT.",
            AnalysisDepth.STANDARD,
            {"market_snapshot", "financial_statements", "sec_company_facts", "recent_news"},
        ),
        (
            "Review the Q2 2025 earnings call transcript for MSFT.",
            AnalysisDepth.STANDARD,
            {
                "market_snapshot",
                "financial_statements",
                "sec_company_facts",
                "recent_news",
                "earnings_transcript",
            },
        ),
        (
            "Detailed analysis of MSFT.",
            AnalysisDepth.DETAILED,
            {"market_snapshot", "financial_statements", "sec_company_facts", "recent_news"},
        ),
        (
            "Detailed intrinsic value for MSFT.",
            AnalysisDepth.DETAILED,
            {
                "market_snapshot",
                "financial_statements",
                "sec_company_facts",
                "recent_news",
                "discounted_cash_flow",
            },
        ),
    ]
    true_positive = selected_total = expected_total = 0
    from financial_analyst.llm import _safe_fallback_plan

    for query, depth, expected in cases:
        plan = _safe_fallback_plan(
            ResearchRequest(query=query, ticker="MSFT", analysis_depth=depth),
            "MSFT",
            available,
            3 if depth is AnalysisDepth.QUICK else 6 if depth is AnalysisDepth.STANDARD else 8,
            (),
        )
        selected = {step.tool_name for step in plan.steps}
        true_positive += len(selected & expected)
        selected_total += len(selected)
        expected_total += len(expected)
    return (
        true_positive / max(selected_total, 1),
        true_positive / max(expected_total, 1),
        len(cases),
    )


def _retrieval_quality() -> tuple[float, int]:
    document = _evaluation_document()
    cases = [
        ("revenue sales", 1),
        ("profitability earnings", 1),
        ("cash and borrowing risk", 2),
        ("liquidity leverage", 2),
        ("regulatory compliance threat", 3),
        ("unrelated weather forecast", None),
    ]
    correct = 0
    for query, expected_page in cases:
        hits = hybrid_retrieve([document], query, limit=2)
        observed_pages = {hit.page_number for hit in hits}
        correct += not hits if expected_page is None else expected_page in observed_pages
    return correct / len(cases), len(cases)


def _citation_quality() -> tuple[float, float, float, int]:
    evidence = [
        EvidenceRef(evidence_id=f"ev-{index}", source="fixture", metric=f"m{index}")
        for index in range(5)
    ]
    claims = [
        Claim(
            claim_id=f"claim-{index}",
            text=f"Fact {index}",
            category="fixture",
            evidence_ids=[f"ev-{index}"],
            support_status=SupportStatus.UNSUPPORTED,
            confidence_category=ConfidenceCategory.INSUFFICIENT,
        )
        for index in range(5)
    ]
    claims.append(
        Claim(
            claim_id="unsupported",
            text="Unsupported fact",
            category="fixture",
            evidence_ids=["missing"],
            support_status=SupportStatus.VERIFIED,
            confidence_category=ConfidenceCategory.HIGH,
        )
    )
    verified = verify_claims(claims, evidence)
    cited = [claim for claim in verified if claim.evidence_ids]
    valid_cited = [claim for claim in cited if claim.support_status is SupportStatus.VERIFIED]
    supported_expected = 5
    unsupported = verified[-1]
    unsupported_accepted = unsupported.support_status is SupportStatus.VERIFIED
    return (
        len(valid_cited) / len(cited),
        len(valid_cited) / supported_expected,
        float(unsupported_accepted),
        len(claims),
    )


def _consistency_quality() -> tuple[float, int]:
    analysis_date = datetime(2026, 1, 1, tzinfo=UTC)
    base = "2026-01-01\n## Research Conclusion\nEvidence remains limited."
    cases = [
        (base, True),
        (f"{base}\nBUY the shares.", False),
        (f"{base}\n## Research Conclusion\nRepeated.", False),
        (f"{base}\nThe event completed on 2027-01-01.", False),
        (f"{base}\nNews, not transcripts, is shown.", True),
        (f"{base}\n## Disclaimer\nEducational only.", True),
    ]
    correct = 0
    for report, expected_complete in cases:
        validation = validate_report(
            report=report,
            data=[],
            claims=[],
            sources=[],
            analysis_date=analysis_date,
        )
        correct += validation.report_complete is expected_complete
    return correct / len(cases), len(cases)


def _conflict_quality() -> tuple[float, int]:
    cases = [
        (100.0, 100.0, False),
        (100.0, 100.5, False),
        (100.0, 110.0, True),
        (100.0, 80.0, True),
    ]
    correct = 0
    for provider_value, sec_value, expected_conflict in cases:
        statements = _reconciliation_statements(provider_value)
        sec = _reconciliation_sec(sec_value)
        _, records = reconcile_financial_sources(statements, sec)
        observed = records[0].conflict_status is Availability.CONFLICT
        correct += observed is expected_conflict
    return correct / len(cases), len(cases)


def _provider_format_quality() -> tuple[float, int]:
    payloads = [
        '{"selected_tools":["market_snapshot"],"purposes":{}}',
        '```json\n{"selected_tools":["financial_statements"],"purposes":{}}\n```',
        'Plan: {"selected_tools":["sec_company_facts"],"purposes":{}}',
        '{"selected_tools":["recent_news"],"requested_outputs":["news"]}',
        "malformed output",
    ]
    tools = tuple(
        StructuredTool.from_function(
            func=lambda ticker: ticker,
            name=name,
            description=name,
        )
        for name in (
            "market_snapshot",
            "financial_statements",
            "sec_company_facts",
            "recent_news",
        )
    )
    successes = 0
    for content in payloads:
        model = _FixturePlanner(content)
        plan = create_research_plan(
            model,
            ResearchRequest(query="Analyze MSFT.", ticker="MSFT"),
            "MSFT",
            tools,
        )
        successes += bool(plan.steps) and plan.planning_method in {
            "structured_json",
            "safe_fallback",
        }
    return successes / len(payloads), len(payloads)


class _FixturePlanner:
    def __init__(self, content: str) -> None:
        self.content = content

    def bind_tools(self, tools: list[Any]) -> Any:
        raise NotImplementedError

    def invoke(self, messages: list[Any]) -> Any:
        return type("Response", (), {"content": self.content})()


def _sec_entry(
    accession: str,
    start: str | None,
    end: str,
    form: str,
    filed: str,
) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "val": 1.0,
        "accn": accession,
        "fy": 2025,
        "fp": "FY" if form.startswith("10-K") else "Q2",
        "form": form,
        "filed": filed,
    }


def _evaluation_document() -> UploadedDocument:
    texts = [
        "Revenue and sales increased while earnings and profit margins improved.",
        "Cash liquidity remains adequate, but debt borrowings create refinancing risk.",
        "Regulatory compliance exposure creates a material legal threat.",
    ]
    return UploadedDocument(
        document_id="evaluation",
        safe_filename="evaluation.pdf",
        page_count=3,
        chunks=[
            DocumentChunk(
                document_id="evaluation",
                safe_filename="evaluation.pdf",
                page_number=index,
                chunk_index=0,
                text=text,
                character_end=len(text),
            )
            for index, text in enumerate(texts, start=1)
        ],
    )


def _reconciliation_statements(value: float) -> DataResult:
    return DataResult(
        name="financial_statements",
        status=Availability.AVAILABLE,
        source="yfinance",
        values={
            "annual_periods": [
                {
                    "period_end": "2025-12-31",
                    "currency": "USD",
                    "revenue": value,
                }
            ]
        },
    )


def _reconciliation_sec(value: float) -> DataResult:
    return DataResult(
        name="sec_company_facts",
        status=Availability.AVAILABLE,
        source="SEC",
        values={
            "facts": {
                "revenue": {
                    "concept": "Revenues",
                    "unit": "USD",
                    "annual": {
                        "value": value,
                        "period_end": "2025-12-31",
                        "duration_days": 365,
                    },
                }
            }
        },
    )


def _print_summary(summary: EvaluationSummary) -> None:
    print(f"Offline evaluation tasks: {summary.task_count}")
    print(f"{'Metric':36} {'Measured':>10} {'Threshold':>10}")
    print("-" * 60)
    for name, value in summary.metrics.items():
        print(f"{name:36} {value:10.3f} {THRESHOLDS[name]:10.3f}")
    print("-" * 60)
    print("PASS" if summary.passed else f"FAIL: {', '.join(summary.failed_thresholds)}")


if __name__ == "__main__":
    result = run_evaluation()
    _print_summary(result)
    raise SystemExit(0 if result.passed else 1)
