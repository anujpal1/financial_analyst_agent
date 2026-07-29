"""Provider-agnostic LangGraph workflow for deterministic collection and reporting."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from financial_analyst.config import AppSettings
from financial_analyst.llm import validate_tool_calling
from financial_analyst.models import (
    Availability,
    DataResult,
    EvidenceRef,
    ResearchRequest,
    ResearchResult,
    utc_now,
)
from financial_analyst.reporting import build_report, detect_conflicts
from financial_analyst.security import new_session_id
from financial_analyst.tickers import resolve_ticker
from financial_analyst.tools import build_tool_registry, tool_mapping


class ResearchState(TypedDict, total=False):
    request: ResearchRequest
    session_id: str
    ticker: str
    data: list[DataResult]
    analysis_date: Any
    report_markdown: str
    evidence_quality: str


def build_research_graph(
    *,
    llm: BaseChatModel,
    settings: AppSettings,
    tools: Sequence[BaseTool] | None = None,
    require_tool_calling: bool = True,
) -> Any:
    """Compile the same workflow for every supported provider."""

    registry = tuple(tools or build_tool_registry(settings))
    mapping = tool_mapping(registry)
    if require_tool_calling:
        validate_tool_calling(llm, registry)

    def validate_node(state: ResearchState) -> ResearchState:
        request = state["request"]
        return {
            "ticker": resolve_ticker(request.query, request.ticker),
            "analysis_date": utc_now(),
        }

    def collect_node(state: ResearchState) -> ResearchState:
        request = state["request"]
        ticker = state["ticker"]
        results = [
            mapping["market_snapshot"].invoke({"ticker": ticker}),
            mapping["financial_statements"].invoke({"ticker": ticker}),
            mapping["price_history"].invoke({"ticker": ticker}),
            mapping["recent_news"].invoke({"ticker": ticker}),
            mapping["sec_company_facts"].invoke({"ticker": ticker}),
        ]

        query_lower = request.query.lower()
        if any(word in query_lower for word in ("peer", "competitor", "relative valuation")):
            results.append(mapping["peer_comparison"].invoke({"ticker": ticker}))
        if any(word in query_lower for word in ("transcript", "earnings call")):
            period = _transcript_period(request.query)
            if period:
                year, quarter = period
                results.append(
                    mapping["earnings_transcript"].invoke(
                        {"ticker": ticker, "year": year, "quarter": quarter}
                    )
                )
            else:
                results.append(
                    DataResult.unavailable(
                        name="earnings_transcript",
                        source="Transcript request validation",
                        message=(
                            "Transcript requested, but a year and quarter such as "
                            "'Q2 2025' were not provided."
                        ),
                        missing_fields=["year", "quarter"],
                        content_type="transcript",
                    )
                )

        if request.documents:
            results.append(_document_result(request))

        if any(
            phrase in query_lower
            for phrase in ("dcf", "valuation", "intrinsic value", "fair value")
        ):
            statements = next(
                (item for item in results if item.name == "financial_statements"),
                None,
            )
            results.append(_run_valuation(mapping, request, statements))

        conflict = detect_conflicts(results)
        if conflict:
            results.append(conflict)
        return {"data": results}

    def report_node(state: ResearchState) -> ResearchState:
        report, quality = build_report(
            llm=llm,
            request=state["request"],
            ticker=state["ticker"],
            data=state["data"],
            analysis_date=state["analysis_date"],
        )
        return {"report_markdown": report, "evidence_quality": quality}

    graph = StateGraph(ResearchState)
    graph.add_node("validate_request", validate_node)
    graph.add_node("collect_evidence", collect_node)
    graph.add_node("build_report", report_node)
    graph.add_edge(START, "validate_request")
    graph.add_edge("validate_request", "collect_evidence")
    graph.add_edge("collect_evidence", "build_report")
    graph.add_edge("build_report", END)
    return graph.compile()


def run_research(
    graph: Any,
    request: ResearchRequest,
    *,
    session_id: str | None = None,
) -> ResearchResult:
    """Run one isolated research request and return a typed result."""

    active_session_id = session_id or new_session_id()
    state = graph.invoke(
        {
            "request": request,
            "session_id": active_session_id,
        },
        config={"configurable": {"thread_id": active_session_id}},
    )
    return ResearchResult(
        session_id=active_session_id,
        ticker=state["ticker"],
        analysis_date=state["analysis_date"],
        report_markdown=state["report_markdown"],
        data=state["data"],
        evidence_quality=state["evidence_quality"],
    )


def _run_valuation(
    mapping: dict[str, BaseTool],
    request: ResearchRequest,
    statements: DataResult | None,
) -> DataResult:
    values = statements.values if statements else {}
    return mapping["discounted_cash_flow"].invoke(
        {
            "base_free_cash_flow": values.get("free_cash_flow"),
            "growth_rate": request.dcf_growth_rate,
            "discount_rate": request.dcf_discount_rate,
            "terminal_growth_rate": request.dcf_terminal_growth_rate,
            "cash": values.get("cash"),
            "debt": values.get("debt"),
            "diluted_shares": values.get("diluted_shares"),
            "currency": values.get("currency") or "USD",
            "period_end": values.get("cash_flow_period_end"),
        }
    )


def _transcript_period(query: str) -> tuple[int, int] | None:
    match = re.search(r"\bQ([1-4])\s*[-/]?\s*(20\d{2})\b", query, re.IGNORECASE)
    if match:
        return int(match.group(2)), int(match.group(1))
    match = re.search(r"\b(20\d{2})\s*[-/]?\s*Q([1-4])\b", query, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _document_result(request: ResearchRequest) -> DataResult:
    terms = {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]{3,}", request.query)
        if word.lower() not in {"about", "from", "that", "this", "with"}
    }
    scored = []
    for document in request.documents:
        for chunk in document.chunks:
            lower_text = chunk.text.lower()
            score = sum(lower_text.count(term) for term in terms)
            scored.append((score, chunk))
    selected = sorted(
        scored,
        key=lambda item: (item[0], -item[1].page_number, -item[1].chunk_index),
        reverse=True,
    )[:8]
    if not selected:
        return DataResult.unavailable(
            name="uploaded_documents",
            source="User-uploaded PDF",
            message="No extractable uploaded-document evidence was available.",
            content_type="uploaded_document",
        )
    excerpts = [
        {
            "document_id": chunk.document_id,
            "filename": chunk.safe_filename,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
        }
        for _, chunk in selected
    ]
    return DataResult(
        name="uploaded_documents",
        status=Availability.AVAILABLE,
        source="User-uploaded PDF",
        values={"excerpts": excerpts},
        evidence=[
            EvidenceRef(
                source=f"Uploaded PDF: {chunk.safe_filename}",
                page_number=chunk.page_number,
            )
            for _, chunk in selected
        ],
        content_type="uploaded_document",
    )
