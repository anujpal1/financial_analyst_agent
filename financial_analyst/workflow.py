"""Bounded single-agent research planning, deterministic execution, and reporting."""

from __future__ import annotations

import re
from collections.abc import Sequence
from time import perf_counter
from typing import Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, START, StateGraph

from financial_analyst import __version__
from financial_analyst.analytics import (
    build_data_quality,
)
from financial_analyst.config import AppSettings
from financial_analyst.documents import hybrid_retrieve
from financial_analyst.evidence import build_evidence_catalog
from financial_analyst.llm import create_research_plan
from financial_analyst.models import (
    AnalysisDepth,
    Availability,
    DataResult,
    EvidenceRef,
    PlanStep,
    ResearchPlan,
    ResearchRequest,
    ResearchResult,
    RunManifest,
    ToolCallRecord,
    utc_now,
)
from financial_analyst.reporting import build_validated_report
from financial_analyst.sec import reconcile_financial_sources
from financial_analyst.security import new_session_id
from financial_analyst.tools import build_tool_registry, tool_mapping

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
_TICKER_STOP_WORDS = {
    "A",
    "AI",
    "AND",
    "DCF",
    "ETF",
    "FOR",
    "IN",
    "NEWS",
    "OF",
    "ON",
    "SEC",
    "THE",
}


class TickerValidationError(ValueError):
    """Raised when no safe ticker can be resolved."""


def normalize_ticker(value: str) -> str:
    """Normalize and validate a ticker before any provider receives it."""

    normalized = value.strip().upper().lstrip("$")
    if not _TICKER_PATTERN.fullmatch(normalized):
        raise TickerValidationError(
            "Ticker must be 1-10 letters/numbers and may include '.' or '-'."
        )
    return normalized


def resolve_ticker(query: str, explicit_ticker: str | None = None) -> str:
    """Resolve an explicit symbol or a small deterministic company-name map."""

    if explicit_ticker and explicit_ticker.strip():
        return normalize_ticker(explicit_ticker)
    exchange_match = _EXCHANGE_TICKER_PATTERN.search(query)
    if exchange_match:
        return normalize_ticker(exchange_match.group(1))
    known_symbols = [
        ticker
        for ticker in dict.fromkeys(_COMMON_COMPANIES.values())
        if re.search(rf"\b{re.escape(ticker)}\b", query, re.IGNORECASE)
    ]
    if known_symbols:
        return known_symbols[0]
    lower_query = query.lower()
    company_matches = [ticker for name, ticker in _COMMON_COMPANIES.items() if name in lower_query]
    if company_matches:
        return company_matches[0]
    symbols = [
        match for match in _QUERY_TICKER_PATTERN.findall(query) if match not in _TICKER_STOP_WORDS
    ]
    if symbols:
        return normalize_ticker(symbols[0])
    raise TickerValidationError(
        "Enter a ticker symbol or mention a supported company name in the question."
    )


class ResearchState(TypedDict, total=False):
    request: ResearchRequest
    session_id: str
    ticker: str
    analysis_date: Any
    started_at: float
    research_plan: ResearchPlan
    replan_count: int
    data: list[DataResult]
    tool_calls: list[ToolCallRecord]
    evidence_gaps: list[str]
    reconciliations: list[Any]
    report_markdown: str
    evidence_quality: str
    evidence_quality_detail: Any
    dashboard: Any
    historical_analysis: Any
    scorecard: Any
    claims: Any
    calculations: Any
    evidence: Any
    sources: Any
    data_quality: Any
    validation: Any
    run_manifest: RunManifest


def build_research_graph(
    *,
    llm: BaseChatModel,
    settings: AppSettings,
    tools: Sequence[BaseTool] | None = None,
    require_tool_calling: bool | None = None,
    provider_name: str = "Configured provider",
    model_name: str = "Configured model",
) -> Any:
    """Compile a bounded planner/executor graph for one controlled research agent."""

    del require_tool_calling
    registry = tuple(tools or build_tool_registry(settings))
    mapping = tool_mapping(registry)
    planning_tools = (*registry, _document_planning_tool())

    def validate_node(state: ResearchState) -> ResearchState:
        request = state["request"]
        return {
            "ticker": resolve_ticker(request.query, request.ticker),
            "analysis_date": utc_now(),
            "started_at": perf_counter(),
            "data": [],
            "tool_calls": [],
            "replan_count": 0,
        }

    def plan_node(state: ResearchState) -> ResearchState:
        plan = create_research_plan(
            llm,
            state["request"],
            state["ticker"],
            planning_tools,
        )
        return {"research_plan": _enforce_mode_contract(plan, state["request"], planning_tools)}

    def execute_node(state: ResearchState) -> ResearchState:
        data = list(state.get("data", []))
        calls = list(state.get("tool_calls", []))
        attempted = {call.tool_name for call in calls}
        updated_steps: list[PlanStep] = []
        for step in state["research_plan"].steps:
            if step.tool_name in attempted:
                updated_steps.append(step.model_copy(update={"status": "skipped"}))
                continue
            if step.tool_name == "discounted_cash_flow":
                updated_steps.append(step)
                continue
            start = perf_counter()
            result = _execute_step(
                step,
                state["request"],
                state["ticker"],
                mapping,
            )
            latency = (perf_counter() - start) * 1000
            data.append(result)
            call_status = (
                "completed"
                if result.status is Availability.AVAILABLE
                else "partial"
                if result.status in {Availability.PARTIAL, Availability.STALE}
                else "failed"
            )
            calls.append(
                ToolCallRecord(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    status=call_status,
                    latency_ms=round(latency, 3),
                    source_status=result.status,
                    message=result.message,
                )
            )
            updated_steps.append(
                step.model_copy(
                    update={
                        "status": call_status,
                        "outcome": result.message or result.status.value,
                        "latency_ms": round(latency, 3),
                    }
                )
            )
        return {
            "data": data,
            "tool_calls": calls,
            "research_plan": state["research_plan"].model_copy(update={"steps": updated_steps}),
        }

    def assess_node(state: ResearchState) -> ResearchState:
        gaps = _evidence_gaps(state["request"], state["data"])
        return {
            "evidence_gaps": gaps,
            "research_plan": state["research_plan"].model_copy(update={"gaps": gaps}),
        }

    def route_after_assessment(state: ResearchState) -> str:
        if (
            state["request"].analysis_depth is AnalysisDepth.DETAILED
            and state.get("evidence_gaps")
            and state.get("replan_count", 0) < 1
        ):
            return "replan"
        return "calculate"

    def replan_node(state: ResearchState) -> ResearchState:
        revised = create_research_plan(
            llm,
            state["request"],
            state["ticker"],
            planning_tools,
            gap_feedback=state.get("evidence_gaps", []),
        )
        revised = _merge_plan_steps(state["research_plan"], revised)
        return {
            "research_plan": _enforce_mode_contract(
                revised,
                state["request"],
                planning_tools,
            ),
            "replan_count": state.get("replan_count", 0) + 1,
        }

    def calculate_node(state: ResearchState) -> ResearchState:
        data = list(state["data"])
        plan = state["research_plan"]
        statements = _result(data, "financial_statements")
        sec = _result(data, "sec_company_facts")
        canonical, reconciliations = reconcile_financial_sources(statements, sec)
        data.append(canonical)
        if _planned(state["research_plan"], "discounted_cash_flow"):
            start = perf_counter()
            dcf = _run_valuation(mapping, state["request"], canonical)
            latency = (perf_counter() - start) * 1000
            data.append(dcf)
            state["tool_calls"].append(
                ToolCallRecord(
                    step_id=_step_id(state["research_plan"], "discounted_cash_flow"),
                    tool_name="discounted_cash_flow",
                    status=(
                        "completed"
                        if dcf.status is Availability.AVAILABLE
                        else "partial"
                        if dcf.status is Availability.PARTIAL
                        else "failed"
                    ),
                    latency_ms=round(latency, 3),
                    source_status=dcf.status,
                    message=dcf.message,
                )
            )
            plan = plan.model_copy(
                update={
                    "steps": [
                        step.model_copy(
                            update={
                                "status": (
                                    "completed"
                                    if dcf.status is Availability.AVAILABLE
                                    else "partial"
                                    if dcf.status is Availability.PARTIAL
                                    else "failed"
                                ),
                                "outcome": dcf.message or dcf.status.value,
                                "latency_ms": round(latency, 3),
                            }
                        )
                        if step.tool_name == "discounted_cash_flow"
                        else step
                        for step in plan.steps
                    ]
                }
            )
        conflicts = [item for item in reconciliations if item.unresolved_warning]
        if conflicts:
            data.append(
                DataResult(
                    name="cross_source_conflicts",
                    status=Availability.CONFLICT,
                    source="Canonical source reconciliation",
                    values={"conflicts": [item.model_dump(mode="json") for item in conflicts]},
                    message=(
                        "Comparable official and provider values differ; SEC remains canonical "
                        "and alternatives are preserved."
                    ),
                    content_type="data_quality",
                )
            )
        return {
            "data": data,
            "reconciliations": reconciliations,
            "research_plan": plan,
        }

    def report_node(state: ResearchState) -> ResearchState:
        artifacts = build_validated_report(
            llm=llm,
            request=state["request"],
            ticker=state["ticker"],
            data=state["data"],
            analysis_date=state["analysis_date"],
        )
        evidence = build_evidence_catalog(state["data"])
        elapsed = (perf_counter() - state["started_at"]) * 1000
        usable = sum(
            item.status in {Availability.AVAILABLE, Availability.PARTIAL, Availability.STALE}
            for item in state["data"]
        )
        source_times = {
            item.name: min(
                (ref.retrieved_at for ref in item.evidence),
                default=state["analysis_date"],
            )
            for item in state["data"]
        }
        manifest = RunManifest(
            run_id=state["session_id"],
            analysis_date=state["analysis_date"],
            timezone=str(state["analysis_date"].tzinfo),
            ticker=state["ticker"],
            analysis_mode=state["request"].analysis_depth,
            llm_provider=provider_name,
            model_name=model_name,
            planning_method=state["research_plan"].planning_method,
            tools_selected=[step.tool_name for step in state["research_plan"].steps],
            tool_calls=state["tool_calls"],
            source_retrieval_timestamps=source_times,
            source_statuses={item.name: item.status for item in state["data"]},
            llm_calls=1 + state.get("replan_count", 0) + artifacts.llm_calls,
            input_tokens=artifacts.input_tokens,
            output_tokens=artifacts.output_tokens,
            total_runtime_ms=round(elapsed, 3),
            data_completeness=usable / max(len(state["data"]), 1),
            evidence_quality=artifacts.evidence_quality.label.value,
            validation_passed=artifacts.validation.report_complete,
            application_version=__version__,
        )
        return {
            "report_markdown": artifacts.report_markdown,
            "evidence_quality": artifacts.evidence_quality.label.value,
            "evidence_quality_detail": artifacts.evidence_quality,
            "dashboard": artifacts.dashboard,
            "historical_analysis": artifacts.historical_analysis,
            "scorecard": artifacts.scorecard,
            "claims": artifacts.claims,
            "calculations": artifacts.calculations,
            "evidence": evidence,
            "sources": artifacts.sources,
            "data_quality": build_data_quality(state["data"]),
            "validation": artifacts.validation,
            "run_manifest": manifest,
        }

    graph = StateGraph(ResearchState)
    graph.add_node("validate_request", validate_node)
    graph.add_node("create_research_plan", plan_node)
    graph.add_node("execute_selected_tools", execute_node)
    graph.add_node("assess_evidence_gaps", assess_node)
    graph.add_node("revise_plan", replan_node)
    graph.add_node("calculate_and_reconcile", calculate_node)
    graph.add_node("draft_and_verify_report", report_node)
    graph.add_edge(START, "validate_request")
    graph.add_edge("validate_request", "create_research_plan")
    graph.add_edge("create_research_plan", "execute_selected_tools")
    graph.add_edge("execute_selected_tools", "assess_evidence_gaps")
    graph.add_conditional_edges(
        "assess_evidence_gaps",
        route_after_assessment,
        {"replan": "revise_plan", "calculate": "calculate_and_reconcile"},
    )
    graph.add_edge("revise_plan", "execute_selected_tools")
    graph.add_edge("calculate_and_reconcile", "draft_and_verify_report")
    graph.add_edge("draft_and_verify_report", END)
    return graph.compile()


def run_research(
    graph: Any,
    request: ResearchRequest,
    *,
    session_id: str | None = None,
) -> ResearchResult:
    """Run one isolated, bounded research request and return a typed result."""

    active_session_id = session_id or new_session_id()
    state = graph.invoke(
        {"request": request, "session_id": active_session_id},
        config={
            "configurable": {"thread_id": active_session_id},
            "recursion_limit": 18,
        },
    )
    return ResearchResult(
        session_id=active_session_id,
        ticker=state["ticker"],
        analysis_date=state["analysis_date"],
        report_markdown=state["report_markdown"],
        data=state["data"],
        evidence_quality=state["evidence_quality"],
        evidence_quality_detail=state["evidence_quality_detail"],
        dashboard=state["dashboard"],
        historical_analysis=state["historical_analysis"],
        scorecard=state["scorecard"],
        claims=state["claims"],
        calculations=state["calculations"],
        evidence=state["evidence"],
        sources=state["sources"],
        data_quality=state["data_quality"],
        validation=state["validation"],
        research_plan=state["research_plan"],
        reconciliations=state.get("reconciliations", []),
        run_manifest=state["run_manifest"],
    )


def _execute_step(
    step: PlanStep,
    request: ResearchRequest,
    ticker: str,
    mapping: dict[str, BaseTool],
) -> DataResult:
    if step.tool_name == "search_uploaded_documents":
        return _document_result(request)
    if step.tool_name == "earnings_transcript":
        period = _transcript_period(request.query)
        if not period:
            return DataResult.unavailable(
                name="earnings_transcript",
                source="Transcript request validation",
                message="A transcript request requires a quarter and year such as Q2 2025.",
                missing_fields=["year", "quarter"],
                content_type="transcript",
            )
        year, quarter = period
        return mapping[step.tool_name].invoke({"ticker": ticker, "year": year, "quarter": quarter})
    tool = mapping.get(step.tool_name)
    if tool is None:
        return DataResult.unavailable(
            name=step.tool_name,
            source="Research plan validation",
            message="The planner selected a tool outside the executable allowlist.",
        )
    return tool.invoke({"ticker": ticker})


def _run_valuation(
    mapping: dict[str, BaseTool],
    request: ResearchRequest,
    statements: DataResult,
) -> DataResult:
    values = statements.values
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
            "projection_years": request.dcf_projection_years,
        }
    )


def _document_result(request: ResearchRequest) -> DataResult:
    hits = hybrid_retrieve(
        request.documents,
        request.query,
        limit={
            AnalysisDepth.QUICK: 3,
            AnalysisDepth.STANDARD: 6,
            AnalysisDepth.DETAILED: 8,
        }[request.analysis_depth],
    )
    if not hits:
        return DataResult.unavailable(
            name="uploaded_documents",
            source="Page-aware hybrid document retrieval",
            message="No uploaded-document passage exceeded the relevance threshold.",
            content_type="uploaded_document",
        )
    return DataResult(
        name="uploaded_documents",
        status=Availability.AVAILABLE,
        source="Page-aware hybrid semantic and lexical retrieval",
        values={
            "retrieval_method": "local_concept_embedding+bm25",
            "excerpts": [
                {
                    "document_id": hit.document_id,
                    "filename": hit.safe_filename,
                    "page_number": hit.page_number,
                    "chunk_id": hit.chunk_id,
                    "character_start": hit.character_start,
                    "character_end": hit.character_end,
                    "text": hit.text,
                    "lexical_score": hit.lexical_score,
                    "semantic_score": hit.semantic_score,
                    "hybrid_score": hit.hybrid_score,
                    "untrusted_instruction_detected": hit.contains_instruction_like_text,
                }
                for hit in hits
            ],
        },
        evidence=[
            EvidenceRef(
                source=f"Uploaded PDF: {hit.safe_filename}",
                source_type="uploaded_document",
                provider="User upload",
                title=hit.safe_filename,
                page_number=hit.page_number,
                excerpt=hit.text[:500],
                definition="Untrusted user-supplied document evidence",
            )
            for hit in hits
        ],
        content_type="uploaded_document",
    )


def _enforce_mode_contract(
    plan: ResearchPlan,
    request: ResearchRequest,
    tools: Sequence[BaseTool],
) -> ResearchPlan:
    available = {tool.name for tool in tools}
    required = ["market_snapshot", "financial_statements"]
    if request.analysis_depth is not AnalysisDepth.QUICK:
        required.extend(["sec_company_facts", "recent_news"])
    query = request.query.casefold()
    allowed_conditional = {
        "discounted_cash_flow": any(
            term in query for term in ("dcf", "valuation", "intrinsic value", "fair value")
        ),
        "earnings_transcript": any(term in query for term in ("transcript", "earnings call")),
        "search_uploaded_documents": bool(request.documents),
    }
    if any(term in query for term in ("transcript", "earnings call")):
        required.append("earnings_transcript")
    if request.documents and (
        request.analysis_depth is AnalysisDepth.DETAILED
        or any(term in query for term in ("document", "pdf", "filing", "uploaded"))
    ):
        required.append("search_uploaded_documents")
    if any(term in query for term in ("dcf", "valuation", "intrinsic value", "fair value")):
        required.append("discounted_cash_flow")
    by_name = {
        step.tool_name: step
        for step in plan.steps
        if step.tool_name in available and allowed_conditional.get(step.tool_name, True)
    }
    for name in required:
        if name in available and name not in by_name:
            by_name[name] = PlanStep(
                step_id=f"step-policy-{len(by_name) + 1}",
                tool_name=name,
                purpose=f"{request.analysis_depth.value} mode source contract",
                inputs={"ticker": plan.ticker},
                required=True,
            )
    ordered_names = [
        *[name for name in required if name in by_name],
        *[name for name in by_name if name not in required],
    ]
    ordered = [by_name[name] for name in ordered_names[: plan.maximum_tool_budget]]
    return plan.model_copy(
        update={
            "steps": ordered,
            "valuation_requested": any(
                step.tool_name == "discounted_cash_flow" for step in ordered
            ),
            "document_retrieval_requested": any(
                step.tool_name == "search_uploaded_documents" for step in ordered
            ),
            "news_requested": any(step.tool_name == "recent_news" for step in ordered),
            "transcript_requested": any(
                step.tool_name == "earnings_transcript" for step in ordered
            ),
        }
    )


def _merge_plan_steps(existing: ResearchPlan, revised: ResearchPlan) -> ResearchPlan:
    by_name = {step.tool_name: step for step in existing.steps}
    for step in revised.steps:
        by_name.setdefault(step.tool_name, step)
    return revised.model_copy(
        update={
            "steps": list(by_name.values())[: revised.maximum_tool_budget],
            "revision_count": existing.revision_count + 1,
        }
    )


def _evidence_gaps(request: ResearchRequest, data: list[DataResult]) -> list[str]:
    by_name = {item.name: item for item in data}
    gaps = []
    for name in ("market_snapshot", "financial_statements"):
        if not by_name.get(name) or by_name[name].status is Availability.UNAVAILABLE:
            gaps.append(f"Required {name.replace('_', ' ')} is unavailable.")
    if request.analysis_depth is not AnalysisDepth.QUICK:
        sec = by_name.get("sec_company_facts")
        if not sec or sec.status is Availability.UNAVAILABLE:
            gaps.append("Comparable official SEC filing facts are unavailable.")
    if request.documents and request.analysis_depth is AnalysisDepth.DETAILED:
        document = by_name.get("uploaded_documents")
        if not document or document.status is Availability.UNAVAILABLE:
            gaps.append("No relevant uploaded-document evidence was retrieved.")
    return gaps


def _transcript_period(query: str) -> tuple[int, int] | None:
    match = re.search(r"\bQ([1-4])\s*[-/]?\s*(20\d{2})\b", query, re.IGNORECASE)
    if match:
        return int(match.group(2)), int(match.group(1))
    match = re.search(r"\b(20\d{2})\s*[-/]?\s*Q([1-4])\b", query, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _document_planning_tool() -> BaseTool:
    def search_uploaded_documents(query: str) -> str:
        return query

    return StructuredTool.from_function(
        func=search_uploaded_documents,
        name="search_uploaded_documents",
        description=(
            "Select page-aware hybrid semantic and lexical evidence from uploaded PDFs. "
            "Planning only; deterministic execution receives in-memory documents."
        ),
    )


def _result(data: list[DataResult], name: str) -> DataResult | None:
    return next((item for item in data if item.name == name), None)


def _planned(plan: ResearchPlan, tool_name: str) -> bool:
    return any(step.tool_name == tool_name for step in plan.steps)


def _step_id(plan: ResearchPlan, tool_name: str) -> str:
    return next(
        (step.step_id for step in plan.steps if step.tool_name == tool_name),
        f"step-{tool_name}",
    )
