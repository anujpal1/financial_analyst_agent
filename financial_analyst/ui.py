"""Streamlit result presentation for the financial research workbench."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import streamlit as st

from financial_analyst.models import Availability, DashboardMetric


def inject_theme() -> None:
    """Apply a restrained, high-contrast light theme across Streamlit widgets."""

    st.markdown(
        """
        <style>
        :root {
          --page:#f7f8fa; --sidebar:#f1f4f6; --surface:#fff; --text:#1f2933;
          --muted:#5f6c78; --navy:#315a78; --navy-hover:#274a65;
          --border:#d9e0e5; --amber-bg:#fff8e7; --amber-text:#704f14;
        }
        html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"] {
          background:var(--page)!important; color:var(--text)!important;
        }
        [data-testid="stSidebar"] {
          background:var(--sidebar)!important; border-right:1px solid var(--border);
        }
        [data-testid="stSidebarContent"] { padding-top:1rem; }
        .block-container { padding:1rem 1rem 3rem; max-width:1320px; }
        .stMarkdown,.stMarkdown p,.stMarkdown li,[data-testid="stText"],
        [data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] p,
        h1,h2,h3,h4,h5,h6 { color:var(--text)!important; }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p { color:var(--muted)!important; }
        h1,h2,h3,h4,h5,h6 { letter-spacing:-.015em; }
        .workbench-header {
          background:var(--surface); border:1px solid var(--border); border-radius:8px;
          padding:1.15rem 1.3rem; margin-bottom:1.1rem;
        }
        .workbench-header h1 {
          font-size:1.4rem; font-weight:650; line-height:1.25; margin:0;
        }
        .workbench-header p { color:var(--muted)!important; margin:.4rem 0 .8rem; }
        .header-meta {
          color:var(--text)!important; display:flex; flex-wrap:wrap;
          gap:.45rem 1.4rem; font-size:.82rem;
        }
        .header-meta span { color:var(--text)!important; }
        .header-meta strong { color:var(--navy)!important; }
        .disclaimer-line { color:var(--muted)!important; font-size:.76rem; margin-top:.65rem; }
        .stTextInput input,.stTextArea textarea,.stNumberInput input,
        [data-baseweb="select"]>div {
          background:var(--surface)!important; border-color:var(--border)!important;
          color:var(--text)!important; -webkit-text-fill-color:var(--text)!important;
        }
        .stTextInput input::placeholder,.stTextArea textarea::placeholder {
          color:#87929c!important; opacity:1;
        }
        [data-baseweb="select"] span,[data-baseweb="select"] svg,
        [data-testid="stExpander"] summary,[data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary svg {
          color:var(--text)!important; fill:var(--text)!important;
        }
        [data-baseweb="popover"],[role="listbox"],[role="option"] {
          background:var(--surface)!important; color:var(--text)!important;
        }
        [role="option"]:hover { background:#eaf0f4!important; }
        [data-testid="stFileUploaderDropzone"] {
          background:var(--surface)!important; border:1px dashed #b9c4cc!important;
        }
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small { color:var(--muted)!important; }
        [data-testid="stFileUploaderDropzone"] button {
          background:var(--surface)!important; border:1px solid var(--border)!important;
          color:var(--navy)!important;
        }
        [data-testid="stExpander"] {
          background:var(--surface)!important; border:1px solid var(--border)!important;
          border-radius:7px;
        }
        [data-testid="stAlert"] {
          border:1px solid #ead9a5!important; background:var(--amber-bg)!important;
          color:var(--amber-text)!important;
        }
        [data-testid="stAlert"] p,[data-testid="stAlert"] svg {
          color:var(--amber-text)!important; fill:var(--amber-text)!important;
        }
        .stButton>button,.stDownloadButton>button { border-radius:6px; font-weight:600; }
        .stButton>button[kind="primary"] {
          background:var(--navy)!important; border-color:var(--navy)!important;
          color:#fff!important;
        }
        .stButton>button[kind="primary"]:hover {
          background:var(--navy-hover)!important; border-color:var(--navy-hover)!important;
        }
        .stButton>button:disabled {
          background:#dbe3e8!important; border-color:#d2dbe1!important;
          color:#65737f!important; opacity:1!important;
        }
        .stCheckbox label span,.stSlider label,
        .stSlider [data-testid="stThumbValue"] { color:var(--text)!important; }
        div[data-testid="stMetric"] {
          background:var(--surface); border:1px solid var(--border); border-radius:6px;
          padding:.75rem .85rem; min-height:116px;
        }
        div[data-testid="stMetric"] label { color:var(--muted); }
        div[data-testid="stMetricValue"] { color:var(--text)!important; font-size:1.35rem; }
        [data-baseweb="tab-list"] { gap:.35rem; }
        [data-baseweb="tab"] { background:transparent; color:var(--muted)!important; }
        [data-baseweb="tab"][aria-selected="true"] { color:var(--navy)!important; }
        hr { border-color:var(--border)!important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(metrics: list[DashboardMetric]) -> None:
    """Render the most decision-useful deterministic metrics first."""

    preferred = [
        "price",
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
        "dcf_range",
        "upside",
        "evidence",
        "completeness",
    ]
    by_key = {metric.key: metric for metric in metrics}
    ordered = [by_key[key] for key in preferred if key in by_key]
    for start in range(0, len(ordered), 4):
        columns = st.columns(4)
        for column, metric in zip(columns, ordered[start : start + 4], strict=False):
            with column:
                st.metric(metric.label, metric.formatted_value)
                detail = " | ".join(
                    part
                    for part in (
                        f"Period: {metric.period}" if metric.period else None,
                        f"Source: {metric.source}" if metric.source else None,
                    )
                    if part
                )
                if metric.detail:
                    detail = f"{detail} | {metric.detail}" if detail else metric.detail
                st.caption(detail or "Deterministic structured value")


def render_data_quality(rows: Iterable[Any]) -> None:
    """Render source status and freshness without hiding partial results."""

    table = [
        {
            "Dataset": row.dataset,
            "Status": row.status.value,
            "Source": row.source,
            "Period": row.period or "Unavailable",
            "Retrieved": (
                row.retrieved_at.strftime("%Y-%m-%d %H:%M UTC")
                if row.retrieved_at
                else "Unavailable"
            ),
            "Warning": row.warning or "",
        }
        for row in rows
    ]
    if table:
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No data-quality records are available for this run.")


def render_market_chart(result: Any) -> None:
    """Render price history from the same canonical object used in comparisons."""

    market = next((item for item in result.data if item.name == "market_snapshot"), None)
    points = market.values.get("history", []) if market else []
    if not points:
        st.info("Canonical market-price history is unavailable for this run.")
        return
    frame = pd.DataFrame(points)
    frame["date"] = pd.to_datetime(frame["date"])
    st.line_chart(frame, x="date", y="close", color="#294b68")
    st.caption(
        "Six-month daily closes from the canonical yfinance market object used "
        "for price comparisons."
    )


def _metric_frame(result: Any, names: tuple[str, ...]) -> pd.DataFrame:
    history = result.historical_analysis
    if not history:
        return pd.DataFrame()
    selected = {metric.name: metric for metric in history.metrics if metric.name in names}
    periods = sorted({item["period"] for metric in selected.values() for item in metric.values})
    rows = []
    for period in periods:
        row: dict[str, Any] = {"Annual period": period}
        for name, metric in selected.items():
            row[name] = next(
                (item["value"] for item in metric.values if item["period"] == period),
                None,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def render_financial_charts(result: Any) -> None:
    """Render compact annual trend views from reconciled facts."""

    chart_specs = (
        ("Revenue and net-income trend", ("Revenue", "Net income")),
        (
            "Operating cash flow and free-cash-flow trend",
            ("Operating cash flow", "Free cash flow"),
        ),
        ("Cash versus debt trend", ("Cash", "Debt")),
        (
            "Revenue growth and margin trend",
            ("Revenue growth", "Net margin", "Free-cash-flow margin"),
        ),
    )
    for title, names in chart_specs:
        frame = _metric_frame(result, names)
        if frame.empty or len(frame.columns) < 2:
            continue
        st.markdown(f"#### {title}")
        st.line_chart(frame, x="Annual period", y=list(frame.columns[1:]))
    if not result.historical_analysis or not result.historical_analysis.periods:
        st.info(
            "Historical annual charts are unavailable because annual periods were not retrieved."
        )


def render_scorecard(result: Any) -> None:
    """Render the transparent heuristic score and every contribution trace."""

    scorecard = result.scorecard
    if not scorecard:
        st.info("The scorecard is unavailable.")
        return
    columns = st.columns(3)
    for index, component in enumerate(scorecard.components):
        with columns[index % 3]:
            st.markdown(f"#### {component.name}")
            if component.score is None:
                st.warning(component.explanation)
            else:
                st.metric("Component score", f"{component.score:.1f} / 100")
                st.progress(component.score / 100)
            if component.missing_metrics:
                st.caption(f"Missing: {', '.join(component.missing_metrics)}")
            with st.expander("Scoring trace"):
                st.caption(component.explanation)
                if component.contributions:
                    st.dataframe(
                        [item.model_dump(mode="json") for item in component.contributions],
                        use_container_width=True,
                        hide_index=True,
                    )
    overall = (
        f"{scorecard.overall_score:.1f} / 100"
        if scorecard.overall_score is not None
        else "Not scored - insufficient data"
    )
    st.info(f"Overall research score: {overall}. {scorecard.overall_explanation}")


def render_valuation(result: Any) -> None:
    """Render FCFE inputs, scenarios, sensitivity, and limitations."""

    dcf = next((item for item in result.data if item.name == "discounted_cash_flow"), None)
    if not dcf:
        st.info("DCF was not requested in this analysis.")
        return
    if dcf.status is Availability.UNAVAILABLE:
        st.warning(dcf.message or "DCF is unavailable.")
        return
    values = dcf.values
    st.caption(
        f"Method: {values.get('method', 'Unavailable')} | "
        f"{values.get('cash_flow_definition', 'Cash-flow definition unavailable')}"
    )
    inputs = values.get("inputs", {})
    market = next((item for item in result.data if item.name == "market_snapshot"), None)
    price = (
        market.values.get("price")
        if market and market.status in {Availability.AVAILABLE, Availability.PARTIAL}
        else None
    )
    per_share = {
        item.get("name"): item.get("per_share_value")
        for item in values.get("scenarios", [])
        if item.get("per_share_value") is not None
    }
    summary_columns = st.columns(4)
    summary_columns[0].metric(
        "Latest price", f"{price:,.2f}" if price is not None else "Unavailable"
    )
    for column, name in zip(summary_columns[1:], ("Bear", "Base", "Bull"), strict=False):
        value = per_share.get(name)
        column.metric(f"{name} value", f"{value:,.2f}" if value is not None else "Unavailable")
    base = per_share.get("Base")
    if price is None:
        st.caption("Comparison unavailable because the canonical market price was not retrieved.")
    elif base is None:
        st.caption(
            "Comparison unavailable because the base-case per-share value was not calculated."
        )
    elif price < min(per_share.values()):
        st.caption("The latest price appears below the modelled range under these assumptions.")
    elif price > max(per_share.values()):
        st.caption("The latest price appears above the modelled range under these assumptions.")
    else:
        st.caption("The latest price appears within the modelled range under these assumptions.")
    left, right = st.columns([1, 2])
    with left:
        st.markdown("### Model inputs")
        st.dataframe(
            [
                {"Input": "Base free cash flow", "Value": values.get("base_free_cash_flow")},
                {"Input": "Method", "Value": values.get("method")},
                {"Input": "Base period", "Value": values.get("period_end")},
                {"Input": "Projection years", "Value": values.get("projection_years")},
                {"Input": "Growth rate", "Value": inputs.get("growth_rate")},
                {"Input": "Discount rate", "Value": inputs.get("discount_rate")},
                {"Input": "Terminal growth", "Value": inputs.get("terminal_growth_rate")},
                {"Input": "Cash (context only)", "Value": inputs.get("cash")},
                {"Input": "Debt (context only)", "Value": inputs.get("debt")},
                {"Input": "Diluted shares", "Value": inputs.get("diluted_shares")},
                {"Input": "Currency", "Value": values.get("currency")},
            ],
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("### Bear, base, and bull scenarios")
        st.dataframe(values.get("scenarios", []), use_container_width=True, hide_index=True)
    sensitivity = values.get("sensitivity")
    if sensitivity:
        st.markdown("### DCF sensitivity - implied value per share")
        frame = pd.DataFrame(
            sensitivity["values"],
            index=[f"{value:.1%}" for value in sensitivity["terminal_growth_rates"]],
            columns=[f"{value:.1%}" for value in sensitivity["discount_rates"]],
        )
        frame.index.name = "Terminal growth / discount rate"
        st.dataframe(frame.style.format("{:,.2f}", na_rep="Invalid"), use_container_width=True)
        st.caption(
            "Invalid combinations are blank because terminal growth must remain below "
            "the discount rate. DCF outcomes are assumption-sensitive."
        )
    for warning in values.get("warnings", []):
        st.warning(warning)


def render_evidence(result: Any) -> None:
    """Render report validation and claim-level support records."""

    if result.validation:
        validation = result.validation
        if validation.blocking_errors:
            st.error("Report blocked by consistency validation; a partial report is shown.")
        elif validation.warnings:
            st.warning("Report passed with validation warnings.")
        else:
            st.success("Consistency validation passed.")
        with st.expander("Validation details"):
            st.write("Passed checks")
            st.write(validation.passed_checks)
            if validation.warnings:
                st.write("Warnings")
                st.json([item.model_dump(mode="json") for item in validation.warnings])
            if validation.blocking_errors:
                st.write("Blocking errors")
                st.json([item.model_dump(mode="json") for item in validation.blocking_errors])
            st.caption(f"Controlled regeneration attempted: {validation.regeneration_attempted}")
    evidence_by_id = {item.evidence_id: item for item in result.evidence}
    if not result.claims:
        st.info("No structured factual claims were produced.")
        return
    for claim in result.claims:
        label = f"{claim.support_status.value}: {claim.text}"
        with st.expander(label):
            st.write(f"Category: {claim.category}")
            st.write(f"Confidence category: {claim.confidence_category.value}")
            st.write(f"Calculation: {claim.calculation_id or 'Not applicable'}")
            st.write(f"Verification: {claim.verification_reason or 'Not recorded'}")
            if claim.conflict_status:
                st.error("This claim has conflicting evidence.")
            refs = [evidence_by_id[item] for item in claim.evidence_ids if item in evidence_by_id]
            if refs:
                st.dataframe(
                    [
                        {
                            "Evidence ID": item.evidence_id,
                            "Source": item.source,
                            "Metric": item.metric,
                            "Value": item.value,
                            "Unit": item.unit,
                            "Period": item.period_end,
                            "Form": item.form,
                            "Page": item.page_number,
                            "URL": item.url,
                        }
                        for item in refs
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning("No direct evidence record is attached; inspect the calculation trace.")


def render_sources(result: Any) -> None:
    """Render deduplicated source records and data-quality states."""

    if result.sources:
        st.dataframe(
            [item.model_dump(mode="json") for item in result.sources],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No source records are available.")
    st.markdown("### Data quality")
    render_data_quality(result.data_quality)


def render_research_plan(result: Any) -> None:
    """Render safe plan outcomes, reconciliation, and compact provenance."""

    plan = result.research_plan
    if not plan:
        st.info("No research-plan trace is available.")
        return
    st.caption(
        f"Planning path: {plan.planning_method} | Tool budget: "
        f"{plan.maximum_tool_budget} | Revisions: {plan.revision_count}"
    )
    st.dataframe(
        [
            {
                "Step": step.step_id,
                "Tool": step.tool_name,
                "Purpose": step.purpose,
                "Required": step.required,
                "Status": step.status,
                "Latency ms": step.latency_ms,
                "Outcome": step.outcome,
            }
            for step in plan.steps
        ],
        use_container_width=True,
        hide_index=True,
    )
    if plan.gaps:
        st.warning("Evidence gaps: " + "; ".join(plan.gaps))
    if result.reconciliations:
        st.markdown("### Canonical source reconciliation")
        st.dataframe(
            [item.model_dump(mode="json") for item in result.reconciliations],
            use_container_width=True,
            hide_index=True,
        )
    manifest = result.run_manifest
    if manifest:
        st.markdown("### Run summary")
        columns = st.columns(4)
        columns[0].metric("Runtime", f"{manifest.total_runtime_ms / 1000:.2f}s")
        columns[1].metric("LLM calls", str(manifest.llm_calls))
        columns[2].metric("Tool calls", str(len(manifest.tool_calls)))
        columns[3].metric("Completeness", f"{manifest.data_completeness:.0%}")
        st.caption(
            f"Run {manifest.run_id[:12]} | {manifest.llm_provider} / "
            f"{manifest.model_name} | Validation "
            f"{'passed' if manifest.validation_passed else 'blocked'}"
        )
