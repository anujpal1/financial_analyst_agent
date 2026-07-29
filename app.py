"""Streamlit entry point for the Evidence-Grounded Financial Research Workbench."""

from __future__ import annotations

import hashlib
import html
from collections.abc import Iterable
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import SecretStr, ValidationError

from financial_analyst.config import (
    DEFAULT_MODELS,
    AppSettings,
    LLMProvider,
    ProviderConfig,
    environment_api_key,
)
from financial_analyst.documents import DocumentValidationError, extract_pdf_upload
from financial_analyst.llm import (
    ProviderConfigurationError,
    create_chat_model,
    test_connection,
    validate_tool_calling,
)
from financial_analyst.models import (
    AnalysisDepth,
    Availability,
    DashboardMetric,
    ResearchRequest,
)
from financial_analyst.security import configure_logging, new_session_id, safe_error_message
from financial_analyst.tools import build_tool_registry
from financial_analyst.workflow import build_research_graph, run_research

st.set_page_config(
    page_title="Financial Research Workbench",
    page_icon="FR",
    layout="wide",
    initial_sidebar_state="expanded",
)

settings = AppSettings()
logger = configure_logging(settings.cache_directory / "logs")

_STATUS_COLORS = {
    Availability.AVAILABLE: "#2f6b55",
    Availability.PARTIAL: "#9a6b19",
    Availability.UNAVAILABLE: "#8b3e3e",
    Availability.STALE: "#9a6b19",
    Availability.CONFLICT: "#8b3e3e",
    Availability.INVALID: "#8b3e3e",
}


def _initialize_session() -> None:
    defaults: dict[str, Any] = {
        "session_id": new_session_id(),
        "report": "",
        "result": None,
        "provider_status": "Not tested",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_session() -> None:
    for key in list(st.session_state):
        del st.session_state[key]
    st.rerun()


def _extract_pdf_for_session(filename: str, content: bytes, max_size_mb: int) -> Any:
    """Reuse parsed pages only within this session and keep the cache small."""

    cache: dict[str, Any] = st.session_state.setdefault("parsed_pdf_cache", {})
    digest = hashlib.sha256(content).hexdigest()
    key = f"{digest}:{max_size_mb}:{filename}"
    if key not in cache:
        if len(cache) >= 4:
            oldest_key = next(iter(cache))
            del cache[oldest_key]
        cache[key] = extract_pdf_upload(
            filename,
            content,
            max_size_mb=max_size_mb,
        )
    return cache[key]


def _provider_configuration(
    *,
    provider: LLMProvider,
    model_name: str,
    api_key_text: str,
    temperature: float,
    ollama_base_url: str,
) -> ProviderConfig:
    ui_key = SecretStr(api_key_text.strip()) if api_key_text.strip() else None
    selected_key = ui_key or environment_api_key(settings, provider)
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        api_key=selected_key,
        temperature=temperature,
        ollama_base_url=ollama_base_url,
        request_timeout_seconds=settings.request_timeout_seconds,
    )


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --paper: #f7f5f0;
            --surface: #fffefa;
            --slate: #26323d;
            --muted: #65717c;
            --navy: #294b68;
            --border: #d9dde0;
            --positive: #2f6b55;
            --negative: #8b3e3e;
            --amber: #9a6b19;
        }
        .stApp { background: var(--paper); color: var(--slate); }
        [data-testid="stSidebar"] { background: #eef0ef; border-right: 1px solid var(--border); }
        .block-container { padding-top: 1.35rem; padding-bottom: 3rem; max-width: 1500px; }
        .workbench-header {
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 4px solid var(--navy);
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(38, 50, 61, 0.05);
        }
        .workbench-header h1 { font-size: 1.45rem; margin: 0; color: var(--slate); }
        .workbench-header p { color: var(--muted); margin: .35rem 0 .7rem 0; }
        .header-meta { display: flex; flex-wrap: wrap; gap: .5rem 1.4rem; font-size: .82rem; }
        .header-meta strong { color: var(--navy); }
        .disclaimer-line { color: var(--muted); font-size: .76rem; margin-top: .65rem; }
        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: .75rem .85rem;
            box-shadow: 0 2px 7px rgba(38, 50, 61, 0.04);
            min-height: 116px;
        }
        div[data-testid="stMetric"] label { color: var(--muted); }
        div[data-testid="stMetricValue"] { color: var(--slate); font-size: 1.35rem; }
        .metric-caption { color: var(--muted); font-size: .72rem; margin-top: -.55rem; }
        .status-pill {
            display: inline-block; border: 1px solid var(--border); background: var(--surface);
            padding: .2rem .55rem; border-radius: 999px; font-size: .75rem;
        }
        h2, h3 { color: var(--slate); letter-spacing: -.01em; }
        .stButton > button[kind="primary"] { background: var(--navy); border-color: var(--navy); }
        [data-baseweb="tab-list"] { gap: .35rem; }
        [data-baseweb="tab"] { background: transparent; border-radius: 4px 4px 0 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(provider: LLMProvider, model_name: str) -> None:
    result = st.session_state.result
    timestamp = (
        result.analysis_date.strftime("%Y-%m-%d %H:%M UTC") if result else "Awaiting analysis"
    )
    status = st.session_state.provider_status
    provider_text = html.escape(provider.value)
    model_text = html.escape(model_name)
    status_text = html.escape(status)
    timestamp_text = html.escape(timestamp)
    st.markdown(
        f"""
        <div class="workbench-header">
          <h1>Evidence-Grounded Financial Research Workbench</h1>
          <p>Structured company research with deterministic analytics, valuation,
             claim-level evidence, and constrained qualitative synthesis.</p>
          <div class="header-meta">
            <span><strong>Provider</strong> {provider_text} / {model_text}</span>
            <span><strong>Status</strong> {status_text}</span>
            <span><strong>Data timestamp</strong> {timestamp_text}</span>
          </div>
          <div class="disclaimer-line">Informational and educational research only;
          not financial advice or a recommendation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dashboard(metrics: list[DashboardMetric]) -> None:
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
                detail = " · ".join(
                    part
                    for part in (
                        f"Period: {metric.period}" if metric.period else None,
                        f"Source: {metric.source}" if metric.source else None,
                    )
                    if part
                )
                if metric.detail:
                    detail = f"{detail} · {metric.detail}" if detail else metric.detail
                caption = detail or "Deterministic structured value"
                st.caption(caption)


def _render_data_quality(rows: Iterable[Any]) -> None:
    table = []
    for row in rows:
        table.append(
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
        )
    if table:
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No data-quality records are available for this run.")


def _render_market_chart(result: Any) -> None:
    market = next((item for item in result.data if item.name == "market_snapshot"), None)
    points = market.values.get("history", []) if market else []
    if not points:
        st.info("Canonical market-price history is unavailable for this run.")
        return
    frame = pd.DataFrame(points)
    frame["date"] = pd.to_datetime(frame["date"])
    st.line_chart(frame, x="date", y="close", color="#294b68")
    st.caption(
        "Six-month daily closes from the same canonical yfinance market object used "
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


def _render_financial_charts(result: Any) -> None:
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


def _render_scorecard(result: Any) -> None:
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
        else "Not scored — insufficient data"
    )
    st.info(f"Overall research score: {overall}. {scorecard.overall_explanation}")


def _render_valuation(result: Any) -> None:
    dcf = next((item for item in result.data if item.name == "discounted_cash_flow"), None)
    if not dcf:
        st.info("DCF was not requested in this analysis.")
        return
    if dcf.status is Availability.UNAVAILABLE:
        st.warning(dcf.message or "DCF is unavailable.")
        return
    values = dcf.values
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
                {"Input": "Base period", "Value": values.get("period_end")},
                {"Input": "Projection years", "Value": values.get("projection_years")},
                {"Input": "Growth rate", "Value": inputs.get("growth_rate")},
                {"Input": "Discount rate", "Value": inputs.get("discount_rate")},
                {"Input": "Terminal growth", "Value": inputs.get("terminal_growth_rate")},
                {"Input": "Cash", "Value": inputs.get("cash")},
                {"Input": "Debt", "Value": inputs.get("debt")},
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
        st.markdown("### DCF sensitivity — implied value per share")
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


def _render_evidence(result: Any) -> None:
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


def _render_sources(result: Any) -> None:
    if result.sources:
        st.dataframe(
            [item.model_dump(mode="json") for item in result.sources],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No source records are available.")
    st.markdown("### Data quality")
    _render_data_quality(result.data_quality)


_initialize_session()
_inject_theme()

with st.sidebar:
    st.subheader("Workbench controls")
    st.caption(f"Session {st.session_state.session_id[:8]}")

    with st.expander("1. LLM configuration", expanded=True):
        provider = LLMProvider(st.selectbox("LLM provider", [item.value for item in LLMProvider]))
        suggested_models = DEFAULT_MODELS[provider]
        selected_model = st.selectbox("Model", suggested_models)
        use_custom_model = st.checkbox("Use a custom model name")
        custom_model = (
            st.text_input("Custom model", placeholder="provider-model-name")
            if use_custom_model
            else ""
        )
        model_name = custom_model.strip() or selected_model
        api_key_text = ""
        if provider is not LLMProvider.OLLAMA:
            api_key_text = st.text_input(
                f"{provider.value} API key",
                type="password",
                help="Held only in this Streamlit session; never written by the app.",
                key=f"session_key_{provider.value}",
            )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
        )
        ollama_base_url = settings.ollama_base_url
        if provider is LLMProvider.OLLAMA:
            ollama_base_url = st.text_input(
                "Ollama base URL",
                value=settings.ollama_base_url,
            )
        if (provider is LLMProvider.GEMINI and model_name.lower().startswith("gemini-3")) or (
            provider is LLMProvider.OPENAI
            and model_name.lower().startswith(("gpt-5", "o1", "o3", "o4"))
        ):
            st.caption("This model family ignores temperature; it is omitted from the request.")

    try:
        provider_config = _provider_configuration(
            provider=provider,
            model_name=model_name,
            api_key_text=api_key_text,
            temperature=temperature,
            ollama_base_url=ollama_base_url,
        )
        configuration_error = None
    except ValidationError as error:
        provider_config = None
        configuration_error = error.errors()[0]["msg"]
    cloud_key_missing = provider is not LLMProvider.OLLAMA and (
        provider_config is None or provider_config.api_key is None
    )

    provider_signature = (
        provider.value,
        model_name,
        temperature,
        ollama_base_url,
        bool(api_key_text.strip()),
    )
    if st.session_state.get("provider_signature") != provider_signature:
        st.session_state.provider_signature = provider_signature
        st.session_state.provider_status = "Not tested"

    if configuration_error:
        st.error(configuration_error)
    elif cloud_key_missing:
        st.warning("Enter a cloud API key before testing or running analysis.")
    elif st.button("Test connection", use_container_width=True):
        try:
            model = create_chat_model(provider_config)
            validate_tool_calling(model, build_tool_registry(settings))
            secrets = (
                [provider_config.api_key.get_secret_value()]
                if provider_config.api_key is not None
                else []
            )
            st.session_state.provider_status = test_connection(model, secrets=secrets)
            st.success(st.session_state.provider_status)
        except ProviderConfigurationError as error:
            st.session_state.provider_status = str(error)
            st.error(st.session_state.provider_status)
    st.caption(f"Connection status: {st.session_state.provider_status}")

    with st.expander("2. Optional data sources"):
        sec_user_agent = st.text_input(
            "SEC User-Agent",
            value=settings.sec_user_agent or "",
            placeholder="FinancialResearch/1.0 you@example.com",
            help="Application name plus contact email; required for SEC access.",
        )
        fmp_key_text = st.text_input(
            "FMP API key (optional transcripts)",
            type="password",
            key="session_fmp_api_key",
        )
        st.caption("Core market and annual statement data use yfinance.")

    with st.expander("3. Analysis settings"):
        st.caption(
            "Quick skips news. Standard includes relevant news. Detailed uses the same "
            "verified sources with the fullest report."
        )
        st.caption("Valuation runs only when the question asks for it.")

    with st.expander("4. Session controls"):
        st.write("Reset clears results, uploads, UI-entered keys, and per-run data clients.")
        if st.button("Clear / reset session", use_container_width=True):
            _clear_session()

_render_header(provider, model_name)

input_left, input_right = st.columns([3, 1])
with input_left:
    query = st.text_area(
        "Research question",
        placeholder=(
            "Example: Analyze MSFT's annual financial direction, cash generation, "
            "material risks, and a DCF valuation."
        ),
        height=105,
    )
with input_right:
    ticker = st.text_input("Ticker (optional)")
    analysis_depth = AnalysisDepth(
        st.selectbox(
            "Analysis depth",
            [item.value for item in AnalysisDepth],
            index=1,
        )
    )
    uploaded_pdf = st.file_uploader("Supporting PDF (optional)", type=["pdf"])

with st.expander("DCF assumptions"):
    st.caption("Used only when the research question requests valuation.")
    assumption_columns = st.columns(3)
    with assumption_columns[0]:
        dcf_growth_rate = st.number_input(
            "Base annual FCF growth",
            min_value=-0.50,
            max_value=0.50,
            value=0.05,
            step=0.005,
            format="%.3f",
        )
    with assumption_columns[1]:
        dcf_discount_rate = st.number_input(
            "Base discount rate",
            min_value=0.01,
            max_value=0.50,
            value=0.10,
            step=0.005,
            format="%.3f",
        )
    with assumption_columns[2]:
        dcf_terminal_growth_rate = st.number_input(
            "Base terminal growth",
            min_value=-0.10,
            max_value=0.10,
            value=0.025,
            step=0.005,
            format="%.3f",
        )

analysis_disabled = bool(configuration_error or cloud_key_missing)
run_label = "Refresh analysis" if st.session_state.result else "Run analysis"
run_clicked = st.button(
    run_label,
    type="primary",
    disabled=analysis_disabled,
    use_container_width=True,
)

if run_clicked:
    st.session_state.report = ""
    st.session_state.result = None
    if not query.strip():
        st.error("Enter a research question.")
    else:
        try:
            runtime_settings = settings.model_copy(
                update={
                    "sec_user_agent": sec_user_agent.strip() or None,
                    "fmp_api_key": (
                        SecretStr(fmp_key_text.strip())
                        if fmp_key_text.strip()
                        else settings.fmp_api_key
                    ),
                }
            )
            documents = []
            if uploaded_pdf is not None:
                documents.append(
                    _extract_pdf_for_session(
                        uploaded_pdf.name,
                        uploaded_pdf.getvalue(),
                        runtime_settings.upload_size_limit_mb,
                    )
                )
            request = ResearchRequest(
                query=query.strip(),
                ticker=ticker.strip() or None,
                documents=documents,
                dcf_growth_rate=dcf_growth_rate,
                dcf_discount_rate=dcf_discount_rate,
                dcf_terminal_growth_rate=dcf_terminal_growth_rate,
                analysis_depth=analysis_depth,
            )
            with st.status("Running evidence-grounded analysis...", expanded=True) as status:
                st.write("Validating provider and ticker")
                model = create_chat_model(provider_config)
                tools = build_tool_registry(runtime_settings)
                st.write("Collecting and normalizing source data")
                graph = build_research_graph(
                    llm=model,
                    settings=runtime_settings,
                    tools=tools,
                )
                result = run_research(
                    graph,
                    request,
                    session_id=st.session_state.session_id,
                )
                st.write("Calculating analytics and validating claims")
                status.update(label="Analysis complete", state="complete")
            st.session_state.result = result
            st.session_state.report = result.report_markdown
        except (
            DocumentValidationError,
            ProviderConfigurationError,
            ValidationError,
            ValueError,
        ) as error:
            logger.info("Analysis validation failure: %s", type(error).__name__)
            st.error(safe_error_message(error, context="Analysis could not run"))
        except Exception as error:
            logger.exception("Unexpected analysis failure: %s", type(error).__name__)
            st.error(
                "Analysis failed unexpectedly. Review the local application log for "
                "technical details; credentials and document text are not logged."
            )

if st.session_state.result:
    result = st.session_state.result
    if result.validation and not result.validation.report_complete:
        st.error(
            "Report blocked by consistency validation. A clearly labelled partial report "
            "is available for inspection."
        )
    st.markdown("## Executive research dashboard")
    if result.dashboard:
        _render_dashboard(result.dashboard.metrics)
    else:
        st.info("Dashboard metrics are unavailable for this run.")

    (
        overview_tab,
        financials_tab,
        valuation_tab,
        evidence_tab,
        sources_tab,
        report_tab,
    ) = st.tabs(
        [
            "Overview",
            "Financials",
            "Valuation",
            "Evidence",
            "Sources",
            "Full report",
        ]
    )
    with overview_tab:
        st.markdown("### Canonical market history")
        _render_market_chart(result)
        st.markdown("### Data quality")
        _render_data_quality(result.data_quality)
        if result.evidence_quality_detail:
            quality = result.evidence_quality_detail
            st.caption(
                f"Evidence quality: {quality.label.value}; deterministic source coverage "
                f"{quality.coverage_score:.1f}/100. {quality.freshness_summary}"
            )
    with financials_tab:
        if result.historical_analysis and result.historical_analysis.observations:
            st.markdown("### Deterministic observations")
            for observation in result.historical_analysis.observations:
                st.write(f"- {observation}")
        _render_financial_charts(result)
        st.markdown("### Financial scorecard")
        _render_scorecard(result)
    with valuation_tab:
        _render_valuation(result)
    with evidence_tab:
        _render_evidence(result)
    with sources_tab:
        _render_sources(result)
    with report_tab:
        st.markdown(st.session_state.report)
        st.download_button(
            "Download Markdown report",
            data=st.session_state.report.encode("utf-8"),
            file_name=f"{result.ticker}_financial_research.md",
            mime="text/markdown",
        )
