"""Streamlit entry point for the agentic financial research workbench."""

from __future__ import annotations

import hashlib
import html
import json
from typing import Any

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
)
from financial_analyst.models import (
    AnalysisDepth,
    ResearchRequest,
)
from financial_analyst.security import configure_logging, new_session_id, safe_error_message
from financial_analyst.tools import build_tool_registry
from financial_analyst.ui import (
    inject_theme,
    render_dashboard,
    render_data_quality,
    render_evidence,
    render_financial_charts,
    render_market_chart,
    render_research_plan,
    render_scorecard,
    render_sources,
    render_valuation,
)
from financial_analyst.workflow import build_research_graph, run_research

st.set_page_config(
    page_title="Agentic Financial Research Workbench",
    page_icon="FR",
    layout="wide",
    initial_sidebar_state="expanded",
)

settings = AppSettings()
logger = configure_logging(settings.cache_directory / "logs")


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
          <h1>Agentic Evidence-Grounded Financial Research Workbench</h1>
          <p>Model-directed research planning, canonical financial reconciliation,
             deterministic FCFE valuation, and verified evidence.</p>
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


_initialize_session()
inject_theme()

with st.sidebar:
    st.subheader("Research settings")
    st.caption(f"Session · {st.session_state.session_id[:8]}")

    with st.expander("LLM configuration", expanded=True):
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

    with st.expander("Optional data sources"):
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

    with st.expander("Analysis settings"):
        st.caption(
            "Quick uses core market and annual facts. Standard adds SEC reconciliation and "
            "news. Detailed adds bounded evidence-gap replanning and hybrid document evidence."
        )
        st.caption("Valuation runs only when the question asks for it.")

    with st.expander("Session controls"):
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
    assumption_columns = st.columns(4)
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
    with assumption_columns[3]:
        dcf_projection_years = st.number_input(
            "Projection years",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
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
                dcf_projection_years=int(dcf_projection_years),
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
                    provider_name=provider.value,
                    model_name=model_name,
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
        render_dashboard(result.dashboard.metrics)
    else:
        st.info("Dashboard metrics are unavailable for this run.")

    (
        overview_tab,
        financials_tab,
        valuation_tab,
        plan_tab,
        evidence_tab,
        sources_tab,
        report_tab,
    ) = st.tabs(
        [
            "Overview",
            "Financials",
            "Valuation",
            "Research plan",
            "Evidence",
            "Sources",
            "Full report",
        ]
    )
    with overview_tab:
        st.markdown("### Canonical market history")
        render_market_chart(result)
        st.markdown("### Data quality")
        render_data_quality(result.data_quality)
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
        render_financial_charts(result)
        st.markdown("### Financial scorecard")
        render_scorecard(result)
    with valuation_tab:
        render_valuation(result)
    with plan_tab:
        render_research_plan(result)
    with evidence_tab:
        render_evidence(result)
    with sources_tab:
        render_sources(result)
    with report_tab:
        st.markdown(st.session_state.report)
        st.download_button(
            "Download Markdown report",
            data=st.session_state.report.encode("utf-8"),
            file_name=f"{result.ticker}_financial_research.md",
            mime="text/markdown",
        )
        evidence_payload = {
            "claims": [item.model_dump(mode="json") for item in result.claims],
            "evidence": [item.model_dump(mode="json") for item in result.evidence],
            "calculations": [item.model_dump(mode="json") for item in result.calculations],
            "reconciliations": [item.model_dump(mode="json") for item in result.reconciliations],
        }
        st.download_button(
            "Download evidence JSON",
            data=json.dumps(evidence_payload, indent=2, default=str).encode("utf-8"),
            file_name=f"{result.ticker}_evidence.json",
            mime="application/json",
        )
        if result.run_manifest:
            st.download_button(
                "Download run manifest JSON",
                data=result.run_manifest.model_dump_json(indent=2).encode("utf-8"),
                file_name=f"{result.ticker}_run_manifest.json",
                mime="application/json",
            )
