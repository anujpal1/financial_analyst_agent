"""Streamlit entry point for the local financial research application."""

from __future__ import annotations

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
    validate_tool_calling,
)
from financial_analyst.models import Availability, ResearchRequest
from financial_analyst.security import (
    configure_logging,
    new_session_id,
    safe_error_message,
)
from financial_analyst.tools import build_tool_registry
from financial_analyst.workflow import build_research_graph, run_research

st.set_page_config(
    page_title="Evidence-Aware Financial Research",
    page_icon="📊",
    layout="wide",
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


_initialize_session()

with st.sidebar:
    st.header("Configuration")
    st.caption(f"Session: {st.session_state.session_id[:8]}")

    provider = LLMProvider(st.selectbox("LLM provider", [item.value for item in LLMProvider]))
    suggested_models = DEFAULT_MODELS[provider]
    selected_model = st.selectbox("Model", suggested_models)
    use_custom_model = st.checkbox("Use a custom model name")
    custom_model = (
        st.text_input("Custom model", placeholder="provider-model-name") if use_custom_model else ""
    )
    model_name = custom_model.strip() or selected_model

    api_key_text = ""
    if provider is not LLMProvider.OLLAMA:
        api_key_text = st.text_input(
            f"{provider.value} API key",
            type="password",
            help="Kept only in this Streamlit session and never written by the app.",
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

    with st.expander("Provider setup"):
        if provider is LLMProvider.OLLAMA:
            st.write("Start Ollama locally and pull the selected model. No API key is required.")
        else:
            variable = {
                LLMProvider.OPENAI: "OPENAI_API_KEY",
                LLMProvider.GEMINI: "GOOGLE_API_KEY",
                LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            }[provider]
            st.write(
                f"Paste a key above or set `{variable}` before starting Streamlit. "
                "A pasted key takes precedence."
            )

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
    if configuration_error:
        st.error(configuration_error)
    elif cloud_key_missing:
        st.warning("A cloud API key is required before analysis.")
    else:
        st.success(f"Selected provider: {provider.value} · {model_name}")

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

    if (provider is LLMProvider.GEMINI and model_name.lower().startswith("gemini-3")) or (
        provider is LLMProvider.OPENAI
        and model_name.lower().startswith(("gpt-5", "o1", "o3", "o4"))
    ):
        st.caption(
            "This model family does not use the temperature value; the provider "
            "factory omits that parameter."
        )

    if st.button(
        "Test connection",
        use_container_width=True,
        disabled=bool(configuration_error or cloud_key_missing),
    ):
        try:
            model = create_chat_model(provider_config)
            validate_tool_calling(model, build_tool_registry(settings))
            connection_secrets = (
                [provider_config.api_key.get_secret_value()]
                if provider_config.api_key is not None
                else []
            )
            st.session_state.provider_status = test_connection(
                model,
                secrets=connection_secrets,
            )
            st.success(st.session_state.provider_status)
        except ProviderConfigurationError as error:
            st.session_state.provider_status = str(error)
            st.error(st.session_state.provider_status)

    st.caption(f"Connection status: {st.session_state.provider_status}")

    with st.expander("Advanced data sources"):
        sec_user_agent = st.text_input(
            "SEC User-Agent",
            value=settings.sec_user_agent or "",
            help="Use an application name plus contact email. Required only for SEC access.",
        )
        fmp_key_text = st.text_input(
            "FMP API key (optional transcripts)",
            type="password",
            key="session_fmp_api_key",
        )
        st.caption("Core market and statement data use yfinance; these fields are optional.")

    if st.button("Clear / reset session", use_container_width=True):
        _clear_session()

st.title("Evidence-Aware Financial Research")
st.write(
    "A local-first financial research workflow that combines provider-labelled market data, "
    "official SEC facts, transparent valuation, optional uploaded PDFs, and a selected LLM."
)
st.warning(
    "For informational and educational use only. This application does not provide "
    "financial advice or guaranteed investment recommendations."
)

query = st.text_area(
    "Research question",
    placeholder=(
        "Example: Analyze MSFT cash generation, risks, and a DCF valuation. "
        "Identify missing or conflicting evidence."
    ),
    height=110,
)
ticker = st.text_input("Ticker (optional if clearly stated in the question)")
uploaded_pdf = st.file_uploader("Supporting PDF (optional)", type=["pdf"])

with st.expander("DCF assumptions"):
    st.caption("Used only when the question requests a DCF or valuation.")
    dcf_growth_rate = st.number_input(
        "Base annual FCF growth",
        min_value=-0.50,
        max_value=0.50,
        value=0.05,
        step=0.005,
        format="%.3f",
    )
    dcf_discount_rate = st.number_input(
        "Base discount rate",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.005,
        format="%.3f",
    )
    dcf_terminal_growth_rate = st.number_input(
        "Base terminal growth",
        min_value=-0.10,
        max_value=0.10,
        value=0.025,
        step=0.005,
        format="%.3f",
    )

analysis_disabled = bool(configuration_error or cloud_key_missing)
run_clicked = st.button(
    "Run analysis",
    type="primary",
    disabled=analysis_disabled,
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
                    extract_pdf_upload(
                        uploaded_pdf.name,
                        uploaded_pdf.getvalue(),
                        max_size_mb=runtime_settings.upload_size_limit_mb,
                    )
                )

            request = ResearchRequest(
                query=query.strip(),
                ticker=ticker.strip() or None,
                documents=documents,
                dcf_growth_rate=dcf_growth_rate,
                dcf_discount_rate=dcf_discount_rate,
                dcf_terminal_growth_rate=dcf_terminal_growth_rate,
            )
            with st.status("Running evidence-aware analysis...", expanded=True) as status:
                st.write("Validating provider and ticker...")
                model = create_chat_model(provider_config)
                tools = build_tool_registry(runtime_settings)
                st.write("Collecting provider-labelled evidence...")
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
                st.write("Rendering and checking the report...")
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
                "technical details; no credentials or document text are logged."
            )

if st.session_state.report:
    result = st.session_state.result
    report_tab, evidence_tab, chart_tab = st.tabs(
        ["Report", "Sources and evidence", "Market chart"]
    )
    with report_tab:
        st.markdown(st.session_state.report)
        st.download_button(
            "Download Markdown report",
            data=st.session_state.report.encode("utf-8"),
            file_name=f"{result.ticker}_financial_research.md",
            mime="text/markdown",
        )
    with evidence_tab:
        for item in result.data:
            icon = "✅" if item.status is Availability.AVAILABLE else "⚠️"
            with st.expander(f"{icon} {item.name} — {item.status.value}"):
                st.write(f"Source: {item.source}")
                if item.message:
                    st.write(item.message)
                if item.evidence:
                    st.json([entry.model_dump(mode="json") for entry in item.evidence])
    with chart_tab:
        history = next(
            (item for item in result.data if item.name == "price_history"),
            None,
        )
        if history and history.status is not Availability.UNAVAILABLE:
            st.line_chart(history.values["points"], x="date", y="close")
            st.caption("Six-month daily close data from Yahoo Finance via yfinance.")
        else:
            st.info("No chart data is available for this run.")
