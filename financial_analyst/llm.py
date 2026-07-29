"""Centralized construction and validation of supported chat models."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from financial_analyst.config import (
    AppSettings,
    LLMProvider,
    ProviderConfig,
    environment_api_key,
)
from financial_analyst.models import (
    AnalysisDepth,
    PlanStep,
    ProviderCapabilities,
    ResearchPlan,
    ResearchRequest,
)
from financial_analyst.security import safe_error_message


class ProviderConfigurationError(ValueError):
    """Raised when a provider cannot be configured safely."""


def _require_cloud_key(config: ProviderConfig) -> str:
    if config.api_key is None or not config.api_key.get_secret_value().strip():
        raise ProviderConfigurationError(
            f"{config.provider.value} requires an API key. Enter it in the sidebar "
            "or set the matching environment variable."
        )
    return config.api_key.get_secret_value()


def create_chat_model(config: ProviderConfig) -> BaseChatModel:
    """Build one provider model behind LangChain's common chat-model interface."""

    common: dict[str, Any] = {
        "model": config.model_name,
    }
    if _supports_temperature(config):
        common["temperature"] = config.temperature
    try:
        if config.provider is LLMProvider.OPENAI:
            return ChatOpenAI(
                **common,
                api_key=_require_cloud_key(config),
                timeout=config.request_timeout_seconds,
                max_retries=0,
            )
        if config.provider is LLMProvider.GEMINI:
            return ChatGoogleGenerativeAI(
                **common,
                google_api_key=_require_cloud_key(config),
                timeout=config.request_timeout_seconds,
                max_retries=0,
            )
        if config.provider is LLMProvider.ANTHROPIC:
            return ChatAnthropic(
                **common,
                anthropic_api_key=_require_cloud_key(config),
                timeout=config.request_timeout_seconds,
                max_retries=0,
            )
        if config.provider is LLMProvider.OLLAMA:
            return ChatOllama(
                **common,
                base_url=config.ollama_base_url,
                client_kwargs={"timeout": config.request_timeout_seconds},
            )
    except ProviderConfigurationError:
        raise
    except Exception as error:
        secrets = [config.api_key.get_secret_value()] if config.api_key is not None else []
        raise ProviderConfigurationError(
            safe_error_message(
                error,
                context=f"Could not configure {config.provider.value}",
                secrets=secrets,
            )
        ) from error

    raise ProviderConfigurationError(f"Unsupported LLM provider: {config.provider!s}")


def provider_capabilities(provider: LLMProvider) -> ProviderCapabilities:
    """Return configured capabilities used to choose a guarded planning path."""

    cloud = provider is not LLMProvider.OLLAMA
    return ProviderCapabilities(
        supports_native_tools=cloud,
        supports_structured_output=True,
        supports_streaming=True,
        supports_usage_metadata=cloud,
        supports_system_instructions=True,
        maximum_configured_context=None,
        local=not cloud,
        api_key_required=cloud,
    )


def create_research_plan(
    model: BaseChatModel,
    request: ResearchRequest,
    ticker: str,
    tools: Sequence[BaseTool],
    *,
    gap_feedback: Sequence[str] = (),
) -> ResearchPlan:
    """Ask the selected model to choose allowlisted tools, then validate its plan."""

    budget = _mode_budget(request.analysis_depth)
    available = [tool.name for tool in tools]
    prompt = _planning_prompt(request, ticker, available, budget, gap_feedback)
    try:
        bound = model.bind_tools(list(tools))
        response = bound.invoke(
            [
                SystemMessage(
                    content=(
                        "Select only the read-only research tools needed for the objective. "
                        "Do not execute tools, provide chain-of-thought, or invent inputs."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            return _plan_from_tool_calls(
                request,
                ticker,
                available,
                tool_calls,
                budget,
                gap_feedback,
            )
        content = _message_text(getattr(response, "content", ""))
        if content:
            return _parse_structured_plan(
                request,
                ticker,
                available,
                content,
                budget,
                gap_feedback,
            )
    except (NotImplementedError, AttributeError, TypeError, ValueError):
        pass

    try:
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "Return one strict JSON research plan. Select only allowlisted tools. "
                        "Do not include explanations outside JSON or private reasoning."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        return _parse_structured_plan(
            request,
            ticker,
            available,
            _message_text(getattr(response, "content", "")),
            budget,
            gap_feedback,
        )
    except Exception as error:
        fallback_gaps = [
            *gap_feedback,
            f"Planner fallback after {type(error).__name__}.",
        ]
        return _safe_fallback_plan(
            request,
            ticker,
            available,
            budget,
            fallback_gaps,
        )


def test_connection(model: BaseChatModel, *, secrets: Sequence[str] = ()) -> str:
    """Make a user-requested minimal provider call and return a safe status."""

    try:
        response = model.invoke(
            [HumanMessage(content="Reply with exactly: connection ok")],
        )
        if not getattr(response, "content", None):
            raise ProviderConfigurationError("The provider returned an empty response.")
        return "Connection successful."
    except ProviderConfigurationError:
        raise
    except Exception as error:
        raise ProviderConfigurationError(
            safe_error_message(
                error,
                context="Connection test failed",
                secrets=secrets,
            )
        ) from error


def _supports_temperature(config: ProviderConfig) -> bool:
    """Avoid parameters that current reasoning-model APIs reject or ignore."""

    model = config.model_name.lower()
    if config.provider is LLMProvider.GEMINI and model.startswith("gemini-3"):
        return False
    return not (
        config.provider is LLMProvider.OPENAI
        and (model.startswith("gpt-5") or model.startswith(("o1", "o3", "o4")))
    )


def _planning_prompt(
    request: ResearchRequest,
    ticker: str,
    available: list[str],
    budget: int,
    gaps: Sequence[str],
) -> str:
    return (
        f"Objective: {request.query}\nTicker: {ticker}\nMode: {request.analysis_depth.value}\n"
        f"Uploaded documents: {bool(request.documents)}\nAllowlisted tools: {available}\n"
        f"Maximum tool calls: {budget}\nEvidence gaps: {list(gaps)}\n"
        "Return JSON with selected_tools (list), purposes (object keyed by tool), "
        "requested_outputs, required_metrics, required_periods, and expected_evidence. "
        "Quick should be minimal. Standard should normally include SEC and news. Detailed "
        "may add transcripts and uploaded-document evidence when relevant. Select DCF only "
        "when valuation is requested."
    )


def _parse_structured_plan(
    request: ResearchRequest,
    ticker: str,
    available: list[str],
    content: str,
    budget: int,
    gaps: Sequence[str],
) -> ResearchPlan:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("Planner did not return a JSON object.")
    payload = json.loads(match.group(0))
    selected = payload.get("selected_tools")
    if not isinstance(selected, list):
        raise ValueError("Planner JSON is missing selected_tools.")
    purposes = payload.get("purposes") if isinstance(payload.get("purposes"), dict) else {}
    tools = [str(name) for name in selected if str(name) in available][:budget]
    if not tools:
        raise ValueError("Planner selected no allowlisted tool.")
    return _make_plan(
        request,
        ticker,
        tools,
        purposes,
        budget,
        "structured_json",
        gaps,
        requested_outputs=payload.get("requested_outputs", []),
        required_metrics=payload.get("required_metrics", []),
        required_periods=payload.get("required_periods", []),
        expected_evidence=payload.get("expected_evidence", []),
    )


def _plan_from_tool_calls(
    request: ResearchRequest,
    ticker: str,
    available: list[str],
    calls: list[dict[str, Any]],
    budget: int,
    gaps: Sequence[str],
) -> ResearchPlan:
    selected = [str(call.get("name", "")) for call in calls]
    selected = [name for name in selected if name in available][:budget]
    if not selected:
        raise ValueError("No allowlisted native tool calls were selected.")
    purposes = {
        str(call.get("name")): str((call.get("args") or {}).get("purpose") or "Model selected")
        for call in calls
        if str(call.get("name")) in selected
    }
    return _make_plan(
        request,
        ticker,
        selected,
        purposes,
        budget,
        "native_tools",
        gaps,
    )


def _safe_fallback_plan(
    request: ResearchRequest,
    ticker: str,
    available: list[str],
    budget: int,
    gaps: Sequence[str],
) -> ResearchPlan:
    query = request.query.casefold()
    selected = ["market_snapshot", "financial_statements"]
    if request.analysis_depth is not AnalysisDepth.QUICK:
        selected.extend(["sec_company_facts", "recent_news"])
    if any(term in query for term in ("transcript", "earnings call")):
        selected.append("earnings_transcript")
    if any(term in query for term in ("dcf", "valuation", "intrinsic value", "fair value")):
        selected.append("discounted_cash_flow")
    selected = [name for name in dict.fromkeys(selected) if name in available][:budget]
    return _make_plan(
        request,
        ticker,
        selected,
        {name: "Safe deterministic fallback after invalid planner output" for name in selected},
        budget,
        "safe_fallback",
        gaps,
    )


def _make_plan(
    request: ResearchRequest,
    ticker: str,
    tools: list[str],
    purposes: dict[str, Any],
    budget: int,
    method: str,
    gaps: Sequence[str],
    *,
    requested_outputs: Any = (),
    required_metrics: Any = (),
    required_periods: Any = (),
    expected_evidence: Any = (),
) -> ResearchPlan:
    steps = [
        PlanStep(
            step_id=f"step-{index}",
            tool_name=name,
            purpose=str(purposes.get(name) or f"Collect {name.replace('_', ' ')} evidence"),
            inputs={"ticker": ticker},
            required=name in {"market_snapshot", "financial_statements"},
        )
        for index, name in enumerate(tools, start=1)
    ]
    return ResearchPlan(
        research_objective=request.query,
        ticker=ticker,
        requested_outputs=_string_list(requested_outputs),
        required_metrics=_string_list(required_metrics),
        required_periods=_string_list(required_periods),
        steps=steps,
        valuation_requested="discounted_cash_flow" in tools,
        document_retrieval_requested=bool(request.documents),
        news_requested="recent_news" in tools,
        transcript_requested="earnings_transcript" in tools,
        expected_evidence=_string_list(expected_evidence),
        maximum_tool_budget=budget,
        planning_method=method,
        revision_count=1 if gaps else 0,
        gaps=list(gaps),
    )


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list | tuple) else []


def _mode_budget(depth: AnalysisDepth) -> int:
    return {
        AnalysisDepth.QUICK: 3,
        AnalysisDepth.STANDARD: 6,
        AnalysisDepth.DETAILED: 8,
    }[depth]


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in content
            if isinstance(item, str | dict)
        )
    return str(content)


def _live_smoke_main() -> int:
    """Run one explicitly requested, minimal live provider compatibility call."""

    import argparse

    parser = argparse.ArgumentParser(description="Optional live LLM provider smoke test")
    parser.add_argument("--provider", required=True, choices=[item.value for item in LLMProvider])
    parser.add_argument("--model", required=True)
    arguments = parser.parse_args()
    provider = LLMProvider(arguments.provider)
    settings = AppSettings()
    config = ProviderConfig(
        provider=provider,
        model_name=arguments.model,
        api_key=environment_api_key(settings, provider),
        ollama_base_url=settings.ollama_base_url,
        request_timeout_seconds=settings.request_timeout_seconds,
    )
    model = create_chat_model(config)
    print(test_connection(model))
    capabilities = provider_capabilities(provider)
    print(
        f"provider={provider.value} model={arguments.model} "
        f"native_tools={capabilities.supports_native_tools} "
        f"structured_output={capabilities.supports_structured_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_live_smoke_main())
