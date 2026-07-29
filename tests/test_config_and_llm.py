from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from financial_analyst.config import AppSettings, LLMProvider, ProviderConfig
from financial_analyst.llm import (
    ProviderConfigurationError,
    create_chat_model,
    create_research_plan,
    provider_capabilities,
)
from financial_analyst.llm import test_connection as run_connection_test
from financial_analyst.models import AnalysisDepth, ResearchRequest
from financial_analyst.tools import build_tool_registry


class FakeModel:
    def __init__(self) -> None:
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> FakeModel:
        self.bound_tools = tools
        return self

    def invoke(self, messages: list[Any]) -> Any:
        return type("Response", (), {"content": "connection ok"})()


def _config(provider: LLMProvider, key: str | None = "secret-value") -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        model_name="test-model",
        api_key=SecretStr(key) if key else None,
    )


def test_configuration_loads_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "17")
    settings = AppSettings(_env_file=None)
    assert settings.ollama_model == "qwen3:8b"
    assert settings.request_timeout_seconds == 17


@pytest.mark.parametrize(
    ("provider", "constructor_name", "key_argument"),
    [
        (LLMProvider.OPENAI, "ChatOpenAI", "api_key"),
        (LLMProvider.GEMINI, "ChatGoogleGenerativeAI", "google_api_key"),
        (LLMProvider.ANTHROPIC, "ChatAnthropic", "anthropic_api_key"),
    ],
)
def test_cloud_provider_factory_selection(
    monkeypatch: pytest.MonkeyPatch,
    provider: LLMProvider,
    constructor_name: str,
    key_argument: str,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = FakeModel()

    def constructor(**kwargs: Any) -> FakeModel:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(f"financial_analyst.llm.{constructor_name}", constructor)
    assert create_chat_model(_config(provider)) is sentinel
    assert captured["model"] == "test-model"
    assert captured[key_argument] == "secret-value"


def test_ollama_factory_does_not_require_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = FakeModel()

    def constructor(**kwargs: Any) -> FakeModel:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("financial_analyst.llm.ChatOllama", constructor)
    assert create_chat_model(_config(LLMProvider.OLLAMA, key=None)) is sentinel
    assert captured["base_url"] == "http://localhost:11434"
    assert "api_key" not in captured


@pytest.mark.parametrize(
    "provider",
    [LLMProvider.OPENAI, LLMProvider.GEMINI, LLMProvider.ANTHROPIC],
)
def test_missing_cloud_api_key_is_clear(provider: LLMProvider) -> None:
    with pytest.raises(ProviderConfigurationError, match="requires an API key"):
        create_chat_model(_config(provider, key=None))


def test_unsupported_provider_is_rejected() -> None:
    invalid = ProviderConfig.model_construct(
        provider="Unsupported",
        model_name="test-model",
        api_key=None,
        temperature=0.1,
        ollama_base_url="http://localhost:11434",
        request_timeout_seconds=20,
    )
    with pytest.raises(ProviderConfigurationError, match="Unsupported"):
        create_chat_model(invalid)


def test_invalid_model_configuration_is_rejected() -> None:
    with pytest.raises(ValidationError, match="whitespace"):
        ProviderConfig(
            provider=LLMProvider.OLLAMA,
            model_name="bad model",
        )


def test_provider_exception_does_not_leak_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-super-secret-value"

    def constructor(**kwargs: Any) -> FakeModel:
        raise ValueError(f"request failed?apikey={secret}")

    monkeypatch.setattr("financial_analyst.llm.ChatOpenAI", constructor)
    with pytest.raises(ProviderConfigurationError) as captured:
        create_chat_model(_config(LLMProvider.OPENAI, key=secret))
    assert secret not in str(captured.value)
    assert "[REDACTED]" in str(captured.value)


def test_connection_probe_and_provider_capabilities() -> None:
    model = FakeModel()
    assert run_connection_test(model) == "Connection successful."
    cloud = provider_capabilities(LLMProvider.OPENAI)
    local = provider_capabilities(LLMProvider.OLLAMA)
    assert cloud.supports_native_tools
    assert cloud.api_key_required
    assert local.local
    assert not local.api_key_required


def test_structured_planner_selects_allowlisted_tools() -> None:
    class PlannerModel(FakeModel):
        def bind_tools(self, tools: list[Any]) -> Any:
            raise NotImplementedError

        def invoke(self, messages: list[Any]) -> Any:
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"selected_tools":["market_snapshot","financial_statements"],'
                        '"purposes":{"market_snapshot":"price"},'
                        '"requested_outputs":["dashboard"],"required_metrics":["price"],'
                        '"required_periods":["latest"],"expected_evidence":["market"]}'
                    )
                },
            )()

    plan = create_research_plan(
        PlannerModel(),
        ResearchRequest(
            query="Analyze MSFT.",
            ticker="MSFT",
            analysis_depth=AnalysisDepth.QUICK,
        ),
        "MSFT",
        build_tool_registry(AppSettings(_env_file=None)),
    )
    assert plan.planning_method == "structured_json"
    assert [step.tool_name for step in plan.steps] == [
        "market_snapshot",
        "financial_statements",
    ]


def test_current_gemini_model_omits_deprecated_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def constructor(**kwargs: Any) -> FakeModel:
        captured.update(kwargs)
        return FakeModel()

    monkeypatch.setattr("financial_analyst.llm.ChatGoogleGenerativeAI", constructor)
    config = _config(LLMProvider.GEMINI)
    config.model_name = "gemini-3.5-flash"
    create_chat_model(config)
    assert "temperature" not in captured
