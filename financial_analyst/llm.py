"""Centralized construction and validation of supported chat models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from financial_analyst.config import LLMProvider, ProviderConfig
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


def validate_tool_calling(model: BaseChatModel, tools: Sequence[BaseTool] | None = None) -> None:
    """Verify that a selected model exposes LangChain's tool-binding interface."""

    if not hasattr(model, "bind_tools"):
        raise ProviderConfigurationError(
            "This model does not expose the tool-calling interface required by the application."
        )

    validation_tools = list(tools or [_connection_probe_tool()])
    try:
        model.bind_tools(validation_tools)
    except Exception as error:
        raise ProviderConfigurationError(
            "The selected model cannot bind financial tools. Choose a tool-capable model."
        ) from error


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


def _connection_probe(value: str) -> str:
    """Return a value unchanged; used only for non-network compatibility validation."""

    return value


def _connection_probe_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_connection_probe,
        name="connection_probe",
        description="Compatibility probe that returns its input unchanged.",
    )


def _supports_temperature(config: ProviderConfig) -> bool:
    """Avoid parameters that current reasoning-model APIs reject or ignore."""

    model = config.model_name.lower()
    if config.provider is LLMProvider.GEMINI and model.startswith("gemini-3"):
        return False
    return not (
        config.provider is LLMProvider.OPENAI
        and (model.startswith("gpt-5") or model.startswith(("o1", "o3", "o4")))
    )
