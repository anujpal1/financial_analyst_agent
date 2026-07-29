"""Typed application and provider configuration."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMProvider(StrEnum):
    """Supported chat-model providers."""

    OPENAI = "OpenAI"
    GEMINI = "Google Gemini"
    ANTHROPIC = "Anthropic"
    OLLAMA = "Ollama"


DEFAULT_MODELS: dict[LLMProvider, tuple[str, ...]] = {
    LLMProvider.OPENAI: ("gpt-4.1-mini", "gpt-5-mini", "gpt-4.1"),
    LLMProvider.GEMINI: (
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.5-pro",
    ),
    LLMProvider.ANTHROPIC: (
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5-20250929",
    ),
    LLMProvider.OLLAMA: ("llama3.1:8b", "qwen3:8b", "mistral:7b"),
}


class AppSettings(BaseSettings):
    """Environment-backed settings; the UI may override provider credentials in memory."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    sec_user_agent: str | None = None
    fmp_api_key: SecretStr | None = None

    request_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    retry_count: int = Field(default=2, ge=0, le=5)
    upload_size_limit_mb: int = Field(default=10, ge=1, le=50)
    cache_directory: Path = PROJECT_ROOT / ".cache"


class ProviderConfig(BaseModel):
    """Provider choices for one Streamlit session."""

    provider: LLMProvider
    model_name: str
    api_key: SecretStr | None = None
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    ollama_base_url: str = "http://localhost:11434"
    request_timeout_seconds: float = Field(default=20.0, gt=0, le=120)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 200:
            raise ValueError("Choose a valid model name.")
        if any(character.isspace() for character in normalized):
            raise ValueError("Model names cannot contain whitespace.")
        return normalized

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("Ollama base URL must start with http:// or https://.")
        return normalized


def environment_api_key(settings: AppSettings, provider: LLMProvider) -> SecretStr | None:
    """Return the environment key for a cloud provider."""

    return {
        LLMProvider.OPENAI: settings.openai_api_key,
        LLMProvider.GEMINI: settings.google_api_key,
        LLMProvider.ANTHROPIC: settings.anthropic_api_key,
        LLMProvider.OLLAMA: None,
    }[provider]
