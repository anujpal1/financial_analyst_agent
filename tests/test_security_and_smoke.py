from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from pydantic import SecretStr
from streamlit.testing.v1 import AppTest

import financial_analyst
from financial_analyst.config import LLMProvider, ProviderConfig
from financial_analyst.security import redact_text


def test_redaction_masks_common_credentials() -> None:
    secret = "sk-abcdefgh12345678"
    text = redact_text(
        f"Authorization: Bearer abcdefgh token={secret} apikey=private-value",
        [secret],
    )
    assert secret not in text
    assert "private-value" not in text
    assert "[REDACTED]" in text


def test_api_key_configuration_is_memory_only(tmp_path: Path) -> None:
    before = list(tmp_path.iterdir())
    config = ProviderConfig(
        provider=LLMProvider.OPENAI,
        model_name="test-model",
        api_key=SecretStr("session-only-secret"),
    )
    assert config.api_key.get_secret_value() == "session-only-secret"
    assert list(tmp_path.iterdir()) == before


def test_all_application_modules_import() -> None:
    module_names = [
        item.name
        for item in pkgutil.walk_packages(
            financial_analyst.__path__,
            prefix="financial_analyst.",
        )
    ]
    assert module_names
    for module_name in module_names:
        importlib.import_module(module_name)


def test_streamlit_application_smoke() -> None:
    app_path = Path(__file__).parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20)
    app.run()
    assert not app.exception
    assert any(
        "Evidence-Grounded Financial Research Workbench" in element.value
        for element in app.markdown
    )
    assert len(app.button) >= 2
