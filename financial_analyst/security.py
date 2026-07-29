"""Credential redaction, safe logging, and session identifiers."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CREDENTIAL_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|authorization|token|secret)(\s*[=:]\s*)([^\s&,;]+)"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\b(?:sk|pk|key)-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(apikey=)[^&\s]+"),
)


def redact_text(text: str, secrets: Iterable[str] = ()) -> str:
    """Redact known secrets and common credential formats from text."""

    sanitized = str(text)
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(secret, "[REDACTED]")
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.groups >= 2:
            sanitized = pattern.sub(r"\1\2[REDACTED]", sanitized)
        else:
            sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized


def safe_error_message(error: BaseException, *, context: str, secrets: Iterable[str] = ()) -> str:
    """Return a concise error suitable for UI display."""

    detail = redact_text(str(error), secrets).replace("\n", " ").strip()
    if len(detail) > 240:
        detail = f"{detail[:237]}..."
    return f"{context}: {detail or type(error).__name__}"


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_text(record.getMessage())
        record.args = ()
        return True


def configure_logging(log_directory: Path) -> logging.Logger:
    """Create a small rotating local log without prompt or document contents."""

    logger = logging.getLogger("financial_analyst")
    if logger.handlers:
        return logger

    log_directory.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_directory / "application.log",
        maxBytes=500_000,
        backupCount=2,
        encoding="utf-8",
    )
    handler.addFilter(_RedactingFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def new_session_id() -> str:
    """Return an opaque identifier unique to one UI session."""

    return uuid.uuid4().hex
