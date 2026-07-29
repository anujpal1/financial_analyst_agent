"""Shared test safeguards and fixtures."""

from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def block_live_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that accidentally attempts a real network connection."""

    def blocked_connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("Live network access is forbidden in automated tests.")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    monkeypatch.setattr(socket, "create_connection", blocked_connect)
