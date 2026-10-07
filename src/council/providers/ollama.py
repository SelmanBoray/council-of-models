"""Ollama provider (``ollama:<model>``) via Ollama's OpenAI-compatible ``/v1`` API.

No API key is needed. The server is ``OLLAMA_HOST`` (default ``http://localhost:11434``),
read at call time.
"""

from __future__ import annotations

import os

from council.providers.openai_compat import DEFAULT_TIMEOUT_S, OpenAICompatProvider

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def ollama_base_url(host: str | None = None) -> str:
    """Build the ``/v1`` base URL from ``host`` or the ``OLLAMA_HOST`` env var.

    Accepts ``localhost:11434``, ``http://host:port``, with or without a trailing
    ``/`` or ``/v1``.
    """
    raw = host if host is not None else os.environ.get("OLLAMA_HOST", "")
    raw = (raw or "").strip() or DEFAULT_OLLAMA_HOST
    if "://" not in raw:
        raw = f"http://{raw}"
    raw = raw.rstrip("/")
    return raw if raw.endswith("/v1") else f"{raw}/v1"


class OllamaProvider(OpenAICompatProvider):
    """Local models served by Ollama."""

    def __init__(self, host: str | None = None, *, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        super().__init__("", None, "ollama", timeout=timeout)
        self._host = host

    @property
    def base_url(self) -> str:
        return ollama_base_url(self._host)

    def _connect_hint(self) -> str:
        return " (is Ollama running? start it with `ollama serve` or set OLLAMA_HOST)"
