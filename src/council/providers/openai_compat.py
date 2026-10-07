"""Shared client for OpenAI-compatible ``/chat/completions`` APIs.

Used by the OpenRouter, OpenAI and Ollama providers.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import httpx

from council.errors import ProviderError
from council.models import Completion
from council.providers.base import Message

DEFAULT_TIMEOUT_S = 120.0
OPENAI_BASE_URL = "https://api.openai.com/v1"
_EXCERPT_CHARS = 500


def _excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _content_to_text(content: Any) -> str:
    """``message.content`` may be a string, ``None`` or a list of content parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


class OpenAICompatProvider:
    """Async client for an OpenAI-compatible chat-completions endpoint.

    Args:
        base_url: API root, e.g. ``https://openrouter.ai/api/v1``.
        api_key_env: Name of the env var holding the API key, or ``None`` when no key
            is needed (Ollama). The key is read at call time, never at construction.
        name: Provider name used in error messages (``openrouter``, ``openai``...).
        extra_headers: Additional HTTP headers sent with every request.
        max_tokens_param: Body field for the output-token limit. OpenAI's own API
            requires ``max_completion_tokens`` for current models; others accept
            ``max_tokens``.
        extra_body: Additional JSON fields merged into every request body.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key_env: str | None,
        name: str,
        extra_headers: Mapping[str, str] | None = None,
        *,
        max_tokens_param: str = "max_tokens",
        extra_body: Mapping[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._base_url = base_url
        self.api_key_env = api_key_env
        self.name = name
        self.extra_headers = dict(extra_headers or {})
        self.max_tokens_param = max_tokens_param
        self.extra_body = dict(extra_body or {})
        self.timeout = timeout

    @property
    def base_url(self) -> str:
        return self._base_url.rstrip("/")

    def _api_key(self) -> str | None:
        if self.api_key_env is None:
            return None
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ProviderError(
                f"{self.api_key_env} is not set. Set this environment variable to use "
                f"'{self.name}:' models."
            )
        return key

    def build_payload(
        self,
        model: str,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [dict(m) for m in messages],
            self.max_tokens_param: max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(self.extra_body)
        return payload

    def _connect_hint(self) -> str:
        return ""

    async def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> Completion:
        api_key = self._api_key()
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"{self.base_url}/chat/completions"
        payload = self.build_payload(
            model, messages, max_tokens=max_tokens, temperature=temperature
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"{self.name}: request for model {model!r} timed out after {self.timeout:g}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"{self.name}: request to {url} failed: {exc!s}{self._connect_hint()}"
            ) from exc

        if not response.is_success:
            raise ProviderError(
                f"{self.name}: HTTP {response.status_code} for model {model!r}: "
                f"{_excerpt(response.text)}",
                status_code=response.status_code,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"{self.name}: invalid JSON from {url}: {_excerpt(response.text)}"
            ) from exc
        return self.parse_response(data, model)

    def parse_response(self, data: Any, model: str) -> Completion:
        if not isinstance(data, Mapping):
            raise ProviderError(f"{self.name}: unexpected response for model {model!r}")
        choices = data.get("choices") or []
        if not choices:
            error = data.get("error")
            if isinstance(error, Mapping):
                error = error.get("message") or error
            detail = _excerpt(str(error)) if error else "response contained no choices"
            raise ProviderError(f"{self.name}: model {model!r} failed: {detail}")
        choice = choices[0] if isinstance(choices[0], Mapping) else {}
        message = choice.get("message") or {}
        text = _content_to_text(message.get("content"))
        if not text.strip():
            refusal = message.get("refusal")
            if refusal:
                raise ProviderError(f"{self.name}: model {model!r} refused: {_excerpt(refusal)}")
            raise ProviderError(
                f"{self.name}: model {model!r} returned an empty answer "
                f"(finish_reason={choice.get('finish_reason')!r})"
            )
        usage = data.get("usage") or {}
        if not isinstance(usage, Mapping):
            usage = {}
        return Completion(
            text=text,
            input_tokens=_as_int(usage.get("prompt_tokens")),
            output_tokens=_as_int(usage.get("completion_tokens")),
            cost_usd=_as_float(usage.get("cost")),
        )


class OpenAIProvider(OpenAICompatProvider):
    """OpenAI's own API (``openai:<model>``), key from ``OPENAI_API_KEY``."""

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        super().__init__(
            OPENAI_BASE_URL,
            "OPENAI_API_KEY",
            "openai",
            max_tokens_param="max_completion_tokens",
            timeout=timeout,
        )
