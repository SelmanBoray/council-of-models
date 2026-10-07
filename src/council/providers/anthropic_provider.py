"""Anthropic provider (``anthropic:<model>``) using the official ``anthropic`` SDK.

The SDK client is created lazily on the first call, so a missing
``ANTHROPIC_API_KEY`` surfaces as a :class:`ProviderError` at call time, not at import.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from council.errors import ProviderError
from council.models import Completion
from council.providers.base import Message

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

DEFAULT_TIMEOUT_S = 120.0
_EXCERPT_CHARS = 500


def _excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def split_messages(messages: list[Message]) -> tuple[str | None, list[dict[str, str]]]:
    """Separate system messages (joined) from the user/assistant conversation.

    The Messages API takes the system prompt as the top-level ``system`` argument,
    not as a message.
    """
    system_parts: list[str] = []
    conversation: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            if content.strip():
                system_parts.append(content)
        elif role in ("user", "assistant"):
            conversation.append({"role": role, "content": content})
        else:
            raise ProviderError(f"anthropic: unsupported message role {role!r}")
    if not conversation:
        raise ProviderError("anthropic: at least one user message is required")
    system = "\n\n".join(system_parts) if system_parts else None
    return system, conversation


class AnthropicProvider:
    """Claude models through ``AsyncAnthropic().messages.create``."""

    name = "anthropic"

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT_S, max_retries: int = 2) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: AsyncAnthropic | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    def _get_client(self) -> AsyncAnthropic:
        loop = asyncio.get_running_loop()
        if self._client is not None and self._client_loop is loop:
            return self._client
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
        if not api_key and not auth_token:
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. Set this environment variable to use "
                "'anthropic:' models (or use 'openrouter:anthropic/...' instead)."
            )
        from anthropic import AsyncAnthropic

        credentials: dict[str, Any] = (
            {"api_key": api_key} if api_key else {"auth_token": auth_token}
        )
        # A client is bound to the event loop it was first used on; recreate it if
        # this provider instance is reused from a different loop.
        self._client = AsyncAnthropic(
            **credentials, timeout=self.timeout, max_retries=self.max_retries
        )
        self._client_loop = loop
        return self._client

    async def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> Completion:
        import anthropic

        client = self._get_client()
        system, conversation = split_messages(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": conversation,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            # SDK 1.x removed `temperature` from messages.create(); older models still
            # honour it in the request body (current Opus models reject it with a 400).
            kwargs["extra_body"] = {"temperature": temperature}

        try:
            response = await client.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"anthropic: HTTP {exc.status_code} for model {model!r}: "
                f"{_excerpt(getattr(exc, 'message', '') or str(exc))}",
                status_code=exc.status_code,
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderError(
                f"anthropic: request for model {model!r} timed out after {self.timeout:g}s"
            ) from exc
        except anthropic.AnthropicError as exc:
            raise ProviderError(f"anthropic: request for model {model!r} failed: {exc}") from exc

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            suffix = f" (category: {category})" if category else ""
            raise ProviderError(f"anthropic: model {model!r} refused to answer{suffix}")

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text.strip():
            raise ProviderError(
                f"anthropic: model {model!r} returned no text "
                f"(stop_reason={response.stop_reason!r})"
            )
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            cost_usd=None,
        )
