"""Provider protocol and model-id helpers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from council.models import Completion

Message = dict[str, str]
"""A chat message: ``{"role": "system" | "user" | "assistant", "content": str}``."""


@runtime_checkable
class Provider(Protocol):
    """Anything that can turn chat messages into a :class:`Completion`."""

    name: str

    async def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> Completion: ...


def split_model_id(model_id: str) -> tuple[str, str]:
    """Split ``"<provider>:<model>"`` on the first colon.

    >>> split_model_id("openrouter:openai/gpt-5.1")
    ('openrouter', 'openai/gpt-5.1')
    >>> split_model_id("ollama:llama3.1:8b")
    ('ollama', 'llama3.1:8b')
    """
    if not isinstance(model_id, str) or ":" not in model_id:
        raise ValueError(
            f"Invalid model id {model_id!r}: expected '<provider>:<model>', "
            "e.g. 'openrouter:openai/gpt-5.1' or 'fake:alpha'"
        )
    provider, model = model_id.split(":", 1)
    provider, model = provider.strip().lower(), model.strip()
    if not provider or not model:
        raise ValueError(
            f"Invalid model id {model_id!r}: both provider and model must be non-empty"
        )
    return provider, model
