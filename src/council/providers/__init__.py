"""Model providers. Use :func:`get_provider` to obtain one by name."""

from __future__ import annotations

from council.providers.base import Message, Provider, split_model_id

PROVIDER_NAMES: tuple[str, ...] = ("openrouter", "anthropic", "openai", "ollama", "fake")

# Environment variables each provider reads (``None`` = no key needed).
PROVIDER_KEY_ENV: dict[str, str | None] = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "ollama": None,
    "fake": None,
}


def get_provider(name: str) -> Provider:
    """Return a new provider instance for ``name``.

    Constructing a provider never touches the network and never checks API keys;
    a missing key raises :class:`council.errors.ProviderError` when ``complete`` is
    called.
    """
    key = (name or "").strip().lower()
    if key == "openrouter":
        from council.providers.openrouter import OpenRouterProvider

        return OpenRouterProvider()
    if key == "anthropic":
        from council.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if key == "openai":
        from council.providers.openai_compat import OpenAIProvider

        return OpenAIProvider()
    if key == "ollama":
        from council.providers.ollama import OllamaProvider

        return OllamaProvider()
    if key == "fake":
        from council.providers.fake import FakeProvider

        return FakeProvider()
    raise ValueError(f"Unknown provider {name!r}; expected one of: {', '.join(PROVIDER_NAMES)}")


__all__ = [
    "PROVIDER_KEY_ENV",
    "PROVIDER_NAMES",
    "Message",
    "Provider",
    "get_provider",
    "split_model_id",
]
