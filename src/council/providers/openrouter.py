"""OpenRouter provider (``openrouter:<vendor>/<model>``), key from ``OPENROUTER_API_KEY``."""

from __future__ import annotations

from council.providers.openai_compat import DEFAULT_TIMEOUT_S, OpenAICompatProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/SelmanBoray/council-of-models",
    "X-Title": "council-of-models",
}


class OpenRouterProvider(OpenAICompatProvider):
    """OpenRouter chat completions. ``usage.cost`` in responses becomes ``cost_usd``."""

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        super().__init__(
            OPENROUTER_BASE_URL,
            "OPENROUTER_API_KEY",
            "openrouter",
            OPENROUTER_HEADERS,
            # Ask OpenRouter to include usage accounting (tokens + cost) in the response.
            extra_body={"usage": {"include": True}},
            timeout=timeout,
        )
