from __future__ import annotations

import json
import re
from types import SimpleNamespace

import httpx
import pytest
import respx

from council import prompts
from council.council import make_labels, parse_ranking
from council.errors import ProviderError
from council.providers import PROVIDER_NAMES, Provider, get_provider, split_model_id
from council.providers.anthropic_provider import AnthropicProvider, split_messages
from council.providers.fake import FakeProvider, answer_digest, identify_author
from council.providers.ollama import OllamaProvider, ollama_base_url
from council.providers.openai_compat import OpenAIProvider
from council.providers.openrouter import OpenRouterProvider

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MESSAGES = [{"role": "user", "content": "What is 2+2?"}]


def _chat_response(text: str = "4", **usage: object) -> dict:
    return {
        "id": "gen-1",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, **usage},
    }


# ----------------------------------------------------------------- ids & registry


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("openrouter:openai/gpt-5.1", ("openrouter", "openai/gpt-5.1")),
        ("openrouter:anthropic/claude-sonnet-4.5", ("openrouter", "anthropic/claude-sonnet-4.5")),
        ("anthropic:claude-opus-5-5", ("anthropic", "claude-opus-5-5")),
        ("ollama:llama3.1:8b", ("ollama", "llama3.1:8b")),
        ("fake:alpha", ("fake", "alpha")),
        (" OpenRouter:openai/gpt-5.1 ", ("openrouter", "openai/gpt-5.1")),
    ],
)
def test_split_model_id(model_id, expected):
    assert split_model_id(model_id) == expected


@pytest.mark.parametrize("bad", ["gpt-5.1", "", ":gpt", "openai:", "  :  "])
def test_split_model_id_rejects_invalid(bad):
    with pytest.raises(ValueError):
        split_model_id(bad)


@pytest.mark.parametrize("name", PROVIDER_NAMES)
def test_get_provider_returns_provider_without_keys(name):
    provider = get_provider(name)
    assert isinstance(provider, Provider)
    assert provider.name == name


def test_get_provider_is_case_insensitive():
    assert isinstance(get_provider("FAKE"), FakeProvider)


def test_get_provider_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("gemini-direct")


# ------------------------------------------------------------------------- fake


async def test_fake_answer_is_deterministic_and_anonymous():
    fake = FakeProvider()
    first = await fake.complete("alpha", MESSAGES)
    second = await fake.complete("alpha", MESSAGES)
    other = await fake.complete("beta", MESSAGES)
    assert first == second
    assert other.text != first.text
    for result, name in ((first, "alpha"), (other, "beta")):
        assert result.text.startswith('Answer to "What is 2+2?": ')
        assert "fake:" not in result.text and "[fake" not in result.text
        assert name not in result.text  # nothing in the text reveals the author
        assert result.text.endswith(f"(digest {answer_digest(name, 'What is 2+2?')})")
        assert 2 <= result.text.count(". ") + 1 <= 3  # 2-3 short sentences
    assert first.cost_usd == 0.0
    assert isinstance(first.input_tokens, int) and isinstance(first.output_tokens, int)


@pytest.mark.parametrize(
    "question", ["What is 2+2?", "Is a hot dog a sandwich?", "Is P = NP?", "Which is best?"]
)
async def test_fake_demo_trio_never_share_wording(question):
    fake = FakeProvider()
    messages = [{"role": "user", "content": question}]
    texts = [(await fake.complete(name, messages)).text for name in ("alpha", "beta", "gamma")]
    without_digest = {re.sub(r"\(digest [0-9a-f]{8}\)", "", text) for text in texts}
    assert len(without_digest) == 3


async def test_fake_answer_depends_on_question():
    fake = FakeProvider()
    a = await fake.complete("alpha", [{"role": "user", "content": "Question one?"}])
    b = await fake.complete("alpha", [{"role": "user", "content": "Question two?"}])
    assert a.text != b.text


async def test_fake_fail_model_raises():
    with pytest.raises(ProviderError, match="simulated failure"):
        await FakeProvider().complete("fail", MESSAGES)


async def _fake_review(
    reviewer: str, labeled_authors: list[tuple[str, str]], question: str
) -> list[str]:
    fake = FakeProvider()
    labeled = []
    for label, author in labeled_authors:
        answer = await fake.complete(author, [{"role": "user", "content": question}])
        labeled.append((label, answer.text))
    review = await fake.complete(reviewer, prompts.review_messages(question, labeled))
    assert prompts.RANKING_MARKER in review.text
    order = parse_ranking(review.text, [label for label, _ in labeled])
    authors = dict(labeled_authors)
    return [authors[label] for label in order]


# The fake's shared "quality" order depends on the question: alpha is best for the
# first question, middle for the second and worst for the third, so self-preference
# is exercised in every position.
@pytest.mark.parametrize("question", ["What is 2+2?", "Is P = NP?", "Which is best?"])
@pytest.mark.parametrize(
    "layout",
    [
        [("Response A", "alpha"), ("Response B", "beta"), ("Response C", "gamma")],
        [("Response A", "gamma"), ("Response B", "alpha"), ("Response C", "beta")],
        [("Response A", "beta"), ("Response B", "gamma"), ("Response C", "alpha")],
    ],
)
async def test_fake_reviews_are_biased_and_label_independent(layout, question):
    alpha_view = await _fake_review("alpha", layout, question)
    beta_view = await _fake_review("beta", layout, question)
    gamma_view = await _fake_review("gamma", layout, question)
    assert alpha_view[0] == "alpha"  # alpha prefers itself
    assert beta_view[-1] == "alpha" and gamma_view[-1] == "alpha"  # others rank alpha last
    assert beta_view == gamma_view  # shared "quality" order among the others
    assert alpha_view[1:] == beta_view[:-1]  # apart from the bias, alpha agrees
    # Independent of which label each author got:
    canonical = [("Response A", "alpha"), ("Response B", "beta"), ("Response C", "gamma")]
    assert alpha_view == await _fake_review("alpha", canonical, question)


async def test_fake_review_order_varies_by_question():
    canonical = [("Response A", "alpha"), ("Response B", "beta"), ("Response C", "gamma")]
    views = {
        tuple(await _fake_review("beta", canonical, q))
        for q in ("What is 2+2?", "Is P = NP?", "Which is best?", "Why is the sky blue?")
    }
    assert len(views) > 1


async def test_fake_synthesis_mentions_question():
    fake = FakeProvider()
    messages = prompts.synthesis_messages(
        "What is 2+2?",
        [("Response A", "fake:beta", "four")],
        [("fake:beta", ["fake:beta"], "only answer")],
        [("fake:beta", 1.0)],
    )
    result = await fake.complete("alpha", messages)
    assert result.text.startswith("SYNTHESIS by alpha:")
    assert "What is 2+2?" in result.text


async def test_fake_debate_mentions_round_and_keeps_identity():
    fake = FakeProvider()
    own = await fake.complete("alpha", MESSAGES)
    other = await fake.complete("beta", MESSAGES)
    messages = prompts.debate_messages(
        "What is 2+2?", own.text, [("Response A", other.text)], round_=2, total_rounds=3
    )
    result = await fake.complete("alpha", messages)
    assert result.text.startswith('Revised answer (debate round 2) to "What is 2+2?": ')
    assert "fake:" not in result.text and "alpha" not in result.text
    assert identify_author(result.text, "What is 2+2?") == "alpha"


async def test_identify_author_uses_digest_and_registry():
    fake = FakeProvider()
    question = "Tabs or spaces?"
    delta = await fake.complete("delta", [{"role": "user", "content": question}])
    assert identify_author(delta.text, question) == "delta"  # registered by complete()
    assert identify_author(delta.text, "another question") is None
    assert identify_author("no digest token here", question) is None
    stranger = f"Plain text (digest {answer_digest('never-seen-xyz', question)})"
    assert identify_author(stranger, question) is None
    assert identify_author(stranger, question, candidates=["never-seen-xyz"]) == "never-seen-xyz"


async def _review_bodies(reviewer: str, question: str, bodies: list[str]) -> list[str]:
    labels = make_labels(len(bodies))
    messages = prompts.review_messages(question, list(zip(labels, bodies, strict=True)))
    raw = (await FakeProvider().complete(reviewer, messages)).text
    order = parse_ranking(raw, labels)
    return [bodies[labels.index(label)] for label in order]


@pytest.mark.parametrize("question", ["What is 2+2?", "Which is best?"])
async def test_fake_review_places_unknown_authors_after_identified(question):
    fake = FakeProvider()
    msgs = [{"role": "user", "content": question}]
    alpha, beta, gamma = [(await fake.complete(n, msgs)).text for n in ("alpha", "beta", "gamma")]
    unknown_1 = "A human wrote this answer."
    unknown_2 = f"Unregistered author (digest {answer_digest('never-seen-xyz', question)})"
    bodies = [unknown_1, alpha, unknown_2, gamma, beta]

    beta_view = await _review_bodies("beta", question, bodies)
    assert sorted(beta_view[:2]) == sorted([beta, gamma])  # identified non-alpha first
    assert sorted(beta_view[2:4]) == sorted([unknown_1, unknown_2])  # then unknown
    assert beta_view[4] == alpha  # alpha still last

    alpha_view = await _review_bodies("alpha", question, bodies)
    assert alpha_view[0] == alpha
    assert alpha_view[1:3] == beta_view[:2]
    assert alpha_view[3:] == beta_view[2:4]

    # Deterministic and independent of label positions.
    assert await _review_bodies("beta", question, list(reversed(bodies))) == beta_view


# ------------------------------------------------------------------- openrouter


@respx.mock
async def test_openrouter_request_and_response(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    route = respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, json=_chat_response("Four.", cost=0.00042))
    )
    completion = await OpenRouterProvider().complete("openai/gpt-5.1", MESSAGES, max_tokens=256)

    assert completion.text == "Four."
    assert completion.input_tokens == 12
    assert completion.output_tokens == 3
    assert completion.cost_usd == pytest.approx(0.00042)

    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer sk-or-test"
    assert request.headers["HTTP-Referer"] == "https://github.com/SelmanBoray/council-of-models"
    assert request.headers["X-Title"] == "council-of-models"
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-5.1"
    assert body["messages"] == MESSAGES
    assert body["max_tokens"] == 256
    assert "temperature" not in body


@respx.mock
async def test_openrouter_passes_temperature_and_tolerates_missing_cost(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    route = respx.post(OPENROUTER_URL).mock(return_value=httpx.Response(200, json=_chat_response()))
    completion = await OpenRouterProvider().complete("x-ai/grok-4", MESSAGES, temperature=0.3)
    assert json.loads(route.calls.last.request.content)["temperature"] == 0.3
    assert completion.cost_usd is None


@respx.mock
async def test_openrouter_http_error_raises_provider_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
    )
    with pytest.raises(ProviderError, match="HTTP 429") as excinfo:
        await OpenRouterProvider().complete("openai/gpt-5.1", MESSAGES)
    assert excinfo.value.status_code == 429
    assert "Rate limit exceeded" in str(excinfo.value)


@respx.mock
async def test_openrouter_error_body_with_200_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    respx.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, json={"error": {"message": "Provider returned error"}})
    )
    with pytest.raises(ProviderError, match="Provider returned error"):
        await OpenRouterProvider().complete("openai/gpt-5.1", MESSAGES)


@respx.mock
async def test_openrouter_empty_content_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    payload = _chat_response()
    payload["choices"][0]["message"]["content"] = None
    payload["choices"][0]["finish_reason"] = "length"
    respx.post(OPENROUTER_URL).mock(return_value=httpx.Response(200, json=payload))
    with pytest.raises(ProviderError, match="empty answer.*length"):
        await OpenRouterProvider().complete("openai/gpt-5.1", MESSAGES)


@respx.mock(assert_all_called=False)
async def test_openrouter_missing_key_fails_at_call_time_without_network(respx_mock):
    route = respx_mock.post(OPENROUTER_URL).mock(
        return_value=httpx.Response(200, json=_chat_response())
    )
    provider = OpenRouterProvider()  # constructing must not fail
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        await provider.complete("openai/gpt-5.1", MESSAGES)
    assert not route.called


@respx.mock
async def test_openrouter_timeout_raises_provider_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    respx.post(OPENROUTER_URL).mock(side_effect=httpx.ReadTimeout)
    with pytest.raises(ProviderError, match="timed out"):
        await OpenRouterProvider().complete("openai/gpt-5.1", MESSAGES)


# ------------------------------------------------------------------ openai / ollama


@respx.mock
async def test_openai_uses_max_completion_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=_chat_response()))
    await OpenAIProvider().complete("gpt-5.1", MESSAGES, max_tokens=100)
    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["max_completion_tokens"] == 100 and "max_tokens" not in body
    assert request.headers["Authorization"] == "Bearer sk-test"


@respx.mock
async def test_ollama_default_host_needs_no_key():
    route = respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=_chat_response("4")))
    completion = await OllamaProvider().complete("llama3.1", MESSAGES)
    assert completion.text == "4"
    assert "Authorization" not in route.calls.last.request.headers
    assert json.loads(route.calls.last.request.content)["model"] == "llama3.1"


@respx.mock
async def test_ollama_host_from_env_at_call_time(monkeypatch):
    provider = OllamaProvider()
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box:11500")
    route = respx.post("http://gpu-box:11500/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response())
    )
    await provider.complete("qwen3:32b", MESSAGES)
    assert route.called


@respx.mock
async def test_ollama_connection_error_has_hint():
    respx.post(OLLAMA_URL).mock(side_effect=httpx.ConnectError)
    with pytest.raises(ProviderError, match="is Ollama running"):
        await OllamaProvider().complete("llama3.1", MESSAGES)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        (None, "http://localhost:11434/v1"),
        ("", "http://localhost:11434/v1"),
        ("localhost:11434", "http://localhost:11434/v1"),
        ("http://10.0.0.5:11434/", "http://10.0.0.5:11434/v1"),
        ("https://ollama.example.com/v1", "https://ollama.example.com/v1"),
    ],
)
def test_ollama_base_url(host, expected):
    assert ollama_base_url(host) == expected


# --------------------------------------------------------------------- anthropic


def test_split_messages_moves_system_to_kwarg():
    system, conversation = split_messages(
        [
            {"role": "system", "content": "Be terse."},
            {"role": "system", "content": "Answer in English."},
            {"role": "user", "content": "Hi"},
        ]
    )
    assert system == "Be terse.\n\nAnswer in English."
    assert conversation == [{"role": "user", "content": "Hi"}]


def test_split_messages_without_system():
    system, conversation = split_messages(MESSAGES)
    assert system is None and conversation == MESSAGES


@pytest.mark.parametrize(
    "messages",
    [[{"role": "system", "content": "only system"}], [{"role": "tool", "content": "x"}]],
)
def test_split_messages_rejects_invalid(messages):
    with pytest.raises(ProviderError):
        split_messages(messages)


async def test_anthropic_missing_key_errors_at_call_time_not_construction():
    provider = AnthropicProvider()  # must not raise without a key
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        await provider.complete("claude-opus-5-5", MESSAGES)


class _StubMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _stub_provider(response) -> tuple[AnthropicProvider, _StubMessages]:
    provider = AnthropicProvider()
    messages = _StubMessages(response)
    provider._get_client = lambda: SimpleNamespace(messages=messages)  # no network
    return provider, messages


def _message(blocks, stop_reason="end_turn"):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(input_tokens=20, output_tokens=7),
    )


async def test_anthropic_extracts_text_blocks_and_usage():
    response = _message(
        [
            SimpleNamespace(type="thinking", thinking="hmm"),
            SimpleNamespace(type="text", text="Four"),
            SimpleNamespace(type="text", text="."),
        ]
    )
    provider, stub = _stub_provider(response)
    completion = await provider.complete(
        "claude-opus-5-5",
        [{"role": "system", "content": "Be terse."}, *MESSAGES],
        max_tokens=512,
    )
    assert completion.text == "Four."
    assert (completion.input_tokens, completion.output_tokens) == (20, 7)
    assert completion.cost_usd is None
    assert stub.kwargs["system"] == "Be terse."
    assert stub.kwargs["messages"] == MESSAGES
    assert stub.kwargs["max_tokens"] == 512
    assert "temperature" not in stub.kwargs and "extra_body" not in stub.kwargs


async def test_anthropic_temperature_goes_to_extra_body():
    provider, stub = _stub_provider(_message([SimpleNamespace(type="text", text="ok")]))
    await provider.complete("claude-haiku-4-5", MESSAGES, temperature=0.2)
    assert stub.kwargs["extra_body"] == {"temperature": 0.2}
    assert "system" not in stub.kwargs


async def test_anthropic_refusal_raises():
    provider, _ = _stub_provider(_message([], stop_reason="refusal"))
    with pytest.raises(ProviderError, match="refused"):
        await provider.complete("claude-opus-5-5", MESSAGES)
