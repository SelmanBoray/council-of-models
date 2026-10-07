"""Shared fixtures. Every test is offline: provider env vars are cleared for each test."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from council.config import CouncilConfig
from council.models import Answer, CouncilRun, Review, Synthesis

FAKE_MEMBERS = ["fake:alpha", "fake:beta", "fake:gamma"]
PROVIDER_ENV_VARS = (
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "OLLAMA_HOST",
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a developer's real API keys leak into tests."""
    for name in PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_config(tmp_path: Path) -> CouncilConfig:
    return CouncilConfig(
        members=list(FAKE_MEMBERS), chairman="fake:alpha", data_dir=tmp_path / "data"
    )


@pytest.fixture
def make_run() -> Callable[..., CouncilRun]:
    """Factory for hand-built runs: ``make_run(rankings=[[...], ...], **overrides)``."""

    def factory(
        *,
        rankings: list[list[str]] | None = None,
        members: list[str] | None = None,
        **overrides: Any,
    ) -> CouncilRun:
        members = list(members or FAKE_MEMBERS)
        rankings = rankings if rankings is not None else [list(members) for _ in members]
        reviews = [
            Review(
                reviewer=members[i % len(members)],
                ranking=ranking,
                rationale="because",
                raw="because\nRANKING: ...",
            )
            for i, ranking in enumerate(rankings)
        ]
        fields: dict[str, Any] = {
            "id": "run000000001",
            "question": "What is 2+2?",
            "created_at": "2026-10-07T12:00:00.000Z",
            "members": members,
            "chairman": members[0],
            "answers": [Answer(model=m, text=f"answer from {m}", latency_s=0.1) for m in members],
            "reviews": reviews,
            "synthesis": Synthesis(chairman=members[0], text="4", latency_s=0.2),
            "labels": {f"Response {chr(65 + i)}": m for i, m in enumerate(members)},
        }
        fields.update(overrides)
        return CouncilRun(**fields)

    return factory
