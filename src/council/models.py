"""Pydantic data models shared by every part of council-of-models.

These shapes are a public contract: the CLI, the MCP server, the web API and the
frontend all serialize/deserialize them. Change them only together with the contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field


class Completion(BaseModel):
    """Raw result of a single provider call."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class Answer(BaseModel):
    """One member's answer to the question (initial or revised during debate)."""

    model: str
    text: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None


class Review(BaseModel):
    """One member's ranking of all (anonymized) answers.

    ``ranking`` holds de-anonymized model ids, best -> worst, and includes the
    reviewer's own answer. It is empty when the review failed or was unparseable
    (``error`` is set in that case).
    """

    reviewer: str
    ranking: list[str]
    rationale: str
    raw: str
    error: str | None = None


class Synthesis(BaseModel):
    """The chairman's final answer."""

    chairman: str
    text: str
    latency_s: float
    cost_usd: float | None = None


class CouncilRun(BaseModel):
    """Everything that happened while the council answered one question."""

    id: str
    question: str
    created_at: str
    members: list[str]
    chairman: str
    answers: list[Answer]
    reviews: list[Review]
    synthesis: Synthesis | None
    debate_rounds: int = 0
    debate_answers: list[list[Answer]] = Field(default_factory=list)
    labels: dict[str, str]
    human_vote: str | None = None
    total_cost_usd: float | None = None

    def council_ranking(self) -> list[str]:
        """Aggregate ranking, best -> worst.

        Each model's score is its mean 1-based position over all reviews with a
        non-empty ranking. Ties are broken alphabetically. Members that were never
        ranked (e.g. their answer failed) come last, alphabetically.
        """
        return aggregate_ranking(
            (review.ranking for review in self.reviews), candidates=self.members
        )


class CouncilEvent(BaseModel):
    """Progress event emitted by :meth:`council.council.Council.ask`."""

    stage: Literal["answer", "debate", "review", "synthesis"]
    model: str
    status: Literal["start", "done", "error"]
    round: int = 0
    message: str | None = None


def mean_positions(rankings: Iterable[list[str]]) -> dict[str, float]:
    """Mean 1-based position of every model that appears in at least one ranking."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for ranking in rankings:
        if not ranking:
            continue
        for position, model in enumerate(ranking, start=1):
            totals[model] = totals.get(model, 0.0) + position
            counts[model] = counts.get(model, 0) + 1
    return {model: totals[model] / counts[model] for model in totals}


def aggregate_ranking(rankings: Iterable[list[str]], candidates: Iterable[str] = ()) -> list[str]:
    """Order models by mean position (ties alphabetical); unranked candidates go last."""
    means = mean_positions(rankings)
    ranked = sorted(means, key=lambda model: (means[model], model))
    unranked = sorted({c for c in candidates if c not in means})
    return ranked + unranked
