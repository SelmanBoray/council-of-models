from __future__ import annotations

import pytest
from pydantic import ValidationError

from council.models import (
    Answer,
    CouncilEvent,
    CouncilRun,
    Review,
    aggregate_ranking,
    mean_positions,
)


def test_council_ranking_orders_by_mean_position(make_run):
    run = make_run(
        members=["m:a", "m:b", "m:c"],
        rankings=[["m:a", "m:b", "m:c"], ["m:b", "m:a", "m:c"], ["m:a", "m:c", "m:b"]],
    )
    # means: a=(1+2+1)/3=1.33, b=(2+1+3)/3=2.0, c=(3+3+2)/3=2.67
    assert run.council_ranking() == ["m:a", "m:b", "m:c"]


def test_council_ranking_breaks_ties_alphabetically(make_run):
    run = make_run(
        members=["m:zeta", "m:alpha"],
        rankings=[["m:zeta", "m:alpha"], ["m:alpha", "m:zeta"]],
    )
    assert run.council_ranking() == ["m:alpha", "m:zeta"]


def test_council_ranking_ignores_empty_rankings(make_run):
    run = make_run(
        members=["m:a", "m:b"],
        rankings=[["m:b", "m:a"], []],
    )
    run.reviews[1].error = "could not parse"
    assert run.council_ranking() == ["m:b", "m:a"]


def test_council_ranking_puts_never_ranked_members_last_alphabetically(make_run):
    run = make_run(
        members=["m:x", "m:z", "m:w", "m:y"],
        rankings=[["m:y", "m:x"], ["m:y", "m:x"]],
    )
    assert run.council_ranking() == ["m:y", "m:x", "m:w", "m:z"]


def test_council_ranking_without_reviews_is_alphabetical(make_run):
    run = make_run(members=["m:c", "m:a", "m:b"], rankings=[])
    assert run.reviews == []
    assert run.council_ranking() == ["m:a", "m:b", "m:c"]


def test_council_ranking_uses_mean_over_appearances_only(make_run):
    # m:c appears in only one review, at position 1 -> mean 1.0 beats m:a (1.5).
    run = make_run(
        members=["m:a", "m:b", "m:c"],
        rankings=[["m:a", "m:b"], ["m:c", "m:a", "m:b"]],
    )
    assert mean_positions(r.ranking for r in run.reviews) == {
        "m:a": 1.5,
        "m:b": 2.5,
        "m:c": 1.0,
    }
    assert run.council_ranking() == ["m:c", "m:a", "m:b"]


def test_aggregate_ranking_helper_includes_candidates():
    assert aggregate_ranking([["x", "y"]], candidates=["y", "q", "x"]) == ["x", "y", "q"]


def test_run_json_round_trip(make_run):
    run = make_run(
        debate_rounds=1, debate_answers=[[Answer(model="fake:alpha", text="r", latency_s=0)]]
    )
    run.human_vote = "fake:beta"
    run.total_cost_usd = 0.0123
    restored = CouncilRun.model_validate_json(run.model_dump_json(indent=2))
    assert restored == run
    assert restored.council_ranking() == run.council_ranking()


def test_run_defaults():
    run = CouncilRun(
        id="abc",
        question="q",
        created_at="2026-01-01T00:00:00Z",
        members=["fake:a"],
        chairman="fake:a",
        answers=[],
        reviews=[],
        synthesis=None,
        labels={},
    )
    assert run.debate_rounds == 0
    assert run.debate_answers == []
    assert run.human_vote is None
    assert run.total_cost_usd is None


def test_review_and_answer_optional_fields():
    review = Review(reviewer="fake:a", ranking=[], rationale="", raw="")
    answer = Answer(model="fake:a", text="hi", latency_s=0.5)
    assert review.error is None
    assert answer.error is None and answer.cost_usd is None


def test_council_event_validation():
    event = CouncilEvent(stage="answer", model="fake:alpha", status="start")
    assert event.round == 0 and event.message is None
    with pytest.raises(ValidationError):
        CouncilEvent(stage="voting", model="fake:alpha", status="start")
    with pytest.raises(ValidationError):
        CouncilEvent(stage="answer", model="fake:alpha", status="pending")
