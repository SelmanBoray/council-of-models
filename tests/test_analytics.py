from __future__ import annotations

import json

import pytest

from council.analytics import (
    agreement_matrix,
    council_top,
    human_vs_council,
    kendall_tau,
    leaderboard,
    self_preference,
    summary,
)
from council.council import Council
from council.models import Review

ALPHA, BETA, GAMMA = "fake:alpha", "fake:beta", "fake:gamma"
SUMMARY_KEYS = {
    "n_runs",
    "total_cost_usd",
    "leaderboard",
    "self_preference",
    "agreement_matrix",
    "human_vs_council",
}


# --------------------------------------------------------------------- kendall_tau


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (["x", "y", "z"], ["x", "y", "z"], 1.0),
        (["x", "y", "z"], ["z", "y", "x"], -1.0),
        (["x", "y", "z"], ["x", "z", "y"], 1 / 3),  # C=2, D=1 over 3 pairs
        (["x", "y", "z", "w"], ["y", "x", "z", "w"], 4 / 6),  # one swapped pair of 6
        (["x", "y", "z"], ["y", "x"], -1.0),  # only the shared models count
        (["x", "y"], ["x", "q"], None),  # fewer than two shared models
        ([], [], None),
    ],
)
def test_kendall_tau(a, b, expected):
    result = kendall_tau(a, b)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)
        assert kendall_tau(b, a) == pytest.approx(expected)


# --------------------------------------------------------------------- leaderboard


def test_leaderboard_mean_rank_wins_and_appearances(make_run):
    run1 = make_run(
        id="r1",
        rankings=[[BETA, ALPHA, GAMMA], [BETA, GAMMA, ALPHA], [ALPHA, BETA, GAMMA]],
    )
    run2 = make_run(
        id="r2",
        rankings=[[GAMMA, BETA, ALPHA], [GAMMA, ALPHA, BETA], []],
    )
    board = leaderboard([run1, run2])
    by_model = {row["model"]: row for row in board}

    # beta: 1, 1, 2 | 2, 3 -> 9 / 5
    assert by_model[BETA] == {"model": BETA, "mean_rank": 1.8, "wins": 1, "appearances": 5}
    # gamma: 3, 2, 3 | 1, 1 -> 10 / 5
    assert by_model[GAMMA] == {"model": GAMMA, "mean_rank": 2.0, "wins": 1, "appearances": 5}
    # alpha: 2, 3, 1 | 3, 2 -> 11 / 5
    assert by_model[ALPHA] == {"model": ALPHA, "mean_rank": 2.2, "wins": 0, "appearances": 5}
    assert [row["model"] for row in board] == [BETA, GAMMA, ALPHA]


def test_leaderboard_ties_sorted_by_model_and_rounded(make_run):
    run = make_run(rankings=[[GAMMA, ALPHA, BETA], [ALPHA, GAMMA, BETA], [BETA, ALPHA, GAMMA]])
    board = leaderboard([run])
    # alpha (2+1+2)/3, gamma (1+2+3)/3, beta (3+3+1)/3
    assert [(row["model"], row["mean_rank"]) for row in board] == [
        (ALPHA, 1.6667),
        (GAMMA, 2.0),
        (BETA, 2.3333),
    ]
    tied = make_run(rankings=[[GAMMA, BETA, ALPHA], [BETA, GAMMA, ALPHA]])
    assert [row["model"] for row in leaderboard([tied])] == [BETA, GAMMA, ALPHA]


def test_leaderboard_ignores_runs_without_rankings(make_run):
    failed = make_run(rankings=[[], [], []])
    no_reviews = make_run(rankings=[])
    assert leaderboard([failed, no_reviews]) == []
    assert leaderboard([]) == []


# ----------------------------------------------------------------- self_preference


def test_self_preference_hand_computed(make_run):
    # reviewers are alpha, beta, gamma in that order (see conftest.make_run)
    run1 = make_run(
        rankings=[[ALPHA, BETA, GAMMA], [GAMMA, BETA, ALPHA], [GAMMA, ALPHA, BETA]],
    )
    run2 = make_run(
        rankings=[[BETA, ALPHA, GAMMA], [BETA, ALPHA, GAMMA], [GAMMA, BETA, ALPHA]],
    )
    prefs = self_preference([run1, run2])
    # alpha: run1 others (3, 2) -> 2.5 - 1 = 1.5 ; run2 others (2, 3) -> 2.5 - 2 = 0.5
    assert prefs[ALPHA] == pytest.approx(1.0)
    # beta: run1 others (2, 3) -> 2.5 - 2 = 0.5 ; run2 others (1, 2) -> 1.5 - 1 = 0.5
    assert prefs[BETA] == pytest.approx(0.5)
    # gamma: run1 others (3, 1) -> 2 - 1 = 1 ; run2 others (3, 3) -> 3 - 1 = 2
    assert prefs[GAMMA] == pytest.approx(1.5)
    assert list(prefs) == [GAMMA, ALPHA, BETA]  # most self-favoring first


def test_self_preference_requires_own_and_peer_rankings(make_run):
    # gamma's review failed; only alpha ranked, beta's ranking is empty too
    lonely = make_run(rankings=[[ALPHA, BETA, GAMMA], [], []])
    assert self_preference([lonely]) == {}
    # beta ranks, gamma does not: alpha and beta each have one peer ranking
    pair = make_run(rankings=[[ALPHA, BETA, GAMMA], [BETA, ALPHA, GAMMA], []])
    assert self_preference([pair]) == {ALPHA: 1.0, BETA: 1.0}


def test_self_preference_negative_when_self_critical(make_run):
    run = make_run(rankings=[[BETA, GAMMA, ALPHA], [BETA, ALPHA, GAMMA], [BETA, ALPHA, GAMMA]])
    prefs = self_preference([run])
    assert prefs[ALPHA] == pytest.approx(2 - 3)  # peers say 2, alpha says 3
    assert prefs[BETA] == 0.0
    assert prefs[GAMMA] == pytest.approx(2.5 - 3)  # peers say 2 and 3, gamma says 3


def test_self_preference_rounds_to_four_decimals(make_run):
    run1 = make_run(rankings=[[ALPHA, BETA, GAMMA], [BETA, ALPHA, GAMMA], [BETA, ALPHA, GAMMA]])
    run2 = make_run(rankings=[[ALPHA, BETA, GAMMA], [ALPHA, BETA, GAMMA], [ALPHA, BETA, GAMMA]])
    run3 = make_run(rankings=[[ALPHA, BETA, GAMMA], [ALPHA, BETA, GAMMA], [ALPHA, BETA, GAMMA]])
    # alpha: 1, 0, 0 -> 1/3
    assert self_preference([run1, run2, run3])[ALPHA] == 0.3333


# ---------------------------------------------------------------- agreement_matrix


def test_agreement_matrix_values_diagonal_and_symmetry(make_run):
    run1 = make_run(rankings=[[ALPHA, BETA, GAMMA], [ALPHA, BETA, GAMMA], [GAMMA, BETA, ALPHA]])
    run2 = make_run(rankings=[[ALPHA, BETA, GAMMA], [ALPHA, GAMMA, BETA], [GAMMA, BETA, ALPHA]])
    matrix = agreement_matrix([run1, run2])

    assert list(matrix) == [ALPHA, BETA, GAMMA]
    for reviewer in matrix:
        assert matrix[reviewer][reviewer] == 1.0
        for other, value in matrix[reviewer].items():
            assert matrix[other][reviewer] == value
    # alpha-beta: run1 1.0, run2 1/3 -> 2/3
    assert matrix[ALPHA][BETA] == 0.6667
    # alpha-gamma: -1 in both runs
    assert matrix[ALPHA][GAMMA] == -1.0
    # beta-gamma: run1 -1.0; run2 [a,g,b] vs [g,b,a]: (a,g) D, (a,b) D, (g,b) C -> -1/3
    assert matrix[BETA][GAMMA] == pytest.approx(round((-1 + (-1 / 3)) / 2, 4))


def test_agreement_matrix_skips_failed_reviews_and_missing_pairs(make_run):
    run1 = make_run(rankings=[[ALPHA, BETA, GAMMA], [ALPHA, BETA, GAMMA], []])
    run2 = make_run(rankings=[[ALPHA, BETA, GAMMA], [], [GAMMA, BETA, ALPHA]])
    run3 = make_run(rankings=[[], [ALPHA, GAMMA, BETA], []])
    matrix = agreement_matrix([run1, run2, run3])
    assert matrix == {
        ALPHA: {ALPHA: 1.0, BETA: 1.0, GAMMA: -1.0},
        BETA: {ALPHA: 1.0, BETA: 1.0},
        GAMMA: {ALPHA: -1.0, GAMMA: 1.0},
    }
    assert agreement_matrix([]) == {}


# ---------------------------------------------------------------- human_vs_council


def test_human_vs_council(make_run):
    rankings = [[BETA, ALPHA, GAMMA], [BETA, GAMMA, ALPHA], [ALPHA, BETA, GAMMA]]
    agree = make_run(id="a", rankings=rankings, human_vote=BETA)
    disagree = make_run(id="b", rankings=rankings, human_vote=GAMMA)
    disagree2 = make_run(id="c", rankings=rankings, human_vote=GAMMA)
    no_vote = make_run(id="d", rankings=rankings)
    unranked = make_run(id="e", rankings=[[], [], []], human_vote=ALPHA)

    result = human_vs_council([agree, disagree, disagree2, no_vote, unranked])
    assert result == {
        "n": 3,
        "agreement_rate": 0.3333,
        "per_model_human_wins": {GAMMA: 2, BETA: 1},
    }
    assert human_vs_council([no_vote, unranked]) == {
        "n": 0,
        "agreement_rate": None,
        "per_model_human_wins": {},
    }


def test_council_top_ignores_runs_without_rankings(make_run):
    assert council_top(make_run(rankings=[[], [], []])) is None
    assert council_top(make_run(rankings=[[GAMMA, BETA, ALPHA]])) == GAMMA


# -------------------------------------------------------------------------- summary


def test_summary_shape_and_cost(make_run):
    runs = [
        make_run(id="a", total_cost_usd=0.25, human_vote=ALPHA),
        make_run(id="b", total_cost_usd=None),
        make_run(id="c", total_cost_usd=0.5),
    ]
    result = summary(runs)
    assert set(result) == SUMMARY_KEYS
    assert result["n_runs"] == 3
    assert result["total_cost_usd"] == pytest.approx(0.75)
    assert result["leaderboard"] == leaderboard(runs)
    assert result["self_preference"] == self_preference(runs)
    assert result["agreement_matrix"] == agreement_matrix(runs)
    assert result["human_vs_council"] == human_vs_council(runs)
    json.dumps(result)  # must be JSON-serializable for the web API / MCP server


def test_summary_empty_and_unknown_cost(make_run):
    assert summary([]) == {
        "n_runs": 0,
        "total_cost_usd": None,
        "leaderboard": [],
        "self_preference": {},
        "agreement_matrix": {},
        "human_vs_council": {"n": 0, "agreement_rate": None, "per_model_human_wins": {}},
    }
    assert summary([make_run(total_cost_usd=None)])["total_cost_usd"] is None


def test_summary_accepts_generators(make_run):
    runs = [make_run(id="a"), make_run(id="b")]
    assert summary(run for run in runs) == summary(runs)


def test_failed_review_with_error_is_ignored(make_run):
    run = make_run(rankings=[[ALPHA, BETA, GAMMA], [BETA, ALPHA, GAMMA]])
    run.reviews.append(
        Review(reviewer=GAMMA, ranking=[], rationale="", raw="???", error="unparseable")
    )
    assert {row["model"]: row["appearances"] for row in leaderboard([run])} == {
        ALPHA: 2,
        BETA: 2,
        GAMMA: 2,
    }
    assert GAMMA not in self_preference([run])
    assert GAMMA not in agreement_matrix([run])


# ------------------------------------------------------- end to end with the fake council

QUESTIONS = [
    "Is a hot dog a sandwich?",
    "What is the capital of Australia?",
    "Should I use tabs or spaces?",
    "Explain Kendall tau in one sentence.",
    "Why is the sky blue?",
]


async def test_fake_council_bias_signal(fake_config):
    council = Council(fake_config)
    runs = [await council.ask(question) for question in QUESTIONS]

    for run in runs:  # alpha favors itself by +2.0 in every single run
        assert self_preference([run])[ALPHA] == 2.0

    prefs = self_preference(runs)
    assert prefs[ALPHA] == 2.0
    for member in (BETA, GAMMA):
        assert 0.0 <= prefs[member] <= 1.0
    assert list(prefs)[0] == ALPHA

    matrix = agreement_matrix(runs)
    assert matrix[BETA][GAMMA] == 1.0  # the unbiased fakes share one quality order
    assert matrix[ALPHA][BETA] == pytest.approx(-0.3333)  # alpha: self first, peers: alpha last

    board = leaderboard(runs)
    by_model = {row["model"]: row for row in board}
    assert by_model[ALPHA]["mean_rank"] == 2.3333  # (1 + 3 + 3) / 3 every run
    assert by_model[ALPHA]["wins"] == 0  # peers always rank alpha last
    assert board[0]["model"] != ALPHA
    assert sum(row["wins"] for row in board) == len(QUESTIONS)
    assert all(row["appearances"] == 3 * len(QUESTIONS) for row in board)

    result = summary(runs)
    assert result["n_runs"] == len(QUESTIONS)
    assert result["total_cost_usd"] == 0.0
    assert result["human_vs_council"]["n"] == 0
