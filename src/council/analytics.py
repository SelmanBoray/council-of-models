"""Judge-bias analytics over stored council runs.

Every function is pure (no I/O) and works on any iterable of :class:`CouncilRun`.
Only reviews with a non-empty ``ranking`` count; failed or unparseable reviews are
ignored everywhere. Ranks are 1-based (1 = best).

* :func:`leaderboard` - how the council rated each model overall.
* :func:`self_preference` - does a model rank its own answer better than its peers do?
* :func:`agreement_matrix` - how similarly do two reviewers order the answers (Kendall tau)?
* :func:`human_vs_council` - how often does the human's pick match the council's #1?
* :func:`summary` - all of the above in one JSON-serializable dict (consumed by the web
  frontend and the MCP server, so its keys are part of the public contract).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import Any

from council.models import CouncilRun

_DIGITS = 4


def _rankings(run: CouncilRun) -> list[tuple[str, list[str]]]:
    """``(reviewer, ranking)`` for every review of ``run`` with a non-empty ranking.

    If a reviewer somehow appears twice, only its first non-empty review counts.
    """
    seen: set[str] = set()
    result: list[tuple[str, list[str]]] = []
    for review in run.reviews:
        if review.ranking and review.reviewer not in seen:
            seen.add(review.reviewer)
            result.append((review.reviewer, list(review.ranking)))
    return result


def _positions(ranking: Sequence[str]) -> dict[str, int]:
    """Model -> 1-based position (first occurrence wins if a model is repeated)."""
    positions: dict[str, int] = {}
    for position, model in enumerate(ranking, start=1):
        positions.setdefault(model, position)
    return positions


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def council_top(run: CouncilRun) -> str | None:
    """The council's #1 for ``run``, or ``None`` if no review produced a ranking.

    :meth:`CouncilRun.council_ranking` also lists never-ranked members (alphabetically),
    so it is non-empty even when every review failed; this helper only trusts it when
    at least one real ranking exists.
    """
    if not _rankings(run):
        return None
    ranking = run.council_ranking()
    return ranking[0] if ranking else None


def kendall_tau(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Kendall tau between two rankings, over the models both of them contain.

    Rankings are strict orders (no ties), so tau-b reduces to
    ``(concordant - discordant) / (n * (n - 1) / 2)``. Returns ``None`` when fewer
    than two models are shared.
    """
    pos_a, pos_b = _positions(a), _positions(b)
    common = [model for model in pos_a if model in pos_b]
    n = len(common)
    if n < 2:
        return None
    concordant = discordant = 0
    for x, y in combinations(common, 2):
        sign = (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y])
        if sign > 0:
            concordant += 1
        elif sign < 0:
            discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)


def leaderboard(runs: Iterable[CouncilRun]) -> list[dict[str, Any]]:
    """Per-model ``{model, mean_rank, wins, appearances}``, best mean rank first.

    ``mean_rank`` is the mean 1-based position over every review that ranked the
    model, ``appearances`` the number of such reviews, and ``wins`` the number of runs
    where the model is the council's aggregate #1. Ties are broken by model id.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    wins: dict[str, int] = {}
    for run in runs:
        rankings = _rankings(run)
        if not rankings:
            continue
        for _reviewer, ranking in rankings:
            for model, position in _positions(ranking).items():
                totals[model] = totals.get(model, 0.0) + position
                counts[model] = counts.get(model, 0) + 1
        top = run.council_ranking()[0]
        wins[top] = wins.get(top, 0) + 1
    rows = [
        {
            "model": model,
            "mean_rank": round(totals[model] / counts[model], _DIGITS),
            "wins": wins.get(model, 0),
            "appearances": counts[model],
        }
        for model in totals
    ]
    rows.sort(key=lambda row: (row["mean_rank"], row["model"]))
    return rows


def self_preference(runs: Iterable[CouncilRun]) -> dict[str, float]:
    """Per reviewer: mean over runs of (mean rank others gave it) - (rank it gave itself).

    Positive means the model rates its own answer better than its peers do. A run
    counts for model M only if M produced a non-empty ranking that contains itself and
    at least one other reviewer's non-empty ranking contains M. Sorted from most to
    least self-favoring (ties by model id); values rounded to 4 decimals.
    """
    diffs: dict[str, list[float]] = {}
    for run in runs:
        rankings = _rankings(run)
        positions = {reviewer: _positions(ranking) for reviewer, ranking in rankings}
        for reviewer, own in positions.items():
            if reviewer not in own:
                continue
            others = [
                pos[reviewer]
                for other, pos in positions.items()
                if other != reviewer and reviewer in pos
            ]
            if not others:
                continue
            diffs.setdefault(reviewer, []).append(_mean(others) - own[reviewer])
    result = {model: round(_mean(values), _DIGITS) for model, values in diffs.items()}
    return dict(sorted(result.items(), key=lambda item: (-item[1], item[0])))


def agreement_matrix(runs: Iterable[CouncilRun]) -> dict[str, dict[str, float]]:
    """Reviewer x reviewer Kendall tau, averaged over the runs both of them reviewed.

    Symmetric, with ``1.0`` on the diagonal for every reviewer that produced at least
    one ranking. A pair that never shared a run (or shared fewer than two ranked
    models) has no entry. Keys are sorted; values rounded to 4 decimals.
    """
    reviewers: set[str] = set()
    taus: dict[tuple[str, str], list[float]] = {}
    for run in runs:
        rankings = sorted(_rankings(run))
        reviewers.update(reviewer for reviewer, _ in rankings)
        for (a, ranking_a), (b, ranking_b) in combinations(rankings, 2):
            tau = kendall_tau(ranking_a, ranking_b)
            if tau is not None:
                taus.setdefault((a, b), []).append(tau)
    matrix: dict[str, dict[str, float]] = {reviewer: {} for reviewer in sorted(reviewers)}
    for reviewer in matrix:
        matrix[reviewer][reviewer] = 1.0
    for (a, b), values in taus.items():
        mean = round(_mean(values), _DIGITS)
        matrix[a][b] = mean
        matrix[b][a] = mean
    return {reviewer: dict(sorted(row.items())) for reviewer, row in matrix.items()}


def human_vs_council(runs: Iterable[CouncilRun]) -> dict[str, Any]:
    """How often the human's vote matches the council's aggregate #1.

    Only runs with a ``human_vote`` and at least one non-empty review ranking count.
    Returns ``{"n", "agreement_rate", "per_model_human_wins"}``; ``agreement_rate`` is
    ``None`` when ``n == 0``. ``per_model_human_wins`` counts the human's votes per
    model over those runs (most votes first).
    """
    n = agreed = 0
    votes: dict[str, int] = {}
    for run in runs:
        if not run.human_vote:
            continue
        top = council_top(run)
        if top is None:
            continue
        n += 1
        agreed += run.human_vote == top
        votes[run.human_vote] = votes.get(run.human_vote, 0) + 1
    return {
        "n": n,
        "agreement_rate": round(agreed / n, _DIGITS) if n else None,
        "per_model_human_wins": dict(sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def summary(runs: Iterable[CouncilRun]) -> dict[str, Any]:
    """Everything above in one JSON-serializable dict.

    Keys: ``n_runs``, ``total_cost_usd`` (sum of known run costs, ``None`` if no run
    has a cost), ``leaderboard``, ``self_preference``, ``agreement_matrix``,
    ``human_vs_council``.
    """
    runs = list(runs)
    costs = [run.total_cost_usd for run in runs if run.total_cost_usd is not None]
    return {
        "n_runs": len(runs),
        "total_cost_usd": round(sum(costs), 8) if costs else None,
        "leaderboard": leaderboard(runs),
        "self_preference": self_preference(runs),
        "agreement_matrix": agreement_matrix(runs),
        "human_vs_council": human_vs_council(runs),
    }


__all__ = [
    "agreement_matrix",
    "council_top",
    "human_vs_council",
    "kendall_tau",
    "leaderboard",
    "self_preference",
    "summary",
]
