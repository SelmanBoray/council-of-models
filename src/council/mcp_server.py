"""MCP server: lets Claude Code, Cursor and other MCP clients "ask the council".

Run it over stdio with the ``council-mcp`` entry point (or ``python -m
council.mcp_server``). Every tool returns a JSON string; expected failures (bad
config, unknown run id, invalid arguments, provider errors) come back as
``{"error": "..."}`` instead of crashing the server.

Environment variables:

* ``COUNCIL_CONFIG`` - path to a ``council.toml``; otherwise the usual lookup of
  :func:`council.config.load_config` applies (``./council.toml`` relative to the
  server's working directory, then ``~/.config/council/config.toml``, then defaults).
* ``COUNCIL_FAKE=1`` - offline mode: ``ask_council`` uses the deterministic fake
  members ``fake:alpha, fake:beta, fake:gamma`` with ``fake:alpha`` as chairman.
* Provider keys (``OPENROUTER_API_KEY``, ``ANTHROPIC_API_KEY``, ...) as for the CLI;
  :func:`main` also loads a ``.env`` file from the working directory without
  overriding variables that are already set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise ImportError(
        "The MCP server needs the 'mcp' extra: pip install 'council-of-models[mcp]'"
    ) from exc

from council.config import CouncilConfig, load_config, load_dotenv_file
from council.council import Council
from council.errors import CouncilError
from council.models import Answer, CouncilRun
from council.store import RunStore

CONFIG_ENV = "COUNCIL_CONFIG"
FAKE_ENV = "COUNCIL_FAKE"
FAKE_MEMBERS = ["fake:alpha", "fake:beta", "fake:gamma"]
FAKE_CHAIRMAN = "fake:alpha"
ANSWER_PREVIEW_CHARS = 500
ELLIPSIS = "…"
MAX_DEBATE_ROUNDS = 5
MAX_HISTORY = 200

INSTRUCTIONS = """\
council-of-models asks a council of several frontier LLMs the same question. Every
member answers, then each member anonymously ranks all answers (its own included),
and a chairman model synthesizes one final answer that takes the peer reviews into
account.

Use ask_council for hard, ambiguous or high-stakes questions where a second opinion
from several different models is worth it: design trade-offs, tricky debugging
hypotheses, factual questions where models often disagree, reviewing a plan. Do not
use it for simple lookups or anything you can answer confidently yourself: every
call makes several paid model calls (one answer and one review per member, plus the
synthesis, plus one extra call per member for each debate round) and can take a
minute or more. Include all necessary context in the question itself; the council
cannot see the user's files or this conversation.

council_history and council_stats are free and read-only (past runs, leaderboard,
judge-bias analytics). council_vote records which member's answer the human
preferred; use it only when the user explicitly states a preference.
"""

mcp = FastMCP("council-of-models", instructions=INSTRUCTIONS)


# --------------------------------------------------------------------------- helpers


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _error(message: str) -> str:
    return _dumps({"error": message})


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _truncate(text: str, limit: int = ANSWER_PREVIEW_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + ELLIPSIS


def _load_config() -> CouncilConfig:
    """Config from ``$COUNCIL_CONFIG`` (or the default lookup), with the fake override."""
    raw_path = os.environ.get(CONFIG_ENV, "").strip()
    config = load_config(Path(raw_path) if raw_path else None)
    if _truthy(os.environ.get(FAKE_ENV)):
        config = config.model_copy(
            update={"members": list(FAKE_MEMBERS), "chairman": FAKE_CHAIRMAN}
        )
    return config


def _store(config: CouncilConfig) -> RunStore:
    return RunStore(config.data_dir)


def _answer_view(answer: Answer) -> dict[str, Any]:
    return {"model": answer.model, "text": _truncate(answer.text), "error": answer.error}


def run_summary(run: CouncilRun) -> dict[str, Any]:
    """Compact, JSON-ready view of a run for MCP clients (answers truncated)."""
    summary: dict[str, Any] = {
        "run_id": run.id,
        "synthesis": run.synthesis.text if run.synthesis is not None else None,
        "chairman": run.chairman,
        "council_ranking": run.council_ranking(),
        "answers": [_answer_view(answer) for answer in run.answers],
        "reviews": [
            {"reviewer": review.reviewer, "ranking": review.ranking, "error": review.error}
            for review in run.reviews
        ],
        "total_cost_usd": run.total_cost_usd,
    }
    if run.debate_answers:
        # The reviews ranked these revised answers, not the initial ones.
        summary["final_answers"] = [_answer_view(a) for a in run.debate_answers[-1]]
    return summary


def history_item(run: CouncilRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "question": run.question,
        "created_at": run.created_at,
        "council_ranking": run.council_ranking(),
        "human_vote": run.human_vote,
    }


# ----------------------------------------------------------------------------- tools


@mcp.tool(
    title="Ask the council",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
    ),
    structured_output=False,
)
async def ask_council(
    question: Annotated[
        str,
        Field(
            description=(
                "The full, self-contained question. Include all context the models need; "
                "they cannot see your files or conversation."
            )
        ),
    ],
    debate_rounds: Annotated[
        int,
        Field(
            description=(
                "Optional rounds (0-5) in which members read each other's answers and "
                "revise their own before ranking. Each round costs one extra call per "
                "member. Default 0."
            )
        ),
    ] = 0,
) -> str:
    """Ask a council of several LLMs and get a peer-reviewed, synthesized answer.

    Every council member answers the question independently, each member then ranks
    all answers anonymously (its own included), and a chairman model writes the
    final answer using the answers and the reviews. The run is saved to the local
    history.

    This costs money (several paid model calls) and can take a minute or more, so
    use it for hard or ambiguous questions where multiple independent opinions help.

    Returns a JSON object: run_id, synthesis (the chairman's final answer; null if
    the chairman failed), chairman, council_ranking (model ids, best first),
    answers (per model: model, text truncated to 500 characters, error), reviews
    (per reviewer: reviewer, ranking best first, error), total_cost_usd, and
    final_answers when debate rounds were used. On failure: {"error": "..."}.
    """
    if not isinstance(debate_rounds, int) or not 0 <= debate_rounds <= MAX_DEBATE_ROUNDS:
        return _error(f"debate_rounds must be an integer from 0 to {MAX_DEBATE_ROUNDS}")
    try:
        config = _load_config().model_copy(update={"debate_rounds": debate_rounds})
        run = await Council(config).ask(question)
    except (CouncilError, ValueError, OSError) as exc:
        return _error(str(exc))
    result = run_summary(run)
    try:
        _store(config).save(run)
    except (OSError, ValueError) as exc:
        result["save_error"] = f"the run could not be saved: {exc}"
    return _dumps(result)


@mcp.tool(
    title="Council statistics",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    structured_output=False,
)
async def council_stats() -> str:
    """Aggregate statistics over all saved council runs (free, read-only, no model calls).

    Returns a JSON object with n_runs, total_cost_usd, leaderboard (per model: mean
    rank given by its peers, wins, appearances; best first), self_preference (per
    model: how much better it ranks its own answer than its peers do; positive means
    it favors itself), agreement_matrix (reviewer x reviewer Kendall tau of their
    rankings) and human_vs_council (how often the human's vote matched the council's
    top pick). On failure: {"error": "..."}.
    """
    try:
        runs = _store(_load_config()).list()
    except (CouncilError, ValueError, OSError) as exc:
        return _error(str(exc))
    try:
        from council.analytics import summary
    except ImportError as exc:
        return _error(f"analytics are not available: {exc}")
    try:
        return _dumps(summary(runs))
    except (CouncilError, ValueError) as exc:
        return _error(str(exc))


@mcp.tool(
    title="Council history",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    structured_output=False,
)
async def council_history(
    limit: Annotated[
        int, Field(description=f"Maximum number of runs to return (1-{MAX_HISTORY}).")
    ] = 10,
) -> str:
    """List the most recent saved council runs, newest first (free, read-only).

    Returns a JSON list of {id, question, created_at, council_ranking, human_vote}.
    Use an id with council_vote. On failure: {"error": "..."}.
    """
    if not isinstance(limit, int) or not 1 <= limit <= MAX_HISTORY:
        return _error(f"limit must be an integer from 1 to {MAX_HISTORY}")
    try:
        runs = _store(_load_config()).list(limit=limit)
    except (CouncilError, ValueError, OSError) as exc:
        return _error(str(exc))
    return _dumps([history_item(run) for run in runs])


@mcp.tool(
    title="Record human vote",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    structured_output=False,
)
async def council_vote(
    run_id: Annotated[str, Field(description="Run id from ask_council or council_history.")],
    model: Annotated[
        str,
        Field(description="Model id of the member whose answer the human preferred."),
    ],
) -> str:
    """Record which council member's answer the human judged best for a saved run.

    Only call this when the user explicitly says which answer they preferred. Votes
    feed the human-vs-council agreement in council_stats. Voting again replaces the
    previous vote. Returns {run_id, human_vote}; on failure: {"error": "..."} (for
    example an unknown run id or a model that was not a member of that run).
    """
    try:
        store = _store(_load_config())
        run = store.set_human_vote(run_id.strip(), model.strip())
    except KeyError:
        return _error(f"run not found: {run_id}")
    except (CouncilError, ValueError, OSError) as exc:
        return _error(str(exc))
    return _dumps({"run_id": run.id, "human_vote": run.human_vote})


def main() -> None:
    """Entry point (``council-mcp``): serve the tools over stdio.

    A ``.env`` file in the working directory is loaded first (existing environment
    variables win), so MCP clients that launch ``council-mcp`` directly get the same
    API keys as the ``council`` CLI.
    """
    load_dotenv_file()
    mcp.run()


if __name__ == "__main__":
    main()
