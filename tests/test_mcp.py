"""MCP server tests.

The tools are exercised through a real MCP client: an in-process session over
memory streams (``mcp.shared.memory``) for most tests, plus one stdio smoke test
that spawns ``python -m council.mcp_server`` the way Claude Code or Cursor would.
Everything runs offline on the fake provider.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from council import analytics, mcp_server  # noqa: E402
from council.errors import CouncilError  # noqa: E402
from council.models import Answer, CouncilRun, Review  # noqa: E402
from council.store import RunStore  # noqa: E402

FAKE_MEMBERS = ["fake:alpha", "fake:beta", "fake:gamma"]
CONFIG_MEMBERS = ["fake:delta", "fake:epsilon"]
TOOL_NAMES = {"ask_council", "council_stats", "council_history", "council_vote"}
SUMMARY_KEYS = {
    "n_runs",
    "total_cost_usd",
    "leaderboard",
    "self_preference",
    "agreement_matrix",
    "human_vs_council",
}
QUESTION = "Should a small team pick PostgreSQL or SQLite for an internal tool?"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def config_path(tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A council.toml with a *different* fake council; COUNCIL_FAKE=1 must override it."""
    path = tmp_path / "council.toml"
    path.write_text(
        f"members = {json.dumps(CONFIG_MEMBERS)}\n"
        f'chairman = "{CONFIG_MEMBERS[0]}"\n'
        f"data_dir = '{data_dir.as_posix()}'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("COUNCIL_CONFIG", str(path))
    monkeypatch.setenv("COUNCIL_FAKE", "1")
    return path


async def call(session: ClientSession, name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Call a tool over MCP and decode its single JSON text block."""
    result = await session.call_tool(name, arguments or {})
    assert not result.isError, result.content
    assert len(result.content) == 1
    return json.loads(result.content[0].text)


# ------------------------------------------------------------------ discovery


async def test_lists_all_tools_with_descriptions(config_path: Path) -> None:
    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        listed = await session.list_tools()
    tools = {tool.name: tool for tool in listed.tools}
    assert TOOL_NAMES <= set(tools)
    for name in TOOL_NAMES:
        assert tools[name].description
    ask_schema = tools["ask_council"].inputSchema
    assert ask_schema["required"] == ["question"]
    assert ask_schema["properties"]["debate_rounds"]["default"] == 0
    assert tools["council_stats"].annotations.readOnlyHint is True
    assert "ask_council" in (mcp_server.mcp.instructions or "")


# ------------------------------------------------------------ end-to-end flow


async def test_ask_history_vote_flow(config_path: Path, data_dir: Path) -> None:
    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        run = await call(session, "ask_council", {"question": QUESTION})

        assert "error" not in run
        run_id = run["run_id"]
        assert run_id
        assert sorted(run["council_ranking"]) == FAKE_MEMBERS
        assert run["chairman"] == "fake:alpha"
        assert run["synthesis"].startswith("SYNTHESIS by alpha")
        assert [answer["model"] for answer in run["answers"]] == FAKE_MEMBERS
        assert all(answer["error"] is None and answer["text"] for answer in run["answers"])
        assert sorted(review["reviewer"] for review in run["reviews"]) == FAKE_MEMBERS
        assert all(sorted(review["ranking"]) == FAKE_MEMBERS for review in run["reviews"])
        assert run["total_cost_usd"] == 0.0
        assert "final_answers" not in run
        assert (data_dir / "runs" / f"{run_id}.json").is_file()

        history = await call(session, "council_history")
        assert len(history) == 1
        assert history[0] == {
            "id": run_id,
            "question": QUESTION,
            "created_at": history[0]["created_at"],
            "council_ranking": run["council_ranking"],
            "human_vote": None,
        }

        vote = await call(session, "council_vote", {"run_id": run_id, "model": "fake:beta"})
        assert vote == {"run_id": run_id, "human_vote": "fake:beta"}
        assert RunStore(data_dir).load(run_id).human_vote == "fake:beta"
        history = await call(session, "council_history", {"limit": 5})
        assert history[0]["human_vote"] == "fake:beta"

        bad_model = await call(session, "council_vote", {"run_id": run_id, "model": "fake:zeta"})
        assert set(bad_model) == {"error"}
        assert "not a member" in bad_model["error"]

        bad_run = await call(session, "council_vote", {"run_id": "nope123", "model": "fake:beta"})
        assert bad_run == {"error": "run not found: nope123"}

    assert RunStore(data_dir).load(run_id).human_vote == "fake:beta"


async def test_debate_rounds_adds_final_answers(config_path: Path) -> None:
    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        run = await call(session, "ask_council", {"question": QUESTION, "debate_rounds": 1})
    assert [answer["model"] for answer in run["final_answers"]] == FAKE_MEMBERS
    assert all("debate round 1" in answer["text"] for answer in run["final_answers"])
    assert sorted(run["council_ranking"]) == FAKE_MEMBERS


async def test_without_fake_env_uses_configured_council(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COUNCIL_FAKE")
    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        run = await call(session, "ask_council", {"question": QUESTION})
    assert sorted(run["council_ranking"]) == CONFIG_MEMBERS
    assert run["chairman"] == CONFIG_MEMBERS[0]


# --------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"question": "   "}, "question must not be empty"),
        ({"question": QUESTION, "debate_rounds": -1}, "debate_rounds must be"),
        ({"question": QUESTION, "debate_rounds": 6}, "debate_rounds must be"),
    ],
)
async def test_ask_council_invalid_arguments(
    config_path: Path, data_dir: Path, arguments: dict[str, Any], message: str
) -> None:
    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        result = await call(session, "ask_council", arguments)
    assert set(result) == {"error"}
    assert message in result["error"]
    assert RunStore(data_dir).list() == []


async def test_council_error_is_returned_not_raised(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(self: Any, question: str, **kwargs: Any) -> CouncilRun:
        raise CouncilError("all providers are down")

    monkeypatch.setattr(mcp_server.Council, "ask", boom)
    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        result = await call(session, "ask_council", {"question": QUESTION})
    assert result == {"error": "all providers are down"}


async def test_missing_config_file_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COUNCIL_CONFIG", str(tmp_path / "missing.toml"))
    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        for name, arguments in [
            ("ask_council", {"question": QUESTION}),
            ("council_history", {}),
            ("council_vote", {"run_id": "abc", "model": "fake:alpha"}),
            ("council_stats", {}),
        ]:
            result = await call(session, name, arguments)
            assert set(result) == {"error"}, name
            assert "Config file not found" in result["error"], name


async def test_save_failure_still_returns_the_answer(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_save(self: RunStore, run: CouncilRun) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(RunStore, "save", broken_save)
    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        run = await call(session, "ask_council", {"question": QUESTION})
    assert run["run_id"]
    assert run["synthesis"]
    assert "disk full" in run["save_error"]


async def test_history_limit_validation_and_empty_store(config_path: Path) -> None:
    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        assert await call(session, "council_history") == []
        result = await call(session, "council_history", {"limit": 0})
    assert set(result) == {"error"}


# ---------------------------------------------------------------------- stats


async def test_stats_passes_stored_runs_to_analytics_summary(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[CouncilRun]] = []

    def stub_summary(runs: list[CouncilRun]) -> dict[str, Any]:
        seen.append(runs)
        return {"n_runs": len(runs), "stub": True}

    monkeypatch.setattr(analytics, "summary", stub_summary, raising=False)
    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        run = await call(session, "ask_council", {"question": QUESTION})
        stats = await call(session, "council_stats")
    assert stats == {"n_runs": 1, "stub": True}
    assert [r.id for r in seen[0]] == [run["run_id"]]


async def test_stats_with_real_analytics(config_path: Path) -> None:
    if not callable(getattr(analytics, "summary", None)):
        pytest.skip("council.analytics.summary is not implemented yet")
    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        await call(session, "ask_council", {"question": QUESTION})
        stats = await call(session, "council_stats")
    assert "error" not in stats
    assert SUMMARY_KEYS <= set(stats)
    assert stats["n_runs"] == 1


# --------------------------------------------------------------- run summary


def test_truncate() -> None:
    assert mcp_server._truncate("x" * 500) == "x" * 500
    assert mcp_server._truncate("x" * 501) == "x" * 500 + "…"


def test_run_summary_handles_failed_chairman_and_reviews(
    make_run: Callable[..., CouncilRun],
) -> None:
    run = make_run(
        synthesis=None,
        answers=[
            Answer(model="fake:alpha", text="a" * 800, latency_s=0.1),
            Answer(model="fake:beta", text="", latency_s=0.1, error="ProviderError: boom"),
            Answer(model="fake:gamma", text="short", latency_s=0.1),
        ],
        reviews=[
            Review(reviewer="fake:alpha", ranking=[], rationale="", raw="??", error="unparseable"),
            Review(
                reviewer="fake:gamma",
                ranking=["fake:gamma", "fake:alpha"],
                rationale="",
                raw="RANKING: A > B",
            ),
        ],
    )
    summary = mcp_server.run_summary(run)
    assert summary["synthesis"] is None
    assert summary["answers"][0]["text"] == "a" * 500 + "…"
    assert summary["answers"][1] == {
        "model": "fake:beta",
        "text": "",
        "error": "ProviderError: boom",
    }
    assert summary["reviews"][0] == {
        "reviewer": "fake:alpha",
        "ranking": [],
        "error": "unparseable",
    }
    assert summary["council_ranking"] == ["fake:gamma", "fake:alpha", "fake:beta"]
    json.dumps(summary)  # JSON-serializable


# ------------------------------------------------------------------ entry point


def test_main_loads_dotenv_before_serving(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    new_key, kept_key = "COUNCIL_TEST_DOTENV_NEW", "COUNCIL_TEST_DOTENV_KEPT"
    assert new_key not in os.environ
    monkeypatch.setenv(kept_key, "from-environment")
    (tmp_path / ".env").write_text(
        f"{new_key}=from-dotenv\n{kept_key}=from-dotenv\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    seen: dict[str, str | None] = {}

    def fake_run(*args: Any, **kwargs: Any) -> None:
        seen[new_key] = os.environ.get(new_key)
        seen[kept_key] = os.environ.get(kept_key)

    monkeypatch.setattr(mcp_server.mcp, "run", fake_run)
    try:
        mcp_server.main()
    finally:
        os.environ.pop(new_key, None)
    assert seen == {new_key: "from-dotenv", kept_key: "from-environment"}
    assert new_key not in os.environ


# ---------------------------------------------------------------- stdio smoke


async def test_stdio_server_end_to_end(tmp_path: Path, config_path: Path) -> None:
    """Spawn the real server process over stdio, like an MCP client config would."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "council.mcp_server"],
        env={"COUNCIL_FAKE": "1", "COUNCIL_CONFIG": str(config_path)},
        cwd=str(tmp_path),
    )
    with (tmp_path / "server-stderr.log").open("w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                listed = await session.list_tools()
                run = await call(session, "ask_council", {"question": QUESTION})
    assert init.serverInfo.name == "council-of-models"
    assert init.instructions and "ask_council" in init.instructions
    assert TOOL_NAMES <= {tool.name for tool in listed.tools}
    assert sorted(run["council_ranking"]) == FAKE_MEMBERS
