from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from council import cli
from council.cli import app
from council.config import CouncilConfig
from council.models import CouncilRun
from council.store import RunStore

FAKE_MEMBERS = ["fake:alpha", "fake:beta", "fake:gamma"]
QUESTION = "Is a hot dog a sandwich?"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run every CLI call from an empty tmp cwd and undo the .env loader's changes.

    The CLI loads ``./.env`` at startup straight into ``os.environ``; without this, a
    developer's real ``.env`` (or one written by a test) would leak into other tests.
    """
    monkeypatch.chdir(tmp_path)
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def runner() -> CliRunner:
    # Wide, colourless, non-interactive output so assertions do not depend on the
    # developer's terminal.
    return CliRunner(env={"COLUMNS": "200", "FORCE_COLOR": None, "TTY_COMPATIBLE": None})


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def config_file(tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)  # nothing may ever land in the repository
    path = tmp_path / "council.toml"
    path.write_text(
        'members = ["fake:alpha", "fake:beta", "fake:gamma"]\n'
        'chairman = "fake:alpha"\n'
        f'data_dir = "{data_dir.as_posix()}"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def invoke(runner: CliRunner, config_file: Path):
    def _invoke(*args: str):
        return runner.invoke(app, [*args, "--config", str(config_file)])

    return _invoke


def _ask_json(invoke, question: str = QUESTION, *extra: str) -> CouncilRun:
    result = invoke("ask", question, "--fake", "--json", *extra)
    assert result.exit_code == 0, result.output
    return CouncilRun.model_validate_json(result.stdout)


# ---------------------------------------------------------------------------- ask


def test_ask_fake_json_prints_only_the_run_and_saves_it(invoke, data_dir):
    result = invoke("ask", QUESTION, "--fake", "--json")
    assert result.exit_code == 0, result.output

    payload = json.loads(result.stdout)  # stdout is pure JSON
    run = CouncilRun.model_validate(payload)
    assert run.question == QUESTION
    assert run.members == FAKE_MEMBERS
    assert run.chairman == "fake:alpha"
    assert run.synthesis is not None
    assert len(run.reviews) == 3 and all(review.ranking for review in run.reviews)

    saved = data_dir / "runs" / f"{run.id}.json"
    assert saved.is_file()
    assert CouncilRun.model_validate_json(saved.read_text(encoding="utf-8")) == run


def test_ask_rich_output(invoke, data_dir):
    result = invoke("ask", QUESTION, "--fake")
    assert result.exit_code == 0, result.output
    out = result.output
    (run,) = RunStore(data_dir).list()

    for stage in ("Answers", "Reviews", "Synthesis"):  # final state of the live progress
        assert stage in out
    assert "Council ranking" in out
    assert "1*" in out  # alpha ranks its own answer first, flagged with *
    assert f'Answer to "{QUESTION}"' in out
    assert "SYNTHESIS by alpha" in out
    assert f"Run id: {run.id}" in out
    assert f"council vote {run.id} <model>" in out
    assert "Traceback" not in out


def test_ask_prints_markup_like_text_verbatim(invoke):
    # The fake echoes the question into every answer and the synthesis; Rich must not
    # treat the brackets as style markup (which would hide them or raise).
    question = "What does [bold red]this[/bold red] or [fake:alpha] mean?"
    result = invoke("ask", question, "--fake")
    assert result.exit_code == 0, result.output
    assert "[bold red]this[/bold red]" in result.output
    assert "[fake:alpha]" in result.output


def test_ask_truncates_answers_unless_full(invoke, monkeypatch):
    monkeypatch.setattr(cli, "ANSWER_PREVIEW_CHARS", 20)
    truncated = invoke("ask", QUESTION, "--fake")
    assert truncated.exit_code == 0, truncated.output
    answer_body = f'Answer to "{QUESTION}"'  # the fake answers' text after the signature
    assert "truncated" in truncated.output
    assert answer_body not in truncated.output

    full = invoke("ask", QUESTION, "--fake", "--full")
    assert full.exit_code == 0, full.output
    assert "truncated" not in full.output
    assert answer_body in full.output


def test_ask_with_debate(invoke):
    result = invoke("ask", QUESTION, "--fake", "--debate", "1")
    assert result.exit_code == 0, result.output
    assert "Debate 1" in result.output
    assert "Revised answer (debate round 1)" in result.output

    run = _ask_json(invoke, QUESTION, "--debate", "1")
    assert run.debate_rounds == 1 and len(run.debate_answers) == 1


def test_ask_members_and_chairman_override(invoke):
    result = invoke(
        "ask", QUESTION, "--members", "fake:one, fake:two", "--chairman", "fake:two", "--json"
    )
    assert result.exit_code == 0, result.output
    run = CouncilRun.model_validate_json(result.stdout)
    assert run.members == ["fake:one", "fake:two"]
    assert run.chairman == "fake:two"


def test_ask_all_members_failing_exits_1_without_traceback(invoke, data_dir):
    result = invoke("ask", QUESTION, "--members", "fake:fail1,fake:fail2")
    assert result.exit_code == 1
    assert "no council member produced an answer" in result.output
    assert "fake:fail1" in result.output and "simulated failure" in result.output
    assert "Traceback" not in result.output
    assert RunStore(data_dir).list() == []  # nothing useful to save


def test_ask_failed_member_and_failed_chairman_still_render(invoke, data_dir):
    result = invoke(
        "ask", QUESTION, "--members", "fake:alpha,fake:beta,fake:fail", "--chairman", "fake:fail"
    )
    assert result.exit_code == 0, result.output
    assert "simulated failure" in result.output  # failed member's panel
    assert "(not ranked)" in result.output
    assert "chairman fake:fail failed" in result.output
    (run,) = RunStore(data_dir).list()
    assert run.synthesis is None


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--members", "no-colon-here"], "invalid council settings"),
        (["--members", "bogus:model,fake:alpha"], "unknown provider"),
        (["--members", ","], "invalid council settings"),
    ],
)
def test_ask_invalid_settings(invoke, args, message):
    result = invoke("ask", QUESTION, *args)
    assert result.exit_code == 1
    assert message in result.output
    assert "Traceback" not in result.output


def test_ask_empty_question(invoke):
    result = invoke("ask", "   ", "--fake")
    assert result.exit_code == 1
    assert "must not be empty" in result.output


def test_missing_config_file(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["history", "--config", str(tmp_path / "nope.toml")])
    assert result.exit_code == 1
    assert "Config file not found" in result.output


# ------------------------------------------------------------------ history / show


def test_history_lists_runs(invoke):
    empty = invoke("history")
    assert empty.exit_code == 0
    assert "No runs yet" in empty.output

    first = _ask_json(invoke, "First question?")
    second = _ask_json(invoke, "Second question?")
    result = invoke("history")
    assert result.exit_code == 0, result.output
    assert first.id in result.output and second.id in result.output
    assert "First question?" in result.output
    assert result.output.index(second.id) < result.output.index(first.id)  # newest first

    limited = invoke("history", "--limit", "1")
    assert second.id in limited.output and first.id not in limited.output


def test_show_run(invoke):
    run = _ask_json(invoke)
    result = invoke("show", run.id)
    assert result.exit_code == 0, result.output
    assert QUESTION in result.output
    assert "Review by fake:alpha" in result.output
    assert "Council ranking" in result.output
    assert "SYNTHESIS by alpha" in result.output

    as_json = invoke("show", run.id, "--json")
    assert as_json.exit_code == 0
    assert CouncilRun.model_validate_json(as_json.stdout) == run


@pytest.mark.parametrize("run_id", ["doesnotexist", "../../etc/passwd"])
def test_show_unknown_run_id(invoke, run_id):
    result = invoke("show", run_id)
    assert result.exit_code == 1
    assert "not found" in result.output
    assert "Traceback" not in result.output


# -------------------------------------------------------------------- vote / stats


def test_vote_then_stats_counts_human_vote(invoke, data_dir):
    run = _ask_json(invoke)
    before = invoke("stats")
    assert "n=0" in before.output

    result = invoke("vote", run.id, "fake:beta")
    assert result.exit_code == 0, result.output
    assert "Recorded" in result.output
    assert RunStore(data_dir).load(run.id).human_vote == "fake:beta"

    stats = invoke("stats")
    assert stats.exit_code == 0, stats.output
    assert "Human vs council (n=1)" in stats.output
    assert "Leaderboard" in stats.output
    assert "Self-preference" in stats.output
    assert "Kendall tau" in stats.output
    assert "+2.00" in stats.output  # fake:alpha always favors itself by two ranks


def test_vote_accepts_short_name(invoke, data_dir):
    run = _ask_json(invoke)
    result = invoke("vote", run.id, "gamma")
    assert result.exit_code == 0, result.output
    assert RunStore(data_dir).load(run.id).human_vote == "fake:gamma"


def test_vote_rejects_non_member_and_unknown_run(invoke, data_dir):
    run = _ask_json(invoke)
    result = invoke("vote", run.id, "fake:delta")
    assert result.exit_code == 1
    assert "not a member" in result.output
    assert "Traceback" not in result.output
    assert RunStore(data_dir).load(run.id).human_vote is None

    missing = invoke("vote", "nope", "fake:alpha")
    assert missing.exit_code == 1
    assert "not found" in missing.output


def test_stats_empty_store(invoke):
    result = invoke("stats")
    assert result.exit_code == 0
    assert "No runs yet" in result.output


def test_stats_json_matches_summary_shape(invoke):
    _ask_json(invoke)
    _ask_json(invoke, "Another question?")
    result = invoke("stats", "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert set(data) == {
        "n_runs",
        "total_cost_usd",
        "leaderboard",
        "self_preference",
        "agreement_matrix",
        "human_vs_council",
    }
    assert data["n_runs"] == 2
    assert data["self_preference"]["fake:alpha"] == 2.0


# -------------------------------------------------------------------------- models


def test_models_reports_keys_without_printing_them(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "keys.toml"
    path.write_text(
        'members = ["openrouter:openai/gpt-5.1", "anthropic:claude-opus-5-5", '
        '"openai:gpt-5.1", "ollama:llama3.1", "fake:alpha"]\n'
        'chairman = "anthropic:claude-opus-5-5"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-TOPSECRET-1234")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-TOPSECRET-5678")
    monkeypatch.setenv("OLLAMA_HOST", "http://user:hunter2@example.invalid:11434")

    result = runner.invoke(app, ["models", "--config", str(path)])
    assert result.exit_code == 0, result.output
    out = result.output
    for secret in ("TOPSECRET", "hunter2", "example.invalid"):
        assert secret not in out
    assert "openrouter:openai/gpt-5.1" in out
    assert "chairman" in out
    assert "OPENROUTER_API_KEY" in out and "ANTHROPIC_API_KEY" in out
    assert "OPENAI_API_KEY" in out and "not set" in out
    assert "no key needed" in out  # ollama and fake
    assert "✓" in out and "✗" in out  # ✓ and ✗


def test_models_all_keys_present(invoke):
    result = invoke("models")
    assert result.exit_code == 0, result.output
    assert "fake:alpha" in result.output
    assert "Every provider in use has what it needs" in result.output
    assert "Env file: none" in result.output


# ----------------------------------------------------------------------------- .env


def test_cli_loads_dotenv_from_cwd(runner, tmp_path, real_council_toml):
    assert "OPENROUTER_API_KEY" not in os.environ  # cleared by conftest
    (tmp_path / ".env").write_text(
        '# keys for council\nOPENROUTER_API_KEY="sk-or-FROMDOTENV-42"\nOPENAI_API_KEY=\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["models", "--config", str(real_council_toml)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "FROMDOTENV" not in out  # names only, never values
    assert "loaded OPENROUTER_API_KEY" in out
    assert "OPENROUTER_API_KEY" in out and "✓ set" in out
    assert "ANTHROPIC_API_KEY" in out and "not set" in out
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-FROMDOTENV-42"
    assert "OPENAI_API_KEY" not in os.environ  # empty values are ignored


def test_cli_real_environment_wins_over_dotenv(runner, tmp_path, monkeypatch, real_council_toml):
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-dotenv\n", encoding="utf-8")
    result = runner.invoke(app, ["models", "--config", str(real_council_toml)])
    assert result.exit_code == 0, result.output
    assert "nothing loaded" in result.output
    assert os.environ["OPENROUTER_API_KEY"] == "from-shell"


def test_dotenv_is_loaded_before_ask(runner, tmp_path, config_file, data_dir):
    """A COUNCIL_* style variable from .env is visible to every command, e.g. ask."""
    (tmp_path / ".env").write_text("COUNCIL_TEST_FROM_DOTENV=1\n", encoding="utf-8")
    result = runner.invoke(app, ["ask", QUESTION, "--json", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert os.environ.get("COUNCIL_TEST_FROM_DOTENV") == "1"


# ------------------------------------------------------------------- serve / mcp


def test_serve_missing_extra(runner, monkeypatch):
    monkeypatch.setitem(sys.modules, "council.web.server", None)  # import -> ImportError
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1
    assert 'pip install "council-of-models[web]"' in result.output
    assert "Traceback" not in result.output


@pytest.fixture
def fake_web(monkeypatch):
    """Stand-ins for council.web.server and uvicorn that record how they were called."""
    calls: dict = {}
    sentinel = object()

    def create_app(config=None, *, dist_dir=None):
        calls["config"] = config
        return sentinel

    def run(app, host, port):
        calls.update(app=app, host=host, port=port)

    server = types.ModuleType("council.web.server")
    server.create_app = create_app
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = run
    monkeypatch.setitem(sys.modules, "council.web.server", server)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    return calls, sentinel


@pytest.fixture
def real_council_toml(tmp_path, data_dir) -> Path:
    path = tmp_path / "real.toml"
    path.write_text(
        'members = ["openrouter:openai/gpt-5.1", "anthropic:claude-opus-5-5"]\n'
        'chairman = "anthropic:claude-opus-5-5"\n'
        f'data_dir = "{data_dir.as_posix()}"\n',
        encoding="utf-8",
    )
    return path


def test_serve_starts_uvicorn_with_config(runner, fake_web, real_council_toml, data_dir):
    calls, sentinel = fake_web
    result = runner.invoke(
        app,
        ["serve", "--host", "0.0.0.0", "--port", "9123", "--config", str(real_council_toml)],
    )
    assert result.exit_code == 0, result.output
    assert calls["app"] is sentinel
    assert (calls["host"], calls["port"]) == ("0.0.0.0", 9123)
    config = calls["config"]
    assert isinstance(config, CouncilConfig)
    assert config.members == ["openrouter:openai/gpt-5.1", "anthropic:claude-opus-5-5"]
    assert config.data_dir == data_dir
    assert "http://localhost:9123" in result.output  # URL printed before uvicorn starts


def test_serve_fake_overrides_the_council(runner, fake_web, real_council_toml, data_dir):
    calls, _ = fake_web
    result = runner.invoke(app, ["serve", "--fake", "--config", str(real_council_toml)])
    assert result.exit_code == 0, result.output
    config = calls["config"]
    assert config.members == FAKE_MEMBERS
    assert config.chairman == "fake:alpha"
    assert config.data_dir == data_dir  # everything else comes from the config file
    assert (calls["host"], calls["port"]) == ("127.0.0.1", 8000)  # defaults
    assert "http://127.0.0.1:8000" in result.output
    assert "fake:alpha, fake:beta, fake:gamma" in result.output


def test_serve_with_real_web_extra(real_council_toml, data_dir, monkeypatch):
    """The real create_app accepts the CLI's config (no uvicorn involved)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("COUNCIL_FAKE", raising=False)
    from council.web.server import create_app

    settings = cli._apply_overrides(
        cli._load_config(real_council_toml), members=None, chairman=None, debate=None, fake=True
    )
    response = TestClient(create_app(settings)).get("/api/config")
    assert response.status_code == 200
    assert response.json()["members"] == FAKE_MEMBERS


def test_mcp_missing_extra(runner, monkeypatch):
    monkeypatch.setitem(sys.modules, "council.mcp_server", None)
    result = runner.invoke(app, ["mcp"])
    assert result.exit_code == 1
    assert 'pip install "council-of-models[mcp]"' in result.output


def test_mcp_runs_server_main(runner, monkeypatch):
    called = []
    server = types.ModuleType("council.mcp_server")
    server.main = lambda: called.append(True)
    monkeypatch.setitem(sys.modules, "council.mcp_server", server)
    result = runner.invoke(app, ["mcp"])
    assert result.exit_code == 0, result.output
    assert called == [True]
    assert result.stdout == ""  # stdout belongs to the MCP protocol


# -------------------------------------------------------------------------- misc


def test_version_and_help(runner):
    version = runner.invoke(app, ["--version"])
    assert version.exit_code == 0
    assert "council-of-models" in version.output

    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    for command in ("ask", "history", "show", "vote", "stats", "models", "serve", "mcp"):
        assert command in help_result.output


def test_short_names_fall_back_to_full_id_on_collision():
    names = cli._short_names(
        ["openrouter:openai/gpt-5.1", "openai:gpt-5.1", "fake:alpha", "ollama:llama3.1:8b"]
    )
    assert names == {
        "openrouter:openai/gpt-5.1": "openrouter:openai/gpt-5.1",
        "openai:gpt-5.1": "openai:gpt-5.1",
        "fake:alpha": "alpha",
        "ollama:llama3.1:8b": "llama3.1:8b",
    }
