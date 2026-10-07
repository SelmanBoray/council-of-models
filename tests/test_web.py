"""Web API tests: offline (fake provider), tmp data dir, FastAPI TestClient."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from council.config import CouncilConfig  # noqa: E402
from council.models import CouncilEvent, CouncilRun  # noqa: E402
from council.store import RunStore  # noqa: E402
from council.web import server  # noqa: E402
from council.web.server import create_app  # noqa: E402

FAKE_MEMBERS = ["fake:alpha", "fake:beta", "fake:gamma"]
INDEX_MARKER = "<!-- test-index -->"


@pytest.fixture(autouse=True)
def _clean_web_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COUNCIL_FAKE", raising=False)
    monkeypatch.delenv("COUNCIL_DIST_DIR", raising=False)


@pytest.fixture
def dist_dir(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        f"<!doctype html><html><body>{INDEX_MARKER}<div id=root></div></body></html>",
        encoding="utf-8",
    )
    (dist / "assets" / "app.js").write_text("console.log('council');", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg></svg>", encoding="utf-8")
    return dist


@pytest.fixture
def client(fake_config: CouncilConfig, dist_dir: Path) -> Iterator[TestClient]:
    with TestClient(create_app(fake_config, dist_dir=dist_dir)) as test_client:
        yield test_client


@pytest.fixture
def store(fake_config: CouncilConfig) -> RunStore:
    return RunStore(fake_config.data_dir)


def parse_sse(body: str) -> list[dict[str, Any]]:
    """Strictly parse the stream: every event is exactly one ``data: <json>`` line."""
    assert body.endswith("\n\n"), "stream must end with a blank line"
    events: list[dict[str, Any]] = []
    for block in body.split("\n\n")[:-1]:
        if block.startswith(":"):  # keep-alive comment
            continue
        assert "\n" not in block, f"multi-line SSE event: {block!r}"
        assert block.startswith("data: "), block
        events.append(json.loads(block.removeprefix("data: ")))
    return events


def ask(client: TestClient, question: str, debate_rounds: int = 0) -> tuple[Any, list[dict]]:
    response = client.post(
        "/api/ask",
        json={"question": question, "debate_rounds": debate_rounds},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200, response.text
    return response, parse_sse(response.text)


# ------------------------------------------------------------------------- config


def test_config_endpoint(client: TestClient) -> None:
    response = client.get("/api/config")
    assert response.status_code == 200
    assert response.json() == {
        "members": FAKE_MEMBERS,
        "chairman": "fake:alpha",
        "debate_rounds": 0,
    }


def test_council_fake_env_overrides_members(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dist_dir: Path
) -> None:
    monkeypatch.setenv("COUNCIL_FAKE", "1")
    real = CouncilConfig(
        members=["openrouter:openai/gpt-5.1", "anthropic:claude-opus-5-5"],
        chairman="anthropic:claude-opus-5-5",
        debate_rounds=2,
        data_dir=tmp_path / "data",
    )
    app = create_app(real, dist_dir=dist_dir)
    assert app.state.config.data_dir == tmp_path / "data"
    with TestClient(app) as test_client:
        assert test_client.get("/api/config").json() == {
            "members": FAKE_MEMBERS,
            "chairman": "fake:alpha",
            "debate_rounds": 2,
        }


# ---------------------------------------------------------------------------- ask


def test_ask_streams_events_then_done_and_saves_run(client: TestClient, store: RunStore) -> None:
    response, events = ask(client, "What is the capital of France?", debate_rounds=2)

    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert "content-encoding" not in response.headers

    final = events[-1]
    assert final["stage"] == "done"
    assert isinstance(final["run_id"], str) and final["run_id"]
    progress = events[:-1]
    assert progress, "expected progress events before done"
    for event in progress:
        CouncilEvent.model_validate(event)  # every progress event is a CouncilEvent
    stages = {event["stage"] for event in progress}
    assert {"answer", "debate", "review", "synthesis"} <= stages
    assert sorted({e["round"] for e in progress if e["stage"] == "debate"}) == [1, 2]
    assert {e["round"] for e in progress if e["stage"] != "debate"} == {0}
    assert all(e["stage"] not in {"done", "error"} for e in progress)

    run = store.load(final["run_id"])
    assert run.question == "What is the capital of France?"
    assert run.debate_rounds == 2
    assert len(run.debate_answers) == 2
    assert run.synthesis is not None


def test_ask_saved_run_is_listed_and_fetchable(client: TestClient, store: RunStore) -> None:
    _, events = ask(client, "ping")
    run_id = events[-1]["run_id"]

    runs = client.get("/api/runs").json()
    assert isinstance(runs, list) and len(runs) == 1
    row = runs[0]
    assert set(row) == {"id", "question", "created_at", "council_ranking", "human_vote"}
    stored = store.load(run_id)
    assert row == {
        "id": run_id,
        "question": "ping",
        "created_at": stored.created_at,
        "council_ranking": stored.council_ranking(),
        "human_vote": None,
    }
    assert sorted(row["council_ranking"]) == sorted(FAKE_MEMBERS)

    response = client.get(f"/api/runs/{run_id}")
    assert response.status_code == 200
    assert response.json() == stored.model_dump(mode="json")
    assert CouncilRun.model_validate(response.json()) == stored


@pytest.mark.parametrize(
    "body",
    [
        {"question": "", "debate_rounds": 0},
        {"question": "   ", "debate_rounds": 0},
        {"question": "ok", "debate_rounds": 6},
        {"question": "ok", "debate_rounds": -1},
        {"debate_rounds": 0},
    ],
)
def test_ask_rejects_invalid_body(client: TestClient, body: dict, store: RunStore) -> None:
    response = client.post("/api/ask", json=body)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert store.list() == []


def test_ask_debate_rounds_defaults_to_zero(client: TestClient) -> None:
    response = client.post("/api/ask", json={"question": "no rounds given"})
    events = parse_sse(response.text)
    assert events[-1]["stage"] == "done"
    assert all(event["stage"] != "debate" for event in events)


def test_ask_fatal_error_sends_error_event_and_no_done(
    client: TestClient, store: RunStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExplodingCouncil:
        def __init__(self, config: CouncilConfig) -> None:
            self.config = config

        async def ask(self, question: str, *, on_event: Callable[[CouncilEvent], None]) -> None:
            on_event(CouncilEvent(stage="answer", model="fake:alpha", status="start"))
            raise RuntimeError("boom\nsecond line")

    monkeypatch.setattr(server, "Council", ExplodingCouncil)
    _, events = ask(client, "will fail")

    assert events[0] == {
        "stage": "answer",
        "model": "fake:alpha",
        "status": "start",
        "round": 0,
        "message": None,
    }
    assert events[-1] == {"stage": "error", "message": "RuntimeError: boom\nsecond line"}
    assert all(event["stage"] != "done" for event in events)
    assert store.list() == []


def test_ask_member_failure_is_an_event_not_a_fatal_error(tmp_path: Path, dist_dir: Path) -> None:
    config = CouncilConfig(
        members=["fake:alpha", "fake:fail", "fake:gamma"],
        chairman="fake:alpha",
        data_dir=tmp_path / "data",
    )
    with TestClient(create_app(config, dist_dir=dist_dir)) as test_client:
        _, events = ask(test_client, "partial failure")
    errors = [e for e in events if e.get("status") == "error"]
    assert errors and all(e["model"] == "fake:fail" and e["message"] for e in errors)
    assert events[-1]["stage"] == "done"


async def test_stream_cancels_council_when_client_goes_away(store: RunStore) -> None:
    cancelled = asyncio.Event()

    class SlowCouncil:
        async def ask(self, question: str, *, on_event: Callable[[CouncilEvent], None]) -> None:
            on_event(CouncilEvent(stage="answer", model="fake:alpha", status="start"))
            try:
                await asyncio.Event().wait()  # a model call that never returns
            except asyncio.CancelledError:
                cancelled.set()
                raise

    stream = server._council_stream(SlowCouncil(), "q", store)  # type: ignore[arg-type]
    first = await anext(stream)
    assert json.loads(first.removeprefix("data: "))["stage"] == "answer"
    await stream.aclose()  # what Starlette does when the client disconnects
    await asyncio.wait_for(cancelled.wait(), timeout=2)
    assert store.list() == []


async def test_stream_sends_keepalive_comments_while_waiting(
    store: RunStore, make_run: Callable[..., CouncilRun], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "KEEPALIVE_INTERVAL_S", 0.01)
    run = make_run(id="slowrun")

    class SlowCouncil:
        async def ask(self, question: str, *, on_event: Callable[[CouncilEvent], None]) -> Any:
            await asyncio.sleep(0.1)
            return run

    chunks = [chunk async for chunk in server._council_stream(SlowCouncil(), "q", store)]  # type: ignore[arg-type]
    assert ": keep-alive\n\n" in chunks
    events = parse_sse("".join(chunks))
    assert events == [{"stage": "done", "run_id": "slowrun"}]
    assert store.load("slowrun") == run


# --------------------------------------------------------------------------- runs


def test_runs_list_newest_first_with_limit(
    client: TestClient, store: RunStore, make_run: Callable[..., CouncilRun]
) -> None:
    for i, created in enumerate(["2026-10-01T00:00:00.000Z", "2026-10-03T00:00:00.000Z"]):
        store.save(make_run(id=f"run{i}", created_at=created, question=f"q{i}"))
    store.save(make_run(id="run2", created_at="2026-10-02T00:00:00.000Z", question="q2"))

    assert [r["id"] for r in client.get("/api/runs").json()] == ["run1", "run2", "run0"]
    assert [r["id"] for r in client.get("/api/runs?limit=2").json()] == ["run1", "run2"]
    assert client.get("/api/runs?limit=0").status_code == 422


def test_runs_list_empty(client: TestClient) -> None:
    response = client.get("/api/runs?limit=50")
    assert response.status_code == 200
    assert response.json() == []


def test_get_run_unknown_is_404_json(client: TestClient) -> None:
    for run_id in ("doesnotexist", "bad.id"):
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 404
        assert "detail" in response.json()


# --------------------------------------------------------------------------- vote


def test_vote_happy_path(
    client: TestClient, store: RunStore, make_run: Callable[..., CouncilRun]
) -> None:
    store.save(make_run(id="votable"))
    response = client.post("/api/runs/votable/vote", json={"model": "fake:beta"})
    assert response.status_code == 200
    body = response.json()
    assert body["human_vote"] == "fake:beta"
    assert CouncilRun.model_validate(body).id == "votable"
    assert store.load("votable").human_vote == "fake:beta"
    listed = client.get("/api/runs").json()
    assert listed[0]["human_vote"] == "fake:beta"


def test_vote_unknown_run_is_404(client: TestClient) -> None:
    response = client.post("/api/runs/nope/vote", json={"model": "fake:alpha"})
    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


def test_vote_model_not_in_run_is_400(
    client: TestClient, store: RunStore, make_run: Callable[..., CouncilRun]
) -> None:
    store.save(make_run(id="votable"))
    response = client.post("/api/runs/votable/vote", json={"model": "fake:delta"})
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str)
    assert store.load("votable").human_vote is None


# -------------------------------------------------------------------------- stats


def test_stats_returns_analytics_summary(
    client: TestClient,
    store: RunStore,
    make_run: Callable[..., CouncilRun],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store.save(make_run(id="r1"))
    store.save(make_run(id="r2", created_at="2026-10-08T00:00:00.000Z"))
    import council.analytics as analytics

    seen: list[list[str]] = []

    def fake_summary(runs: list[CouncilRun]) -> dict[str, Any]:
        seen.append([run.id for run in runs])
        return {
            "n_runs": len(runs),
            "total_cost_usd": None,
            "leaderboard": [{"model": "fake:alpha", "mean_rank": 1.0, "wins": 2, "appearances": 2}],
            "self_preference": {"fake:alpha": math.nan},
            "agreement_matrix": {"fake:alpha": {"fake:beta": math.inf}},
            "human_vs_council": {"n": 0, "agreement_rate": None, "per_model_human_wins": {}},
        }

    monkeypatch.setattr(analytics, "summary", fake_summary, raising=False)
    response = client.get("/api/stats")
    assert response.status_code == 200
    body = response.json()
    assert seen == [["r2", "r1"]]
    assert body["n_runs"] == 2
    assert body["leaderboard"][0]["model"] == "fake:alpha"
    assert body["self_preference"] == {"fake:alpha": None}  # NaN is not valid JSON
    assert body["agreement_matrix"] == {"fake:alpha": {"fake:beta": None}}


def test_stats_with_real_analytics_if_available(
    client: TestClient, store: RunStore, make_run: Callable[..., CouncilRun]
) -> None:
    import council.analytics as analytics

    if not callable(getattr(analytics, "summary", None)):
        pytest.skip("council.analytics.summary not implemented yet")
    store.save(make_run(id="r1"))
    response = client.get("/api/stats")
    assert response.status_code == 200
    assert {
        "n_runs",
        "total_cost_usd",
        "leaderboard",
        "self_preference",
        "agreement_matrix",
        "human_vs_council",
    } <= set(response.json())


# ----------------------------------------------------------------------- frontend


@pytest.mark.parametrize("path", ["/", "/history", "/stats", "/runs/abc123", "/no/such/page"])
def test_spa_fallback_serves_index(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert INDEX_MARKER in response.text


def test_static_files_are_served(client: TestClient) -> None:
    js = client.get("/assets/app.js")
    assert js.status_code == 200
    assert js.text == "console.log('council');"
    assert "javascript" in js.headers["content-type"]
    assert client.get("/assets/missing.js").status_code == 404
    favicon = client.get("/favicon.svg")
    assert favicon.status_code == 200
    assert favicon.text == "<svg></svg>"


def test_spa_does_not_serve_files_outside_dist(client: TestClient, dist_dir: Path) -> None:
    (dist_dir.parent / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    for path in ("/..%2Fsecret.txt", "/%2e%2e/secret.txt", "/assets/..%2F..%2Fsecret.txt"):
        response = client.get(path)
        assert "TOP SECRET" not in response.text, path


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_unknown_api_path_is_json_404(client: TestClient, method: str) -> None:
    response = client.request(method, "/api/unknown")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Not Found"}


def test_frontend_not_built_page(fake_config: CouncilConfig, tmp_path: Path) -> None:
    missing = tmp_path / "no-dist"
    with TestClient(create_app(fake_config, dist_dir=missing)) as test_client:
        for path in ("/", "/history", "/runs/abc"):
            response = test_client.get(path)
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/html")
            assert "frontend not built" in response.text
            assert "npm run build" in response.text
        api = test_client.get("/api/unknown")
        assert api.status_code == 404
        assert api.json() == {"detail": "Not Found"}
        assert test_client.get("/api/config").status_code == 200


def test_dist_dir_env_override(
    monkeypatch: pytest.MonkeyPatch, fake_config: CouncilConfig, dist_dir: Path
) -> None:
    monkeypatch.setenv("COUNCIL_DIST_DIR", str(dist_dir))
    app = create_app(fake_config)
    assert app.state.dist_dir == dist_dir.resolve()
    with TestClient(app) as test_client:
        assert INDEX_MARKER in test_client.get("/history").text


def test_default_dist_dir_is_repo_frontend_dist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert server.DEFAULT_DIST_DIR == repo_root / "frontend" / "dist"
    assert server.resolve_dist_dir() == server.DEFAULT_DIST_DIR


# --------------------------------------------------------------------------- cors


def test_cors_allows_vite_dev_origin(client: TestClient) -> None:
    response = client.options(
        "/api/ask",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = client.get("/api/config", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers
