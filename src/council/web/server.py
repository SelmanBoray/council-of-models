"""FastAPI backend for the council web UI.

Routes (see the project contract)::

    GET  /api/config              -> {members, chairman, debate_rounds}
    POST /api/ask                 -> text/event-stream of CouncilEvent JSON, then
                                     {"stage": "done", "run_id": ...}
                                     (or {"stage": "error", "message": ...} on failure)
    GET  /api/runs?limit=50       -> [{id, question, created_at, council_ranking, human_vote}]
    GET  /api/runs/{run_id}       -> CouncilRun
    POST /api/runs/{run_id}/vote  -> CouncilRun   (body {"model": "<model id>"})
    GET  /api/stats               -> council.analytics.summary(runs)

Everything else (GET) serves the built React app from ``frontend/dist``: real files
(``/assets/*``, ``/favicon.svg``) are served as-is and any other path falls back to
``index.html`` so client-side routes such as ``/runs/<id>`` or ``/history`` work when
opened directly. Unknown ``/api/*`` paths stay JSON 404s.

Environment:
    COUNCIL_FAKE=1          replace the council with the offline fake one
                            (fake:alpha, fake:beta, fake:gamma; chairman fake:alpha).
    COUNCIL_DIST_DIR=path   serve the frontend from this directory instead of the
                            repository's ``frontend/dist``.

Run it with ``council serve`` or ``uvicorn council.web.server:get_app --factory``.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints

from council import __version__
from council.config import CouncilConfig, load_config
from council.council import Council
from council.models import CouncilEvent, CouncilRun
from council.store import RunStore

logger = logging.getLogger(__name__)

FAKE_ENV = "COUNCIL_FAKE"
DIST_DIR_ENV = "COUNCIL_DIST_DIR"
FAKE_MEMBERS: tuple[str, ...] = ("fake:alpha", "fake:beta", "fake:gamma")
FAKE_CHAIRMAN = "fake:alpha"
MAX_DEBATE_ROUNDS = 5
DEFAULT_RUNS_LIMIT = 50
MAX_RUNS_LIMIT = 1000
KEEPALIVE_INTERVAL_S = 15.0
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

# src/council/web/server.py -> parents[3] is the repository root.
DEFAULT_DIST_DIR = Path(__file__).resolve().parents[3] / "frontend" / "dist"

_STREAM_END = object()


# ------------------------------------------------------------------ request/response


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    debate_rounds: int = Field(default=0, ge=0, le=MAX_DEBATE_ROUNDS)


class VoteRequest(BaseModel):
    model: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ConfigInfo(BaseModel):
    members: list[str]
    chairman: str
    debate_rounds: int


class RunSummary(BaseModel):
    id: str
    question: str
    created_at: str
    council_ranking: list[str]
    human_vote: str | None


# --------------------------------------------------------------------------- helpers


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def fake_config(config: CouncilConfig) -> CouncilConfig:
    """``config`` with the offline fake council (data_dir and the rest are kept)."""
    data = config.model_dump()
    data.update(members=list(FAKE_MEMBERS), chairman=FAKE_CHAIRMAN)
    return CouncilConfig.model_validate(data)


def resolve_dist_dir(dist_dir: Path | None = None) -> Path:
    """Explicit ``dist_dir`` -> ``$COUNCIL_DIST_DIR`` -> repository ``frontend/dist``."""
    if dist_dir is not None:
        return Path(dist_dir).expanduser().resolve()
    env = os.environ.get(DIST_DIR_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_DIST_DIR


def _format_error(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _sse(payload: dict[str, Any]) -> str:
    """One SSE event: a single ``data:`` line of compact JSON plus a blank line."""
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _not_built_page(dist_dir: Path) -> str:
    where = html.escape(str(dist_dir))
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Council of Models</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 42rem; margin: 3rem auto;
 padding: 0 1rem; line-height: 1.5">
<h1>Council of Models</h1>
<p>frontend not built: run <code>npm run build</code> in frontend/</p>
<p>Looked for <code>{where}</code>. The API itself is up: see <a href="/docs">/docs</a>.</p>
</body>
</html>
"""


# --------------------------------------------------------------------------- app


def create_app(config: CouncilConfig | None = None, *, dist_dir: Path | None = None) -> FastAPI:
    """Build the FastAPI app.

    Args:
        config: Council configuration; defaults to :func:`council.config.load_config`.
            With ``COUNCIL_FAKE=1`` the members/chairman are replaced by the fake council.
        dist_dir: Built frontend directory; defaults to ``$COUNCIL_DIST_DIR`` or the
            repository's ``frontend/dist``. Checked once, at app creation.
    """
    if config is None:
        config = load_config()
    if _env_flag(FAKE_ENV):
        config = fake_config(config)
    dist = resolve_dist_dir(dist_dir)
    store = RunStore(config.data_dir)

    app = FastAPI(
        title="Council of Models",
        version=__version__,
        description="Ask a council of LLMs, let them rank each other, get a synthesis.",
    )
    app.state.config = config
    app.state.store = store
    app.state.dist_dir = dist
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _add_api_routes(app, config, store)
    _add_frontend_routes(app, dist)
    return app


def get_app() -> FastAPI:
    """App factory with default settings (``uvicorn council.web.server:get_app --factory``)."""
    return create_app()


def _add_api_routes(app: FastAPI, config: CouncilConfig, store: RunStore) -> None:
    @app.get("/api/config", response_model=ConfigInfo)
    def get_config() -> ConfigInfo:
        return ConfigInfo(
            members=list(config.members),
            chairman=config.chairman,
            debate_rounds=config.debate_rounds,
        )

    @app.post(
        "/api/ask",
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def ask(body: AskRequest) -> StreamingResponse:
        council = Council(config.model_copy(update={"debate_rounds": body.debate_rounds}))
        return StreamingResponse(
            _council_stream(council, body.question, store),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @app.get("/api/runs", response_model=list[RunSummary])
    def list_runs(
        limit: int = Query(DEFAULT_RUNS_LIMIT, ge=1, le=MAX_RUNS_LIMIT),
    ) -> list[RunSummary]:
        return [
            RunSummary(
                id=run.id,
                question=run.question,
                created_at=run.created_at,
                council_ranking=run.council_ranking(),
                human_vote=run.human_vote,
            )
            for run in store.list(limit=limit)
        ]

    @app.get("/api/runs/{run_id}", response_model=CouncilRun)
    def get_run(run_id: str) -> CouncilRun:
        try:
            return store.load(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}") from None

    @app.post("/api/runs/{run_id}/vote", response_model=CouncilRun)
    def vote(run_id: str, body: VoteRequest) -> CouncilRun:
        try:
            return store.set_human_vote(run_id, body.model)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        try:
            from council.analytics import summary
        except ImportError as exc:  # analytics module missing or incomplete
            raise HTTPException(status_code=503, detail=f"analytics unavailable: {exc}") from exc
        # The dict response model is serialized by pydantic, which writes NaN/inf as null.
        return summary(store.list())

    @app.api_route(
        "/api/{rest:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    def api_not_found(rest: str) -> None:
        raise HTTPException(status_code=404, detail="Not Found")


async def _council_stream(council: Council, question: str, store: RunStore) -> AsyncIterator[str]:
    """Run the council in a task and relay its progress as SSE events.

    Ends with ``done`` (after the run is saved) or ``error``. If the client goes away
    the generator is cancelled and the council run is cancelled with it (nothing is saved).
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()

    def on_event(event: CouncilEvent) -> None:
        queue.put_nowait(event.model_dump())

    task = asyncio.create_task(council.ask(question, on_event=on_event))
    task.add_done_callback(lambda _task: queue.put_nowait(_STREAM_END))
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_INTERVAL_S)
            except TimeoutError:
                yield ": keep-alive\n\n"  # SSE comment; keeps idle proxies from closing us
                continue
            if item is _STREAM_END:
                break
            yield _sse(item)
        run = task.result()
        await asyncio.to_thread(store.save, run)
    except Exception as exc:
        logger.exception("council run failed")
        yield _sse({"stage": "error", "message": _format_error(exc)})
        return
    finally:
        if not task.done():
            task.cancel()
        elif not task.cancelled():
            task.exception()  # mark retrieved so asyncio does not log it again
    yield _sse({"stage": "done", "run_id": run.id})


def _add_frontend_routes(app: FastAPI, dist: Path) -> None:
    index = dist / "index.html"
    if not index.is_file():
        page = _not_built_page(dist)

        @app.get("/{full_path:path}", include_in_schema=False, response_class=HTMLResponse)
        def frontend_not_built(full_path: str) -> HTMLResponse:
            return HTMLResponse(page)

        return

    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        if full_path:
            candidate = (dist / full_path).resolve()
            if candidate.is_relative_to(dist) and candidate.is_file():
                return FileResponse(candidate)
        # Client-side route (/, /history, /runs/<id>, ...): let React Router handle it.
        return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})
