"""JSON file store for council runs: ``<data_dir>/runs/<run_id>.json``."""

from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from council.models import CouncilRun

logger = logging.getLogger(__name__)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _timestamp(created_at: str) -> float:
    try:
        return datetime.fromisoformat(created_at).timestamp()
    except (TypeError, ValueError):
        return 0.0


class RunStore:
    """Persist :class:`CouncilRun` objects as pretty-printed JSON files.

    The ``runs`` directory is created lazily on the first save. Run ids are
    restricted to ``[A-Za-z0-9_-]`` so ids coming from URLs cannot escape the
    directory.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.runs_dir = self.data_dir / "runs"

    def _path(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
            raise KeyError(run_id)
        return self.runs_dir / f"{run_id}.json"

    def save(self, run: CouncilRun) -> Path:
        """Write ``run`` (atomically) and return the file path."""
        try:
            path = self._path(run.id)
        except KeyError:
            raise ValueError(
                f"Invalid run id {run.id!r}: use letters, digits, '-' or '_'"
            ) from None
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(run.model_dump_json(indent=2), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return path

    def load(self, run_id: str) -> CouncilRun:
        """Return the run with ``run_id``; raise KeyError if it does not exist."""
        path = self._path(run_id)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise KeyError(run_id) from None
        return CouncilRun.model_validate_json(text)

    def list(self, limit: int | None = None) -> list[CouncilRun]:
        """All stored runs, newest first (by ``created_at``). Unreadable files are skipped."""
        if not self.runs_dir.is_dir():
            return []
        runs: list[CouncilRun] = []
        for path in self.runs_dir.glob("*.json"):
            try:
                runs.append(CouncilRun.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                logger.warning("Skipping unreadable run file %s: %s", path, exc)
        runs.sort(key=lambda run: (_timestamp(run.created_at), run.id), reverse=True)
        if limit is not None:
            runs = runs[: max(limit, 0)]
        return runs

    def set_human_vote(self, run_id: str, model: str) -> CouncilRun:
        """Record the human's pick for best answer; ``model`` must be a run member."""
        run = self.load(run_id)
        if model not in run.members:
            raise ValueError(
                f"{model!r} is not a member of run {run_id}; "
                f"choose one of: {', '.join(run.members)}"
            )
        run.human_vote = model
        self.save(run)
        return run

    def delete(self, run_id: str) -> None:
        """Delete a stored run; raise KeyError if it does not exist."""
        path = self._path(run_id)
        try:
            path.unlink()
        except FileNotFoundError:
            raise KeyError(run_id) from None
