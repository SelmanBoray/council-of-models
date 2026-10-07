"""Council orchestration: answer -> (debate) -> anonymous review -> chairman synthesis."""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Literal

from council import prompts
from council.config import CouncilConfig
from council.models import (
    Answer,
    Completion,
    CouncilEvent,
    CouncilRun,
    Review,
    Synthesis,
    aggregate_ranking,
    mean_positions,
)
from council.providers import Message, Provider, get_provider, split_model_id

logger = logging.getLogger(__name__)

EventCallback = Callable[[CouncilEvent], None]
ProviderFactory = Callable[[str], Provider]
Stage = Literal["answer", "debate", "review", "synthesis"]
CallResult = tuple[Completion | None, float, str | None]

UNPARSEABLE_RANKING_ERROR = "could not parse a ranking from the review"


# --------------------------------------------------------------------------- labels


def make_labels(count: int) -> list[str]:
    """``["Response A", "Response B", ...]`` (at most 26)."""
    if count > 26:
        raise ValueError("at most 26 anonymous labels are supported")
    return [f"Response {chr(ord('A') + i)}" for i in range(count)]


def assign_labels(models: Sequence[str], seed: str) -> dict[str, str]:
    """Shuffle ``models`` with a RNG seeded by ``seed`` and map label -> model id."""
    shuffled = list(models)
    random.Random(seed).shuffle(shuffled)
    return dict(zip(make_labels(len(shuffled)), shuffled, strict=True))


# ---------------------------------------------------------------------- ranking parse

_ANCHORED_MARKER_RE = re.compile(r"^[ \t>#\-]*(?:final[ \t]+)?ranking[ \t]*:", re.I | re.M)
_ANY_MARKER_RE = re.compile(r"ranking[ \t]*:", re.I)
_PURE_LINE_LEFTOVER_RE = re.compile(r"^[\s\d.)\]\[(>,;:|/=\-–—→]*$")


def _label_key(label: str) -> str:
    stripped = label.strip()
    return stripped.split()[-1].upper() if stripped else ""


def _key_pattern(keys: Sequence[str]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9'])(?:response\s+)?({alternatives})(?![A-Za-z0-9'])", re.I)


def _extract(segment: str, pattern: re.Pattern[str], key_to_label: dict[str, str]) -> list[str]:
    found: list[str] = []
    for match in pattern.finditer(segment):
        label = key_to_label.get(match.group(1).upper())
        if label is not None and label not in found:
            found.append(label)
    return found


def _is_pure_ranking_line(line: str, pattern: re.Pattern[str]) -> bool:
    """True if ``line`` holds only label references, numbering and separators."""
    leftover = pattern.sub("", line)
    leftover = re.sub(r"(?i)\b(?:final\s+)?ranking\b", "", leftover)
    return bool(_PURE_LINE_LEFTOVER_RE.match(leftover))


def _from_marker(text: str, pattern: re.Pattern[str], key_to_label: dict[str, str]) -> list[str]:
    markers = list(_ANCHORED_MARKER_RE.finditer(text)) or list(_ANY_MARKER_RE.finditer(text))
    for marker in reversed(markers):
        lines = text[marker.end() :].split("\n")
        found = _extract(lines[0], pattern, key_to_label)
        if found:
            return found
        # Multi-line list after the marker, e.g. "FINAL RANKING:\n1. Response C\n2. ..."
        collected: list[str] = []
        for line in lines[1:]:
            if not line.strip():
                if collected:
                    break
                continue
            line_labels = _extract(line, pattern, key_to_label)
            if not line_labels:
                break
            collected.extend(label for label in line_labels if label not in collected)
        if collected:
            return collected
    return []


def _from_tail(
    text: str, pattern: re.Pattern[str], key_to_label: dict[str, str], n_labels: int
) -> list[str]:
    """No marker: accept a trailing ranking-only block, or a last line with >= 2 refs."""
    lines = text.strip().split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return []
    block: list[str] = []
    for line in reversed(lines):
        if (
            line.strip()
            and _is_pure_ranking_line(line, pattern)
            and _extract(line, pattern, key_to_label)
        ):
            block.insert(0, line)
        else:
            break
    collected: list[str] = []
    for line in block:
        collected.extend(
            label for label in _extract(line, pattern, key_to_label) if label not in collected
        )
    if collected and (len(collected) >= 2 or n_labels == 1 or len(block) == 1):
        return collected
    last = lines[-1]
    explicit = re.compile(pattern.pattern.replace(r"(?:response\s+)?", r"response\s+"), re.I)
    refs = _extract(last, explicit, key_to_label)
    return refs if len(refs) >= min(2, n_labels) else []


def parse_ranking(raw: str, labels: list[str]) -> list[str]:
    """Parse a reviewer's output into an ordered list of ``labels`` (best first).

    Tolerant of ``RANKING: A > C > B``, ``RANKING: Response B, Response A``,
    lowercase letters, markdown emphasis, numbered lists after ``FINAL RANKING:`` and,
    without any marker, a trailing ranking-only line such as ``B > A > C`` or
    ``Response B, Response A``. The last ``RANKING:`` line wins. Labels the reviewer
    left out are appended in ``labels`` order. Returns ``[]`` when nothing can be
    parsed.
    """
    if not raw or not labels:
        return []
    key_to_label: dict[str, str] = {}
    for label in labels:
        key = _label_key(label)
        if key:
            key_to_label.setdefault(key, label)
    if not key_to_label:
        return []
    pattern = _key_pattern(list(key_to_label))
    text = raw.replace("\r\n", "\n").replace("*", "").replace("`", "")
    found = _from_marker(text, pattern, key_to_label)
    if not found:
        found = _from_tail(text, pattern, key_to_label, len(labels))
    if not found:
        return []
    return found + [label for label in labels if label not in found]


def extract_rationale(raw: str) -> str:
    """Everything before the final ``RANKING:`` line (the whole text if there is none)."""
    text = raw.replace("\r\n", "\n")
    markers = list(_ANCHORED_MARKER_RE.finditer(text.replace("*", " ").replace("`", " ")))
    if not markers:
        return text.strip()
    return text[: markers[-1].start()].strip()


# --------------------------------------------------------------------------- council


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _format_error(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class _Session:
    """Per-``ask`` state: provider cache, event sink and cost accumulator."""

    def __init__(self, factory: ProviderFactory, on_event: EventCallback | None) -> None:
        self._factory = factory
        self._on_event = on_event
        self._providers: dict[str, Provider] = {}
        self.costs: list[float | None] = []

    def resolve(self, model_id: str) -> tuple[Provider, str]:
        provider_name, model_name = split_model_id(model_id)
        if provider_name not in self._providers:
            self._providers[provider_name] = self._factory(provider_name)
        return self._providers[provider_name], model_name

    def emit(
        self,
        stage: Stage,
        model: str,
        status: Literal["start", "done", "error"],
        *,
        round_: int = 0,
        message: str | None = None,
    ) -> None:
        if self._on_event is None:
            return
        event = CouncilEvent(stage=stage, model=model, status=status, round=round_, message=message)
        try:
            self._on_event(event)
        except Exception:  # a broken progress callback must not break the run
            logger.exception("on_event callback raised for %s", event)


class Council:
    """Runs one question through the council.

    Args:
        config: Members, chairman, debate rounds, max tokens.
        provider_factory: Maps a provider name (``"openrouter"``, ``"fake"``...) to a
            :class:`Provider`. Defaults to :func:`council.providers.get_provider`;
            tests inject their own.
    """

    def __init__(
        self, config: CouncilConfig, *, provider_factory: ProviderFactory = get_provider
    ) -> None:
        self.config = config
        self.provider_factory = provider_factory

    async def ask(self, question: str, *, on_event: EventCallback | None = None) -> CouncilRun:
        """Answer ``question`` with the full council pipeline and return the run.

        Individual model failures never raise: they are recorded in ``Answer.error``
        / ``Review.error`` (failed members are excluded from later stages), and a
        failed chairman leaves ``synthesis`` as ``None``. ``on_event`` receives a
        start and a done/error :class:`CouncilEvent` for every model call.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("question must not be empty")
        config = self.config
        session = _Session(self.provider_factory, on_event)
        run = CouncilRun(
            id=uuid.uuid4().hex[:12],
            question=question,
            created_at=_utc_now(),
            members=list(config.members),
            chairman=config.chairman,
            answers=[],
            reviews=[],
            synthesis=None,
            debate_rounds=config.debate_rounds,
            labels={},
        )

        # Stage 1: every member answers.
        run.answers = await self._answer_stage(
            session,
            "answer",
            [(member, prompts.answer_messages(question)) for member in run.members],
        )
        working = [answer for answer in run.answers if answer.error is None]

        # Stage 2: optional debate rounds; revised answers replace the working set.
        for round_no in range(1, config.debate_rounds + 1):
            calls = []
            for answer in working:
                others = [other for other in working if other.model != answer.model]
                random.Random(f"{run.id}:debate:{round_no}:{answer.model}").shuffle(others)
                labeled = list(zip(make_labels(len(others)), (o.text for o in others), strict=True))
                messages = prompts.debate_messages(
                    question,
                    answer.text,
                    labeled,
                    round_=round_no,
                    total_rounds=config.debate_rounds,
                )
                calls.append((answer.model, messages))
            revised = await self._answer_stage(session, "debate", calls, round_=round_no)
            run.debate_answers.append(revised)
            working = [answer for answer in revised if answer.error is None]

        # Stage 3: anonymous peer review of the final working set.
        if working:
            run.labels = assign_labels([answer.model for answer in working], seed=run.id)
        if len(working) >= 2:
            run.reviews = await self._review_stage(session, question, working, run.labels)

        # Stage 4: chairman synthesis.
        if working:
            run.synthesis = await self._synthesis_stage(session, run, working)

        known_costs = [cost for cost in session.costs if cost is not None]
        run.total_cost_usd = round(sum(known_costs), 8) if known_costs else None
        return run

    # ------------------------------------------------------------------ helpers

    async def _call(
        self,
        session: _Session,
        stage: Stage,
        model_id: str,
        messages: list[Message],
        *,
        round_: int = 0,
        emit_done: bool = True,
    ) -> CallResult:
        session.emit(stage, model_id, "start", round_=round_)
        started = time.perf_counter()
        try:
            provider, model_name = session.resolve(model_id)
            completion = await provider.complete(
                model_name, messages, max_tokens=self.config.max_tokens
            )
        except Exception as exc:
            latency = time.perf_counter() - started
            error = _format_error(exc)
            logger.info("%s call to %s failed: %s", stage, model_id, error)
            session.emit(stage, model_id, "error", round_=round_, message=error)
            return None, latency, error
        latency = time.perf_counter() - started
        session.costs.append(completion.cost_usd)
        if emit_done:
            session.emit(stage, model_id, "done", round_=round_)
        return completion, latency, None

    @staticmethod
    async def _gather(calls: Sequence[Awaitable[CallResult]]) -> list[CallResult]:
        results = await asyncio.gather(*calls, return_exceptions=True)
        normalized: list[CallResult] = []
        for result in results:
            if isinstance(result, BaseException):
                if not isinstance(result, Exception):  # CancelledError, KeyboardInterrupt...
                    raise result
                normalized.append((None, 0.0, _format_error(result)))
            else:
                normalized.append(result)
        return normalized

    async def _answer_stage(
        self,
        session: _Session,
        stage: Stage,
        calls: Sequence[tuple[str, list[Message]]],
        *,
        round_: int = 0,
    ) -> list[Answer]:
        results = await self._gather(
            [
                self._call(session, stage, model, messages, round_=round_)
                for model, messages in calls
            ]
        )
        answers: list[Answer] = []
        for (model, _), (completion, latency, error) in zip(calls, results, strict=True):
            if completion is None:
                answers.append(
                    Answer(model=model, text="", latency_s=round(latency, 3), error=error)
                )
            else:
                answers.append(
                    Answer(
                        model=model,
                        text=completion.text,
                        latency_s=round(latency, 3),
                        input_tokens=completion.input_tokens,
                        output_tokens=completion.output_tokens,
                        cost_usd=completion.cost_usd,
                    )
                )
        return answers

    async def _review_stage(
        self,
        session: _Session,
        question: str,
        working: Sequence[Answer],
        labels: dict[str, str],
    ) -> list[Review]:
        text_by_model = {answer.model: answer.text for answer in working}
        label_list = list(labels)
        messages = prompts.review_messages(
            question, [(label, text_by_model[labels[label]]) for label in label_list]
        )
        reviewers = [answer.model for answer in working]
        results = await self._gather(
            [
                self._call(session, "review", reviewer, messages, emit_done=False)
                for reviewer in reviewers
            ]
        )
        reviews: list[Review] = []
        for reviewer, (completion, _latency, error) in zip(reviewers, results, strict=True):
            if completion is None:
                reviews.append(
                    Review(reviewer=reviewer, ranking=[], rationale="", raw="", error=error)
                )
                continue
            raw = completion.text
            order = parse_ranking(raw, label_list)
            if not order:
                session.emit("review", reviewer, "error", message=UNPARSEABLE_RANKING_ERROR)
                reviews.append(
                    Review(
                        reviewer=reviewer,
                        ranking=[],
                        rationale=extract_rationale(raw),
                        raw=raw,
                        error=UNPARSEABLE_RANKING_ERROR,
                    )
                )
                continue
            session.emit("review", reviewer, "done")
            reviews.append(
                Review(
                    reviewer=reviewer,
                    ranking=[labels[label] for label in order],
                    rationale=extract_rationale(raw),
                    raw=raw,
                )
            )
        return reviews

    async def _synthesis_stage(
        self, session: _Session, run: CouncilRun, working: Sequence[Answer]
    ) -> Synthesis | None:
        text_by_model = {answer.model: answer.text for answer in working}
        answers = [
            (label, model, text_by_model[model])
            for label, model in run.labels.items()
            if model in text_by_model
        ]
        good_reviews = [review for review in run.reviews if review.ranking]
        means = mean_positions(review.ranking for review in good_reviews)
        ranking = [
            (model, means[model])
            for model in aggregate_ranking(review.ranking for review in good_reviews)
        ]
        messages = prompts.synthesis_messages(
            run.question,
            answers,
            [(review.reviewer, review.ranking, review.rationale) for review in good_reviews],
            ranking,
        )
        completion, latency, _error = await self._call(session, "synthesis", run.chairman, messages)
        if completion is None:
            return None
        return Synthesis(
            chairman=run.chairman,
            text=completion.text,
            latency_s=round(latency, 3),
            cost_usd=completion.cost_usd,
        )
