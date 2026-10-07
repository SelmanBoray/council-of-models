"""``council`` command line: ask the council, browse runs, vote, and inspect judge bias.

Commands: ``ask``, ``history``, ``show``, ``vote``, ``stats``, ``models``, ``serve``,
``mcp``. Optional extras (web server, MCP server) are imported lazily, so the core CLI
works without them.

All model-produced and user-provided text is rendered through :class:`rich.text.Text`
or :class:`rich.markdown.Markdown`, never as Rich markup, so brackets such as
``[fake:alpha]`` in an answer are printed verbatim.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from pydantic import ValidationError
from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from council import __version__, analytics
from council.config import (
    DOTENV_FILENAME,
    CouncilConfig,
    find_config_file,
    load_config,
    load_dotenv_file,
)
from council.council import Council
from council.errors import CouncilError
from council.models import Answer, CouncilEvent, CouncilRun, Review, mean_positions
from council.providers import PROVIDER_KEY_ENV, PROVIDER_NAMES, split_model_id
from council.store import RunStore

FAKE_MEMBERS = ("fake:alpha", "fake:beta", "fake:gamma")
FAKE_CHAIRMAN = "fake:alpha"
ANSWER_PREVIEW_CHARS = 600
QUESTION_PREVIEW_CHARS = 70
# Providers whose credentials may come from more than one environment variable.
EXTRA_KEY_ENV: dict[str, tuple[str, ...]] = {"anthropic": ("ANTHROPIC_AUTH_TOKEN",)}

app = typer.Typer(
    name="council",
    help=(
        "Ask a council of LLMs. Every member answers, anonymously ranks all answers "
        "(its own included), and a chairman synthesizes the final answer. "
        "Stored runs feed judge-bias analytics: self-preference, reviewer agreement, "
        "human-vs-council agreement."
    ),
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        dir_okay=False,
        help="Path to a council.toml (default: ./council.toml, then "
        "~/.config/council/config.toml, then built-in defaults).",
    ),
]
JsonOption = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON only.")]


# ------------------------------------------------------------------ console helpers


@dataclass(frozen=True)
class _Symbols:
    unicode: bool
    ok: str
    fail: str
    pending: str
    dash: str
    ellipsis: str
    dot: str
    bar: str
    spinner: str


_UNICODE = _Symbols(True, "✓", "✗", "…", "—", "…", "·", "█", "dots")
_ASCII = _Symbols(False, "OK", "x", "..", "-", "...", "|", "#", "line")


def _symbols(console: Console) -> _Symbols:
    """Unicode glyphs when the output can encode them, ASCII otherwise (Windows pipes)."""
    return _UNICODE if _is_utf8(console.encoding) else _ASCII


def _is_utf8(encoding: str | None) -> bool:
    return (encoding or "").lower().replace("-", "").replace("_", "") == "utf8"


def _out() -> Console:
    return Console(highlight=False)


def _err() -> Console:
    return Console(stderr=True, highlight=False)


def _make_streams_safe() -> None:
    """Never crash on characters the output encoding cannot represent.

    On Windows, piped output uses the ANSI code page; replacing unencodable characters
    beats a UnicodeEncodeError halfway through printing a run.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure) and not _is_utf8(getattr(stream, "encoding", None)):
            try:
                reconfigure(errors="replace")
            except (OSError, ValueError):  # pragma: no cover - exotic streams
                pass


def _print_json(text: str) -> None:
    """Write JSON to stdout, always as UTF-8 (RFC 8259) whatever the console code page."""
    stream = sys.stdout
    buffer = getattr(stream, "buffer", None)
    if buffer is not None and not _is_utf8(getattr(stream, "encoding", None)):
        stream.flush()
        buffer.write((text + "\n").encode("utf-8"))
        buffer.flush()
    else:
        typer.echo(text)


def _fail(
    message: str, *, hint: str | None = None, details: Iterable[str] = (), code: int = 1
) -> NoReturn:
    """Print a readable error (no traceback) to stderr and exit with ``code``."""
    console = _err()
    console.print(Text.assemble(("Error: ", "bold red"), message))
    for line in details:
        console.print(Text("  " + line))
    if hint:
        console.print(Text(hint, style="dim"))
    raise typer.Exit(code)


def _explain(console: Console, text: str) -> None:
    console.print(Text(text, style="dim"))


def _one_line(text: str, limit: int, sym: _Symbols) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - len(sym.ellipsis)].rstrip() + sym.ellipsis


def _short_names(models: Iterable[str]) -> dict[str, str]:
    """``openrouter:openai/gpt-5.1`` -> ``gpt-5.1``; colliding short names keep the full id."""
    unique = list(dict.fromkeys(models))
    shorts: dict[str, str] = {}
    for model in unique:
        try:
            _, name = split_model_id(model)
        except ValueError:
            name = model
        shorts[model] = name.rsplit("/", 1)[-1]
    counts = Counter(shorts.values())
    return {model: (short if counts[short] == 1 else model) for model, short in shorts.items()}


def _format_time(created_at: str) -> str:
    try:
        return datetime.fromisoformat(created_at).astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return created_at


def _format_cost(cost: float | None, sym: _Symbols) -> str:
    return sym.dash if cost is None else f"${cost:.4f}"


# ------------------------------------------------------------------- config & store


def _load_config(path: Path | None) -> CouncilConfig:
    try:
        return load_config(path)
    except (OSError, ValueError) as exc:
        _fail(str(exc))


def _providers_of(config: CouncilConfig) -> list[str]:
    return list(dict.fromkeys(split_model_id(m)[0] for m in [*config.members, config.chairman]))


def _apply_overrides(
    config: CouncilConfig,
    *,
    members: str | None,
    chairman: str | None,
    debate: int | None,
    fake: bool,
) -> CouncilConfig:
    updates: dict[str, Any] = {}
    if fake:
        if members is not None or chairman is not None:
            _err().print(Text("--fake overrides --members/--chairman.", style="yellow"))
        updates.update(members=list(FAKE_MEMBERS), chairman=FAKE_CHAIRMAN)
    else:
        if members is not None:
            updates["members"] = [m.strip() for m in members.split(",") if m.strip()]
        if chairman is not None:
            updates["chairman"] = chairman.strip()
    if debate is not None:
        updates["debate_rounds"] = debate
    try:
        updated = CouncilConfig.model_validate({**config.model_dump(), **updates})
    except ValidationError as exc:
        _fail(
            "invalid council settings",
            details=[
                f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
                for err in exc.errors()
            ],
        )
    unknown = [p for p in _providers_of(updated) if p not in PROVIDER_NAMES]
    if unknown:
        _fail(
            f"unknown provider(s): {', '.join(unknown)}",
            hint=f"Known providers: {', '.join(PROVIDER_NAMES)} (model ids look like "
            "'openrouter:openai/gpt-5.1' or 'fake:alpha').",
        )
    return updated


def _load_run(store: RunStore, run_id: str) -> CouncilRun:
    try:
        return store.load(run_id.strip())
    except KeyError:
        _fail(
            f"run {run_id!r} not found in {store.runs_dir}",
            hint="List stored runs with: council history",
        )
    except (OSError, ValueError) as exc:
        _fail(f"could not read run {run_id!r}: {exc}")


def _no_runs_message(settings: CouncilConfig) -> str:
    return f'No runs yet in {settings.data_dir}. Try: council ask "..." --fake'


def _final_answers(run: CouncilRun) -> dict[str, Answer]:
    """Latest answer per model: debate revisions replace the initial answers."""
    latest = {answer.model: answer for answer in run.answers}
    for round_answers in run.debate_answers:
        for answer in round_answers:
            latest[answer.model] = answer
    return latest


# ------------------------------------------------------------------- live progress

_STAGE_TITLES = {
    "answer": "Answers",
    "debate": "Debate",
    "review": "Reviews",
    "synthesis": "Synthesis",
}


class _Progress:
    """Collects :class:`CouncilEvent` objects and renders a per-stage status table."""

    def __init__(self, config: CouncilConfig, sym: _Symbols) -> None:
        self.sym = sym
        self.rows: list[tuple[str, int]] = [
            ("answer", 0),
            *(("debate", r) for r in range(1, config.debate_rounds + 1)),
            ("review", 0),
            ("synthesis", 0),
        ]
        self.states: dict[tuple[str, int], dict[str, str]] = {row: {} for row in self.rows}
        self.messages: dict[tuple[str, str], str] = {}
        self.started: dict[tuple[str, int], float] = {}
        self.ended: dict[tuple[str, int], float] = {}
        self.spinners = {row: Spinner(sym.spinner, style="cyan") for row in self.rows}
        self.short = _short_names([*config.members, config.chairman])
        self.finished = False

    def handle(self, event: CouncilEvent) -> None:
        row = (event.stage, event.round if event.stage == "debate" else 0)
        if row not in self.states:  # defensive: an unexpected debate round
            self.rows.insert(len(self.rows) - 2, row)
            self.states[row] = {}
            self.spinners[row] = Spinner(self.sym.spinner, style="cyan")
        self.states[row][event.model] = event.status
        if event.message:
            self.messages[(event.stage, event.model)] = event.message
        now = time.monotonic()
        self.started.setdefault(row, now)
        if event.status != "start":
            self.ended[row] = now

    def error(self, stage: str, model: str) -> str | None:
        return self.messages.get((stage, model))

    def render(self) -> Table:
        sym = self.sym
        table = Table.grid(padding=(0, 2))
        table.add_column(no_wrap=True)
        table.add_column(no_wrap=True)
        table.add_column()
        for row in self.rows:
            stage, round_ = row
            title = _STAGE_TITLES[stage] + (f" {round_}" if stage == "debate" else "")
            states = self.states[row]
            done = sum(1 for s in states.values() if s == "done")
            failed = sum(1 for s in states.values() if s == "error")
            running = len(states) - done - failed
            status: RenderableType
            if not states:
                status = Text("skipped" if self.finished else "waiting", style="dim")
            elif running and not self.finished:
                spinner = self.spinners[row]
                spinner.update(text=Text(f"{done + failed}/{len(states)}"))
                status = spinner
            else:
                elapsed = self.ended.get(row, time.monotonic()) - self.started.get(row, 0.0)
                style = "green" if not failed else ("red" if not done else "yellow")
                glyph = sym.fail if failed else sym.ok
                label = f"{glyph} {done}/{len(states)}" + (" ok" if failed else "")
                status = Text.assemble((label, style), (f"  {elapsed:.1f}s", "dim"))
            models = Text()
            for model, state in states.items():
                if models:
                    models.append("  ")
                glyph, style = {
                    "start": (sym.pending, "cyan"),
                    "done": (sym.ok, "green"),
                    "error": (sym.fail, "red"),
                }[state]
                models.append(self.short.get(model, model) + " ")
                models.append(glyph, style=style)
            table.add_row(Text(title, style="bold"), status, models)
        return table


def _run_council(
    council: Council, question: str, on_event: Callable[[CouncilEvent], None] | None
) -> CouncilRun:
    return asyncio.run(council.ask(question, on_event=on_event))


# ---------------------------------------------------------------------- rendering


def _answer_panel(
    answer: Answer,
    *,
    position: int | None,
    label: str | None,
    full: bool,
    sym: _Symbols,
    truncation_hint: str,
) -> Panel:
    title = Text()
    if position is not None:
        title.append(f"#{position} ", style="bold green" if position == 1 else "bold")
    title.append(answer.model, style="bold")
    if label:
        title.append(f" {sym.dot} {label}", style="dim")
    if answer.error:
        return Panel(
            Text(answer.error, style="red"),
            title=title,
            title_align="left",
            subtitle=Text("failed", style="red"),
            subtitle_align="right",
            border_style="red",
        )
    text = answer.text.strip()
    body: RenderableType
    if not full and len(text) > ANSWER_PREVIEW_CHARS:
        cut = text[:ANSWER_PREVIEW_CHARS].rstrip() + sym.ellipsis
        hint = f"({len(text)} characters, truncated; {truncation_hint})"
        body = Group(Markdown(cut), Text(hint, style="dim"))
    else:
        body = Markdown(text) if text else Text("(empty answer)", style="dim")
    meta = [f"{answer.latency_s:.1f}s"]
    if answer.output_tokens is not None:
        meta.append(f"{answer.output_tokens} tokens")
    if answer.cost_usd is not None:
        meta.append(_format_cost(answer.cost_usd, sym))
    return Panel(
        body,
        title=title,
        title_align="left",
        subtitle=Text(f" {sym.dot} ".join(meta), style="dim"),
        subtitle_align="right",
        border_style="green" if position == 1 else "blue",
    )


def _print_answers(
    console: Console,
    run: CouncilRun,
    answers: dict[str, Answer],
    *,
    full: bool,
    sym: _Symbols,
    truncation_hint: str,
    final: bool = True,
) -> None:
    """One panel per answer, in council-ranking order.

    ``final`` marks the answers the council actually reviewed: only those get their
    anonymous label and council position in the panel title.
    """
    ranked = final and analytics.council_top(run) is not None
    order = run.council_ranking()
    order += [model for model in answers if model not in order]
    label_of = {model: label for label, model in run.labels.items()}
    ranked_models = set(mean_positions(review.ranking for review in run.reviews))
    for position, model in enumerate(order, start=1):
        answer = answers.get(model)
        if answer is None:
            continue
        console.print(
            _answer_panel(
                answer,
                position=position if ranked and model in ranked_models else None,
                label=label_of.get(model) if final else None,
                full=full,
                sym=sym,
                truncation_hint=truncation_hint,
            )
        )


def _ranking_table(run: CouncilRun, sym: _Symbols) -> Table:
    reviews = run.reviews
    means = mean_positions(review.ranking for review in reviews)
    short = _short_names([review.reviewer for review in reviews])
    table = Table(title="Council ranking")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Mean rank", justify="right", no_wrap=True)
    positions: list[dict[str, int]] = []
    for review in reviews:
        header = Text(short[review.reviewer])
        if not review.ranking:
            header.append("\n(failed)", style="red")
        table.add_column(header, justify="center", no_wrap=True)
        positions.append({model: i for i, model in enumerate(review.ranking, start=1)})
    for index, model in enumerate(run.council_ranking(), start=1):
        name = Text(model)
        if run.human_vote == model:
            name.append("  (your pick)", style="magenta")
        if model not in means:
            name.append("  (not ranked)", style="dim")
            table.add_row(
                Text(sym.dash, style="dim"),
                name,
                Text(sym.dash, style="dim"),
                *(Text(sym.dash, style="dim") for _ in reviews),
            )
            continue
        cells: list[Text] = []
        for review, pos in zip(reviews, positions, strict=True):
            position = pos.get(model)
            if position is None:
                cells.append(Text(sym.dash, style="dim"))
            elif review.reviewer == model:
                cells.append(Text(f"{position}*", style="bold yellow"))
            else:
                cells.append(Text(str(position)))
        style = "bold green" if index == 1 else ""
        table.add_row(Text(str(index), style=style), name, Text(f"{means[model]:.2f}"), *cells)
    return table


def _print_rankings(console: Console, run: CouncilRun, sym: _Symbols) -> None:
    if not run.reviews:
        console.print(
            Text("Peer review skipped: fewer than two members produced an answer.", style="dim")
        )
        return
    console.print(_ranking_table(run, sym))
    _explain(
        console,
        "Reviewer columns: the position each reviewer gave the answer (1 = best); "
        "* marks the reviewer's own answer.",
    )
    for review in run.reviews:
        if review.error:
            console.print(
                Text.assemble((f"Review by {review.reviewer} failed: ", "yellow"), review.error)
            )


def _synthesis_panel(run: CouncilRun, sym: _Symbols, chairman_error: str | None) -> Panel:
    if run.synthesis is None:
        message = chairman_error or "The chairman did not produce a final answer."
        return Panel(
            Text(message, style="red"),
            title=Text(f"Final answer {sym.dot} chairman {run.chairman} failed", style="bold red"),
            title_align="left",
            border_style="red",
        )
    synthesis = run.synthesis
    meta = [f"{synthesis.latency_s:.1f}s"]
    if synthesis.cost_usd is not None:
        meta.append(_format_cost(synthesis.cost_usd, sym))
    return Panel(
        Markdown(synthesis.text),
        title=Text(f"Final answer {sym.dot} chairman {synthesis.chairman}", style="bold"),
        title_align="left",
        subtitle=Text(f" {sym.dot} ".join(meta), style="dim"),
        subtitle_align="right",
        border_style="green",
    )


def _review_panel(review: Review, short: dict[str, str], sym: _Symbols) -> Panel:
    if not review.ranking:
        body: RenderableType = Group(
            Text(review.error or "no ranking", style="red"),
            Text(review.raw.strip() or "(no output)", style="dim"),
        )
        subtitle = Text("failed", style="red")
        border = "red"
    else:
        body = Markdown(review.rationale) if review.rationale.strip() else Text("(no rationale)")
        subtitle = Text(" > ".join(short.get(m, m) for m in review.ranking), style="dim")
        border = "blue"
    return Panel(
        body,
        title=Text(f"Review by {review.reviewer}", style="bold"),
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style=border,
    )


def _footer(console: Console, run: CouncilRun, path: Path | None) -> None:
    console.print()
    line = Text.assemble(("Run id: ", "bold"), (run.id, "bold cyan"))
    if path is not None:
        line.append(f"  (saved to {path})", style="dim")
    console.print(line)
    console.print(
        Text.assemble("Record your own pick: ", (f"council vote {run.id} <model>", "bold"))
    )
    short = ", ".join(_short_names(run.members).values())
    _explain(console, f"<model> is a member id or short name: {short}")


# ------------------------------------------------------------------------ commands


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"council-of-models {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show the version."
        ),
    ] = False,
) -> None:
    _make_streams_safe()
    # Provider keys from ./.env; variables already set in the environment win.
    loaded = load_dotenv_file(Path(DOTENV_FILENAME))
    state = ctx.ensure_object(dict)
    state["dotenv_keys"] = sorted(loaded)  # names only: the values are secrets


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to put to the council.")],
    members: Annotated[
        str | None,
        typer.Option(
            "--members",
            "-m",
            help="Comma-separated member model ids, e.g. "
            "'openrouter:openai/gpt-5.1,anthropic:claude-opus-5-5'. Overrides the config.",
        ),
    ] = None,
    chairman: Annotated[
        str | None,
        typer.Option("--chairman", help="Chairman model id. Overrides the config."),
    ] = None,
    debate: Annotated[
        int | None,
        typer.Option("--debate", min=0, help="Debate rounds before review (overrides config)."),
    ] = None,
    config: ConfigOption = None,
    json_output: JsonOption = False,
    fake: Annotated[
        bool,
        typer.Option(
            "--fake",
            help="Offline demo: members fake:alpha,fake:beta,fake:gamma, chairman fake:alpha.",
        ),
    ] = False,
    full: Annotated[
        bool, typer.Option("--full", help="Print answers in full instead of truncated.")
    ] = False,
) -> None:
    """Ask the council a question: answers, (debate), anonymous peer review, synthesis."""
    if not question.strip():
        _fail("the question must not be empty")
    settings = _apply_overrides(
        _load_config(config), members=members, chairman=chairman, debate=debate, fake=fake
    )
    council = Council(settings)
    console = _out()
    sym = _symbols(console)
    progress: _Progress | None = None
    try:
        if json_output:
            run = _run_council(council, question, None)
        else:
            tracker = progress = _Progress(settings, sym)
            with Live(tracker.render(), console=console, refresh_per_second=8) as live:

                def on_event(event: CouncilEvent) -> None:
                    tracker.handle(event)
                    live.update(tracker.render())

                run = _run_council(council, question, on_event)
                tracker.finished = True
                live.update(tracker.render(), refresh=True)
    except KeyboardInterrupt:
        _fail("interrupted", code=130)
    except (CouncilError, ValueError) as exc:
        _fail(str(exc))

    if not any(answer.error is None for answer in run.answers):
        _fail(
            "no council member produced an answer",
            details=[f"{answer.model}: {answer.error}" for answer in run.answers],
            hint="Check provider keys with 'council models', or try '--fake' for an offline demo.",
        )
    try:
        path = RunStore(settings.data_dir).save(run)
    except (OSError, ValueError) as exc:
        _fail(f"could not save the run: {exc}")

    if json_output:
        _print_json(run.model_dump_json(indent=2))
        return

    if not console.is_terminal:  # Live adds the line break itself on a terminal
        console.print()
    console.print()
    _print_answers(
        console,
        run,
        _final_answers(run),
        full=full,
        sym=sym,
        truncation_hint=f"use --full or 'council show {run.id}'",
    )
    if run.debate_answers:
        console.print(
            Text(
                f"Answers above are the revised ones after {len(run.debate_answers)} debate "
                f"round(s); 'council show {run.id}' shows every round.",
                style="dim",
            )
        )
    console.print()
    _print_rankings(console, run, sym)
    console.print()
    chairman_error = progress.error("synthesis", run.chairman) if progress else None
    console.print(_synthesis_panel(run, sym, chairman_error))
    _footer(console, run, path)


@app.command()
def history(
    limit: Annotated[
        int, typer.Option("--limit", "-n", min=1, help="How many runs to show (newest first).")
    ] = 20,
    config: ConfigOption = None,
) -> None:
    """List past runs, newest first."""
    settings = _load_config(config)
    runs = RunStore(settings.data_dir).list(limit=limit)
    console = _out()
    sym = _symbols(console)
    if not runs:
        console.print(Text(_no_runs_message(settings), style="dim"))
        return
    short = _short_names(model for run in runs for model in run.members)
    show_cost = any(run.total_cost_usd for run in runs)
    table = Table(title=f"Council runs ({len(runs)} shown, newest first)", box=box.SIMPLE_HEAD)
    table.add_column("Run id", style="cyan", no_wrap=True)
    table.add_column("Created", no_wrap=True)
    table.add_column("Question", min_width=16)
    table.add_column("Council #1")
    table.add_column("Your pick")
    if show_cost:
        table.add_column("Cost", justify="right", no_wrap=True)
    for run in runs:
        top = analytics.council_top(run)
        top_cell = Text(short.get(top, top) if top else sym.dash, style="" if top else "dim")
        vote = run.human_vote
        vote_cell = Text(short.get(vote, vote) if vote else sym.dash, style="" if vote else "dim")
        if vote and top:
            agree = vote == top
            vote_cell.append(
                f" {sym.ok if agree else sym.fail}", style="green" if agree else "yellow"
            )
        cells = [
            Text(run.id),
            Text(_format_time(run.created_at)),
            Text(_one_line(run.question, QUESTION_PREVIEW_CHARS, sym)),
            top_cell,
            vote_cell,
        ]
        if show_cost:
            cells.append(Text(_format_cost(run.total_cost_usd, sym)))
        table.add_row(*cells)
    console.print(table)
    _explain(console, "Details: council show <run_id>")


@app.command()
def show(
    run_id: Annotated[str, typer.Argument(help="Run id (see 'council history').")],
    json_output: JsonOption = False,
    config: ConfigOption = None,
) -> None:
    """Show everything about one run: answers, debate, reviews, ranking, synthesis."""
    settings = _load_config(config)
    run = _load_run(RunStore(settings.data_dir), run_id)
    if json_output:
        _print_json(run.model_dump_json(indent=2))
        return
    console = _out()
    sym = _symbols(console)

    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold", no_wrap=True)
    meta.add_column()
    meta.add_row("Run id", Text(run.id, style="cyan"))
    meta.add_row("Created", Text(_format_time(run.created_at)))
    meta.add_row("Members", Text(", ".join(run.members)))
    meta.add_row("Chairman", Text(run.chairman))
    meta.add_row("Debate rounds", Text(str(run.debate_rounds)))
    meta.add_row("Cost", Text(_format_cost(run.total_cost_usd, sym)))
    meta.add_row("Your pick", Text(run.human_vote or sym.dash))
    console.print(
        Panel(
            Group(Text(run.question, style="bold"), Text(), meta),
            title="Council run",
            title_align="left",
        )
    )

    hint = "this is the full stored text"
    if run.debate_answers:
        console.print(Rule(Text("Initial answers")))
        _print_answers(
            console,
            run,
            {a.model: a for a in run.answers},
            full=True,
            sym=sym,
            truncation_hint=hint,
            final=False,
        )
        for round_no, round_answers in enumerate(run.debate_answers, start=1):
            console.print(Rule(Text(f"Debate round {round_no}")))
            _print_answers(
                console,
                run,
                {a.model: a for a in round_answers},
                full=True,
                sym=sym,
                truncation_hint=hint,
                final=round_no == len(run.debate_answers),
            )
    else:
        console.print(Rule(Text("Answers")))
        _print_answers(console, run, _final_answers(run), full=True, sym=sym, truncation_hint=hint)

    if run.reviews:
        console.print(Rule(Text("Reviews")))
        short = _short_names(run.members)
        for review in run.reviews:
            console.print(_review_panel(review, short, sym))
    console.print()
    _print_rankings(console, run, sym)
    console.print()
    console.print(_synthesis_panel(run, sym, None))


def _resolve_member(run: CouncilRun, model: str) -> str:
    """Accept the full model id, or an unambiguous short name such as ``beta``/``gpt-5.1``."""
    wanted = model.strip()
    if wanted in run.members:
        return wanted
    short = _short_names(run.members)
    needle = wanted.lower()
    matches = []
    for member in run.members:
        try:
            _, name = split_model_id(member)
        except ValueError:
            name = member
        if needle in {member.lower(), name.lower(), short[member].lower()}:
            matches.append(member)
    if len(matches) == 1:
        return matches[0]
    problem = "is ambiguous" if matches else "is not a member of this run"
    _fail(
        f"{model!r} {problem}",
        hint=f"Choose one of: {', '.join(matches or run.members)}",
    )


@app.command()
def vote(
    run_id: Annotated[str, typer.Argument(help="Run id (see 'council history').")],
    model: Annotated[
        str, typer.Argument(help="The member whose answer you think is best (id or short name).")
    ],
    config: ConfigOption = None,
) -> None:
    """Record which answer you think is best; feeds the human-vs-council statistics."""
    settings = _load_config(config)
    store = RunStore(settings.data_dir)
    run = _load_run(store, run_id)
    member = _resolve_member(run, model)
    try:
        run = store.set_human_vote(run.id, member)
    except KeyError:
        _fail(f"run {run_id!r} not found in {store.runs_dir}")
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    console = _out()
    sym = _symbols(console)
    console.print(
        Text.assemble("Recorded your pick for run ", (run.id, "cyan"), ": ", (member, "bold"))
    )
    top = analytics.council_top(run)
    if top is None:
        console.print(Text("The council produced no ranking for this run.", style="dim"))
    elif top == member:
        console.print(Text(f"{sym.ok} You agree with the council's #1.", style="green"))
    else:
        console.print(
            Text.assemble(
                (f"{sym.fail} The council's #1 was ", "yellow"), (top, "bold"), ".", style="yellow"
            )
        )


def _tau_style(value: float) -> str:
    if value >= 0.5:
        return "green"
    if value >= 0.0:
        return "yellow"
    return "red"


@app.command()
def stats(config: ConfigOption = None, json_output: JsonOption = False) -> None:
    """Judge-bias analytics over all stored runs."""
    settings = _load_config(config)
    runs = RunStore(settings.data_dir).list()
    data = analytics.summary(runs)
    if json_output:
        _print_json(json.dumps(data, indent=2, ensure_ascii=False))
        return
    console = _out()
    sym = _symbols(console)
    if not runs:
        console.print(Text(_no_runs_message(settings), style="dim"))
        return
    console.print(
        Text.assemble(
            ("Council statistics: ", "bold"),
            f"{data['n_runs']} run(s), total cost {_format_cost(data['total_cost_usd'], sym)}",
        )
    )
    console.print()

    board = data["leaderboard"]
    console.print(Text("Leaderboard", style="bold"))
    _explain(
        console,
        "Mean position across all peer reviews (1 = best). Wins = runs where the model "
        "was the council's #1.",
    )
    if board:
        table = Table()
        table.add_column("#", justify="right")
        table.add_column("Model", no_wrap=True)
        table.add_column("Mean rank", justify="right")
        table.add_column("Wins", justify="right")
        table.add_column("Reviews", justify="right")
        for index, row in enumerate(board, start=1):
            table.add_row(
                str(index),
                Text(row["model"]),
                f"{row['mean_rank']:.2f}",
                str(row["wins"]),
                str(row["appearances"]),
            )
        console.print(table)
    else:
        _explain(console, "No successful peer reviews yet.")
    console.print()

    prefs: dict[str, float] = data["self_preference"]
    console.print(Text("Self-preference", style="bold"))
    _explain(
        console,
        "Mean rank the other reviewers gave a model's answer minus the rank it gave "
        "itself. Positive = the model rates its own answer better than its peers do.",
    )
    if prefs:
        widest = max(abs(v) for v in prefs.values()) or 1.0
        table = Table(show_header=True)
        table.add_column("Model", no_wrap=True)
        table.add_column("Bias (ranks)", justify="right", no_wrap=True)
        table.add_column("", no_wrap=True)
        for model, value in prefs.items():
            style = "red" if value > 0.25 else ("green" if value < -0.25 else "yellow")
            bar = sym.bar * max(1 if value else 0, round(abs(value) / widest * 20))
            table.add_row(Text(model), Text(f"{value:+.2f}", style=style), Text(bar, style=style))
        console.print(table)
    else:
        _explain(console, "Not enough reviews yet (needs a model's own ranking plus a peer's).")
    console.print()

    matrix: dict[str, dict[str, float]] = data["agreement_matrix"]
    console.print(Text("Reviewer agreement (Kendall tau)", style="bold"))
    _explain(
        console,
        "How similarly two reviewers order the answers: 1 = same order, 0 = unrelated, "
        "-1 = reversed; averaged over the runs both reviewed.",
    )
    if matrix:
        short = _short_names(matrix)
        table = Table()
        table.add_column("Reviewer", no_wrap=True)
        for reviewer in matrix:
            table.add_column(Text(short[reviewer]), justify="right", no_wrap=True)
        for reviewer, row in matrix.items():
            cells = []
            for other in matrix:
                value = row.get(other)
                cells.append(
                    Text(sym.dash, style="dim")
                    if value is None
                    else Text(f"{value:+.2f}", style=_tau_style(value))
                )
            table.add_row(Text(short[reviewer]), *cells)
        console.print(table)
    else:
        _explain(console, "No successful peer reviews yet.")
    console.print()

    human = data["human_vs_council"]
    if human["n"] == 0:
        console.print(
            Text.assemble(
                ("Human vs council (n=0): ", "bold"),
                "no votes yet. Record one with: council vote <run_id> <model>",
            )
        )
    else:
        rate = human["agreement_rate"]
        console.print(
            Text.assemble(
                (f"Human vs council (n={human['n']}): ", "bold"),
                f"your pick matched the council's #1 in {rate * 100:.1f}% of voted runs.",
            )
        )
        picks = ", ".join(f"{m} {c}" for m, c in human["per_model_human_wins"].items())
        _explain(console, f"Your picks: {picks}")


def _key_status(provider: str) -> tuple[bool, str, str]:
    """``(ok, env var, status)`` for a provider's credentials. Never reads out key values."""
    if provider not in PROVIDER_NAMES:
        return False, "?", "unknown provider"
    env = PROVIDER_KEY_ENV.get(provider)
    if env is None:
        if provider == "ollama":
            return True, "OLLAMA_HOST (optional)", "no key needed (local server)"
        return True, "", "no key needed (offline)"
    names = (env, *EXTRA_KEY_ENV.get(provider, ()))
    present = any(os.environ.get(name, "").strip() for name in names)
    return present, env, "set" if present else "not set"


def _dotenv_line(ctx: typer.Context) -> Text:
    line = Text("Env file: ", style="bold")
    dotenv = Path(DOTENV_FILENAME)
    if not dotenv.is_file():
        line.append("none (no .env in the current directory)", style="dim")
        return line
    keys = ctx.obj.get("dotenv_keys", []) if isinstance(ctx.obj, dict) else []
    line.append(str(dotenv.resolve()))
    detail = (
        f"loaded {', '.join(keys)}"
        if keys
        else "nothing loaded: variables already set in the environment, or empty"
    )
    line.append(f"  ({detail})", style="dim")
    return line


@app.command("models")
def models_command(ctx: typer.Context, config: ConfigOption = None) -> None:
    """Show the configured council and whether each provider's API key is set."""
    try:
        source = find_config_file(config)
    except FileNotFoundError as exc:
        _fail(str(exc))
    settings = _load_config(config)
    console = _out()
    sym = _symbols(console)
    console.print(
        Text.assemble(
            ("Config: ", "bold"),
            str(source) if source else "built-in defaults (no council.toml found)",
        )
    )
    console.print(Text.assemble(("Data dir: ", "bold"), str(settings.data_dir.resolve())))
    console.print(Text.assemble(("Debate rounds: ", "bold"), str(settings.debate_rounds)))
    console.print(_dotenv_line(ctx))
    console.print()

    council_table = Table(title="Council", box=box.SIMPLE_HEAD)
    council_table.add_column("Role", no_wrap=True)
    council_table.add_column("Model")
    for member in settings.members:
        council_table.add_row(Text("member"), Text(member))
    council_table.add_row(Text("chairman", style="bold"), Text(settings.chairman))
    console.print(council_table)

    keys_table = Table(title="Provider keys", box=box.SIMPLE_HEAD)
    keys_table.add_column("Provider", no_wrap=True)
    keys_table.add_column("Env var")
    keys_table.add_column("Status")
    missing: list[str] = []
    for provider in _providers_of(settings):
        ok, env, status = _key_status(provider)
        if not ok:
            missing.append(env if env != "?" else f"{provider} (unknown provider)")
        keys_table.add_row(
            Text(provider),
            Text(env or sym.dash, style="" if env else "dim"),
            Text.assemble((sym.ok if ok else sym.fail, "green" if ok else "red"), " ", status),
        )
    console.print(keys_table)
    if missing:
        _explain(
            console,
            f"Missing: {', '.join(missing)}. Export the variable(s) or put them in ./.env, "
            "or try 'council ask \"...\" --fake' for an offline demo.",
        )
    else:
        console.print(Text(f"{sym.ok} Every provider in use has what it needs.", style="green"))


def _display_url(host: str, port: int) -> str:
    if host in {"0.0.0.0", "::", ""}:
        host = "localhost"
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    return f"http://{host}:{port}"


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[
        int, typer.Option("--port", min=1, max=65535, help="Port to listen on.")
    ] = 8000,
    config: ConfigOption = None,
    fake: Annotated[
        bool,
        typer.Option(
            "--fake",
            help="Serve the offline demo council: fake:alpha,fake:beta,fake:gamma, "
            "chairman fake:alpha.",
        ),
    ] = False,
) -> None:
    """Start the web UI and JSON API (needs the 'web' extra)."""
    try:
        import uvicorn

        from council.web.server import create_app
    except ImportError as exc:
        _fail(
            f"the web server is not available ({exc})",
            hint='Install the web extra: pip install "council-of-models[web]"',
        )
    settings = _apply_overrides(
        _load_config(config), members=None, chairman=None, debate=None, fake=fake
    )
    web_app = create_app(settings)
    # create_app may still swap in the fake council (COUNCIL_FAKE=1); show what it serves.
    effective = getattr(getattr(web_app, "state", None), "config", None)
    if not isinstance(effective, CouncilConfig):
        effective = settings
    console = _out()
    console.print(
        Text.assemble(
            ("Council web UI: ", "bold"),
            (_display_url(host, port), "bold cyan"),
            ("  (Ctrl+C to stop)", "dim"),
        )
    )
    _explain(
        console,
        f"Members: {', '.join(effective.members)}; chairman: {effective.chairman}; "
        f"runs in {effective.data_dir}",
    )
    uvicorn.run(web_app, host=host, port=port)


@app.command("mcp")
def mcp_command() -> None:
    """Run the MCP server over stdio, for Claude Code, Cursor and other MCP clients."""
    try:
        from council.mcp_server import main as mcp_main
    except ImportError as exc:
        _fail(
            f"the MCP server is not available ({exc})",
            hint='Install the MCP extra: pip install "council-of-models[mcp]"',
        )
    mcp_main()


if __name__ == "__main__":  # pragma: no cover
    app()
