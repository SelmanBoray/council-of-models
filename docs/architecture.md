# Architecture

council-of-models is a small Python package (`src/council/`) with three thin surfaces on top: a CLI, an MCP server and a web app. Every surface calls the same `Council` class and reads and writes the same JSON run files.

```text
  CLI (typer)      MCP server (FastMCP, stdio)      web app (FastAPI + React)
       \                      |                              /
        +-------------> Council.ask(question) <-------------+
                              |
             providers: openrouter | anthropic | openai | ollama | fake
                              |
                         CouncilRun  --->  RunStore  --->  data/runs/<id>.json
                                                                 |
                                         analytics (pure)  <-----+
```

## Package layout

| module | responsibility |
| ------ | -------------- |
| `council/models.py` | Pydantic models shared by every surface: `Completion`, `Answer`, `Review`, `Synthesis`, `CouncilRun`, `CouncilEvent`; aggregate ranking helpers |
| `council/config.py` | `CouncilConfig` and `load_config()` (TOML lookup, defaults, validation) |
| `council/council.py` | `Council.ask()`: the four stages; `parse_ranking()` |
| `council/prompts.py` | prompt builders for every stage, plus stable stage markers |
| `council/providers/` | `Provider` protocol, `get_provider()`, one module per provider |
| `council/store.py` | `RunStore`: one JSON file per run |
| `council/analytics.py` | pure functions over a list of runs: leaderboard, self-preference, agreement matrix, human-vs-council, summary |
| `council/cli.py` | the `council` command |
| `council/mcp_server.py` | the `council-mcp` command (MCP over stdio) |
| `council/web/server.py` | FastAPI app behind `council serve` |
| `frontend/` | Vite + React + TypeScript UI, built into `frontend/dist/` |
| `council/errors.py` | `CouncilError` and `ProviderError` |

## Lifecycle of a run

`Council.ask(question, on_event=None)` returns a `CouncilRun`. Within each stage, all model calls run concurrently with `asyncio.gather`. A failing call never aborts the run: it is recorded as `Answer.error` or `Review.error`, and a member whose answer failed takes no part in later stages.

1. **Answer.** Each member receives the question as a single user message.
2. **Debate** (only when `debate_rounds > 0`, repeated that many times). Each surviving member sees the question, its own current answer, and the other members' current answers labeled `Response A, B, ...` in an order shuffled per member and per round. It replies with a complete revised answer. Each round's answers are stored in `run.debate_answers` and replace the working set.
3. **Review.** The surviving models are shuffled with an RNG seeded by the run id and mapped to anonymous labels (`run.labels`, e.g. `{"Response A": "openrouter:openai/gpt-5.1"}`). Every reviewer gets the same prompt: all answers, its own included, under those labels, with instructions to judge on accuracy, completeness, reasoning and usefulness, ignore length, style and position, and end with one line such as `RANKING: B > A > C`. `parse_ranking()` accepts `Response B`, bare letters, commas, numbered lists after `FINAL RANKING:` and markdown emphasis; the last `RANKING:` line wins, and labels a reviewer left out are appended in label order. A reply with no parseable ranking becomes a `Review` with `error` set and an empty `ranking`. Review needs at least two surviving answers.
4. **Synthesis.** The chairman receives the question, every answer with both its label and its model id (so it can resolve the reviewers' references), every usable review (ranking and rationale) and the aggregate ranking with mean positions. It is told to treat the reviews as evidence, not orders. If the chairman fails, `run.synthesis` is `None`.

The aggregate council ranking (`CouncilRun.council_ranking()`) orders models by mean 1-based position across all usable reviews. Ties are broken alphabetically, and members that were never ranked come last.

`run.total_cost_usd` is the sum of the costs providers reported. It is `None` when no call reported a cost, and only covers the calls that did (in practice, OpenRouter calls).

`on_event` receives a `CouncilEvent(stage, model, status, round, message)` with `status="start"` before each call and `"done"` or `"error"` after it. For reviews, `"done"` is sent only once the ranking has been parsed. The CLI progress display and the web app's SSE stream are both built on these events.

## Stored runs

`RunStore(data_dir)` writes each run as pretty-printed JSON to `<data_dir>/runs/<id>.json`, through a temporary file and `os.replace`, so a crash cannot leave a half-written run. Run ids are 12 hex characters; ids from outside (URLs, CLI arguments) must match `[A-Za-z0-9_-]{1,64}`, so they cannot escape the directory. `list()` returns runs newest first and skips files it cannot parse. `set_human_vote(run_id, model)` accepts only a model that is a member of that run.

An abbreviated run:

```json
{
  "id": "38234df62422",
  "question": "Is a hot dog a sandwich?",
  "created_at": "2026-10-07T21:40:12.345Z",
  "members": ["fake:alpha", "fake:beta", "fake:gamma"],
  "chairman": "fake:alpha",
  "answers": [{ "model": "fake:alpha", "text": "...", "latency_s": 0.0, "cost_usd": 0.0, "error": null }],
  "reviews": [{ "reviewer": "fake:beta", "ranking": ["fake:beta", "fake:gamma", "fake:alpha"], "rationale": "...", "raw": "...", "error": null }],
  "synthesis": { "chairman": "fake:alpha", "text": "...", "latency_s": 0.0, "cost_usd": 0.0 },
  "debate_rounds": 0,
  "debate_answers": [],
  "labels": { "Response A": "fake:gamma", "Response B": "fake:alpha", "Response C": "fake:beta" },
  "human_vote": null,
  "total_cost_usd": 0.0
}
```

`Review.ranking` is already de-anonymized: model ids, best first. The labels a reviewer actually saw are kept in `run.labels`.

## Providers

| id prefix | class | transport |
| --------- | ----- | --------- |
| `openrouter:` | `OpenRouterProvider` | OpenAI-compatible `/chat/completions` at `openrouter.ai/api/v1` via `httpx`; asks for usage accounting, so `usage.cost` becomes `cost_usd` |
| `anthropic:` | `AnthropicProvider` | official `anthropic` SDK (`AsyncAnthropic().messages.create`); system messages go to the `system` parameter |
| `openai:` | `OpenAIProvider` | OpenAI Chat Completions via `httpx`, using `max_completion_tokens` |
| `ollama:` | `OllamaProvider` | Ollama's OpenAI-compatible `/v1` API at `OLLAMA_HOST` (default `http://localhost:11434`), no key |
| `fake:` | `FakeProvider` | in-process, deterministic, offline |

`split_model_id()` splits on the first colon only, so `ollama:llama3.1:8b` is provider `ollama`, model `llama3.1:8b`. `get_provider()` never touches the network or checks keys; a missing key raises `ProviderError` when the provider is first called.

### The fake provider

`fake:<name>` recognizes each stage by the markers in `prompts.py` and answers deterministically per question:

- answers are deterministic text ending in an opaque `(digest ...)` token (`sha256(name, question)[:8]`); fake reviewers recover each author by recomputing the digest for the names they know, so they can tell who wrote what regardless of the label shuffle, while no model name is visible and the blind vote stays blind;
- reviews rank answers by a shared, question-dependent "quality" order, with two deliberate biases: `fake:alpha` always ranks itself first, and every other fake reviewer ranks alpha's answer last;
- names starting with `fail` (e.g. `fake:fail`) always raise `ProviderError`, to exercise the error paths;
- cost is `0.0` and latency is effectively zero.

All tests and the `--fake` demo use it. See [judge-bias.md](judge-bias.md) for what its biases look like in the metrics.

## Surfaces

**CLI** (`council`, typer + rich): `ask "question" [--members a,b,c] [--chairman x] [--debate N] [--config path] [--json] [--fake]`, `history [--limit N]`, `show <run_id>`, `vote <run_id> <model>`, `stats`, `models`, `serve [--fake] [--config path] [--host 127.0.0.1] [--port 8000]`, `mcp`. `serve` and `mcp` import their optional dependencies lazily, so the core install works without the `web` and `mcp` extras. The CLI loads a `.env` file from the current directory with a small stdlib parser; variables already set in the environment take precedence and empty values are ignored.

**MCP server** (`council-mcp` or `council mcp`, stdio, built on FastMCP from the official `mcp` package). Every tool returns a JSON string, and expected failures come back as `{"error": "..."}` instead of crashing the server.

| tool | does |
| ---- | ---- |
| `ask_council(question, debate_rounds=0)` | runs the council (0 to 5 debate rounds), saves the run, returns the run id, synthesis, council ranking, reviews, cost and answers truncated to 500 characters |
| `council_stats()` | `analytics.summary` over all stored runs; no model calls |
| `council_history(limit=10)` | recent runs, newest first; no model calls |
| `council_vote(run_id, model)` | records which member's answer the human preferred |

`COUNCIL_CONFIG=<path>` points the server at a config file (otherwise the usual lookup applies, relative to the server's working directory), and `COUNCIL_FAKE=1` swaps in the offline fake council.

**Web API** (FastAPI, `council serve`):

| method and path | returns |
| --------------- | ------- |
| `GET /api/config` | `{members, chairman, debate_rounds}` |
| `POST /api/ask` `{question, debate_rounds?}` | server-sent events: one `CouncilEvent` per message, then `{"stage": "done", "run_id": ...}` (or `{"stage": "error", "message": ...}`) |
| `GET /api/runs?limit=` | run summaries: `{id, question, created_at, council_ranking, human_vote}` |
| `GET /api/runs/{id}` | the full `CouncilRun` |
| `POST /api/runs/{id}/vote` `{model}` | the updated `CouncilRun` |
| `GET /api/stats` | `analytics.summary` |

Any other `GET` path serves the built UI from `frontend/dist/` (falling back to `index.html` for client-side routes such as `/runs/<id>`); if the UI has not been built, a page explains how to build it. `COUNCIL_DIST_DIR=<path>` serves the UI from another directory. `council serve --fake` (or `COUNCIL_FAKE=1` when starting the app another way) swaps in the offline fake council. The app can also be started directly with `uvicorn council.web.server:get_app --factory`.

**Frontend** (`frontend/`, Vite + React + TypeScript): Ask, Run, History and Stats pages. The run page enforces the blind vote: until you vote (or explicitly reveal), answers show only their labels; latency, tokens and cost are hidden because they can identify a model; reviewers are shown as "Author of Response B"; the Final tab is locked because the synthesis may name models; and the History page hides the council's #1 for runs you have not voted on. Revealing without voting keeps the run out of the human-vs-council statistic. In development, `npm run dev` proxies `/api` to `http://localhost:8000`.

## Using it as a library

```python
import asyncio

from council import Council, RunStore, load_config

config = load_config()  # ./council.toml, ~/.config/council/config.toml, or defaults
run = asyncio.run(Council(config).ask("Is a hot dog a sandwich?", on_event=print))

print(run.council_ranking())  # model ids, best first
print(run.synthesis.text if run.synthesis else "the chairman failed")
RunStore(config.data_dir).save(run)  # now it counts in `council stats`
```

`Council` also takes a `provider_factory` argument (provider name to `Provider`), which is how the tests inject custom providers.
