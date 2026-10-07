# Contributing to council-of-models

Thanks for helping. Bug reports, new providers, better analytics and documentation fixes are all welcome. For anything larger than a small fix, please open an issue first so we can agree on the approach.

## Development setup

You need Python 3.11+ and, for the web UI, Node 22.12+.

```bash
git clone https://github.com/SelmanBoray/council-of-models.git
cd council-of-models
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[web,mcp,dev]"
```

Frontend:

```bash
cd frontend
npm install
npm run dev                          # Vite dev server; proxies /api to http://localhost:8000
```

Run `council serve` in a second terminal so the dev server has a backend to talk to. `npm run build` writes `frontend/dist/`, which `council serve` serves at `/`.

## Checks

CI runs exactly these; please run them before opening a pull request.

```bash
ruff check src tests
pytest -q
cd frontend && npm test && npm run build
```

`ruff format` is welcome but not enforced. The line length is 100.

## Ground rules

- **Tests never touch the network.** Use the `fake` provider (`fake:alpha`, `fake:beta`, ...) or mock HTTP with `respx`. `tests/conftest.py` clears all provider API keys for every test.
- **Keep the data shapes stable.** `src/council/models.py` defines `CouncilRun` and friends, which the CLI, MCP server, web API and frontend all read. Changing a field means updating every consumer and the docs in one pull request.
- **Keep the stage markers in the prompts.** The fake provider recognizes stages by the strings in `src/council/prompts.py` (`[COUNCIL REVIEW]`, `RANKING:`, ...). If you reword a prompt, keep the markers.
- **Never log or print API keys.** Read them from the environment at call time, never at import time.
- **Analytics stay pure.** `src/council/analytics.py` takes a list of runs and returns plain data, with no I/O. If you change a formula, update [docs/judge-bias.md](docs/judge-bias.md) in the same pull request.

## Adding a provider

1. Implement the `Provider` protocol from `src/council/providers/base.py`: a `name` attribute and `async def complete(model, messages, *, max_tokens=4096, temperature=None) -> Completion`.
2. Raise `council.errors.ProviderError` with a clear message for a missing key, an HTTP error, a refusal or an empty answer. A provider must not raise for a missing key when it is constructed, only when it is called.
3. Register it in `get_provider`, `PROVIDER_NAMES` and `PROVIDER_KEY_ENV` in `src/council/providers/__init__.py`.
4. If the API is OpenAI-compatible, subclass `OpenAICompatProvider` in `providers/openai_compat.py`, as the OpenRouter, OpenAI and Ollama providers do.
5. Add offline tests (`respx` for HTTP) and a row in the README provider table.

## Pull requests

- Keep each pull request focused on one change and describe what it changes and why.
- Add or update tests for behavior changes.
- Update the README or `docs/` when user-facing behavior changes.

By contributing you agree that your work is released under the [MIT License](LICENSE) and that you will follow the [Code of Conduct](CODE_OF_CONDUCT.md).
