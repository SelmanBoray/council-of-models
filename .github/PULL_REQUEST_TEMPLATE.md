## What this changes

<!-- One or two sentences. Link the issue if there is one: "Closes #123". -->

## Why

## How it was tested

<!-- Commands you ran, e.g. `pytest -q`, `npm test`, or a `council ask ... --fake` run. -->

## Checklist

- [ ] `ruff check src tests` passes
- [ ] `pytest -q` passes (tests stay offline: fake provider or `respx`)
- [ ] `cd frontend && npm test && npm run build` passes (if the frontend changed)
- [ ] README or `docs/` updated for user-facing changes
- [ ] `docs/judge-bias.md` updated if an analytics formula changed
