# Security Policy

## Supported versions

council-of-models is pre-1.0. Security fixes go into the latest release on `main`.

| version | supported |
| ------- | :-------: |
| 0.1.x   |    yes    |

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability**. If that option is not available, contact the maintainer through the details on the [maintainer's GitHub profile](https://github.com/SelmanBoray).

Include what you found, how to reproduce it, and the impact you expect. You can expect a first response within a week. Once a fix is released you will be credited in the release notes, unless you prefer otherwise.

## Things to know when running council-of-models

- **API keys** are read from environment variables at call time. They are never written to stored runs, and `council models` only reports whether a key is set, never its value. Keep `.env` and `council.toml` out of version control (both are in `.gitignore`).
- **The web server has no authentication.** `council serve` is meant for your own machine; do not expose it to a network you do not trust. Anyone who can reach it can spend your API credits and read your stored runs.
- **Stored runs are plain JSON** under `data_dir/runs/`. They contain your questions and every model's answers; treat them as being as sensitive as the questions you ask.
- **Questions go to third parties.** With a real provider, every question and answer is sent to that provider (and, for review and synthesis, to the other council members). Use `ollama:` or `fake:` models for anything that must stay local.
