"""Council configuration and TOML loading.

Lookup order for :func:`load_config`: explicit path -> ``./council.toml`` ->
``~/.config/council/config.toml`` -> built-in defaults. Missing keys fall back to the
defaults below.

:func:`load_dotenv_file` loads provider keys from a ``./.env`` file (the CLI calls it
once at startup).
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from council.providers.base import split_model_id

DEFAULT_MEMBERS: tuple[str, ...] = (
    "openrouter:openai/gpt-5.1",
    "openrouter:google/gemini-3-pro-preview",
    "openrouter:anthropic/claude-sonnet-4.5",
    "openrouter:x-ai/grok-4",
)
DEFAULT_CHAIRMAN = "openrouter:google/gemini-3-pro-preview"
DEFAULT_DEBATE_ROUNDS = 0
DEFAULT_DATA_DIR = "./data"
DEFAULT_MAX_TOKENS = 4096
MAX_MEMBERS = 26  # anonymous labels run from "Response A" to "Response Z"

CONFIG_FILENAME = "council.toml"
DOTENV_FILENAME = ".env"
CONFIG_KEYS = ("members", "chairman", "debate_rounds", "data_dir", "max_tokens")


class CouncilConfig(BaseModel):
    """Who sits on the council and how it runs."""

    members: list[str]
    chairman: str
    debate_rounds: int = 0
    data_dir: Path
    max_tokens: int = 4096

    @field_validator("members")
    @classmethod
    def _check_members(cls, members: list[str]) -> list[str]:
        cleaned = [m.strip() for m in members]
        if not cleaned:
            raise ValueError("the council needs at least one member")
        if len(cleaned) > MAX_MEMBERS:
            raise ValueError(f"at most {MAX_MEMBERS} members are supported, got {len(cleaned)}")
        duplicates = sorted({m for m in cleaned if cleaned.count(m) > 1})
        if duplicates:
            raise ValueError(f"duplicate members: {', '.join(duplicates)}")
        for member in cleaned:
            split_model_id(member)
        return cleaned

    @field_validator("chairman")
    @classmethod
    def _check_chairman(cls, chairman: str) -> str:
        chairman = chairman.strip()
        split_model_id(chairman)
        return chairman

    @field_validator("debate_rounds")
    @classmethod
    def _check_debate_rounds(cls, rounds: int) -> int:
        if rounds < 0:
            raise ValueError("debate_rounds must be >= 0")
        return rounds

    @field_validator("max_tokens")
    @classmethod
    def _check_max_tokens(cls, max_tokens: int) -> int:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        return max_tokens

    @field_validator("data_dir")
    @classmethod
    def _expand_data_dir(cls, data_dir: Path) -> Path:
        return Path(data_dir).expanduser()


def user_config_path() -> Path:
    """``~/.config/council/config.toml`` (resolved at call time)."""
    return Path.home() / ".config" / "council" / "config.toml"


def find_config_file(path: Path | None = None) -> Path | None:
    """Return the config file :func:`load_config` would read, or ``None`` for defaults.

    Raises FileNotFoundError when an explicit ``path`` does not exist.
    """
    if path is not None:
        explicit = Path(path).expanduser()
        if not explicit.is_file():
            raise FileNotFoundError(f"Config file not found: {explicit}")
        return explicit
    for candidate in (Path.cwd() / CONFIG_FILENAME, user_config_path()):
        if candidate.is_file():
            return candidate
    return None


def default_config() -> CouncilConfig:
    return CouncilConfig(
        members=list(DEFAULT_MEMBERS),
        chairman=DEFAULT_CHAIRMAN,
        debate_rounds=DEFAULT_DEBATE_ROUNDS,
        data_dir=Path(DEFAULT_DATA_DIR),
        max_tokens=DEFAULT_MAX_TOKENS,
    )


def load_config(path: Path | None = None) -> CouncilConfig:
    """Load the council configuration (see module docstring for the lookup order).

    If a file sets ``members`` but not ``chairman``, the first member chairs, so a
    custom council never silently depends on the default OpenRouter chairman.

    Raises:
        FileNotFoundError: ``path`` was given but does not exist.
        ValueError: the file is not valid TOML or contains invalid values.
    """
    config_file = find_config_file(path)
    if config_file is None:
        return default_config()

    try:
        with config_file.open("rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {config_file}: {exc}") from exc

    values: dict[str, Any] = {key: data[key] for key in CONFIG_KEYS if key in data}
    if "members" in values and "chairman" not in values and isinstance(values["members"], list):
        if values["members"]:
            values["chairman"] = values["members"][0]
    merged: dict[str, Any] = {
        "members": list(DEFAULT_MEMBERS),
        "chairman": DEFAULT_CHAIRMAN,
        "debate_rounds": DEFAULT_DEBATE_ROUNDS,
        "data_dir": DEFAULT_DATA_DIR,
        "max_tokens": DEFAULT_MAX_TOKENS,
        **values,
    }
    try:
        return CouncilConfig(**merged)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration in {config_file}:\n{exc}") from exc


# ------------------------------------------------------------------------------ .env

_DOTENV_LINE_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def _dotenv_value(raw: str) -> str:
    """Strip optional surrounding quotes; unquoted values end at an inline `` #`` comment."""
    value = raw.strip()
    if value[:1] in {'"', "'"}:
        closing = value.find(value[0], 1)
        if closing != -1:
            return value[1:closing]
    # Search the raw text so "KEY=   # comment" is empty, while "KEY=a#b" stays "a#b".
    return re.split(r"\s#", raw, maxsplit=1)[0].strip()


def load_dotenv_file(path: Path = Path(DOTENV_FILENAME)) -> dict[str, str]:
    """Load ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ`` (stdlib only).

    Blank lines and ``#`` comments are skipped, an ``export`` prefix is accepted, values
    may be wrapped in single or double quotes (no escape processing), and an unquoted
    value ends at an inline `` #`` comment. If a key repeats, the last line wins.
    Variables already present in the environment are never overridden, and keys with an
    empty value are ignored, so a copied ``.env.example`` sets nothing. A missing or
    unreadable file loads nothing.

    Returns:
        The variables that were actually set, as ``{name: value}``. The values are
        usually API keys: callers must never print them.
    """
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return {}
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _DOTENV_LINE_RE.match(stripped)
        if match is None:
            continue
        parsed[match.group(1)] = _dotenv_value(match.group(2))
    loaded: dict[str, str] = {}
    for key, value in parsed.items():
        if not value or key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded
