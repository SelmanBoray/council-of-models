from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from council.config import (
    DEFAULT_CHAIRMAN,
    DEFAULT_MEMBERS,
    CouncilConfig,
    find_config_file,
    load_config,
    load_dotenv_file,
    user_config_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Empty cwd and a fake home directory, so no real config file is picked up."""
    cwd = tmp_path / "cwd"
    home = tmp_path / "home"
    cwd.mkdir()
    home.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return cwd, home


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_without_any_file(isolated):
    config = load_config()
    assert config.members == list(DEFAULT_MEMBERS)
    assert config.chairman == DEFAULT_CHAIRMAN
    assert config.debate_rounds == 0
    assert config.max_tokens == 4096
    assert config.data_dir == Path("data")
    assert find_config_file() is None


def test_default_council_matches_contract(isolated):
    config = load_config()
    assert config.members == [
        "openrouter:openai/gpt-5.1",
        "openrouter:google/gemini-3-pro-preview",
        "openrouter:anthropic/claude-sonnet-4.5",
        "openrouter:x-ai/grok-4",
    ]
    assert config.chairman == "openrouter:google/gemini-3-pro-preview"


def test_local_council_toml_is_used_and_merged_with_defaults(isolated):
    cwd, _ = isolated
    _write(cwd / "council.toml", "debate_rounds = 2\nmax_tokens = 1000\n")
    config = load_config()
    assert config.debate_rounds == 2
    assert config.max_tokens == 1000
    assert config.members == list(DEFAULT_MEMBERS)
    assert find_config_file() == cwd / "council.toml"


def test_user_config_is_used_when_no_local_file(isolated):
    _, home = isolated
    assert user_config_path() == home / ".config" / "council" / "config.toml"
    _write(user_config_path(), 'members = ["fake:a", "fake:b"]\nchairman = "fake:b"\n')
    config = load_config()
    assert config.members == ["fake:a", "fake:b"]
    assert config.chairman == "fake:b"


def test_local_file_wins_over_user_config(isolated):
    cwd, _ = isolated
    _write(user_config_path(), 'chairman = "fake:user"\n')
    _write(cwd / "council.toml", 'chairman = "fake:local"\n')
    assert load_config().chairman == "fake:local"


def test_explicit_path_wins(isolated, tmp_path):
    cwd, _ = isolated
    _write(cwd / "council.toml", 'chairman = "fake:local"\n')
    explicit = _write(tmp_path / "custom.toml", 'chairman = "fake:explicit"\n')
    assert load_config(explicit).chairman == "fake:explicit"


def test_explicit_missing_path_raises(isolated, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")


def test_data_dir_expands_user(isolated):
    cwd, home = isolated
    _write(cwd / "council.toml", 'data_dir = "~/council-data"\n')
    assert load_config().data_dir == home / "council-data"


def test_members_without_chairman_uses_first_member(isolated):
    cwd, _ = isolated
    _write(cwd / "council.toml", 'members = ["ollama:llama3.1", "ollama:qwen3"]\n')
    assert load_config().chairman == "ollama:llama3.1"


def test_unknown_keys_are_ignored(isolated):
    cwd, _ = isolated
    _write(cwd / "council.toml", "debate_rounds = 1\n[web]\nport = 9000\n")
    assert load_config().debate_rounds == 1


@pytest.mark.parametrize(
    "toml_text",
    [
        "members = [",  # invalid TOML
        'members = ["fake:a", "fake:a"]',  # duplicate
        'members = ["gpt-5.1"]',  # missing provider prefix
        "members = []",
        "debate_rounds = -1",
        "max_tokens = 0",
        'chairman = "no-colon"',
    ],
)
def test_invalid_config_raises_value_error(isolated, toml_text):
    cwd, _ = isolated
    _write(cwd / "council.toml", toml_text + "\n")
    with pytest.raises(ValueError):
        load_config()


def test_council_config_validation(tmp_path):
    with pytest.raises(ValidationError):
        CouncilConfig(
            members=[f"fake:m{i}" for i in range(27)], chairman="fake:m0", data_dir=tmp_path
        )
    config = CouncilConfig(members=[" fake:a "], chairman=" fake:a", data_dir="~/x")
    assert config.members == ["fake:a"] and config.chairman == "fake:a"
    assert config.data_dir == Path.home() / "x"


def test_example_config_is_valid(isolated):
    config = load_config(REPO_ROOT / "council.example.toml")
    assert config.members == list(DEFAULT_MEMBERS)
    assert config.chairman == DEFAULT_CHAIRMAN
    assert config.data_dir == Path("data")


# ------------------------------------------------------------------------------ .env


@pytest.fixture
def environ_snapshot():
    """Undo every os.environ change that load_dotenv_file makes during the test."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _absent(monkeypatch, *names: str) -> None:
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_load_dotenv_file_parses_lines(tmp_path, monkeypatch, environ_snapshot):
    expected = {
        "COUNCIL_TEST_PLAIN": "plain-value",
        "COUNCIL_TEST_DOUBLE": "double quoted value",
        "COUNCIL_TEST_SINGLE": "single # not a comment",
        "COUNCIL_TEST_EXPORTED": "exported",
        "COUNCIL_TEST_SPACES": "spaced",
        "COUNCIL_TEST_INLINE": "value",
        "COUNCIL_TEST_QUOTED_INLINE": "quoted",
        "COUNCIL_TEST_HASH": "abc#def",
        "COUNCIL_TEST_EQUALS": "a=b=c",
        "COUNCIL_TEST_REPEAT": "second",
    }
    _absent(monkeypatch, *expected, "BAD_1")
    lines = [
        "# a comment",
        "",
        "   ",
        "COUNCIL_TEST_PLAIN=plain-value",
        'COUNCIL_TEST_DOUBLE="double quoted value"',
        "COUNCIL_TEST_SINGLE='single # not a comment'",
        "export COUNCIL_TEST_EXPORTED=exported",
        "  COUNCIL_TEST_SPACES =  spaced  ",
        "COUNCIL_TEST_INLINE=value # trailing comment",
        'COUNCIL_TEST_QUOTED_INLINE="quoted" # trailing comment',
        "COUNCIL_TEST_HASH=abc#def",
        "COUNCIL_TEST_EQUALS=a=b=c",
        "COUNCIL_TEST_REPEAT=first",
        "COUNCIL_TEST_REPEAT=second",
        "this line is not an assignment",
        "1BAD=starts with a digit",
        "BAD 1=space in key",
    ]
    path = _write(tmp_path / "custom.env", "\n".join(lines) + "\n")

    loaded = load_dotenv_file(path)

    assert loaded == expected
    for name, value in expected.items():
        assert os.environ[name] == value
    assert "BAD_1" not in os.environ and "1BAD" not in os.environ


def test_load_dotenv_file_never_overrides_the_environment(tmp_path, monkeypatch, environ_snapshot):
    _absent(monkeypatch, "COUNCIL_TEST_NEW")
    monkeypatch.setenv("COUNCIL_TEST_SHELL", "from-shell")
    path = _write(
        tmp_path / ".env",
        "COUNCIL_TEST_SHELL=from-file\nCOUNCIL_TEST_NEW=from-file\n",
    )

    assert load_dotenv_file(path) == {"COUNCIL_TEST_NEW": "from-file"}
    assert os.environ["COUNCIL_TEST_SHELL"] == "from-shell"
    assert os.environ["COUNCIL_TEST_NEW"] == "from-file"


def test_load_dotenv_file_ignores_empty_values(tmp_path, monkeypatch, environ_snapshot):
    names = ("COUNCIL_TEST_EMPTY", "COUNCIL_TEST_EMPTY_QUOTED", "COUNCIL_TEST_BLANK")
    _absent(monkeypatch, *names)
    path = _write(
        tmp_path / ".env",
        'COUNCIL_TEST_EMPTY=\nCOUNCIL_TEST_EMPTY_QUOTED=""\nCOUNCIL_TEST_BLANK=   # nothing\n',
    )

    assert load_dotenv_file(path) == {}
    for name in names:
        assert name not in os.environ


def test_load_dotenv_file_defaults_to_cwd_and_handles_bom(isolated, monkeypatch, environ_snapshot):
    cwd, _home = isolated
    _absent(monkeypatch, "COUNCIL_TEST_BOM")
    (cwd / ".env").write_text("COUNCIL_TEST_BOM=yes\n", encoding="utf-8-sig")  # Notepad style

    assert load_dotenv_file() == {"COUNCIL_TEST_BOM": "yes"}
    assert os.environ["COUNCIL_TEST_BOM"] == "yes"


def test_load_dotenv_file_missing_or_unreadable(isolated, tmp_path, environ_snapshot):
    assert load_dotenv_file() == {}  # no .env in the (empty) cwd
    assert load_dotenv_file(tmp_path / "does-not-exist.env") == {}
    assert load_dotenv_file(tmp_path) == {}  # a directory, not a file


def test_env_example_sets_nothing(environ_snapshot):
    # conftest clears the provider variables, so only the file's (empty) values matter
    assert load_dotenv_file(REPO_ROOT / ".env.example") == {}
