from __future__ import annotations

import pytest

from council.council import Council
from council.models import CouncilRun
from council.store import RunStore


def test_runs_dir_created_lazily(tmp_path, make_run):
    store = RunStore(tmp_path / "data")
    assert not (tmp_path / "data").exists()
    assert store.list() == []
    path = store.save(make_run())
    assert path == tmp_path / "data" / "runs" / "run000000001.json"
    assert path.is_file()


def test_save_load_round_trip(tmp_path, make_run):
    store = RunStore(tmp_path)
    run = make_run(question="Türkçe karakterler: ğüşöçİı — and emoji ✓", total_cost_usd=0.5)
    store.save(run)
    loaded = store.load(run.id)
    assert loaded == run
    assert isinstance(loaded, CouncilRun)
    raw = (tmp_path / "runs" / f"{run.id}.json").read_text(encoding="utf-8")
    assert raw.startswith("{\n  ")  # pretty-printed JSON


async def test_round_trip_of_real_fake_run(tmp_path, fake_config):
    run = await Council(fake_config).ask("What is 2+2?")
    store = RunStore(fake_config.data_dir)
    store.save(run)
    loaded = store.load(run.id)
    assert loaded == run
    assert loaded.council_ranking() == run.council_ranking()


def test_load_missing_raises_key_error(tmp_path):
    with pytest.raises(KeyError):
        RunStore(tmp_path).load("doesnotexist")


@pytest.mark.parametrize("bad_id", ["../secrets", "a/b", "..", "", "x" * 65, "run id"])
def test_invalid_ids_are_rejected(tmp_path, bad_id):
    store = RunStore(tmp_path)
    with pytest.raises(KeyError):
        store.load(bad_id)
    with pytest.raises(KeyError):
        store.delete(bad_id)


def test_save_rejects_unsafe_run_id(tmp_path, make_run):
    with pytest.raises(ValueError):
        RunStore(tmp_path).save(make_run(id="../escape"))


def test_list_newest_first_with_limit(tmp_path, make_run):
    store = RunStore(tmp_path)
    store.save(make_run(id="old", created_at="2026-01-01T00:00:00.000Z"))
    store.save(make_run(id="new", created_at="2026-03-01T00:00:00.000Z"))
    store.save(make_run(id="mid", created_at="2026-02-01T00:00:00+00:00"))
    assert [r.id for r in store.list()] == ["new", "mid", "old"]
    assert [r.id for r in store.list(limit=2)] == ["new", "mid"]
    assert store.list(limit=0) == []


def test_list_skips_corrupted_files(tmp_path, make_run):
    store = RunStore(tmp_path)
    store.save(make_run(id="good"))
    (tmp_path / "runs" / "broken.json").write_text("{not json", encoding="utf-8")
    assert [r.id for r in store.list()] == ["good"]


def test_set_human_vote(tmp_path, make_run):
    store = RunStore(tmp_path)
    run = make_run()
    store.save(run)
    updated = store.set_human_vote(run.id, "fake:beta")
    assert updated.human_vote == "fake:beta"
    assert store.load(run.id).human_vote == "fake:beta"


def test_set_human_vote_rejects_non_member(tmp_path, make_run):
    store = RunStore(tmp_path)
    run = make_run()
    store.save(run)
    with pytest.raises(ValueError, match="not a member"):
        store.set_human_vote(run.id, "openrouter:openai/gpt-5.1")
    assert store.load(run.id).human_vote is None


def test_set_human_vote_missing_run(tmp_path):
    with pytest.raises(KeyError):
        RunStore(tmp_path).set_human_vote("nope", "fake:alpha")


def test_delete(tmp_path, make_run):
    store = RunStore(tmp_path)
    run = make_run()
    store.save(run)
    store.delete(run.id)
    with pytest.raises(KeyError):
        store.load(run.id)
    with pytest.raises(KeyError):
        store.delete(run.id)


def test_save_overwrites_and_leaves_no_temp_files(tmp_path, make_run):
    store = RunStore(tmp_path)
    run = make_run()
    store.save(run)
    run.human_vote = "fake:gamma"
    store.save(run)
    assert store.load(run.id).human_vote == "fake:gamma"
    assert [p.name for p in (tmp_path / "runs").iterdir()] == [f"{run.id}.json"]
