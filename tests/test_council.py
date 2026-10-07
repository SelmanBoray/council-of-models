from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest

from council.config import CouncilConfig
from council.council import (
    UNPARSEABLE_RANKING_ERROR,
    Council,
    assign_labels,
    extract_rationale,
    make_labels,
    parse_ranking,
)
from council.errors import ProviderError
from council.models import Completion, CouncilEvent
from council.prompts import RANKING_MARKER, REVIEW_MARKER, SYNTHESIS_MARKER
from council.providers import get_provider

LABELS = ["Response A", "Response B", "Response C"]
A, B, C = LABELS


def _config(tmp_path: Path, members, chairman="fake:alpha", **kwargs) -> CouncilConfig:
    return CouncilConfig(members=members, chairman=chairman, data_dir=tmp_path, **kwargs)


# ------------------------------------------------------------------ parse_ranking


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Some rationale.\nRANKING: A > C > B", [A, C, B]),
        ("RANKING: Response C > Response A > Response B", [C, A, B]),
        ("RANKING: B, A, C", [B, A, C]),
        ("RANKING: C A B", [C, A, B]),
        ("Response B, Response A", [B, A, C]),
        ("ranking: c > a > b", [C, A, B]),
        ("Ranking: response b > response c > response a", [B, C, A]),
        ("**RANKING:** B > A > C", [B, A, C]),
        ("`RANKING: C > B > A`", [C, B, A]),
        ("FINAL RANKING:\n1. Response C\n2. Response A\n3. Response B", [C, A, B]),
        ("FINAL RANKING:\n\n- C\n- B\n- A\n\nThanks!", [C, B, A]),
        ("RANKING: B", [B, A, C]),  # missing labels appended in label order
        ("RANKING: C > B", [C, B, A]),
        ("RANKING: A > A > B", [A, B, C]),  # duplicates ignored
        ("RANKING: D > B > E", [B, A, C]),  # unknown letters ignored
        ("Draft ranking: A > B > C\nOn reflection...\nRANKING: C > B > A", [C, B, A]),
        ("Thoughts.\nRANKING: B > C > A\nI hope this ranking: helps", [B, C, A]),
        ("Analysis...\n\nB > A > C", [B, A, C]),  # no marker, ranking-only last line
        ("1. Response C\n2. Response A\n3. Response B", [C, A, B]),
        ("RANKING: A > C > B\r\n", [A, C, B]),
    ],
)
def test_parse_ranking_variants(raw, expected):
    assert parse_ranking(raw, LABELS) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "garbage",
        "I liked them all equally.",
        "Response C has a typo.",  # a single mention is not a ranking
        "RANKING: none of them",
        "RANKING: D > E > F",
    ],
)
def test_parse_ranking_unparseable_returns_empty(raw):
    assert parse_ranking(raw, LABELS) == []


def test_parse_ranking_with_bare_letter_labels_and_two_responses():
    assert parse_ranking("RANKING: B > A", ["A", "B"]) == ["B", "A"]
    assert parse_ranking("RANKING: A > B", LABELS[:2]) == [A, B]


def test_parse_ranking_empty_labels():
    assert parse_ranking("RANKING: A > B", []) == []


def test_extract_rationale():
    raw = "A is precise.\nB is vague.\n\nRANKING: A > B"
    assert extract_rationale(raw) == "A is precise.\nB is vague."
    assert extract_rationale("**RANKING:** A > B") == ""
    assert extract_rationale("no marker at all") == "no marker at all"


# ------------------------------------------------------------------------ labels


def test_make_labels():
    assert make_labels(3) == LABELS
    assert make_labels(26)[-1] == "Response Z"
    with pytest.raises(ValueError):
        make_labels(27)


def test_assign_labels_is_seeded_and_complete():
    models = ["m:a", "m:b", "m:c", "m:d", "m:e"]
    first = assign_labels(models, seed="run123")
    assert first == assign_labels(models, seed="run123")
    assert list(first) == make_labels(5)
    assert sorted(first.values()) == models
    orders = {tuple(assign_labels(models, seed=f"seed{i}").values()) for i in range(20)}
    assert len(orders) > 1  # actually shuffled


# --------------------------------------------------------------------- full runs


async def test_full_fake_run(fake_config):
    events: list[CouncilEvent] = []
    run = await Council(fake_config).ask("What is 2+2?", on_event=events.append)

    assert re.fullmatch(r"[0-9a-f]{12}", run.id)
    assert run.created_at.endswith("Z")
    datetime.fromisoformat(run.created_at)
    assert run.question == "What is 2+2?"
    assert run.members == fake_config.members and run.chairman == "fake:alpha"

    assert [a.model for a in run.answers] == fake_config.members
    assert all(a.error is None for a in run.answers)
    assert all(a.text.startswith('Answer to "What is 2+2?": ') for a in run.answers)
    assert run.debate_rounds == 0 and run.debate_answers == []

    assert list(run.labels) == LABELS
    assert sorted(run.labels.values()) == sorted(fake_config.members)

    assert [r.reviewer for r in run.reviews] == fake_config.members
    for review in run.reviews:
        assert review.error is None
        assert sorted(review.ranking) == sorted(fake_config.members)
        assert RANKING_MARKER in review.raw
        assert review.rationale and RANKING_MARKER not in review.rationale
    by_reviewer = {r.reviewer: r.ranking for r in run.reviews}
    assert by_reviewer["fake:alpha"][0] == "fake:alpha"
    assert by_reviewer["fake:beta"][-1] == "fake:alpha"
    assert by_reviewer["fake:gamma"][-1] == "fake:alpha"

    ranking = run.council_ranking()
    assert sorted(ranking) == sorted(fake_config.members)
    assert run.synthesis is not None
    assert run.synthesis.chairman == "fake:alpha"
    assert run.synthesis.text.startswith("SYNTHESIS by alpha:")
    assert "What is 2+2?" in run.synthesis.text
    assert run.total_cost_usd == 0.0
    assert run.human_vote is None

    # One start + one done per call: 3 answers, 3 reviews, 1 synthesis.
    assert len(events) == 14
    assert all(e.status in ("start", "done") for e in events)
    for stage, count in (("answer", 3), ("review", 3), ("synthesis", 1)):
        stage_events = [e for e in events if e.stage == stage]
        assert sum(e.status == "start" for e in stage_events) == count
        assert sum(e.status == "done" for e in stage_events) == count


@pytest.mark.parametrize("question", ["Is P = NP?", "Which is best?"])
async def test_fake_alpha_self_preference_signal(fake_config, question):
    # For these questions alpha is not the fake's "best" answer, yet it ranks itself first.
    run = await Council(fake_config).ask(question)
    rank_of_alpha = {r.reviewer: r.ranking.index("fake:alpha") + 1 for r in run.reviews}
    assert rank_of_alpha == {"fake:alpha": 1, "fake:beta": 3, "fake:gamma": 3}


async def test_fake_runs_are_reproducible_per_question(fake_config):
    council = Council(fake_config)
    first = await council.ask("Explain TCP vs UDP")
    second = await council.ask("Explain TCP vs UDP")
    assert first.id != second.id
    assert first.council_ranking() == second.council_ranking()
    assert [a.text for a in first.answers] == [a.text for a in second.answers]


async def test_debate_round_produces_revised_answers(tmp_path):
    config = _config(tmp_path, ["fake:alpha", "fake:beta", "fake:gamma"], debate_rounds=1)
    events: list[CouncilEvent] = []
    run = await Council(config).ask("What is 2+2?", on_event=events.append)

    assert run.debate_rounds == 1
    assert len(run.debate_answers) == 1
    revised = run.debate_answers[0]
    assert [a.model for a in revised] == config.members
    assert all("debate round 1" in a.text for a in revised)
    debate_events = [e for e in events if e.stage == "debate"]
    assert len(debate_events) == 6 and all(e.round == 1 for e in debate_events)
    assert all(e.round == 0 for e in events if e.stage != "debate")
    assert len(run.reviews) == 3 and all(r.error is None for r in run.reviews)


@pytest.mark.parametrize("question", ["Is a hot dog a sandwich?", "Which is best?"])
async def test_fake_answers_do_not_reveal_their_author(tmp_path, question):
    # Blind voting shows answer texts before the reveal: no member id or name may leak.
    members = ["fake:alpha", "fake:beta", "fake:gamma"]
    run = await Council(_config(tmp_path, members, debate_rounds=1)).ask(question)
    texts = [a.text for a in run.answers] + [a.text for a in run.debate_answers[0]]
    for text in texts:
        assert "fake:" not in text and "[fake" not in text
        for member in members:
            assert member.split(":", 1)[1] not in text
    # Authors are still recognised through the digest token: biases are unchanged.
    rank_of_alpha = {r.reviewer: r.ranking.index("fake:alpha") + 1 for r in run.reviews}
    assert rank_of_alpha == {"fake:alpha": 1, "fake:beta": 3, "fake:gamma": 3}


async def test_two_debate_rounds(tmp_path):
    config = _config(tmp_path, ["fake:alpha", "fake:beta"], debate_rounds=2)
    run = await Council(config).ask("Is P = NP?")
    assert len(run.debate_answers) == 2
    assert all("debate round 2" in a.text for a in run.debate_answers[1])


async def test_failing_member_is_excluded_from_later_stages(tmp_path):
    config = _config(tmp_path, ["fake:alpha", "fake:beta", "fake:fail"])
    events: list[CouncilEvent] = []
    run = await Council(config).ask("What is 2+2?", on_event=events.append)

    failed = run.answers[2]
    assert failed.model == "fake:fail"
    assert failed.text == "" and "simulated failure" in failed.error
    assert "fake:fail" not in run.labels.values()
    assert list(run.labels) == ["Response A", "Response B"]
    assert [r.reviewer for r in run.reviews] == ["fake:alpha", "fake:beta"]
    assert all("fake:fail" not in r.ranking for r in run.reviews)
    assert run.council_ranking()[-1] == "fake:fail"
    assert run.synthesis is not None
    errors = [e for e in events if e.status == "error"]
    assert len(errors) == 1
    assert errors[0].stage == "answer" and errors[0].model == "fake:fail"
    assert "simulated failure" in errors[0].message


async def test_member_failing_during_debate_is_excluded(tmp_path):
    class FailOnDebate:
        name = "flaky"

        def __init__(self):
            self.fake = get_provider("fake")

        async def complete(self, model, messages, *, max_tokens=4096, temperature=None):
            if model == "beta" and "[COUNCIL DEBATE]" in messages[-1]["content"]:
                raise ProviderError("boom in debate")
            return await self.fake.complete(model, messages, max_tokens=max_tokens)

    config = _config(tmp_path, ["fake:alpha", "fake:beta", "fake:gamma"], debate_rounds=1)
    run = await Council(config, provider_factory=lambda name: FailOnDebate()).ask("Q?")
    assert run.answers[1].error is None
    assert run.debate_answers[0][1].error == "ProviderError: boom in debate"
    assert sorted(run.labels.values()) == ["fake:alpha", "fake:gamma"]
    assert [r.reviewer for r in run.reviews] == ["fake:alpha", "fake:gamma"]


async def test_unknown_provider_becomes_answer_error(tmp_path):
    config = _config(tmp_path, ["fake:alpha", "fake:beta", "nope:model"])
    run = await Council(config).ask("Q?")
    assert "Unknown provider" in run.answers[2].error
    assert len(run.reviews) == 2


async def test_chairman_failure_leaves_synthesis_none(tmp_path):
    config = _config(tmp_path, ["fake:alpha", "fake:beta"], chairman="fake:fail")
    events: list[CouncilEvent] = []
    run = await Council(config).ask("Q?", on_event=events.append)
    assert run.synthesis is None
    assert len(run.reviews) == 2
    assert events[-1].stage == "synthesis" and events[-1].status == "error"


async def test_all_members_failing(tmp_path):
    config = _config(tmp_path, ["fake:fail1", "fake:fail2"])
    run = await Council(config).ask("Q?")
    assert all(a.error for a in run.answers)
    assert run.labels == {} and run.reviews == [] and run.synthesis is None
    assert run.total_cost_usd is None
    assert run.council_ranking() == ["fake:fail1", "fake:fail2"]


async def test_single_survivor_skips_review_but_synthesizes(tmp_path):
    config = _config(tmp_path, ["fake:alpha", "fake:fail"])
    run = await Council(config).ask("Q?")
    assert run.labels == {"Response A": "fake:alpha"}
    assert run.reviews == []
    assert run.synthesis is not None


class RecordingProvider:
    """Answers without self-identifying text and records every prompt it receives."""

    name = "rec"

    def __init__(self, review_text: str | None = None, cost: float | None = None):
        self.prompts: list[tuple[str, str]] = []
        self.review_text = review_text
        self.cost = cost

    async def complete(self, model, messages, *, max_tokens=4096, temperature=None):
        content = messages[-1]["content"]
        self.prompts.append((model, content))
        if SYNTHESIS_MARKER in content:
            text = "final"
        elif REVIEW_MARKER in content:
            text = self.review_text or "All fine.\nRANKING: A > B > C"
        else:
            text = f"Answer number {len(self.prompts)}"
        return Completion(text=text, input_tokens=1, output_tokens=1, cost_usd=self.cost)


async def test_review_prompt_is_anonymous_and_identical_for_all_reviewers(tmp_path):
    members = ["rec:secret-model-one", "rec:secret-model-two", "rec:secret-model-three"]
    recorder = RecordingProvider()
    run = await Council(
        _config(tmp_path, members, chairman="rec:chair"), provider_factory=lambda n: recorder
    ).ask("Which is best?")

    review_prompts = [(m, p) for m, p in recorder.prompts if REVIEW_MARKER in p]
    assert len(review_prompts) == 3
    assert len({p for _, p in review_prompts}) == 1  # same prompt for every reviewer
    prompt = review_prompts[0][1]
    for member in members:
        assert member not in prompt
        assert member.split(":", 1)[1] not in prompt
    positions = [prompt.index(f'<response label="{label}">') for label in LABELS]
    assert positions == sorted(positions)
    assert "Which is best?" in prompt

    # De-anonymization: "RANKING: A > B > C" maps back through run.labels.
    expected = [run.labels[label] for label in LABELS]
    assert all(r.ranking == expected for r in run.reviews)

    synthesis_prompt = next(p for m, p in recorder.prompts if SYNTHESIS_MARKER in p)
    for member in members:
        assert member in synthesis_prompt  # chairman sees real model ids
    assert "<council_ranking>" in synthesis_prompt


async def test_unparseable_review_sets_error(tmp_path):
    members = ["rec:one", "rec:two"]
    recorder = RecordingProvider(review_text="They are all wonderful.")
    events: list[CouncilEvent] = []
    run = await Council(
        _config(tmp_path, members, chairman="rec:one"), provider_factory=lambda n: recorder
    ).ask("Q?", on_event=events.append)
    assert all(r.ranking == [] and r.error == UNPARSEABLE_RANKING_ERROR for r in run.reviews)
    assert all(r.raw == "They are all wonderful." for r in run.reviews)
    review_events = [e for e in events if e.stage == "review" and e.status != "start"]
    assert [e.status for e in review_events] == ["error", "error"]
    assert run.synthesis is not None
    assert run.council_ranking() == sorted(members)


async def test_total_cost_sums_all_calls(tmp_path):
    recorder = RecordingProvider(cost=0.01)
    config = _config(tmp_path, ["rec:a", "rec:b", "rec:c"], chairman="rec:a")
    run = await Council(config, provider_factory=lambda n: recorder).ask("Q?")
    # 3 answers + 3 reviews + 1 synthesis
    assert run.total_cost_usd == pytest.approx(0.07)
    assert all(a.cost_usd == 0.01 for a in run.answers)
    assert run.synthesis.cost_usd == 0.01


async def test_total_cost_none_when_unknown(tmp_path):
    recorder = RecordingProvider(cost=None)
    config = _config(tmp_path, ["rec:a", "rec:b"], chairman="rec:a")
    run = await Council(config, provider_factory=lambda n: recorder).ask("Q?")
    assert run.total_cost_usd is None


async def test_broken_event_callback_does_not_break_run(fake_config):
    def explode(event):
        raise RuntimeError("UI crashed")

    run = await Council(fake_config).ask("Q?", on_event=explode)
    assert all(a.error is None for a in run.answers)
    assert run.synthesis is not None


async def test_empty_question_rejected(fake_config):
    with pytest.raises(ValueError):
        await Council(fake_config).ask("   ")


async def test_provider_factory_called_once_per_provider(tmp_path):
    calls: list[str] = []

    def factory(name):
        calls.append(name)
        return get_provider(name)

    config = _config(tmp_path, ["fake:alpha", "fake:beta", "fake:gamma"])
    await Council(config, provider_factory=factory).ask("Q?")
    assert calls == ["fake"]
