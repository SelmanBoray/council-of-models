"""Deterministic, offline fake provider (``fake:<name>``) for tests and demos.

Behaviour (stable; other test suites may rely on it):

* Answers are 2-3 short templated sentences of the form
  ``Answer to "<question>": ... (digest 1a2b3c4d)``. They never contain the member's
  name, so blind voting stays blind. The trailing ``(digest ...)`` token is
  ``sha256(name, question)[:8]``; it is the only trace of authorship. The demo trio
  ``alpha``/``beta``/``gamma`` always get different wording for the same question.
* Debate prompts get ``Revised answer (debate round N) to "<question>": ...`` with the
  same digest token as the member's original answer.
* Review prompts (they contain ``RANKING:``) get a short rationale and a final
  ``RANKING: X > Y > Z`` line. The fake reviewer identifies each anonymized answer's
  author by recomputing the digest for every candidate name it knows: its own name,
  ``alpha``/``beta``/``gamma``, and every fake name that has answered in this process.
  A shared, question-dependent "quality" order (sha256 of question and author) decides
  the ranking, with two deliberate biases so judge-bias analytics have signal:
  ``alpha`` always ranks its own answer first (self-preference), and every other fake
  reviewer ranks alpha's answer last. Answers whose author cannot be identified come
  after the identified non-alpha ones (ordered by a hash of their text), before alpha.
  The result does not depend on the random label shuffle, so runs are reproducible per
  question.
* Synthesis prompts get ``"SYNTHESIS by <name>: ..."`` mentioning the question.
* Model names starting with ``fail`` (e.g. ``fake:fail``) raise ProviderError, to
  exercise error paths.
* Zero latency, small token counts, cost 0.0.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from council.errors import ProviderError
from council.models import Completion
from council.prompts import DEBATE_MARKER, RANKING_MARKER, SYNTHESIS_MARKER, label_letter
from council.providers.base import Message

SELF_PREFERRING_MEMBER = "alpha"
DEMO_MEMBERS: tuple[str, ...] = ("alpha", "beta", "gamma")

# Fake member names seen in this process; reviewers try each of them when matching
# an anonymized answer's digest token back to its author.
_KNOWN_NAMES: set[str] = set(DEMO_MEMBERS)

_DIGEST_TOKEN_RE = re.compile(r"\(digest ([0-9a-f]{8})\)")
_RESPONSE_BLOCK_RE = re.compile(r'<response label="([^"]+)">(.*?)</response>', re.S)
_ANSWER_MODEL_RE = re.compile(r'<answer label="[^"]*" model="([^"]+)">')
_QUESTION_RE = re.compile(r"<question>\n?(.*?)\n?</question>", re.S)
_ROUND_RE = re.compile(re.escape(DEBATE_MARKER) + r" round (\d+)")
_TOP_RE = re.compile(r"<council_ranking>\n1\. (\S+)")

# (first sentence, continuing after "Answer to "<question>": ", second sentence)
_STYLES: tuple[tuple[str, str], ...] = (
    (
        "it depends mostly on how strictly the key terms are defined.",
        "Under the most common definition, the conventional answer holds.",
    ),
    (
        "reasoning from first principles gives a clear default answer.",
        "The main caveat is that unusual edge cases can change the conclusion.",
    ),
    (
        "comparing the main alternatives side by side, the simplest explanation fits best.",
        "It is still worth double-checking against a reliable source.",
    ),
    (
        "the practical takeaway matters more than the theory here.",
        "Pick the option whose trade-offs you can actually live with.",
    ),
    (
        "experts do disagree, but the stronger arguments point one way.",
        "The opposite view is a reasonable minority position, not a mistake.",
    ),
)
_COMMENTS = (
    "Accurate, well organized and complete.",
    "Solid, but misses some useful detail.",
    "Reasonable, though less precise than the others.",
    "The least convincing of the set.",
)


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def answer_digest(name: str, question: str) -> str:
    """The ``(digest ...)`` token a fake member puts in its answers to ``question``."""
    return _digest(name, question.strip())[:8]


def register_fake_name(name: str) -> None:
    """Make ``name`` a candidate author for fake reviewers in this process."""
    _KNOWN_NAMES.add(name)


def identify_author(text: str, question: str, candidates: Iterable[str] = ()) -> str | None:
    """Return the fake member whose digest token appears in ``text``, else ``None``.

    Candidates are ``candidates`` plus every registered fake name.
    """
    tokens = _DIGEST_TOKEN_RE.findall(text)
    if not tokens:
        return None
    token = tokens[-1]
    for name in sorted(_KNOWN_NAMES.union(candidates)):
        if answer_digest(name, question) == token:
            return name
    return None


def _style(name: str, question: str) -> tuple[str, str]:
    base = int(_digest(question.strip(), "style")[:8], 16)
    if name in DEMO_MEMBERS:
        offset = DEMO_MEMBERS.index(name)  # the demo trio never shares a style
    else:
        offset = int(_digest(name, "style")[:8], 16)
    return _STYLES[(base + offset) % len(_STYLES)]


def _one_line(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _extract_question(prompt: str) -> str:
    match = _QUESTION_RE.search(prompt)
    return match.group(1).strip() if match else prompt.strip()


def _tokens(text: str) -> int:
    return max(1, len(text.split()))


class FakeProvider:
    """Offline provider; ``model`` is the part after ``fake:`` (e.g. ``"alpha"``)."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name

    async def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> Completion:
        if model.lower().startswith("fail"):
            raise ProviderError(f"fake: simulated failure for model {model!r}")
        register_fake_name(model)
        user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
        prompt = user_turns[-1] if user_turns else ""
        if SYNTHESIS_MARKER in prompt:
            text = self._synthesis(model, prompt)
        elif RANKING_MARKER in prompt:
            text = self._review(model, prompt)
        elif DEBATE_MARKER in prompt:
            text = self._debate(model, prompt)
        else:
            text = self._answer(model, prompt)
        prompt_tokens = sum(_tokens(m.get("content", "")) for m in messages)
        return Completion(
            text=text, input_tokens=prompt_tokens, output_tokens=_tokens(text), cost_usd=0.0
        )

    @staticmethod
    def _answer(model: str, question: str) -> str:
        question = question.strip()
        first, second = _style(model, question)
        return (
            f'Answer to "{_one_line(question, 80)}": {first} {second} '
            f"(digest {answer_digest(model, question)})"
        )

    @staticmethod
    def _debate(model: str, prompt: str) -> str:
        question = _extract_question(prompt)
        match = _ROUND_RE.search(prompt)
        round_no = match.group(1) if match else "?"
        others = len(_RESPONSE_BLOCK_RE.findall(prompt))
        _, second = _style(model, question)
        return (
            f'Revised answer (debate round {round_no}) to "{_one_line(question, 80)}": '
            f"having read {others} other answer(s), I keep my main point and tighten the "
            f"explanation. {second} (digest {answer_digest(model, question)})"
        )

    @staticmethod
    def _review(model: str, prompt: str) -> str:
        question = _extract_question(prompt)
        bodies: dict[str, str] = {}
        authors: dict[str, str | None] = {}
        for label, body in _RESPONSE_BLOCK_RE.findall(prompt):
            if label not in authors:
                bodies[label] = body.strip()
                authors[label] = identify_author(body, question, candidates=(model,))
        if not authors:
            return "I could not find any responses to rank."

        def quality(label: str) -> tuple[int, str]:
            author = authors[label]
            if author is None:
                return (1, _digest(question, bodies[label]))  # unknown: after identified
            if author == SELF_PREFERRING_MEMBER and model != SELF_PREFERRING_MEMBER:
                return (2, "")  # every other fake reviewer ranks alpha last
            return (0, _digest(question, author))

        order = sorted(authors, key=quality)
        if model == SELF_PREFERRING_MEMBER:
            own = [label for label in order if authors[label] == model]
            order = own + [label for label in order if label not in own]
        lines = []
        for position, label in enumerate(order):
            if position == len(order) - 1 and len(order) > 1:
                comment = _COMMENTS[-1]
            else:
                comment = _COMMENTS[min(position, len(_COMMENTS) - 2)]
            lines.append(f"{label}: {comment}")
        ranking = " > ".join(label_letter(label) for label in order)
        return "\n".join(lines) + f"\n\n{RANKING_MARKER} {ranking}"

    @staticmethod
    def _synthesis(model: str, prompt: str) -> str:
        question = _extract_question(prompt)
        answer_count = len(_ANSWER_MODEL_RE.findall(prompt))
        top = _TOP_RE.search(prompt)
        top_text = f", led by the answer from {top.group(1)}," if top else ""
        digest = _digest(model, question, "synthesis")[:8]
        return (
            f'SYNTHESIS by {model}: final answer to "{_one_line(question)}". It combines '
            f"{answer_count} council answer(s){top_text} into one response "
            f"(digest {digest})."
        )
