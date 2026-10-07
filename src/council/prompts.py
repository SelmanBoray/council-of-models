"""Prompt templates for every council stage.

All builders return chat messages (``[{"role": ..., "content": ...}]``). The stage
markers below are stable strings that the offline fake provider uses to recognise
which stage a prompt belongs to; keep them in the prompts if you edit the wording.
"""

from __future__ import annotations

from collections.abc import Sequence

from council.providers.base import Message

RANKING_MARKER = "RANKING:"
"""Prefix of the final line every reviewer must write, e.g. ``RANKING: B > A > C``."""

DEBATE_MARKER = "[COUNCIL DEBATE]"
REVIEW_MARKER = "[COUNCIL REVIEW]"
SYNTHESIS_MARKER = "[COUNCIL CHAIRMAN]"


def _block(tag: str, body: str, **attrs: str) -> str:
    attr_text = "".join(f' {key}="{value}"' for key, value in attrs.items())
    return f"<{tag}{attr_text}>\n{body.strip()}\n</{tag}>"


def label_letter(label: str) -> str:
    """``"Response A"`` -> ``"A"``."""
    stripped = label.strip()
    return stripped.split()[-1] if stripped else label


def answer_messages(question: str) -> list[Message]:
    """Stage 1: the member simply answers the question."""
    return [{"role": "user", "content": question}]


def debate_messages(
    question: str,
    own_answer: str,
    others: Sequence[tuple[str, str]],
    *,
    round_: int,
    total_rounds: int,
) -> list[Message]:
    """Stage 2: show a member its own answer plus the others' (anonymized) answers.

    ``others`` is a sequence of ``(label, answer_text)`` pairs.
    """
    if others:
        other_blocks = "\n\n".join(_block("response", text, label=label) for label, text in others)
    else:
        other_blocks = "(no other answers are available)"
    content = (
        f"{DEBATE_MARKER} round {round_} of {total_rounds}\n\n"
        "You are one member of a council of AI models that all answered the same "
        "question. Below are the question, your current answer, and the current "
        "answers of the other members (anonymized, in random order).\n\n"
        f"{_block('question', question)}\n\n"
        f"{_block('your_answer', own_answer)}\n\n"
        f"<other_answers>\n{other_blocks}\n</other_answers>\n\n"
        "Read the other answers critically. If they reveal a mistake, a gap or a "
        "better approach, improve your answer. If you are still convinced your answer "
        "is best, keep its substance. Do not copy another answer wholesale, and do not "
        "mention the other answers or this debate.\n\n"
        "Reply with your complete revised answer only; it will replace your previous "
        "answer."
    )
    return [{"role": "user", "content": content}]


def review_messages(question: str, labeled_answers: Sequence[tuple[str, str]]) -> list[Message]:
    """Stage 3: ask a member to rank all anonymized answers (its own included).

    ``labeled_answers`` is a sequence of ``(label, answer_text)`` pairs such as
    ``("Response A", "...")``. Model names must never appear here.
    """
    blocks = "\n\n".join(_block("response", text, label=label) for label, text in labeled_answers)
    letters = ", ".join(label_letter(label) for label, _ in labeled_answers)
    count = len(labeled_answers)
    content = (
        f"{REVIEW_MARKER}\n\n"
        "You are one member of a council of AI models. Every member independently "
        "answered the question below. The answers are shown anonymously and in random "
        "order; one of them may be your own.\n\n"
        f"{_block('question', question)}\n\n"
        f"{blocks}\n\n"
        "Evaluate the responses impartially on accuracy, completeness, reasoning and "
        "usefulness to the person who asked. Do not reward length, style or position, "
        "and do not try to guess which response is yours.\n\n"
        "First write a brief evaluation of each response (one to three sentences each). "
        f"Then end your reply with exactly one final line that ranks ALL {count} "
        f"responses from best to worst, using only their letters ({letters}), each "
        "letter exactly once, separated by '>':\n\n"
        f"{RANKING_MARKER} <letter> > <letter> > ... > <letter>\n\n"
        f"Nothing may follow the {RANKING_MARKER} line."
    )
    return [{"role": "user", "content": content}]


def synthesis_messages(
    question: str,
    answers: Sequence[tuple[str, str, str]],
    reviews: Sequence[tuple[str, list[str], str]],
    council_ranking: Sequence[tuple[str, float]] = (),
) -> list[Message]:
    """Stage 4: the chairman writes the final answer.

    ``answers``: ``(label, model_id, text)`` triples (labels let the chairman
    resolve the anonymous references inside reviewers' rationales).
    ``reviews``: ``(reviewer_model_id, ranking_model_ids_best_first, rationale)``.
    ``council_ranking``: ``(model_id, mean_position)`` pairs, best first.
    """
    answer_blocks = "\n\n".join(
        _block("answer", text, label=label, model=model) for label, model, text in answers
    )
    if reviews:
        review_blocks = "\n\n".join(
            _block(
                "review",
                f"Ranking (best to worst): {' > '.join(ranking)}\n"
                f"Rationale:\n{rationale.strip() or '(none given)'}",
                reviewer=reviewer,
            )
            for reviewer, ranking, rationale in reviews
        )
    else:
        review_blocks = "(no reviews are available)"
    parts = [
        f"{SYNTHESIS_MARKER}\n",
        "You are the chairman of a council of AI models. Each member answered the "
        "question below; then every member anonymously reviewed and ranked all of the "
        "answers, its own included. Model names and the anonymous labels the reviewers "
        "saw are both given so you can interpret the reviews.\n",
        _block("question", question) + "\n",
        f"<answers>\n{answer_blocks}\n</answers>\n",
        f"<reviews>\n{review_blocks}\n</reviews>\n",
    ]
    if council_ranking:
        ranking_lines = "\n".join(
            f"{i}. {model} (mean position {mean:.2f})"
            for i, (model, mean) in enumerate(council_ranking, start=1)
        )
        parts.append(f"<council_ranking>\n{ranking_lines}\n</council_ranking>\n")
    parts.append(
        "Write the best possible final answer to the question for the person who asked "
        "it. Build on the strongest points, correct mistakes, and resolve disagreements "
        "on the merits: the reviews and the ranking are evidence, not orders. Answer "
        "directly, as if you were answering yourself. Do not describe the council, the "
        "individual answers, the reviews or the ranking."
    )
    return [{"role": "user", "content": "\n".join(parts)}]
