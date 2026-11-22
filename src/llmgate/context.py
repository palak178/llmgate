"""Fitting a conversation into a context window.

The rules, in priority order, and each exists because the obvious implementation gets it
wrong:

1. **System messages are never dropped.** They carry the instructions that make the reply
   correct. Trimming oldest-first treats a system prompt as the most expendable message in
   the conversation, which is exactly backwards.
2. **The most recent user message is never dropped.** Dropping it answers a different
   question than the one asked.
3. **Otherwise, oldest first.** Recent turns carry more of the state a reply depends on.
4. **Turn pairs are dropped together.** Removing a user message while keeping the assistant
   reply to it leaves an answer to an invisible question, which reads as a non-sequitur to
   the model and degrades the next completion.
5. **If the protected messages alone do not fit, refuse.** Trimming cannot help, so a
   `ContextOverflowError` before the call beats a provider error after one.

Rules 1, 2 and 4 are what separate this from the obvious implementation, which pops the
oldest message until a size check passes. That loop discards the system prompt first, will
drop the question being answered, and splits turn pairs — and if its arithmetic can go
negative it empties the conversation entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

from llmgate.errors import ContextOverflowError
from llmgate.models import Message, ModelSpec
from llmgate.tokens import count_messages


@dataclass(frozen=True, slots=True)
class FitResult:
    """What trimming did, so a caller can log or warn rather than guess."""

    messages: list[Message]
    prompt_tokens: int
    exact: bool
    dropped: int
    """Number of messages removed. Non-zero means the model lost conversational history."""

    @property
    def trimmed(self) -> bool:
        return self.dropped > 0


def fit(
    messages: list[Message],
    *,
    spec: ModelSpec,
    reserve_output_tokens: int,
) -> FitResult:
    """Trim `messages` so prompt + reserved output fits `spec.context_tokens`.

    `reserve_output_tokens` is held back from the window. Without reserving it, a prompt can
    fill the context entirely and leave the model no room to answer — the request succeeds
    and returns an empty or instantly-truncated completion, which is harder to diagnose than
    a refusal.
    """
    if reserve_output_tokens < 1:
        raise ValueError(f"reserve_output_tokens must be >= 1, got {reserve_output_tokens}")
    if reserve_output_tokens > spec.max_output_tokens:
        raise ContextOverflowError(
            f"Reserved {reserve_output_tokens} output tokens but {spec.name} caps output at "
            f"{spec.max_output_tokens}.",
            required=reserve_output_tokens,
            available=spec.max_output_tokens,
        )

    available = spec.context_tokens - reserve_output_tokens
    if available < 1:
        raise ContextOverflowError(
            f"Reserving {reserve_output_tokens} tokens leaves no room in {spec.name}'s "
            f"{spec.context_tokens}-token window.",
            required=reserve_output_tokens,
            available=spec.context_tokens,
        )

    total, exact = count_messages(messages, model=spec.name)
    if total <= available:
        return FitResult(messages=list(messages), prompt_tokens=total, exact=exact, dropped=0)

    protected = _protected_indices(messages)
    kept = list(range(len(messages)))

    # Oldest-first over droppable messages, removing whole turn pairs.
    for index in _drop_order(messages, protected):
        if index not in kept:
            continue
        for victim in _turn_group(messages, index, protected):
            if victim in kept:
                kept.remove(victim)
        candidate = [messages[i] for i in kept]
        total, exact = count_messages(candidate, model=spec.name)
        if total <= available:
            return FitResult(
                messages=candidate,
                prompt_tokens=total,
                exact=exact,
                dropped=len(messages) - len(candidate),
            )

    # Only protected messages remain and they still do not fit.
    remaining = [messages[i] for i in kept]
    total, exact = count_messages(remaining, model=spec.name)
    raise ContextOverflowError(
        f"The system prompt and latest message need {total} tokens but only {available} are "
        f"available in {spec.name} after reserving {reserve_output_tokens} for the reply. "
        f"Shorten the input, or use a model with a larger window.",
        required=total,
        available=available,
    )


def _protected_indices(messages: list[Message]) -> set[int]:
    """Indices that must survive: every system message, and the last user message."""
    protected = {i for i, m in enumerate(messages) if m.role == "system"}
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            protected.add(i)
            break
    return protected


def _drop_order(messages: list[Message], protected: set[int]) -> list[int]:
    return [i for i in range(len(messages)) if i not in protected]


def _turn_group(messages: list[Message], index: int, protected: set[int]) -> list[int]:
    """`index` plus the assistant replies that immediately follow it.

    Dropping a user message without its answer — or an answer without its question — leaves
    the model reading a dangling half-turn.
    """
    group = [index]
    if messages[index].role == "user":
        follower = index + 1
        while (
            follower < len(messages)
            and messages[follower].role == "assistant"
            and follower not in protected
        ):
            group.append(follower)
            follower += 1
    return group
