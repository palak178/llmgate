"""Tests for context fitting.

Each class here maps to a rule the previous version of this library broke. It popped
`memory[0]` until an arithmetic check passed, which meant it could drop the system prompt,
drop the question being asked, split a turn pair, and — when its subtraction went negative —
drop everything.
"""

from __future__ import annotations

import pytest

from llmgate.context import fit
from llmgate.errors import ContextOverflowError
from llmgate.models import Message, ModelSpec, lookup

TINY = lookup("test-tiny")  # 1,000-token window, 256 max output


def filler(tokens: int, label: str = "x") -> str:
    """Text that counts as roughly `tokens` tokens under the heuristic (3.6 chars each)."""
    return (label * 3 + " ") * max(1, int(tokens * 3.6 / 4))


class TestNoTrimNeeded:
    def test_short_conversation_is_untouched(self):
        messages = [Message("system", "Be brief."), Message("user", "Hello?")]
        result = fit(messages, spec=TINY, reserve_output_tokens=100)
        assert result.messages == messages
        assert not result.trimmed
        assert result.dropped == 0

    def test_prompt_tokens_are_reported(self):
        result = fit([Message("user", "Hello?")], spec=TINY, reserve_output_tokens=100)
        assert result.prompt_tokens > 0

    def test_heuristic_counting_is_flagged_inexact(self):
        # Without tiktoken, or for an unknown model, the count is a guess and says so.
        result = fit([Message("user", "Hello?")], spec=TINY, reserve_output_tokens=100)
        assert result.exact is False


class TestProtectedMessages:
    def test_system_message_survives_trimming(self):
        messages = [
            Message("system", "Always answer in French."),
            *[Message("user", filler(120, str(i))) for i in range(10)],
            Message("user", "Final question?"),
        ]
        result = fit(messages, spec=TINY, reserve_output_tokens=100)
        assert result.trimmed
        assert result.messages[0].role == "system"
        assert "French" in result.messages[0].content

    def test_every_system_message_survives(self):
        messages = [
            Message("system", "Rule one."),
            *[Message("user", filler(120, str(i))) for i in range(8)],
            Message("system", "Rule two."),
            Message("user", "Final?"),
        ]
        result = fit(messages, spec=TINY, reserve_output_tokens=100)
        systems = [m.content for m in result.messages if m.role == "system"]
        assert systems == ["Rule one.", "Rule two."]

    def test_latest_user_message_survives(self):
        messages = [
            *[Message("user", filler(150, str(i))) for i in range(10)],
            Message("user", "The actual question?"),
        ]
        result = fit(messages, spec=TINY, reserve_output_tokens=100)
        assert result.messages[-1].content == "The actual question?"

    def test_a_single_huge_latest_message_refuses_rather_than_dropping_it(self):
        # Answering a different question than the one asked is worse than refusing.
        messages = [Message("user", filler(5000))]
        with pytest.raises(ContextOverflowError) as exc:
            fit(messages, spec=TINY, reserve_output_tokens=100)
        assert exc.value.required > exc.value.available
        assert "larger window" in str(exc.value)

    def test_system_plus_question_too_large_refuses(self):
        messages = [Message("system", filler(3000)), Message("user", "Hi?")]
        with pytest.raises(ContextOverflowError):
            fit(messages, spec=TINY, reserve_output_tokens=100)


class TestDropOrder:
    def test_oldest_messages_go_first(self):
        messages = [Message("user", f"turn {i} " + filler(100, str(i))) for i in range(10)] + [
            Message("user", "newest?")
        ]
        result = fit(messages, spec=TINY, reserve_output_tokens=100)
        kept = " ".join(m.content for m in result.messages)
        assert "turn 0" not in kept
        assert "newest?" in kept

    def test_turn_pairs_are_dropped_together(self):
        # An assistant reply with no question above it reads as a non-sequitur.
        messages = []
        for i in range(8):
            messages.append(Message("user", f"q{i} " + filler(70, str(i))))
            messages.append(Message("assistant", f"a{i} " + filler(70, str(i))))
        messages.append(Message("user", "final?"))

        result = fit(messages, spec=TINY, reserve_output_tokens=100)
        kept = [m.content.split()[0] for m in result.messages if m.role != "system"]
        for i in range(8):
            # Never an answer without its question.
            if f"a{i}" in kept:
                assert f"q{i}" in kept, f"a{i} kept without q{i}"

    def test_trimming_reports_how_many_were_dropped(self):
        messages = [Message("user", filler(120, str(i))) for i in range(10)] + [
            Message("user", "last?")
        ]
        result = fit(messages, spec=TINY, reserve_output_tokens=100)
        assert result.dropped == len(messages) - len(result.messages)
        assert result.dropped > 0


class TestReservation:
    def test_output_reservation_is_held_back(self):
        # A prompt that fills the window entirely leaves no room to answer.
        messages = [Message("user", filler(200))]
        small = fit(messages, spec=TINY, reserve_output_tokens=10)
        assert small.prompt_tokens + 10 <= TINY.context_tokens

    def test_reserving_more_than_the_models_output_cap_is_rejected(self):
        with pytest.raises(ContextOverflowError, match="caps output"):
            fit(
                [Message("user", "Hi")], spec=TINY, reserve_output_tokens=TINY.max_output_tokens + 1
            )

    def test_zero_reservation_is_rejected(self):
        with pytest.raises(ValueError, match=">= 1"):
            fit([Message("user", "Hi")], spec=TINY, reserve_output_tokens=0)

    def test_reservation_filling_the_whole_window_is_rejected(self):
        spec = ModelSpec("squeeze", 100, 100, 1.0, 1.0, "2026-11-23")
        with pytest.raises(ContextOverflowError, match="no room"):
            fit([Message("user", "Hi")], spec=spec, reserve_output_tokens=100)


class TestTermination:
    def test_many_small_messages_terminate(self):
        # Per-message framing overhead means this cannot be solved by character count alone.
        messages = [Message("user", "a") for _ in range(500)] + [Message("user", "last?")]
        result = fit(messages, spec=TINY, reserve_output_tokens=100)
        assert len(result.messages) < len(messages)

    def test_large_window_needs_no_trim(self):
        messages = [Message("user", filler(200, str(i))) for i in range(20)]
        result = fit(messages, spec=lookup("gpt-4o-mini"), reserve_output_tokens=1024)
        assert not result.trimmed
