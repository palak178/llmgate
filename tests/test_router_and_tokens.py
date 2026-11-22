"""Tests for model routing, token counting and the catalogue."""

from __future__ import annotations

import pytest

from llmgate import Message, ModelSpec, choose, known_models, lookup, register
from llmgate.errors import ContextOverflowError, UnknownModelError
from llmgate.tokens import (
    CHARS_PER_TOKEN,
    MESSAGE_OVERHEAD_TOKENS,
    count_messages,
    count_text,
    tiktoken_available,
)


class TestTokenCounting:
    def test_longer_text_counts_more(self):
        short, _ = count_text("hello")
        long, _ = count_text("hello " * 100)
        assert long > short

    def test_empty_text_is_zero_or_one(self):
        tokens, _ = count_text("")
        assert tokens == 0

    def test_heuristic_rounds_up(self):
        # Pessimism is the point: an underestimate produces a rejected request, an
        # overestimate only trims slightly more than necessary.
        text = "x" * 100
        tokens, exact = count_text(text)
        if not exact:
            assert tokens >= 100 / CHARS_PER_TOKEN

    def test_heuristic_reports_itself_inexact(self):
        _, exact = count_text("hello", model="definitely-not-a-real-model")
        assert exact is False

    def test_message_framing_is_counted(self):
        # Ignoring per-message overhead makes long conversations of short messages overflow
        # unexpectedly.
        one, _ = count_messages([Message("user", "hi")])
        five, _ = count_messages([Message("user", "hi")] * 5)
        assert five >= one + 4 * MESSAGE_OVERHEAD_TOKENS

    def test_exactness_propagates_across_messages(self):
        _, exact = count_messages([Message("user", "a"), Message("user", "b")], model="not-real")
        assert exact is False

    @pytest.mark.skipif(not tiktoken_available(), reason="tiktoken is not installed")
    def test_tiktoken_is_exact_when_available(self):
        tokens, exact = count_text("hello world", model="gpt-4o-mini")
        assert exact is True
        assert tokens > 0


class TestCatalogue:
    def test_known_models_are_listed(self):
        names = [spec.name for spec in known_models()]
        assert "gpt-4o-mini" in names
        assert names == sorted(names)

    def test_every_entry_records_when_it_was_recorded(self):
        # Staleness you can see beats staleness you cannot.
        assert all(spec.as_of for spec in known_models())

    def test_unknown_model_raises_with_a_useful_message(self):
        with pytest.raises(UnknownModelError) as exc:
            lookup("gpt-5-turbo-ultra")
        assert "register" in str(exc.value)

    def test_register_adds_a_model(self):
        register(ModelSpec("my-local-llama", 8_192, 2_048, 0.0, 0.0, "2026-11-23"))
        spec = lookup("my-local-llama")
        assert spec.context_tokens == 8_192
        assert spec.cost(1000, 1000) == 0.0

    def test_cost_arithmetic(self):
        spec = ModelSpec("priced", 1000, 100, 1.0, 2.0, "2026-11-23")
        # 1M input tokens at $1/M plus 1M output at $2/M.
        assert spec.cost(1_000_000, 1_000_000) == pytest.approx(3.0)
        assert spec.cost(0, 0) == 0.0


class TestRouting:
    def test_chooses_the_cheapest_model_that_fits(self):
        decision = choose(
            [Message("user", "What is the notice period?")],
            candidates=["gpt-4o", "gpt-4o-mini"],
            reserve_output_tokens=256,
        )
        assert decision.model == "gpt-4o-mini"
        assert decision.projected_usd > 0

    def test_skips_models_whose_window_is_too_small(self):
        decision = choose(
            [Message("user", "x" * 40_000)],
            candidates=["test-tiny", "gpt-4o-mini"],
            reserve_output_tokens=256,
        )
        assert decision.model == "gpt-4o-mini"
        assert any(name == "test-tiny" for name, _ in decision.rejected)

    def test_skips_models_the_budget_cannot_afford(self):
        decision = choose(
            [Message("user", "Hi")],
            candidates=["gpt-4o", "gpt-4o-mini"],
            reserve_output_tokens=16_000,
            remaining_usd=0.02,
        )
        assert decision.model == "gpt-4o-mini"
        reasons = dict(decision.rejected)
        assert "exceeds" in reasons["gpt-4o"]

    def test_raises_when_nothing_fits(self):
        with pytest.raises(ContextOverflowError) as exc:
            choose(
                [Message("user", "x" * 40_000)],
                candidates=["test-tiny"],
                reserve_output_tokens=256,
            )
        assert "No candidate model" in str(exc.value)

    def test_raises_when_nothing_is_affordable(self):
        with pytest.raises(ContextOverflowError):
            choose(
                [Message("user", "Hi")],
                candidates=["gpt-4o", "gpt-4o-mini"],
                reserve_output_tokens=1000,
                remaining_usd=1e-9,
            )

    def test_ties_break_on_the_given_order(self):
        register(ModelSpec("tie-a", 10_000, 500, 1.0, 1.0, "2026-11-23"))
        register(ModelSpec("tie-b", 10_000, 500, 1.0, 1.0, "2026-11-23"))
        assert choose([Message("user", "Hi")], candidates=["tie-b", "tie-a"]).model == "tie-b"
        assert choose([Message("user", "Hi")], candidates=["tie-a", "tie-b"]).model == "tie-a"

    def test_empty_candidates_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            choose([Message("user", "Hi")], candidates=[])

    def test_decision_explains_itself(self):
        decision = choose(
            [Message("user", "x" * 40_000)],
            candidates=["test-tiny", "gpt-4o-mini"],
            reserve_output_tokens=256,
        )
        explanation = decision.explain()
        assert "chose gpt-4o-mini" in explanation
        assert "rejected test-tiny" in explanation

    def test_output_reservation_is_clamped_per_candidate(self):
        # test-tiny caps output at 256, so asking for 1024 must not disqualify it on output.
        decision = choose(
            [Message("user", "Hi")], candidates=["test-tiny"], reserve_output_tokens=1024
        )
        assert decision.model == "test-tiny"


class TestMessageValidation:
    def test_valid_roles(self):
        for role in ("system", "user", "assistant"):
            assert Message(role, "x").role == role

    def test_invalid_role_rejected(self):
        with pytest.raises(ValueError, match="Unknown role"):
            Message("tool", "x")
