"""Tests for the client: the refusal order, budgets, usage accounting and conversations."""

from __future__ import annotations

import pytest

from llmgate import (
    Budget,
    Client,
    Conversation,
    FakeProvider,
    Message,
    ModelSpec,
    RetryPolicy,
    register,
)
from llmgate.errors import (
    BudgetExceededError,
    ContextOverflowError,
    ProviderError,
    RetriesExhaustedError,
    RetryableError,
    UnknownModelError,
)


def client(provider=None, **kwargs) -> Client:
    kwargs.setdefault("model", "gpt-4o-mini")
    kwargs.setdefault("sleep", lambda _: None)
    return Client(provider=provider or FakeProvider(), **kwargs)


class TestHappyPath:
    def test_returns_the_providers_text(self):
        reply = client(FakeProvider(replies=["Thirty days."])).complete(
            [Message("user", "How much notice?")]
        )
        assert reply.text == "Thirty days."
        assert reply.model == "gpt-4o-mini"
        assert reply.attempts == 1

    def test_passes_through_model_and_caps(self):
        provider = FakeProvider()
        client(provider, max_output_tokens=500, temperature=0.3).complete([Message("user", "Hi")])
        call = provider.calls[0]
        assert call["model"] == "gpt-4o-mini"
        assert call["max_output_tokens"] == 500
        assert call["temperature"] == 0.3

    def test_per_call_overrides(self):
        provider = FakeProvider()
        client(provider).complete(
            [Message("user", "Hi")], model="gpt-4o", max_output_tokens=64, temperature=1.0
        )
        assert provider.calls[0]["model"] == "gpt-4o"
        assert provider.calls[0]["max_output_tokens"] == 64
        assert provider.calls[0]["temperature"] == 1.0

    def test_output_cap_is_clamped_to_the_models_limit(self):
        provider = FakeProvider()
        client(provider, model="test-tiny", max_output_tokens=99_999).complete(
            [Message("user", "Hi")]
        )
        assert provider.calls[0]["max_output_tokens"] == 256  # test-tiny's cap

    def test_empty_messages_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            client().complete([])

    def test_unknown_model_is_an_error_not_a_guess(self):
        # Guessing a context window produces requests that fail; guessing a price produces a
        # budget that does not bind.
        with pytest.raises(UnknownModelError, match="No catalogue entry"):
            client(model="gpt-9-ultra").complete([Message("user", "Hi")])


class TestUsageAccounting:
    def test_reported_usage_is_not_marked_estimated(self):
        reply = client(FakeProvider(replies=["ok"], report_usage=True)).complete(
            [Message("user", "Hi")]
        )
        assert reply.usage.estimated is False
        assert reply.usage.cost_usd > 0

    def test_missing_provider_usage_falls_back_and_is_flagged(self):
        # A confident zero would be worse than an estimate that admits what it is.
        reply = client(FakeProvider(replies=["ok"], report_usage=False)).complete(
            [Message("user", "Hi")]
        )
        assert reply.usage.estimated is True
        assert reply.usage.prompt_tokens > 0

    def test_budget_accumulates_across_calls(self):
        budget = Budget(limit_usd=10.0)
        c = client(FakeProvider(replies=["a", "b", "c"]), budget=budget)
        for _ in range(3):
            c.complete([Message("user", "Hi")])
        assert budget.calls == 3
        assert budget.spent_usd > 0
        assert budget.usage.total_tokens > 0

    def test_truncation_is_exposed(self):
        reply = client(FakeProvider(replies=["cut off"])).complete([Message("user", "Hi")])
        assert reply.truncated is False

        provider = FakeProvider(replies=["cut off"], finish_reason="length")
        assert client(provider).complete([Message("user", "Hi")]).truncated is True

    @pytest.mark.parametrize("reason", ["length", "max_tokens", "MAX_TOKENS"])
    def test_provider_specific_truncation_spellings(self, reason):
        provider = FakeProvider(replies=["x"], finish_reason=reason)
        assert client(provider).complete([Message("user", "Hi")]).truncated is True


class TestBudgetEnforcement:
    def test_call_is_refused_before_it_is_sent(self):
        # The point of the library: an overrun cannot be un-spent afterwards.
        provider = FakeProvider()
        budget = Budget(limit_usd=0.000001)
        with pytest.raises(BudgetExceededError) as exc:
            client(provider, budget=budget, max_output_tokens=1000).complete(
                [Message("user", "Hi")]
            )
        assert provider.calls == [], "the request was sent despite breaching the budget"
        assert exc.value.projected_usd > exc.value.remaining_usd

    def test_worst_case_not_average_is_used(self):
        # Reserving 16k output tokens on gpt-4o costs ~$0.16 worst case; a budget below that
        # must refuse even though the actual reply will be short.
        provider = FakeProvider(replies=["tiny"])
        budget = Budget(limit_usd=0.01)
        with pytest.raises(BudgetExceededError):
            client(provider, model="gpt-4o", budget=budget, max_output_tokens=16_000).complete(
                [Message("user", "Hi")]
            )
        assert provider.calls == []

    def test_refusal_message_is_actionable(self):
        with pytest.raises(BudgetExceededError, match="Lower max_output_tokens"):
            client(budget=Budget(limit_usd=1e-9)).complete([Message("user", "Hi")])

    def test_no_limit_means_no_enforcement(self):
        c = client(budget=Budget())
        assert c.complete([Message("user", "Hi")]).text == "ok"

    def test_call_count_ceiling(self):
        budget = Budget(max_calls=2)
        c = client(FakeProvider(replies=["a", "b", "c"]), budget=budget)
        c.complete([Message("user", "Hi")])
        c.complete([Message("user", "Hi")])
        with pytest.raises(BudgetExceededError, match="3 of a 2-call limit"):
            c.complete([Message("user", "Hi")])

    def test_budget_bounds_total_spend(self):
        """The invariant: a budget that is checked on the worst case actually binds.

        Each permitted call can cost at most its projected worst case, and a call is only
        permitted when that projection fits the remaining budget. So total spend can never
        exceed the limit — which is the property an after-the-fact accounting layer cannot
        give you.
        """
        register(ModelSpec("pricey", 10_000, 100, 1000.0, 1000.0, "2026-11-23"))
        budget = Budget(limit_usd=0.5)
        # A long reply, so each call costs a meaningful fraction of the budget.
        c = client(
            FakeProvider(replies=["x" * 4000]),
            model="pricey",
            budget=budget,
            max_output_tokens=100,
        )

        sent = 0
        with pytest.raises(BudgetExceededError):
            for _ in range(50):
                c.complete([Message("user", "Hi")])
                sent += 1

        assert sent > 1, "several calls should have been affordable before the ceiling"
        assert budget.spent_usd <= budget.limit_usd, (
            f"spent ${budget.spent_usd:.4f} against a ${budget.limit_usd:.2f} ceiling"
        )


class TestRefusalOrder:
    def test_context_is_checked_before_the_budget(self):
        # Both would fail; the context error is the more specific diagnosis, and neither
        # should reach the provider.
        provider = FakeProvider()
        with pytest.raises(ContextOverflowError):
            client(
                provider,
                model="test-tiny",
                budget=Budget(limit_usd=1e-9),
                max_output_tokens=100,
            ).complete([Message("user", "x" * 20_000)])
        assert provider.calls == []

    def test_oversized_prompt_never_reaches_the_provider(self):
        provider = FakeProvider()
        with pytest.raises(ContextOverflowError):
            client(provider, model="test-tiny", max_output_tokens=100).complete(
                [Message("user", "y" * 20_000)]
            )
        assert provider.calls == []

    def test_history_is_trimmed_rather_than_refused_when_possible(self):
        provider = FakeProvider(replies=["ok"])
        messages = [Message("system", "Be brief.")]
        messages += [Message("user", "z" * 400) for _ in range(12)]
        messages.append(Message("user", "final?"))

        reply = client(provider, model="test-tiny", max_output_tokens=100).complete(messages)
        assert reply.text == "ok"
        sent = provider.calls[0]["messages"]
        assert len(sent) < len(messages)
        assert sent[0].role == "system"
        assert sent[-1].content == "final?"


class TestRetryIntegration:
    def test_transient_failure_then_success(self):
        provider = FakeProvider(replies=[RetryableError("429"), "recovered"])
        reply = client(provider, retry=RetryPolicy(max_attempts=3, base_delay=0)).complete(
            [Message("user", "Hi")]
        )
        assert reply.text == "recovered"
        assert reply.attempts == 2

    def test_permanent_failure_is_not_retried(self):
        provider = FakeProvider(replies=[ProviderError("invalid request")])
        with pytest.raises(ProviderError, match="invalid request"):
            client(provider, retry=RetryPolicy(max_attempts=4, base_delay=0)).complete(
                [Message("user", "Hi")]
            )
        assert len(provider.calls) == 1

    def test_exhausted_retries_raise(self):
        provider = FakeProvider(replies=[RetryableError("503")])
        with pytest.raises(RetriesExhaustedError):
            client(provider, retry=RetryPolicy(max_attempts=3, base_delay=0)).complete(
                [Message("user", "Hi")]
            )
        assert len(provider.calls) == 3

    def test_failed_calls_do_not_consume_budget(self):
        budget = Budget(limit_usd=10.0)
        provider = FakeProvider(replies=[RetryableError("503")])
        with pytest.raises(RetriesExhaustedError):
            client(
                provider, budget=budget, retry=RetryPolicy(max_attempts=2, base_delay=0)
            ).complete([Message("user", "Hi")])
        assert budget.spent_usd == 0.0
        assert budget.calls == 0


class TestConversation:
    def test_history_grows_with_each_turn(self):
        convo = Conversation(client=client(FakeProvider(replies=["one", "two"])))
        convo.ask("first?")
        convo.ask("second?")
        assert [m.role for m in convo.messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert convo.messages[1].content == "one"

    def test_system_prompt_is_inserted_once(self):
        convo = Conversation(client=client(), system="Be terse.")
        convo.ask("hi?")
        assert [m.role for m in convo.messages].count("system") == 1
        assert convo.messages[0].content == "Be terse."

    def test_history_is_not_destructively_truncated(self):
        """The previous version mutated history in place, so one long message erased it."""
        convo = Conversation(
            client=client(model="test-tiny", max_output_tokens=100), system="Be brief."
        )
        for _ in range(12):
            convo.messages.append(Message("user", "q" * 400))
        convo.ask("final?")
        # Full history is retained even though the request itself was trimmed.
        assert len(convo.messages) >= 14

    def test_reset_keeps_the_system_prompt_by_default(self):
        convo = Conversation(client=client(), system="Be terse.")
        convo.ask("hi?")
        convo.reset()
        assert [m.role for m in convo.messages] == ["system"]
        convo.reset(keep_system=False)
        assert convo.messages == []

    def test_spend_is_visible(self):
        convo = Conversation(client=client(budget=Budget(limit_usd=5.0)))
        convo.ask("hi?")
        assert convo.spent_usd > 0
