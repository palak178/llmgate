"""Tests for the retry policy.

The first class is the regression test for the previous version's worst bug: it retried an
invalid-request error, which cannot ever succeed, and slept a flat 60 seconds between
attempts.

`sleep` is injected throughout, so the delay *sequence* is asserted without the suite ever
actually waiting. A retry policy whose tests have to sleep is either slow or untested.
"""

from __future__ import annotations

import random

import pytest

from llmgate.errors import ProviderError, RetriesExhaustedError, RetryableError
from llmgate.retry import RetryPolicy, with_retries


class Recorder:
    """Captures requested delays instead of sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


class TestWhatIsRetried:
    def test_retryable_error_is_retried(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RetryableError("rate limited")
            return "ok"

        sleep = Recorder()
        result, attempts = with_retries(flaky, policy=RetryPolicy(max_attempts=5), sleep=sleep)
        assert result == "ok"
        assert attempts == 3
        assert len(sleep.delays) == 2

    def test_non_retryable_error_fails_immediately(self):
        """The headline fix: an invalid request will fail identically forever."""
        calls = {"n": 0}

        def invalid():
            calls["n"] += 1
            raise ProviderError("invalid request: bad parameter")

        sleep = Recorder()
        with pytest.raises(ProviderError, match="invalid request"):
            with_retries(invalid, policy=RetryPolicy(max_attempts=5), sleep=sleep)
        assert calls["n"] == 1, "a non-retryable error must not be retried"
        assert sleep.delays == []

    def test_unexpected_exceptions_propagate_without_retrying(self):
        def boom():
            raise ValueError("programming error")

        with pytest.raises(ValueError):
            with_retries(boom, policy=RetryPolicy(max_attempts=3), sleep=Recorder())

    def test_success_on_the_first_attempt_never_sleeps(self):
        sleep = Recorder()
        result, attempts = with_retries(lambda: "ok", policy=RetryPolicy(), sleep=sleep)
        assert (result, attempts) == ("ok", 1)
        assert sleep.delays == []

    def test_exhaustion_raises_and_preserves_the_cause(self):
        def always_fails():
            raise RetryableError("still rate limited")

        sleep = Recorder()
        with pytest.raises(RetriesExhaustedError) as exc:
            with_retries(always_fails, policy=RetryPolicy(max_attempts=3), sleep=sleep)
        assert exc.value.attempts == 3
        assert isinstance(exc.value.__cause__, RetryableError)
        assert len(sleep.delays) == 2, "no sleep after the final attempt"


class TestBackoff:
    def test_delay_grows_exponentially(self):
        policy = RetryPolicy(base_delay=1.0, jitter=0.0, max_delay=100.0)
        assert [policy.delay_for(i) for i in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 8.0]

    def test_delay_is_capped(self):
        policy = RetryPolicy(base_delay=1.0, jitter=0.0, max_delay=5.0)
        assert policy.delay_for(10) == 5.0

    def test_jitter_spreads_delays(self):
        # Without jitter every client retries in lockstep and keeps a recovering service down.
        policy = RetryPolicy(base_delay=1.0, jitter=0.5, max_delay=100.0)
        random.seed(0)
        samples = {policy.delay_for(3) for _ in range(40)}
        assert len(samples) > 1
        assert all(2.0 <= d <= 6.0 for d in samples)

    def test_jitter_stays_non_negative(self):
        policy = RetryPolicy(base_delay=1.0, jitter=1.0)
        random.seed(1)
        assert all(policy.delay_for(1) >= 0.0 for _ in range(50))

    def test_retry_after_overrides_computed_backoff(self):
        # The server stating a requirement beats any client-side guess.
        policy = RetryPolicy(base_delay=1.0, jitter=0.0, max_delay=5.0)
        assert policy.delay_for(1, retry_after=42.0) == 42.0

    def test_retry_after_is_honoured_end_to_end(self):
        def rate_limited():
            raise RetryableError("slow down", retry_after=7.5)

        sleep = Recorder()
        with pytest.raises(RetriesExhaustedError):
            with_retries(rate_limited, policy=RetryPolicy(max_attempts=3), sleep=sleep)
        assert sleep.delays == [7.5, 7.5]

    def test_negative_retry_after_is_clamped(self):
        policy = RetryPolicy()
        assert policy.delay_for(1, retry_after=-5.0) == 0.0


class TestPolicyValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_attempts": 0}, "max_attempts"),
            ({"jitter": 1.5}, "jitter"),
            ({"jitter": -0.1}, "jitter"),
            ({"base_delay": -1.0}, "base_delay"),
        ],
    )
    def test_invalid_policies_are_rejected_at_construction(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            RetryPolicy(**kwargs)

    def test_single_attempt_policy_does_not_sleep(self):
        def always_fails():
            raise RetryableError("nope")

        sleep = Recorder()
        with pytest.raises(RetriesExhaustedError):
            with_retries(always_fails, policy=RetryPolicy(max_attempts=1), sleep=sleep)
        assert sleep.delays == []
