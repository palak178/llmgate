"""Exceptions.

The split that matters is `RetryableError` versus everything else, and it is a type rather
than a condition rewritten at each call site. Getting it wrong is costly in both directions:
retrying an invalid request burns attempts on something that will fail identically forever,
while giving up on a rate limit discards a call that would have succeeded a second later.
"""

from __future__ import annotations


class LLMGateError(Exception):
    """Base for every error this package raises."""


class UnknownModelError(LLMGateError):
    """The model is not in the catalogue, so its limits and price are unknown."""


# ---------------------------------------------------------------------------------------
# Refusals: raised *before* a request is sent
# ---------------------------------------------------------------------------------------


class RefusedError(LLMGateError):
    """A request was not sent because it could be proven to fail or overspend.

    The whole point of the library. A refusal is cheap and informative; the alternative is a
    provider error after a round trip, or a bill.
    """


class ContextOverflowError(RefusedError):
    """Prompt plus reserved output exceeds the model's window, and trimming cannot fix it."""

    def __init__(self, message: str, *, required: int, available: int) -> None:
        super().__init__(message)
        self.required = required
        self.available = available


class BudgetExceededError(RefusedError):
    """The projected worst-case cost of this call would breach the session budget."""

    def __init__(self, message: str, *, projected_usd: float, remaining_usd: float) -> None:
        super().__init__(message)
        self.projected_usd = projected_usd
        self.remaining_usd = remaining_usd


# ---------------------------------------------------------------------------------------
# Provider failures
# ---------------------------------------------------------------------------------------


class ProviderError(LLMGateError):
    """A provider call failed and must not be retried.

    The default for an unrecognised failure. Treating unknown errors as permanent means a new
    error type costs one failed call, whereas defaulting to retryable would hammer a provider
    with a request that cannot succeed.
    """


class RetryableError(ProviderError):
    """A transient failure: rate limit, timeout, 5xx. Safe to send again after a delay."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        """Seconds the provider asked us to wait, when it said so.

        Honoured in preference to computed backoff: a server's own number is better
        information than any client-side guess.
        """


class RetriesExhaustedError(ProviderError):
    """Every attempt failed. Carries the last underlying error as `__cause__`."""

    def __init__(self, message: str, *, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts
