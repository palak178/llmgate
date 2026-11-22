"""Retry policy: jittered exponential backoff, and only for errors that can succeed later.

Three decisions, each with a test, and each the opposite of what a naive retry loop does:

* **Retryability is a property of the error type**, and an unrecognised error defaults to
  *not* retryable. An invalid request fails identically forever, so attempting it again only
  adds latency and load.
* **Backoff is exponential and jittered.** A constant delay across many clients produces a
  thundering herd that keeps a recovering service down; jitter decorrelates them and costs
  nothing to implement.
* **`Retry-After` wins.** When a provider states how long to wait, that is better
  information than any client-side computation.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from llmgate.errors import RetriesExhaustedError, RetryableError

log = logging.getLogger("llmgate.retry")

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times to retry, and how long to wait."""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: float = 0.25
    """Fraction of the delay randomised, in [0, 1].

    Full jitter (1.0) decorrelates clients best but makes the wait unpredictable; none (0.0)
    reproduces the herd. A quarter keeps backoff roughly predictable while still spreading a
    fleet of callers apart.
    """

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError(f"jitter must be in [0, 1], got {self.jitter}")
        if self.base_delay < 0:
            raise ValueError(f"base_delay must be >= 0, got {self.base_delay}")

    def delay_for(self, attempt: int, *, retry_after: float | None = None) -> float:
        """Seconds to wait before `attempt` (1-based count of attempts already made).

        A provider's own `Retry-After` wins outright — it is not jittered and not capped by
        `max_delay`, because the server is stating a requirement rather than offering a hint.
        """
        if retry_after is not None:
            return max(0.0, retry_after)
        exponential = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        spread = exponential * self.jitter
        return max(0.0, exponential - spread + random.random() * 2 * spread)


def with_retries(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
    description: str = "request",
) -> tuple[T, int]:
    """Run `operation`, retrying `RetryableError` only. Returns `(result, attempts)`.

    `sleep` is injectable so the tests assert on the delay *sequence* without waiting for it.
    A retry policy whose tests have to actually sleep is a retry policy that is either slow or
    untested, and usually both.
    """
    last: RetryableError | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation(), attempt
        except RetryableError as exc:
            last = exc
            if attempt == policy.max_attempts:
                break
            delay = policy.delay_for(attempt, retry_after=exc.retry_after)
            log.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.2fs",
                description,
                attempt,
                policy.max_attempts,
                exc,
                delay,
            )
            sleep(delay)
        # Every other exception, including ProviderError, propagates on the first failure.

    raise RetriesExhaustedError(
        f"{description} failed after {policy.max_attempts} attempt(s): {last}",
        attempts=policy.max_attempts,
    ) from last
