"""The client, and the order in which it refuses things.

    lookup model  →  fit context  →  check budget  →  send (with retries)  →  record usage

The order is the design. Both checks happen **before** the request leaves the process,
because both failures are predictable from the request itself:

* A prompt that exceeds the context window will be rejected by the provider every time. There
  is no reason to pay a round trip to learn that.
* A call that would breach the budget cannot be un-spent after the fact. Accounting that only
  runs afterwards reports overruns; it does not prevent them.

The budget check uses the **worst case** — the full reserved output, not an expected value —
because a ceiling that holds only on average is not a ceiling.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from llmgate.context import fit
from llmgate.errors import BudgetExceededError
from llmgate.models import Budget, Completion, Message, Role, lookup
from llmgate.providers import Provider, cost_for
from llmgate.retry import RetryPolicy, with_retries

log = logging.getLogger("llmgate")


@dataclass
class Client:
    """A provider-agnostic client with context fitting, budget enforcement and retries."""

    provider: Provider
    model: str
    max_output_tokens: int = 1024
    temperature: float = 0.0
    budget: Budget = field(default_factory=Budget)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    sleep: Callable[[float], None] = time.sleep
    """Injectable so retry behaviour is testable without real delays."""

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Completion:
        """Send `messages` and return the completion, or refuse before sending."""
        if not messages:
            raise ValueError("messages must not be empty")

        name = model or self.model
        spec = lookup(name)
        reserve = min(max_output_tokens or self.max_output_tokens, spec.max_output_tokens)

        # 1. Fit. Raises ContextOverflowError when trimming cannot help.
        fitted = fit(messages, spec=spec, reserve_output_tokens=reserve)
        if fitted.trimmed:
            log.warning(
                "trimmed %d message(s) to fit %s (%d prompt tokens, %s)",
                fitted.dropped,
                name,
                fitted.prompt_tokens,
                "exact" if fitted.exact else "estimated",
            )

        # 2. Budget, on the worst case.
        self._check_budget(spec.cost(fitted.prompt_tokens, reserve), calls=1)

        # 3. Send.
        raw, attempts = with_retries(
            lambda: self.provider.complete(
                fitted.messages,
                model=name,
                max_output_tokens=reserve,
                temperature=self.temperature if temperature is None else temperature,
            ),
            policy=self.retry,
            sleep=self.sleep,
            description=f"{self.provider.name} {name}",
        )

        # 4. Record what it actually cost.
        usage = cost_for(spec, raw, estimated_prompt=fitted.prompt_tokens)
        self.budget.record(usage)
        completion = Completion(
            text=raw.text,
            model=name,
            usage=usage,
            finish_reason=raw.finish_reason,
            attempts=attempts,
        )
        if completion.truncated:
            # Surfaced rather than swallowed: a truncated answer is a correctness problem for
            # whatever parses it, and `finish_reason` is the most-ignored field in the response.
            log.warning(
                "%s stopped at the %d-token output cap; the reply is incomplete",
                name,
                reserve,
            )
        return completion

    def _check_budget(self, projected_usd: float, *, calls: int) -> None:
        if self.budget.max_calls is not None and self.budget.calls + calls > self.budget.max_calls:
            raise BudgetExceededError(
                f"This call would be number {self.budget.calls + calls} of a "
                f"{self.budget.max_calls}-call limit.",
                projected_usd=projected_usd,
                remaining_usd=self.budget.remaining_usd or 0.0,
            )
        remaining = self.budget.remaining_usd
        if remaining is not None and projected_usd > remaining:
            raise BudgetExceededError(
                f"Worst-case cost of this call is ${projected_usd:.4f} but only "
                f"${remaining:.4f} of the ${self.budget.limit_usd:.2f} budget remains. "
                f"Lower max_output_tokens, choose a cheaper model, or raise the budget.",
                projected_usd=projected_usd,
                remaining_usd=remaining,
            )


@dataclass
class Conversation:
    """A client plus message history.

    History is kept in full here and trimmed per request by `Client`, rather than being
    truncated destructively as it grows. The distinction matters: trimming for one call must
    not permanently discard context that a larger model, or a shorter next question, could
    still have used. Trimming the stored history instead would make one oversized message
    permanently destructive.
    """

    client: Client
    system: str | None = None
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.system and not any(m.role == "system" for m in self.messages):
            self.messages.insert(0, Message("system", self.system))

    def ask(self, content: str, **kwargs) -> Completion:
        """Append a user turn, get a reply, append it, return it."""
        self.messages.append(Message("user", content))
        completion = self.client.complete(self.messages, **kwargs)
        self.messages.append(Message("assistant", completion.text))
        return completion

    def add(self, role: Role, content: str) -> None:
        self.messages.append(Message(role, content))

    def reset(self, *, keep_system: bool = True) -> None:
        self.messages = [m for m in self.messages if keep_system and m.role == "system"]

    @property
    def spent_usd(self) -> float:
        return self.client.budget.spent_usd
