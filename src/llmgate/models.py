"""Domain types and the model catalogue.

The catalogue is the uncomfortable part of a library like this: context windows and prices
change, and a hard-coded table is wrong the moment a provider updates it. Three things make
that survivable rather than a trap:

* Every entry carries `as_of`, so a stale number is visibly stale rather than silently wrong.
* `register()` lets a caller override or add a model without waiting for a release, which is
  the real escape hatch.
* Nothing in this package *requires* the catalogue to be right about price. A missing model
  raises `UnknownModelError` instead of guessing — a budget enforced with an invented price
  is worse than no budget, because it reports confidence it has not earned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"Unknown role {self.role!r}")


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts and cost for one call.

    `estimated` records whether `prompt_tokens` came from the provider's response or from
    this package's own estimate. Mixing the two silently would make cost reporting
    untraceable, and the estimate is the one that can be wrong.
    """

    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            # Any estimated component makes the total estimated.
            estimated=self.estimated or other.estimated,
        )


ZERO_USAGE = Usage(prompt_tokens=0, completion_tokens=0, cost_usd=0.0)


@dataclass(frozen=True, slots=True)
class Completion:
    """One model response."""

    text: str
    model: str
    usage: Usage
    finish_reason: str
    attempts: int = 1

    @property
    def truncated(self) -> bool:
        """Whether the model stopped because it hit the output cap.

        Worth a named property: `finish_reason == "length"` is the single most commonly
        ignored field in an LLM response, and a silently truncated answer is a correctness
        bug in whatever consumes it. The spelling differs by provider, so this normalises it.
        """
        return self.finish_reason in {"length", "max_tokens", "MAX_TOKENS"}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """What this package needs to know about a model before calling it."""

    name: str
    context_tokens: int
    """Total window: prompt plus completion. The constraint that makes overflow predictable."""
    max_output_tokens: int
    input_usd_per_million: float
    output_usd_per_million: float
    as_of: str
    """Date these figures were recorded. Staleness you can see beats staleness you cannot."""

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_usd_per_million / 1_000_000
            + completion_tokens * self.output_usd_per_million / 1_000_000
        )


# Recorded 2026-11-23. Verify against your provider's current pricing page before relying on
# these for billing; use `register()` to correct one without waiting for a release.
_CATALOGUE: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        ModelSpec("gpt-4o-mini", 128_000, 16_384, 0.15, 0.60, "2026-11-23"),
        ModelSpec("gpt-4o", 128_000, 16_384, 2.50, 10.00, "2026-11-23"),
        ModelSpec("claude-haiku-4-5", 200_000, 64_000, 1.00, 5.00, "2026-11-23"),
        ModelSpec("claude-sonnet-4-5", 200_000, 64_000, 3.00, 15.00, "2026-11-23"),
        # A deliberately tiny entry, used by the tests to exercise overflow and trimming
        # without constructing a 128k-token fixture.
        ModelSpec("test-tiny", 1_000, 256, 1.00, 2.00, "2026-11-23"),
    ]
}


def register(spec: ModelSpec, *, overwrite: bool = True) -> None:
    """Add or replace a catalogue entry."""
    if not overwrite and spec.name in _CATALOGUE:
        raise ValueError(f"{spec.name} is already registered")
    _CATALOGUE[spec.name] = spec


def lookup(name: str) -> ModelSpec:
    """Catalogue entry for `name`, or raise.

    Deliberately not falling back to a default: a wrong context window produces requests that
    fail at the provider, and a wrong price produces a budget that does not bind. Both are
    worse than being told the model is unknown.
    """
    from llmgate.errors import UnknownModelError

    try:
        return _CATALOGUE[name]
    except KeyError:
        raise UnknownModelError(
            f"No catalogue entry for {name!r}. Known: {', '.join(sorted(_CATALOGUE))}. "
            f"Use llmgate.models.register(ModelSpec(...)) to add it."
        ) from None


def known_models() -> list[ModelSpec]:
    return sorted(_CATALOGUE.values(), key=lambda s: s.name)


@dataclass
class Budget:
    """A spend ceiling for a session, enforced before each call rather than after.

    After-the-fact accounting tells you what you spent. Checking the *projected* worst-case
    cost of a request before sending it is what actually stops an overrun, which is why
    `Client` consults this ahead of every call.
    """

    limit_usd: float | None = None
    spent_usd: float = 0.0
    calls: int = 0
    max_calls: int | None = None
    usage: Usage = field(default=ZERO_USAGE)

    @property
    def remaining_usd(self) -> float | None:
        return None if self.limit_usd is None else max(0.0, self.limit_usd - self.spent_usd)

    def record(self, usage: Usage) -> None:
        self.spent_usd += usage.cost_usd
        self.calls += 1
        self.usage = self.usage + usage
