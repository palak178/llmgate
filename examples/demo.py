#!/usr/bin/env python3
"""The three refusals and one routing decision, with no API key and no network.

    python examples/demo.py

Everything below runs against `FakeProvider`, so the output is the library's own behaviour
rather than a model's. Swap in `OpenAIProvider(api_key=...)` and nothing else changes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llmgate import (
    Budget,
    Client,
    Conversation,
    FakeProvider,
    Message,
    RetryPolicy,
    choose,
)
from llmgate.errors import (
    BudgetExceededError,
    ContextOverflowError,
    ProviderError,
)
from llmgate.retry import RetryableError


def rule(title: str) -> None:
    print(f"\n{title}\n{'─' * len(title)}")


def main() -> int:
    # Line-buffer stdout so the library's own warnings (on stderr, unbuffered) interleave in
    # the right place rather than all appearing before the first heading.
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.WARNING, format="  [llmgate] %(message)s")

    rule("1. A normal call")
    client = Client(
        provider=FakeProvider(replies=["Thirty days' written notice."]),
        model="gpt-4o-mini",
        budget=Budget(limit_usd=5.00),
        max_output_tokens=256,
    )
    reply = client.complete([Message("user", "How much notice must I give?")])
    print(f"  {reply.text}")
    print(
        f"  {reply.usage.total_tokens} tokens · ${reply.usage.cost_usd:.6f} · "
        f"{'estimated' if reply.usage.estimated else 'reported by provider'}"
    )

    rule("2. Refused: the prompt cannot fit, and trimming cannot help")
    tiny = Client(provider=FakeProvider(), model="test-tiny", max_output_tokens=100)
    try:
        tiny.complete([Message("user", "word " * 5000)])
    except ContextOverflowError as exc:
        print(f"  {exc}")
        print(f"  needed {exc.required} tokens, {exc.available} available — nothing was sent")

    rule("3. Refused: the worst case would breach the budget")
    broke = Client(
        provider=FakeProvider(),
        model="gpt-4o",
        budget=Budget(limit_usd=0.01),
        max_output_tokens=16_000,
    )
    try:
        broke.complete([Message("user", "Summarise the entire handbook.")])
    except BudgetExceededError as exc:
        print(f"  {exc}")
        print(f"  no request was made, so nothing was spent (${broke.budget.spent_usd:.4f})")

    rule("4. Retries a rate limit; refuses to retry a bad request")
    flaky = Client(
        provider=FakeProvider(replies=[RetryableError("429 rate limited"), "Recovered."]),
        model="gpt-4o-mini",
        retry=RetryPolicy(max_attempts=3, base_delay=0.0),
        sleep=lambda _: None,
    )
    recovered = flaky.complete([Message("user", "Hello?")])
    print(f"  transient:  {recovered.text!r} after {recovered.attempts} attempt(s)")

    permanent = Client(
        provider=FakeProvider(replies=[ProviderError("400 invalid parameter")]),
        model="gpt-4o-mini",
        retry=RetryPolicy(max_attempts=5, base_delay=0.0),
        sleep=lambda _: None,
    )
    try:
        permanent.complete([Message("user", "Hello?")])
    except ProviderError as exc:
        calls = len(permanent.provider.calls)  # type: ignore[attr-defined]
        print(f"  permanent:  {exc} — tried {calls} time(s), not {5}")

    rule("5. Routing: the cheapest model that actually fits")
    long_prompt = [Message("user", "context " * 8000)]
    decision = choose(
        long_prompt,
        candidates=["test-tiny", "gpt-4o-mini", "gpt-4o"],
        reserve_output_tokens=1024,
    )
    print("  " + decision.explain().replace("\n", "\n  "))

    rule("6. A conversation keeps its history; only the request is trimmed")
    convo = Conversation(
        client=Client(
            provider=FakeProvider(replies=["Noted."]),
            model="test-tiny",
            max_output_tokens=100,
        ),
        system="Answer in one sentence.",
    )
    for i in range(12):
        convo.add("user", f"Fact {i}: " + "detail " * 60)
    convo.ask("What was fact 0?")
    print(f"  history retained: {len(convo.messages)} messages")
    print("  the oversized turns were dropped from the request, not from the history")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
