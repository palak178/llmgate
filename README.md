# llmgate

**A provider-agnostic LLM client that refuses to send a request it can prove will fail.**

[![CI](https://github.com/palak178/llm-wrapper/actions/workflows/ci.yml/badge.svg)](https://github.com/palak178/llm-wrapper/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Zero dependencies](https://img.shields.io/badge/dependencies-0-success)
![License MIT](https://img.shields.io/badge/license-MIT-green)

Two of the ways an LLM call fails are knowable *before* you make it:

- **The prompt does not fit the context window.** The provider will reject it, every time. You
  do not need a round trip to find out.
- **The call would blow the budget.** Money cannot be un-spent. Accounting that runs
  afterwards reports overruns; it does not prevent them.

llmgate checks both before anything leaves the process, and classifies the failures that are
only knowable afterwards into *retry this* and *do not retry this* — because retrying a
malformed request is just a slower error.

```python
from llmgate import Budget, Client, Message, OpenAIProvider

client = Client(
    provider=OpenAIProvider(api_key=...),
    model="gpt-4o-mini",
    budget=Budget(limit_usd=5.00),
)

reply = client.complete([Message("user", "How much notice must I give?")])
print(reply.text, reply.usage.cost_usd, reply.truncated)
```

**No runtime dependencies.** Providers are reached through a `Protocol`, so the package itself
imports nothing — and CI proves it by installing with no extras and running the full happy
path. A library whose job is to stop you overspending should not cost you a dependency tree.

---

## See it refuse

```bash
git clone https://github.com/palak178/llm-wrapper.git llmgate && cd llmgate
pip install -e ".[dev]"
make demo          # no API key, no network
```

```console
2. Refused: the prompt cannot fit, and trimming cannot help
───────────────────────────────────────────────────────────
  The system prompt and latest message need 6952 tokens but only 900 are available in
  test-tiny after reserving 100 for the reply. Shorten the input, or use a model with a
  larger window.
  needed 6952 tokens, 900 available — nothing was sent

3. Refused: the worst case would breach the budget
──────────────────────────────────────────────────
  Worst-case cost of this call is $0.1600 but only $0.0100 of the $0.01 budget remains.
  Lower max_output_tokens, choose a cheaper model, or raise the budget.
  no request was made, so nothing was spent ($0.0000)

4. Retries a rate limit; refuses to retry a bad request
───────────────────────────────────────────────────────
  transient:  'Recovered.' after 2 attempt(s)
  permanent:  400 invalid parameter — tried 1 time(s), not 5
```

---

## What it does

### Budgets are enforced on the worst case

```python
budget = Budget(limit_usd=0.01)
client = Client(provider=p, model="gpt-4o", budget=budget, max_output_tokens=16_000)
client.complete([...])          # BudgetExceededError — and nothing was sent
assert budget.spent_usd == 0.0
```

The check uses the **full reserved output**, not an expected value, because a ceiling that
holds on average is not a ceiling. Since actual output can never exceed the reservation, total
spend can never exceed the limit — and there is a test asserting exactly that invariant across
a run of calls.

`Budget` also takes `max_calls`, for the runaway-loop case where each call is individually
cheap.

### Context fitting that protects the right messages

Trimming oldest-first is the obvious implementation and it is wrong. The rules here, in
priority order:

| Rule | Why |
|---|---|
| System messages are never dropped | They carry the instructions that make the reply correct. Oldest-first treats the system prompt as the *most* expendable message |
| The latest user message is never dropped | Dropping it answers a different question than the one asked |
| Otherwise oldest first | Recent turns carry more of the state the reply depends on |
| Turn pairs drop together | An assistant reply with no question above it reads as a non-sequitur and degrades the next completion |
| If the protected messages alone do not fit, refuse | Trimming cannot help, so a clear error beats a provider rejection |

`Conversation` keeps full history and trims **per request**, so a single long message does not
permanently destroy context that a larger model or a shorter next question could still use.

### Retries that distinguish transient from permanent

```python
RetryPolicy(max_attempts=3, base_delay=0.5, max_delay=30.0, jitter=0.25)
```

Jittered exponential backoff. `RetryableError` (rate limit, timeout, 5xx) is retried;
everything else fails on the first attempt, and an unrecognised error defaults to *not*
retryable — so a new provider error costs one failed call rather than hammering a service with
a request that cannot succeed. A provider's own `Retry-After` wins over computed backoff,
because the server is stating a requirement rather than offering a hint.

`sleep` is injectable, so the tests assert the delay *sequence* without ever waiting.

### Routing: the cheapest model that actually fits

```python
decision = choose(messages, candidates=["gpt-4o-mini", "gpt-4o"], remaining_usd=0.50)
print(decision.explain())
# chose gpt-4o-mini — 8008 prompt tokens (exact), worst case $0.0018
#   rejected test-tiny: needs 18041 tokens, window is 1000
```

Deterministic and explainable. The fashionable alternative — asking a small model whether the
question is hard — adds a call, adds latency, and makes the routing decision itself
unauditable: when a bill looks wrong you cannot reconstruct why a request went to the
expensive model. This is reconstructible from its inputs, and `Decision` records what it
rejected and why.

### Token counting that admits when it is guessing

`count_text()` returns `(tokens, exact)`, not a bare integer, because every downstream
decision depends on whether the number can be trusted. With the `tokens` extra installed,
`tiktoken` gives exact counts. Without it, a character heuristic is used at **3.6 chars/token**
— deliberately pessimistic, so it rounds *up*: an underestimate produces the rejected request
this library exists to prevent, while an overestimate only trims slightly more than necessary.

Per-message framing overhead is counted too. Ignoring it makes a long conversation of short
messages overflow unexpectedly, because the error is per message rather than per character.

> **Honest limit:** the heuristic is tuned for English prose. Code, JSON and non-Latin scripts
> tokenise closer to 2 chars/token, so for those the heuristic still *under*-counts. That is
> the main reason `exact` is surfaced rather than hidden.

---

## The model catalogue

Context windows and prices live in `models.py`. A hard-coded table is wrong the moment a
provider updates it, so three things make that survivable:

- Every entry carries **`as_of`** — staleness you can see beats staleness you cannot.
- **`register()`** overrides or adds a model without waiting for a release.
- An unknown model **raises** rather than falling back to a default. A budget enforced with an
  invented price is worse than no budget, because it reports confidence it has not earned.

```python
from llmgate import ModelSpec, register
register(ModelSpec("my-local-llama", 8192, 2048, 0.0, 0.0, "2026-11-23"))
```

Figures were recorded **2026-11-23**. Verify against your provider's pricing page before
relying on them for billing.

---

## Providers

| Provider | Install | Notes |
|---|---|---|
| `FakeProvider` | built in | Scriptable. Not a mock of an SDK — a real `Provider` whose replies are supplied up front |
| `OpenAIProvider` | `pip install 'llmgate[openai]'` | `openai>=1.0` |
| `AnthropicProvider` | `pip install 'llmgate[anthropic]'` | Handles the separate `system` parameter |

Each adapter has exactly one interesting job: translating that vendor's failures into
`RetryableError` or `ProviderError`. Everything the library promises depends on that
classification, so it is written out per provider rather than pattern-matched on error strings
in one shared place.

Writing your own is five lines — satisfy the `Provider` protocol and nothing else changes.

---

## What is deliberately not built

| Not built | Why |
|---|---|
| **Streaming** | A budget check on the worst case works fine with streaming, but token accounting mid-stream is fiddly and the reservation already bounds cost. Worth adding; not pretended at |
| **Async** | The sync path is the whole library. An async twin would duplicate every module, and the useful version is `asyncio.to_thread` around `complete()` until there is a reason for more |
| **Tool / function calling** | `Message` has three roles and no `tool` role. Adding it properly means tool schemas and result messages, which is a larger surface than this library claims |
| **Caching** | Prompt caching is provider-specific and changes the cost model. Doing it wrong would make the budget numbers silently wrong, which is worse than not doing it |
| **Fallback across providers** | `choose()` picks a model; it does not fail over mid-request. Retrying a different provider after an error needs idempotency guarantees this does not have |
| **A proxy server** | This is a library. If you want one gateway for many services, LiteLLM already does that well |

---

## Testing

```bash
make test     # 88 tests, offline, ~0.2s
make lint
```

Every test runs against `FakeProvider`: no network, no key, no recorded cassettes. CI also
installs with **no extras** and runs the happy path, so the zero-dependency claim is tested
rather than asserted — and runs both `pytest` and `python -m pytest`, which resolve `sys.path`
differently.

The suite is organised around the specific mistakes in [the previous
version](#history): `test_context.py` has a class per protection rule, and `test_retry.py`
opens with the case that proves a non-retryable error is attempted exactly once.

---

## History

This repository held a single 315-line script from 2023. It was rewritten in November 2026,
and not for style — it could not run, and each of its three advertised features was broken:

- **Lines 300–314 executed two live API calls at import time.** `import LLM_wrapper` fired
  network requests and printed to stdout. Disqualifying for anything called a wrapper.
- **"Rate-limit retry handling"** retried `InvalidRequestError`, which can never succeed, and
  slept a flat 60 seconds. The branch meant to handle it was unreachable — nested inside the
  check it was meant to follow.
- **"Conversation history management"** mixed `ChatMessage` objects and raw dicts in one list,
  so the next trim raised `AttributeError` on `msg.content`. It also mutated history in place,
  so one long message silently erased the conversation.
- **"Automatic token-limit truncation"** keyed on `finish_reason == "incomplete"`, which is not
  an OpenAI value (it is `"length"`), so the splitting path was dead code.
- `prioritize_messages` sorted on `msg.timestamp`, a field `ChatMessage` does not have.
  `openai.ChatCompletionChoice` does not exist. `CompletionModelWrapper.set_model` wrote an
  attribute the request path never read, making the setter a silent no-op.
- It targeted `openai<1.0` (`openai.ChatCompletion`, `openai.error.*`) and
  `text-davinci-003`, both removed.

Nothing was salvageable, but the *intent* was right: token limits, history and retries are
genuinely the three things worth wrapping. This is that intent implemented so each piece has a
test that fails when it breaks.

## Licence

MIT — see [LICENSE](LICENSE).
