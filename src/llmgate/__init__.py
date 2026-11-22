"""llmgate — an LLM client that refuses to send a request it can prove will fail.

Context overflow and budget overrun are both predictable from the request itself, so both are
checked before anything leaves the process:

    from llmgate import Budget, Client, FakeProvider, Message

    client = Client(
        provider=FakeProvider(replies=["Thirty days."]),
        model="gpt-4o-mini",
        budget=Budget(limit_usd=5.00),
    )
    reply = client.complete([Message("user", "How much notice is required?")])
    print(reply.text, reply.usage.cost_usd)

Swap `FakeProvider` for `OpenAIProvider(api_key=...)` or `AnthropicProvider(api_key=...)`;
nothing else changes, because both satisfy the same `Provider` protocol.
"""

from __future__ import annotations

from llmgate.client import Client, Conversation
from llmgate.context import FitResult, fit
from llmgate.errors import (
    BudgetExceededError,
    ContextOverflowError,
    LLMGateError,
    ProviderError,
    RefusedError,
    RetriesExhaustedError,
    RetryableError,
    UnknownModelError,
)
from llmgate.models import (
    Budget,
    Completion,
    Message,
    ModelSpec,
    Usage,
    known_models,
    lookup,
    register,
)
from llmgate.providers import (
    AnthropicProvider,
    FakeProvider,
    OpenAIProvider,
    Provider,
    RawCompletion,
)
from llmgate.retry import RetryPolicy, with_retries
from llmgate.router import Decision, choose

__version__ = "1.0.0"

__all__ = [
    "AnthropicProvider",
    "Budget",
    "BudgetExceededError",
    "Client",
    "Completion",
    "ContextOverflowError",
    "Conversation",
    "Decision",
    "FakeProvider",
    "FitResult",
    "LLMGateError",
    "Message",
    "ModelSpec",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "RawCompletion",
    "RefusedError",
    "RetriesExhaustedError",
    "RetryPolicy",
    "RetryableError",
    "UnknownModelError",
    "Usage",
    "__version__",
    "choose",
    "fit",
    "known_models",
    "lookup",
    "register",
    "with_retries",
]
