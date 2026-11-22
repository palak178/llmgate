"""Providers.

`Provider` is a `Protocol`, so no provider SDK is a dependency of this package and the whole
test suite runs against `FakeProvider` with no network and no key.

Each adapter has exactly one interesting job: translating that vendor's failures into
`RetryableError` or `ProviderError`. Everything the library promises — refusing an
overflowing prompt, enforcing a budget, backing off correctly — depends on that
classification being right, so it is done explicitly per provider rather than by pattern
matching on error strings in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from llmgate.errors import ProviderError, RetryableError
from llmgate.models import Message, ModelSpec, Usage


@dataclass(frozen=True, slots=True)
class RawCompletion:
    """What a provider returns, before cost is applied.

    `prompt_tokens`/`completion_tokens` are `None` when the provider did not report them, so
    the client can fall back to its estimate and mark the usage `estimated` rather than
    recording a confident zero.
    """

    text: str
    finish_reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@runtime_checkable
class Provider(Protocol):
    @property
    def name(self) -> str: ...

    def complete(
        self, messages: list[Message], *, model: str, max_output_tokens: int, temperature: float
    ) -> RawCompletion: ...


# ---------------------------------------------------------------------------------------


@dataclass
class FakeProvider:
    """Scriptable provider for tests and local development.

    Not a mock of a specific SDK — it is a real implementation of `Provider` whose responses
    happen to be supplied up front. That keeps the tests asserting on this library's
    behaviour rather than on how a mock was configured.
    """

    replies: list[str | Exception] = field(default_factory=lambda: ["ok"])
    finish_reason: str = "stop"
    report_usage: bool = True
    calls: list[dict] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "fake"

    def complete(
        self, messages: list[Message], *, model: str, max_output_tokens: int, temperature: float
    ) -> RawCompletion:
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            }
        )
        # The last reply repeats, so a retry test can supply one failure and then exercise
        # "still failing" without listing it once per attempt.
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        reply = self.replies[index]
        if isinstance(reply, Exception):
            raise reply
        if not self.report_usage:
            return RawCompletion(text=reply, finish_reason=self.finish_reason)
        # Two details that make this behave like a real provider rather than a convenience:
        #
        # - A non-empty prompt never reports zero tokens, so cost is never spuriously free.
        # - Completion tokens are capped at `max_output_tokens`, because a provider stops at
        #   the cap. Without that, the fake could "spend" more than the client reserved and
        #   break the invariant that actual cost never exceeds the projected worst case.
        prompt_tokens = max(1, sum(len(m.content) for m in messages) // 4)
        completion_tokens = min(max(1, len(reply) // 4), max_output_tokens)
        return RawCompletion(
            text=reply,
            finish_reason=self.finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class OpenAIProvider:
    """OpenAI chat completions. Needs the `openai` extra.

    Note for anyone porting old code: this uses `openai>=1.0`. The module-level
    `openai.ChatCompletion.create` and `openai.error.*` that the previous version of this
    repository was built on were removed in 1.0, which is why none of it ran.
    """

    def __init__(self, api_key: str | None = None, *, base_url: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError("openai is not installed. `pip install 'llmgate[openai]'`") from exc
        if not api_key:
            raise ProviderError("An OpenAI API key is required")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return "openai"

    def complete(
        self, messages: list[Message], *, model: str, max_output_tokens: int, temperature: float
    ) -> RawCompletion:
        import openai

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                max_tokens=max_output_tokens,
                temperature=temperature,
            )
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise RetryableError(str(exc), retry_after=_retry_after(exc)) from exc
        except openai.APIStatusError as exc:
            # 5xx is the server's problem and may clear; 4xx is the request's and will not.
            if exc.status_code >= 500:
                raise RetryableError(str(exc), retry_after=_retry_after(exc)) from exc
            raise ProviderError(f"OpenAI rejected the request ({exc.status_code}): {exc}") from exc
        except openai.OpenAIError as exc:
            raise ProviderError(str(exc)) from exc

        choice = response.choices[0]
        usage = response.usage
        return RawCompletion(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason or "stop",
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )


class AnthropicProvider:
    """Anthropic messages API. Needs the `anthropic` extra."""

    def __init__(self, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                "anthropic is not installed. `pip install 'llmgate[anthropic]'`"
            ) from exc
        if not api_key:
            raise ProviderError("An Anthropic API key is required")
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    def complete(
        self, messages: list[Message], *, model: str, max_output_tokens: int, temperature: float
    ) -> RawCompletion:
        import anthropic

        # Anthropic takes the system prompt as a separate parameter rather than a message.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]

        try:
            message = self._client.messages.create(
                model=model,
                system=system or anthropic.NOT_GIVEN,
                messages=turns,
                max_tokens=max_output_tokens,
                temperature=temperature,
            )
        except (
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
        ) as exc:
            raise RetryableError(str(exc), retry_after=_retry_after(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise RetryableError(str(exc), retry_after=_retry_after(exc)) from exc
            raise ProviderError(
                f"Anthropic rejected the request ({exc.status_code}): {exc}"
            ) from exc
        except anthropic.AnthropicError as exc:
            raise ProviderError(str(exc)) from exc

        return RawCompletion(
            text="".join(b.text for b in message.content if b.type == "text"),
            finish_reason=message.stop_reason or "stop",
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
        )


def _retry_after(exc: object) -> float | None:
    """Parse a `Retry-After` header off a provider exception, if it carries one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        # The header also allows an HTTP date, which is not worth parsing here: falling back
        # to computed backoff is correct and simple.
        return None


def cost_for(spec: ModelSpec, raw: RawCompletion, *, estimated_prompt: int) -> Usage:
    """Build a `Usage` from a provider response, falling back to an estimate."""
    reported = raw.prompt_tokens is not None and raw.completion_tokens is not None
    prompt = raw.prompt_tokens if raw.prompt_tokens is not None else estimated_prompt
    completion = raw.completion_tokens
    if completion is None:
        from llmgate.tokens import count_text

        completion, _ = count_text(raw.text, model=spec.name)
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost_usd=spec.cost(prompt, completion),
        estimated=not reported,
    )
