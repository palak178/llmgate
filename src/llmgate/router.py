"""Model selection: the cheapest candidate that actually fits.

Routing here is deterministic and explainable, not learned and not model-driven. Given a
prompt, a reserved output size and a remaining budget, there is exactly one defensible
answer — the cheapest model whose context window holds the prompt and whose worst-case cost
is affordable — and a function can compute it.

"Ask a small model whether the question is hard" is the fashionable alternative. It adds a
call, adds latency, and makes the routing decision itself unauditable; when a bill is wrong
you cannot reconstruct why a request went to the expensive model. This can be reconstructed
from its inputs, and `Decision` records the reasoning either way.
"""

from __future__ import annotations

from dataclasses import dataclass

from llmgate.errors import ContextOverflowError
from llmgate.models import Message, ModelSpec, lookup
from llmgate.tokens import count_messages


@dataclass(frozen=True, slots=True)
class Decision:
    """A routing outcome, with the numbers behind it."""

    model: str
    prompt_tokens: int
    exact: bool
    projected_usd: float
    considered: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]
    """`(model, reason)` for each candidate passed over, so the choice can be explained."""

    def explain(self) -> str:
        lines = [
            f"chose {self.model} — {self.prompt_tokens} prompt tokens "
            f"({'exact' if self.exact else 'estimated'}), worst case ${self.projected_usd:.4f}"
        ]
        lines.extend(f"  rejected {name}: {reason}" for name, reason in self.rejected)
        return "\n".join(lines)


def choose(
    messages: list[Message],
    *,
    candidates: list[str],
    reserve_output_tokens: int = 1024,
    remaining_usd: float | None = None,
) -> Decision:
    """Pick the cheapest candidate that fits the prompt and the budget.

    Candidates are evaluated by worst-case cost, so the ordering does not depend on how long
    the model's reply happens to be. Ties break on the order given, which lets a caller
    express a preference between equally priced models.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")

    specs = [lookup(name) for name in candidates]
    rejected: list[tuple[str, str]] = []
    best: tuple[float, int, ModelSpec, int, bool] | None = None

    for position, spec in enumerate(specs):
        reserve = min(reserve_output_tokens, spec.max_output_tokens)
        prompt_tokens, exact = count_messages(messages, model=spec.name)

        if prompt_tokens + reserve > spec.context_tokens:
            rejected.append(
                (
                    spec.name,
                    f"needs {prompt_tokens + reserve} tokens, window is {spec.context_tokens}",
                )
            )
            continue

        projected = spec.cost(prompt_tokens, reserve)
        if remaining_usd is not None and projected > remaining_usd:
            rejected.append(
                (spec.name, f"worst case ${projected:.4f} exceeds ${remaining_usd:.4f} remaining")
            )
            continue

        if best is None or (projected, position) < (best[0], best[1]):
            best = (projected, position, spec, prompt_tokens, exact)

    if best is None:
        prompt_tokens, _ = count_messages(messages, model=specs[0].name)
        raise ContextOverflowError(
            "No candidate model can serve this request:\n"
            + "\n".join(f"  {name}: {reason}" for name, reason in rejected),
            required=prompt_tokens + reserve_output_tokens,
            available=max(s.context_tokens for s in specs),
        )

    projected, _, spec, prompt_tokens, exact = best
    return Decision(
        model=spec.name,
        prompt_tokens=prompt_tokens,
        exact=exact,
        projected_usd=projected,
        considered=tuple(candidates),
        rejected=tuple(rejected),
    )
