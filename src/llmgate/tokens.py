"""Token counting, and being honest about when it is a guess.

Two backends. `tiktoken` when it is installed and the model is one it knows; otherwise a
character heuristic. The distinction is carried through the API — `count()` returns
`(tokens, exact)` rather than a bare int — because every downstream decision about fitting a
context window depends on whether the number can be trusted.

The heuristic is deliberately **pessimistic**: it rounds up. An underestimate produces a
request the provider rejects, which is the failure this library exists to prevent; an
overestimate only trims slightly more history than strictly necessary.
"""

from __future__ import annotations

import math
from functools import lru_cache

from llmgate.models import Message

CHARS_PER_TOKEN = 3.6
"""Conservative characters-per-token ratio for the heuristic.

English prose averages roughly 4 characters per token for modern BPE vocabularies. Using 3.6
rounds the token count *up* by about 10%, which biases towards refusing or trimming rather
than towards a request that overflows.

Code, JSON and non-Latin scripts tokenise far less efficiently — closer to 2 — so the
heuristic is still an underestimate for those. That is the main reason `exact` is surfaced:
if you are sending mostly code and have no tiktoken, you should know the number is soft.
"""

MESSAGE_OVERHEAD_TOKENS = 4
"""Per-message framing cost (role markers and separators).

Chat APIs wrap each message in a few tokens of structure. Ignoring it makes a long
conversation of short messages overflow unexpectedly, because the error is per message rather
than per character.
"""

REPLY_PRIMER_TOKENS = 3
"""Tokens the API adds once, to prime the assistant turn."""


@lru_cache(maxsize=32)
def _encoding(model: str):
    """A tiktoken encoding for `model`, or None if unavailable.

    Cached because `encoding_for_model` is slow enough to matter in a loop, and returns None
    rather than raising so the caller can fall back without a try/except at each site.
    """
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # An unrecognised model name. `cl100k_base` is the right guess for most current
        # OpenAI models, but guessing an encoding is still a guess, so this reports inexact.
        return None


def count_text(text: str, *, model: str = "") -> tuple[int, bool]:
    """`(tokens, exact)` for a string."""
    encoding = _encoding(model) if model else None
    if encoding is not None:
        return len(encoding.encode(text)), True
    return math.ceil(len(text) / CHARS_PER_TOKEN), False


def count_messages(messages: list[Message], *, model: str = "") -> tuple[int, bool]:
    """`(tokens, exact)` for a full chat request, including framing overhead."""
    total = REPLY_PRIMER_TOKENS
    exact = True
    for message in messages:
        tokens, message_exact = count_text(message.content, model=model)
        total += tokens + MESSAGE_OVERHEAD_TOKENS
        exact = exact and message_exact
    return total, exact


def tiktoken_available() -> bool:
    """Whether exact counting is possible at all. Used by diagnostics and the CLI."""
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        return False
    return True
