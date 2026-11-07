"""Output guards: deterministic checks on what the model produced.

A citation must name a real knowledge document; a long factual answer with
no grounding at all is flagged as a hallucination risk; and a refusal-shaped
answer is recognised so it counts as a content gap rather than a hallucination.
Streaming means these flag after the fact; they never block.
"""

# Answers longer than this without a single citation get flagged for review.
UNCITED_ANSWER_MIN_CHARS = 200

# An answer that clearly declines or redirects instead of asserting facts.
REFUSAL_MARKERS = (
    "i don't",
    "don't have",
    "doesn't say",
    "doesn't mention",
    "no mention",
    "not mentioned",
    "couldn't find",
    "could not find",
    "don't see",
    "not in the knowledge",
    "no information",
    "not something i",
    "nothing in",
    "no record",
    "not listed",
    "not stated",
    "not specified",
    "doesn't state",
    "doesn't specify",
    "isn't covered",
    "not covered",
    "can't help",
    "cannot help",
    "can only help",
    "i can't",
    "contact ruud",
    "contact him",
    "reach out to ruud",
    "ask ruud directly",
    "just here to talk about ruud",
)


def looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def valid_citation(title: str | None, known_titles: set[str]) -> bool:
    """A citation is only forwarded if it names a real knowledge document."""
    return title is not None and title in known_titles


def uncited_answer(answer: str, *, citations: list[str], tool_rounds: int) -> bool:
    """Groundedness check: a substantive answer that cites nothing, used no
    tools (tool results are grounding too) and is not a refusal."""
    return (
        len(answer) >= UNCITED_ANSWER_MIN_CHARS
        and not citations
        and not tool_rounds
        and not looks_like_refusal(answer)
    )
