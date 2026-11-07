"""Deterministic scoring for the eval datasets.

Pure functions — no model, no I/O — so the harness itself is unit-testable.
"""

from typing import Any

# The refusal heuristic is shared with the agent's own groundedness check so
# evals and runtime agree on what counts as an honest decline.
from app.guardrails.output import looks_like_refusal

__all__ = ["looks_like_refusal", "score_completion", "score_tool_selection"]


def score_tool_selection(
    expected: list[str], actual: list[str], allowed_extra: list[str]
) -> tuple[bool, dict[str, Any]]:
    """Every expected tool called; no disallowed extras (high-risk especially)."""
    expected_set, actual_set = set(expected), set(actual)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set - set(allowed_extra))
    return not missing and not extra, {"missing": missing, "extra": extra}


def score_completion(
    case: dict[str, Any], text: str, citation_count: int
) -> tuple[bool, list[str]]:
    """Check the final answer against the case's declared expectations."""
    failures: list[str] = []
    lowered = text.lower()

    needles = case.get("must_contain_any") or []
    if needles and not any(needle.lower() in lowered for needle in needles):
        failures.append(f"missing all of {needles}")

    for needle in case.get("must_not_contain") or []:
        if needle.lower() in lowered:
            failures.append(f"contains forbidden {needle!r}")

    if case.get("must_cite") and citation_count == 0:
        failures.append("no citations")

    if case.get("expect_refusal") and not looks_like_refusal(text):
        failures.append("expected a refusal, got an answer")

    return not failures, failures
