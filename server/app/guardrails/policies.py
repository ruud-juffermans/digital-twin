"""What the visitor reads when a guard trips. Detection lives in input.py and
output.py; the copy lives here so a deployment can change the wording, or
what happens on a trip, without touching a detector."""

INPUT_REFUSAL = (
    "I can only help with questions about Ruud and his work — happy to answer anything there."
)


PII_REFUSAL = (
    "Please don't share card or account numbers here — I only need a way for "
    "Ruud to reach you, like an email address."
)


OFFTOPIC_REDIRECT = (
    "I'm just here to talk about Ruud and his work — I'll leave the code and "
    "essays to him. Anything you'd like to know about his experience or projects?"
)


BUDGET_REFUSAL = (
    "I've reached my usage budget for today — please come back tomorrow, or "
    "contact Ruud directly via the site."
)
