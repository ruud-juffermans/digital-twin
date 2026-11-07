"""Read-only: Ruud's availability as structured data the model quotes
exactly instead of paraphrasing prose."""

from .base import Tool

# Keep in sync with the knowledge modules.
AVAILABILITY = {
    "status": "open",
    "available_from": "now",
    "looking_for": [
        "fixed-scope data & AI consulting engagements",
        "data platforms, dashboards, AI prototypes, data quality monitoring",
    ],
    "location": "Amsterdam — working with organisations across the Netherlands",
    "contact": "via the contact form on the website",
    "note": "This assistant cannot make commitments; for anything contractual, contact Ruud.",
}


def get_availability() -> Tool:
    return Tool(
        name="get_availability",
        description=(
            "Get Ruud's current availability for roles and freelance work as "
            "structured data. Use for any question about hiring or availability."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        run=lambda: AVAILABILITY,
    )
