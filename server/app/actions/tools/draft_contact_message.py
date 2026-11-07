"""Write action: a message from the visitor to Ruud.

The only tool that changes anything, and even then only a draft: it records
a message, it does not send one. high_risk means the graph pauses for Ruud's
approval before it runs, and the approved draft is archived afterwards.
"""

from .base import Tool


def draft_contact_message() -> Tool:
    return Tool(
        name="draft_contact_message",
        description=(
            "Submit a message from the visitor to Ruud (hiring inquiry, "
            "collaboration, question). Call it whenever the visitor asks you to "
            "send, pass on, or forward something to Ruud, or leaves contact "
            "details for him. Requires Ruud's personal approval before it is "
            "recorded — tell the visitor it was submitted for approval."
        ),
        parameters={
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Short subject line."},
                "message": {"type": "string", "description": "The visitor's message."},
                "sender_contact": {
                    "type": "string",
                    "description": "How Ruud can reach the visitor (email or similar).",
                },
            },
            "required": ["subject", "message", "sender_contact"],
            "additionalProperties": False,
        },
        run=lambda subject, message, sender_contact: {
            "status": "recorded",
            "subject": subject,
            "message": message,
            "sender_contact": sender_contact,
        },
        high_risk=True,
    )
