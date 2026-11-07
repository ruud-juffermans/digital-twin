"""Prompt logging on turn_log: what the visitor asked, what the twin answered,
what it cited and called, plus the prompt/knowledge-base version that
produced the answer and where the chat was embedded. All nullable: the
retention job clears the text columns after TEXT_RETENTION_DAYS and the
numeric telemetry stays.

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-07
"""

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

JSON_VARIANT = sa.JSON().with_variant(sa.dialects.postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    with op.batch_alter_table("turn_log") as batch:
        batch.add_column(sa.Column("visitor_message", sa.Text(), nullable=True))
        batch.add_column(sa.Column("answer", sa.Text(), nullable=True))
        batch.add_column(sa.Column("citations", JSON_VARIANT, nullable=True))
        batch.add_column(sa.Column("tool_calls", JSON_VARIANT, nullable=True))
        batch.add_column(sa.Column("retrieved_chunks", JSON_VARIANT, nullable=True))
        batch.add_column(sa.Column("prompt_version", sa.String(16), nullable=True))
        batch.add_column(sa.Column("kb_version", sa.String(16), nullable=True))
        batch.add_column(sa.Column("locale", sa.String(16), nullable=True))
        batch.add_column(sa.Column("embed", sa.Boolean(), nullable=True))
        batch.create_index("ix_turn_log_created_at", ["created_at"])
        batch.create_index("ix_turn_log_prompt_version", ["prompt_version"])


def downgrade() -> None:
    with op.batch_alter_table("turn_log") as batch:
        batch.drop_index("ix_turn_log_prompt_version")
        batch.drop_index("ix_turn_log_created_at")
        for column in (
            "embed", "locale", "kb_version", "prompt_version", "retrieved_chunks",
            "tool_calls", "citations", "answer", "visitor_message",
        ):
            batch.drop_column(column)
