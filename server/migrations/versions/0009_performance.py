"""Performance telemetry: time to first token, finish reason, retries, an
error type, the run trace (node and tool spans) and length metrics on
turn_log, plus a model_calls table with one row per model round so a
tool-using turn is no longer a single summed number. Nothing here holds
visitor text, so the retention job leaves it alone.

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-07
"""

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

JSON_VARIANT = sa.JSON().with_variant(sa.dialects.postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    with op.batch_alter_table("turn_log") as batch:
        batch.add_column(sa.Column("ttft_ms", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("finish_reason", sa.String(20), nullable=True))
        batch.add_column(sa.Column("retries", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("error_type", sa.String(20), nullable=True))
        batch.add_column(sa.Column("trace", JSON_VARIANT, nullable=True))
        batch.add_column(sa.Column("question_chars", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("context_chars", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("answer_chars", sa.Integer(), nullable=True))
    op.create_table(
        "model_calls",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "turn_id", sa.Integer(), sa.ForeignKey("turn_log.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cached_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ttft_ms", sa.Integer(), nullable=True),
        sa.Column("finish_reason", sa.String(20), nullable=True),
        sa.Column("retries", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tool_call_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_chars", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_model_calls_turn_id", "model_calls", ["turn_id"])
    op.create_index("ix_model_calls_created_at", "model_calls", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_model_calls_created_at", table_name="model_calls")
    op.drop_index("ix_model_calls_turn_id", table_name="model_calls")
    op.drop_table("model_calls")
    with op.batch_alter_table("turn_log") as batch:
        for column in (
            "answer_chars", "context_chars", "question_chars", "trace", "error_type",
            "retries", "finish_reason", "ttft_ms",
        ):
            batch.drop_column(column)
