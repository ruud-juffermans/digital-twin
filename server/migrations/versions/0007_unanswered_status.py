"""Triage state on unanswered questions: open until Ruud marks them handled.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-07
"""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("unanswered_questions") as batch:
        batch.add_column(
            sa.Column("status", sa.String(10), nullable=False, server_default="open")
        )
        batch.create_index("ix_unanswered_questions_status", ["status"])


def downgrade() -> None:
    with op.batch_alter_table("unanswered_questions") as batch:
        batch.drop_index("ix_unanswered_questions_status")
        batch.drop_column("status")
