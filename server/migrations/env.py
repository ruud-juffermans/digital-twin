"""Alembic environment: migrates the app tables declared in app.storage.models.

The LangGraph checkpointer (Commit 22) manages its own tables through
AsyncPostgresSaver.setup() and is deliberately outside alembic's control.
"""

import os

from alembic import context
from sqlalchemy import create_engine, pool

from app.storage.engine import sqlalchemy_url
from app.storage.models import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    url = config.get_main_option("sqlalchemy.url") or os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL is not set — nothing to migrate.")
    return sqlalchemy_url(url)


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
