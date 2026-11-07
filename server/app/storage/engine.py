"""Database engine, sessions and schema migrations.

Postgres is the database of record. With DATABASE_URL unset (tests, quick
bare-metal dev) the app runs on in-memory fallbacks and nothing survives a
restart. Schema changes go through alembic (server/migrations/);
`run_migrations` is called at startup so a deploy is always at head.
"""

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

SERVER_DIR = Path(__file__).resolve().parent.parent.parent


def sqlalchemy_url(database_url: str) -> str:
    """Pin the psycopg3 driver onto a plain postgresql:// URL."""
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


def create_engine_and_sessions(database_url: str) -> tuple[AsyncEngine, async_sessionmaker]:
    engine = create_async_engine(sqlalchemy_url(database_url), pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def run_migrations(database_url: str) -> None:
    """Bring the schema to head. Sync (alembic is sync) — call via to_thread."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(SERVER_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(SERVER_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", sqlalchemy_url(database_url))
    command.upgrade(cfg, "head")

