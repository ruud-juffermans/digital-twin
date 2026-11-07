import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from slowapi.errors import RateLimitExceeded

from .actions.approval import InMemoryApprovalStore, PostgresApprovalStore
from .actions.budget import BudgetTracker
from .admin import router as admin_router
from .admin_read import router as admin_read_router
from .agent.graph import build_agent
from .chat import router as chat_router
from .config import Settings, get_settings
from .context.knowledge import load_knowledge
from .context.prompt import knowledge_version, prompt_version
from .context.retrieval import SemanticRetriever, sync_knowledge_chunks
from .gateway.balances import ProviderBalances
from .gateway.embeddings import create_embedder
from .observability.retention import retention_loop
from .observability.telemetry import TurnRecorder, load_today_spent
from .pipeline.provider import grounded_provider
from .providers import router as providers_router
from .rate_limit import limiter
from .security import install_security
from .storage.engine import create_engine_and_sessions, run_migrations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def _checkpointer(settings: Settings) -> AsyncIterator[BaseCheckpointSaver]:
    """Postgres checkpoints when DATABASE_URL is set; otherwise threads live
    only as long as the process (tests, quick bare-metal dev)."""
    if settings.database_url:
        async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
            await saver.setup()  # manages its own tables, outside alembic
            yield saver
    else:
        yield InMemorySaver()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.knowledge = load_knowledge(settings.knowledge_dir)
    logger.info("Loaded %d knowledge documents", len(app.state.knowledge))
    app.state.llm = grounded_provider(settings, app.state.knowledge)
    app.state.budget = BudgetTracker(settings.daily_budget_usd)
    app.state.provider_balances = ProviderBalances()
    engine = None
    embedder = None
    retention_task: asyncio.Task | None = None
    app.state.sessions = None
    app.state.retriever = None
    if settings.database_url:
        # Schema to head first, then the async engine and the durable stores.
        await asyncio.to_thread(run_migrations, settings.database_url)
        engine, app.state.sessions = create_engine_and_sessions(settings.database_url)
        app.state.approvals = PostgresApprovalStore(app.state.sessions)
        app.state.recorder = TurnRecorder(
            app.state.sessions,
            provider=settings.llm_provider,
            model=settings.active_model,
            prompt_version=prompt_version(),
            kb_version=knowledge_version(app.state.knowledge),
            store_text=settings.text_retention_days > 0,
        )
        retention_task = asyncio.create_task(
            retention_loop(app.state.sessions, settings.text_retention_days)
        )
        # The cap survives restarts: seed today's spend from the ledger.
        app.state.budget.restore(await load_today_spent(app.state.sessions))
        logger.info(
            "Database ready, schema at head (today's spend: $%.4f)", app.state.budget.spent_usd
        )
        if settings.embedding_provider:
            # Semantic search: sync the chunk table, re-embedding only what
            # changed since the last start.
            embedder = create_embedder(settings)
            counts = await sync_knowledge_chunks(
                app.state.sessions, embedder, app.state.knowledge
            )
            app.state.retriever = SemanticRetriever(embedder, app.state.sessions)
            logger.info(
                "Knowledge chunks synced via %s (+%d new, %d kept, -%d stale)",
                embedder.model_id, counts["embedded"], counts["kept"], counts["deleted"],
            )
    else:
        # No database: nothing survives a restart.
        app.state.approvals = InMemoryApprovalStore()
        app.state.recorder = None
        if settings.embedding_provider:
            logger.warning("EMBEDDING_PROVIDER is set but DATABASE_URL is not — ignoring.")
    async with _checkpointer(settings) as checkpointer:
        # get_provider resolves lazily so tests can swap app.state.llm's client
        # without rebuilding the graph.
        app.state.agent = build_agent(
            lambda: app.state.llm,
            app.state.knowledge,
            checkpointer=checkpointer,
            budget=app.state.budget,
            retriever=app.state.retriever,
        )
        logger.info(
            "Server ready (checkpoints: %s, search: %s)",
            "postgres" if settings.database_url else "in-memory",
            "semantic" if app.state.retriever is not None else "term-overlap",
        )
        yield
        if retention_task is not None:
            retention_task.cancel()
        await app.state.llm.close()
        if embedder is not None:
            await embedder.close()
        if engine is not None:
            await engine.dispose()


def create_app() -> FastAPI:
    # No docs endpoints in production: they advertise the schema for free.
    app = FastAPI(
        title="digital-twin server",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.limiter = limiter
    install_security(app, get_settings().frame_ancestors)
    app.include_router(chat_router)
    app.include_router(admin_router)
    app.include_router(admin_read_router)
    app.include_router(providers_router)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limited(_request: Request, _exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit reached — try again in a few minutes."},
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_payload(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": "Invalid messages payload."})

    @app.get("/api/health")
    async def health(request: Request) -> dict:
        return {
            "ok": True,
            "documents": len(request.app.state.knowledge),
            "database": request.app.state.sessions is not None,
        }

    # Production: serve the built client. Declared last, so every /api route
    # above takes precedence over the catch-all static mount.
    static_dir = get_settings().static_dir
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="client")

    return app


app = create_app()
