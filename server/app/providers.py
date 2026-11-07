"""Admin endpoints for the provider accounts (PRD F1): balances from the
gateway, spend per provider from turn_log (the twin's own metering)."""

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from .admin import require_admin
from .config import get_settings
from .gateway.balances import BALANCE_FETCHERS
from .gateway.providers import PROVIDERS
from .storage.models import TurnLogRow

router = APIRouter(prefix="/api/admin")

PROVIDER_PARAM = Query(pattern=r"^[a-z]{1,16}$")
SINCE_PARAM = Query()


async def spend_by_provider(
    sessions: async_sessionmaker, since: datetime | None = None
) -> dict[str, float]:
    query = select(TurnLogRow.provider, func.sum(TurnLogRow.cost_usd)).group_by(TurnLogRow.provider)
    if since is not None:
        query = query.where(TurnLogRow.created_at >= since)
    async with sessions() as session:
        rows = await session.execute(query)
        return {provider: round(float(total or 0.0), 5) for provider, total in rows}


@router.get("/providers")
async def providers(request: Request) -> Any:
    if (denied := require_admin(request)) is not None:
        return denied
    settings = get_settings()
    state = request.app.state
    balances = await state.provider_balances.snapshot(settings)
    now = datetime.now(UTC)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    spend_today: dict[str, float] = {}
    spend_30d: dict[str, float] = {}
    spend_total: dict[str, float] = {}
    if state.sessions is not None:
        spend_today = await spend_by_provider(state.sessions, today)
        spend_30d = await spend_by_provider(state.sessions, now - timedelta(days=30))
        spend_total = await spend_by_provider(state.sessions)
    return {
        "fetched_at": now.isoformat().replace("+00:00", "Z"),
        "active": settings.llm_provider,
        "providers": [
            {
                "id": provider,
                "label": spec["label"],
                "configured": bool(getattr(settings, spec["key_field"])),
                "active": provider == settings.llm_provider,
                "model": getattr(settings, spec["model_field"]),
                "balance_supported": provider in BALANCE_FETCHERS,
                "balance": balances[provider]["balance"],
                "balance_error": balances[provider]["balance_error"],
                "spend": {
                    "today_usd": spend_today.get(provider, 0.0),
                    "last_30d_usd": spend_30d.get(provider, 0.0),
                    "total_usd": spend_total.get(provider, 0.0),
                    "billed_mtd_usd": balances[provider]["billed_mtd_usd"],
                },
            }
            for provider, spec in PROVIDERS.items()
        ],
    }


@router.get("/providers/spend")
async def provider_spend(
    request: Request,
    provider: str = PROVIDER_PARAM,
    since: datetime = SINCE_PARAM,
) -> Any:
    """Metered spend for one provider since a moment: the panel subtracts it
    from the top-ups it recorded to estimate what is left."""
    if (denied := require_admin(request)) is not None:
        return denied
    if provider not in PROVIDERS:
        return JSONResponse(status_code=404, content={"error": "Unknown provider."})
    sessions = request.app.state.sessions
    if sessions is None:
        return {"provider": provider, "since": since.isoformat(), "spent_usd": 0.0, "turns": 0}
    if since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    async with sessions() as session:
        spent, turns = (
            await session.execute(
                select(func.sum(TurnLogRow.cost_usd), func.count()).where(
                    TurnLogRow.provider == provider, TurnLogRow.created_at >= since
                )
            )
        ).one()
    return {
        "provider": provider,
        "since": since.isoformat().replace("+00:00", "Z"),
        "spent_usd": round(float(spent or 0.0), 5),
        "turns": int(turns or 0),
    }
