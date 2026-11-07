"""LLM provider accounts for the admin panel (PRD F1): what is left on each
account and what the twin spent per provider.

Balances are only available where the provider has an API for it:

- DeepSeek: GET https://api.deepseek.com/user/balance
- Moonshot (Kimi): GET https://api.moonshot.ai/v1/users/me/balance
- OpenAI: no balance endpoint; with an *admin* key the organisation Costs API
  gives billed spend month-to-date, which the panel shows instead.
- Gemini: nothing; the panel estimates from its own top-up ledger.

Spend per provider comes from turn_log (the twin's own metering). Balances
are cached in-process for 15 minutes so a dashboard refresh does not hammer
the providers.
"""

import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from .admin import require_admin
from .config import Settings, get_settings
from .storage.models import TurnLogRow

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin")

CACHE_TTL_S = 15 * 60
TIMEOUT_S = 10.0

PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "label": "DeepSeek", "key_field": "deepseek_api_key", "model_field": "deepseek_model",
    },
    "kimi": {
        "label": "Moonshot (Kimi)", "key_field": "moonshot_api_key", "model_field": "kimi_model",
    },
    "openai": {"label": "OpenAI", "key_field": "openai_api_key", "model_field": "openai_model"},
    "gemini": {"label": "Gemini", "key_field": "gemini_api_key", "model_field": "gemini_model"},
}
PROVIDER_PARAM = Query(pattern=r"^[a-z]{1,16}$")
SINCE_PARAM = Query()

Fetcher = Callable[[httpx.AsyncClient, Settings], Awaitable[dict[str, Any]]]


async def _deepseek_balance(client: httpx.AsyncClient, settings: Settings) -> dict[str, Any]:
    response = await client.get(
        "https://api.deepseek.com/user/balance",
        headers={"authorization": f"Bearer {settings.deepseek_api_key}"},
    )
    response.raise_for_status()
    body = response.json()
    infos = body.get("balance_infos") or []
    if not infos:
        raise ValueError("no balance_infos in response")
    # Accounts can carry a CNY and a USD wallet; prefer the USD one.
    info = next((entry for entry in infos if entry.get("currency") == "USD"), infos[0])
    return {
        "available": float(info.get("total_balance") or 0.0),
        "currency": str(info.get("currency") or "USD"),
        "source": "deepseek /user/balance",
    }


async def _moonshot_balance(client: httpx.AsyncClient, settings: Settings) -> dict[str, Any]:
    response = await client.get(
        "https://api.moonshot.ai/v1/users/me/balance",
        headers={"authorization": f"Bearer {settings.moonshot_api_key}"},
    )
    response.raise_for_status()
    data = response.json().get("data") or {}
    if "available_balance" not in data:
        raise ValueError("no available_balance in response")
    return {
        "available": float(data["available_balance"]),
        "currency": "USD",
        "source": "moonshot /users/me/balance",
    }


async def _openai_costs_mtd(client: httpx.AsyncClient, settings: Settings) -> float:
    """Billed spend this calendar month from the organisation Costs API."""
    start = datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    response = await client.get(
        "https://api.openai.com/v1/organization/costs",
        params={"start_time": int(start.timestamp()), "bucket_width": "1d", "limit": 31},
        headers={"authorization": f"Bearer {settings.openai_admin_key}"},
    )
    response.raise_for_status()
    total = 0.0
    for bucket in response.json().get("data") or []:
        for result in bucket.get("results") or []:
            total += float((result.get("amount") or {}).get("value") or 0.0)
    return round(total, 4)


BALANCE_FETCHERS: dict[str, Fetcher] = {
    "deepseek": _deepseek_balance,
    "kimi": _moonshot_balance,
}


class ProviderBalances:
    """Fetches and caches the per-provider account state."""

    def __init__(
        self, transport: httpx.AsyncBaseTransport | None = None, ttl_s: float = CACHE_TTL_S
    ) -> None:
        self._transport = transport
        self._ttl = ttl_s
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=TIMEOUT_S, transport=self._transport)

    async def snapshot(self, settings: Settings) -> dict[str, dict[str, Any]]:
        """{provider: {balance, balance_error, billed_mtd_usd}} for every provider."""
        now = time.monotonic()
        out: dict[str, dict[str, Any]] = {}
        async with self._client() as client:
            for provider, spec in PROVIDERS.items():
                cached = self._cache.get(provider)
                if cached and now - cached[0] < self._ttl:
                    out[provider] = cached[1]
                    continue
                entry = await self._fetch(client, settings, provider, spec)
                self._cache[provider] = (now, entry)
                out[provider] = entry
        return out

    async def _fetch(
        self, client: httpx.AsyncClient, settings: Settings, provider: str, spec: dict[str, str]
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {"balance": None, "balance_error": None, "billed_mtd_usd": None}
        configured = bool(getattr(settings, spec["key_field"]))
        fetcher = BALANCE_FETCHERS.get(provider)
        if fetcher is not None and configured:
            try:
                balance = await fetcher(client, settings)
                entry["balance"] = {
                    **balance,
                    "fetched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                }
            except Exception as exc:  # network, auth, shape — all shown, none fatal
                logger.warning("Balance lookup failed for %s: %s", provider, exc)
                entry["balance_error"] = f"balance lookup failed: {type(exc).__name__}"
        if provider == "openai" and settings.openai_admin_key:
            try:
                entry["billed_mtd_usd"] = await _openai_costs_mtd(client, settings)
            except Exception as exc:
                logger.warning("OpenAI costs lookup failed: %s", exc)
                entry["balance_error"] = f"costs lookup failed: {type(exc).__name__}"
        return entry


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
