"""Provider account state for the admin panel: what is left on each account.

Balances are only available where the provider has an API for it:

- DeepSeek: GET https://api.deepseek.com/user/balance
- Moonshot (Kimi): GET https://api.moonshot.ai/v1/users/me/balance
- OpenAI: no balance endpoint; with an *admin* key the organisation Costs API
  gives billed spend month-to-date, which the panel shows instead.
- Gemini: nothing; the panel estimates from its own top-up ledger.

Balances are cached in-process for 15 minutes so a dashboard refresh does not
hammer the providers.
"""

import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from ..config import Settings
from .providers import PROVIDERS

logger = logging.getLogger(__name__)

CACHE_TTL_S = 15 * 60
TIMEOUT_S = 10.0

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
