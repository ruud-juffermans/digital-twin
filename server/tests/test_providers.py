"""Provider accounts for the admin panel: live balances where the provider
has an API, metered spend per provider from turn_log."""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings, get_settings
from app.db import Base, TurnRecorder
from app.main import create_app
from app.providers import ProviderBalances

ADMIN = {"Authorization": "Bearer test-admin-token"}


@pytest.fixture(autouse=True)
def admin_token(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sessions():
    engine = create_async_engine("sqlite+aiosqlite://")

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(setup())
    yield async_sessionmaker(engine, expire_on_commit=False)
    asyncio.run(engine.dispose())


def provider_transport(calls: list[str], moonshot_status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.url.host}{request.url.path}")
        if request.url.host == "api.deepseek.com":
            assert request.headers["authorization"] == "Bearer ds-key"
            return httpx.Response(200, json={
                "is_available": True,
                "balance_infos": [
                    {"currency": "CNY", "total_balance": "12.00"},
                    {"currency": "USD", "total_balance": "8.50"},
                ],
            })
        if request.url.host == "api.moonshot.ai":
            assert request.headers["authorization"] == "Bearer ms-key"
            return httpx.Response(moonshot_status, json={
                "code": 0, "data": {"available_balance": 3.25}, "status": True,
            })
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_snapshot_fetches_balances_and_caches(sessions):
    calls: list[str] = []
    balances = ProviderBalances(transport=provider_transport(calls))
    settings = Settings()

    first = asyncio.run(balances.snapshot(settings))
    second = asyncio.run(balances.snapshot(settings))

    assert first["deepseek"]["balance"]["available"] == 8.5  # the USD wallet wins
    assert first["deepseek"]["balance"]["currency"] == "USD"
    assert first["kimi"]["balance"]["available"] == 3.25
    assert first["openai"] == {"balance": None, "balance_error": None, "billed_mtd_usd": None}
    assert first["gemini"]["balance"] is None
    assert second == first
    assert sorted(calls) == ["api.deepseek.com/user/balance", "api.moonshot.ai/v1/users/me/balance"]


def test_balance_failure_is_reported_not_raised():
    balances = ProviderBalances(transport=provider_transport([], moonshot_status=401))
    snapshot = asyncio.run(balances.snapshot(Settings()))
    assert snapshot["kimi"]["balance"] is None
    assert "failed" in snapshot["kimi"]["balance_error"]
    assert snapshot["deepseek"]["balance"]["available"] == 8.5


def test_providers_endpoint_joins_balances_and_spend(sessions):
    recorder = TurnRecorder(sessions, provider="deepseek", model="m")
    asyncio.run(recorder.record(thread_id="t", outcome="answered", meta={"costUsd": 0.5},
                                tools_used=[], guard_flags=[], visitor_message="q"))
    with TestClient(create_app()) as client:
        client.app.state.sessions = sessions
        client.app.state.provider_balances = ProviderBalances(transport=provider_transport([]))
        assert client.get("/api/admin/providers").status_code == 401
        body = client.get("/api/admin/providers", headers=ADMIN).json()
        spend = client.get(
            "/api/admin/providers/spend?provider=deepseek&since=2026-01-01T00:00:00Z",
            headers=ADMIN,
        ).json()
        assert client.get("/api/admin/providers/spend?provider=claude&since=2026-01-01T00:00:00Z",
                          headers=ADMIN).status_code == 404

    assert body["active"] == "deepseek"
    by_id = {p["id"]: p for p in body["providers"]}
    assert by_id["deepseek"]["active"] is True
    assert by_id["deepseek"]["configured"] is True
    assert by_id["deepseek"]["balance"]["available"] == 8.5
    assert by_id["deepseek"]["spend"]["total_usd"] == pytest.approx(0.5)
    assert by_id["deepseek"]["spend"]["today_usd"] == pytest.approx(0.5)
    assert by_id["kimi"]["balance_supported"] is True
    assert by_id["openai"]["balance_supported"] is False
    assert by_id["openai"]["configured"] is False
    assert by_id["gemini"]["spend"]["total_usd"] == 0.0
    assert (spend["provider"], spend["turns"]) == ("deepseek", 1)
    assert spend["spent_usd"] == pytest.approx(0.5)
    json.dumps(body)  # serialisable end to end
