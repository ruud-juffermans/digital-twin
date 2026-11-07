from fastapi.testclient import TestClient

from app.api.security import MAX_BODY_BYTES
from app.config import get_settings
from app.main import create_app


def test_security_headers_on_every_response():
    with TestClient(create_app()) as client:
        api = client.get("/api/health")
        missing = client.get("/nope")

    for response in (api, missing):
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        csp = response.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'self'" in csp
        assert "object-src 'none'" in csp
    # Plain http (dev): no HSTS pin.
    assert "Strict-Transport-Security" not in api.headers


def test_hsts_only_behind_tls():
    with TestClient(create_app()) as client:
        response = client.get("/api/health", headers={"x-forwarded-proto": "https"})
    assert response.headers["Strict-Transport-Security"].startswith("max-age=31536000")


def test_frame_ancestors_is_configurable(monkeypatch):
    monkeypatch.setenv("FRAME_ANCESTORS", "'self' https://ruudjuffermans.nl")
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            csp = client.get("/api/health").headers["Content-Security-Policy"]
    finally:
        get_settings.cache_clear()
    assert "frame-ancestors 'self' https://ruudjuffermans.nl" in csp


def test_oversized_body_is_refused_early():
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/chat",
            content=b"x" * (MAX_BODY_BYTES + 1),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json() == {"error": "Request body too large."}


def test_unhandled_errors_never_leak_internals():
    app = create_app()

    @app.get("/api/boom")
    async def boom() -> None:
        raise RuntimeError("secret internal detail")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/boom")

    assert response.status_code == 500
    assert response.json() == {"error": "Unexpected server error."}
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text


def test_docs_endpoints_are_off():
    with TestClient(create_app()) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
