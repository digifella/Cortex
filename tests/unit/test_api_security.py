"""
Unit Tests for API Security Controls
"""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from api.main import verify_token


@pytest.mark.asyncio
async def test_verify_token_allows_when_auth_not_configured(monkeypatch):
    monkeypatch.delenv("CORTEX_API_TOKEN", raising=False)
    monkeypatch.delenv("CORTEX_API_AUTH_REQUIRED", raising=False)

    assert await verify_token(None) is True


@pytest.mark.asyncio
async def test_verify_token_requires_bearer_when_token_configured(monkeypatch):
    monkeypatch.setenv("CORTEX_API_TOKEN", "secret-token")
    monkeypatch.delenv("CORTEX_API_AUTH_REQUIRED", raising=False)

    with pytest.raises(HTTPException) as exc:
        await verify_token(None)

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_token_rejects_invalid_token(monkeypatch):
    monkeypatch.setenv("CORTEX_API_TOKEN", "secret-token")
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-token")

    with pytest.raises(HTTPException) as exc:
        await verify_token(creds)

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_token_accepts_valid_token(monkeypatch):
    monkeypatch.setenv("CORTEX_API_TOKEN", "secret-token")
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="secret-token")

    assert await verify_token(creds) is True

