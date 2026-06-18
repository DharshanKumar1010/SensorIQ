import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_and_login(client: AsyncClient):
    resp = await client.post("/auth/register", json={"email": "test@example.com", "password": "secret123"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "test@example.com"
    assert "id" in body

    resp = await client.post("/auth/login", json={"email": "test@example.com", "password": "secret123"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient):
    await client.post("/auth/register", json={"email": "dup@example.com", "password": "pass"})
    resp = await client.post("/auth/register", json={"email": "dup@example.com", "password": "pass"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_login_bad_password(client: AsyncClient):
    await client.post("/auth/register", json={"email": "user2@example.com", "password": "correct"})
    resp = await client.post("/auth/login", json={"email": "user2@example.com", "password": "wrong"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_requires_auth(client: AsyncClient):
    resp = await client.get("/auth/me")
    assert resp.status_code == 401
