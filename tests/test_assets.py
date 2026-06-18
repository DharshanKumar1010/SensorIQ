import pytest
from httpx import AsyncClient


async def _auth_headers(client: AsyncClient, email: str = "asset_user@example.com") -> dict:
    await client.post("/auth/register", json={"email": email, "password": "pass123"})
    resp = await client.post("/auth/login", json={"email": email, "password": "pass123"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_create_and_list_assets(client: AsyncClient):
    headers = await _auth_headers(client)
    resp = await client.post("/assets", json={"name": "Turbofan-1", "asset_type": "turbofan"}, headers=headers)
    assert resp.status_code == 201
    asset = resp.json()
    assert asset["name"] == "Turbofan-1"

    resp = await client.get("/assets", headers=headers)
    assert resp.status_code == 200
    assert any(a["id"] == asset["id"] for a in resp.json())


@pytest.mark.asyncio
async def test_get_asset_not_found(client: AsyncClient):
    headers = await _auth_headers(client, "asset_user2@example.com")
    import uuid
    resp = await client.get(f"/assets/{uuid.uuid4()}", headers=headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_assets_require_auth(client: AsyncClient):
    resp = await client.get("/assets")
    assert resp.status_code == 401
