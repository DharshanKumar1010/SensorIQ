import pytest
from datetime import datetime, timezone
from httpx import AsyncClient


async def _registered_user_with_asset(client: AsyncClient, email: str):
    await client.post("/auth/register", json={"email": email, "password": "pass123"})
    resp = await client.post("/auth/login", json={"email": email, "password": "pass123"})
    headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    asset = (await client.post("/assets", json={"name": "Motor-A", "asset_type": "motor"}, headers=headers)).json()
    return headers, asset["id"]


@pytest.mark.asyncio
async def test_ingest_batch(client: AsyncClient):
    headers, asset_id = await _registered_user_with_asset(client, "ingest_user@example.com")
    payload = {
        "readings": [
            {"timestamp": datetime.now(timezone.utc).isoformat(), "cycle": i, "sensor_data": {"s1": 1.0, "s2": 0.5}}
            for i in range(1, 4)
        ]
    }
    resp = await client.post(f"/ingest/{asset_id}", json=payload, headers=headers)
    assert resp.status_code == 201
    assert len(resp.json()) == 3


@pytest.mark.asyncio
async def test_ingest_unknown_asset(client: AsyncClient):
    headers, _ = await _registered_user_with_asset(client, "ingest_user2@example.com")
    import uuid
    payload = {"readings": [{"timestamp": datetime.now(timezone.utc).isoformat(), "cycle": 1, "sensor_data": {"s1": 1.0}}]}
    resp = await client.post(f"/ingest/{uuid.uuid4()}", json=payload, headers=headers)
    assert resp.status_code == 404
