import pytest
from datetime import datetime, timezone
from httpx import AsyncClient


async def _setup(client: AsyncClient, email: str):
    await client.post("/auth/register", json={"email": email, "password": "pass123"})
    resp = await client.post("/auth/login", json={"email": email, "password": "pass123"})
    headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    asset = (await client.post("/assets", json={"name": "Fan-1", "asset_type": "fan"}, headers=headers)).json()
    payload = {
        "readings": [
            {"timestamp": datetime.now(timezone.utc).isoformat(), "cycle": i, "sensor_data": {"s1": float(i)}}
            for i in range(1, 6)
        ]
    }
    await client.post(f"/ingest/{asset['id']}", json=payload, headers=headers)
    return headers, asset["id"]


@pytest.mark.asyncio
async def test_get_scored_readings(client: AsyncClient):
    headers, asset_id = await _setup(client, "anomaly_user@example.com")
    resp = await client.get(f"/anomalies/{asset_id}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 5
    assert all("reading" in row for row in data)


@pytest.mark.asyncio
async def test_anomaly_summary(client: AsyncClient):
    headers, asset_id = await _setup(client, "anomaly_user2@example.com")
    resp = await client.get(f"/anomalies/{asset_id}/summary", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_readings"] == 5
    assert body["total_anomalies"] == 0
    assert body["anomaly_rate"] == 0.0


@pytest.mark.asyncio
async def test_anomalies_unknown_asset(client: AsyncClient):
    headers, _ = await _setup(client, "anomaly_user3@example.com")
    import uuid
    resp = await client.get(f"/anomalies/{uuid.uuid4()}", headers=headers)
    assert resp.status_code == 404
