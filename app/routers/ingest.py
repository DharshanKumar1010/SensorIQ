import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.sensor import Asset, AnomalyScore, Reading
from app.models.user import User
from app.schemas.sensor import ReadingBatch, ReadingOut
from app.services.anomaly import MODEL_VERSION, score_reading
from app.services.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/{asset_id}", response_model=list[ReadingOut], status_code=status.HTTP_201_CREATED)
async def ingest_readings(
    asset_id: UUID,
    payload: ReadingBatch,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Asset).where(Asset.id == asset_id, Asset.user_id == current_user.id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")

    readings = [
        Reading(
            asset_id=asset_id,
            timestamp=r.timestamp,
            cycle=r.cycle,
            sensor_data=r.sensor_data,
        )
        for r in payload.readings
    ]
    db.add_all(readings)
    # Flush to assign server-side defaults (created_at) and confirm FK integrity
    # without committing — lets us reference reading.id for anomaly_scores below
    await db.flush()

    anomaly_scores: list[AnomalyScore] = []
    n_anomalies = 0
    for reading in readings:
        try:
            score, is_anomaly = score_reading(reading.sensor_data)
        except FileNotFoundError:
            # Model not yet trained — skip scoring silently; scores can be backfilled later
            logger.debug("Anomaly model unavailable, skipping score for reading %s", reading.id)
            break
        except Exception:
            logger.warning("Scoring failed for reading %s", reading.id, exc_info=True)
            continue
        anomaly_scores.append(
            AnomalyScore(
                reading_id=reading.id,
                model_version=MODEL_VERSION,
                score=score,
                is_anomaly=is_anomaly,
            )
        )
        if is_anomaly:
            n_anomalies += 1

    if anomaly_scores:
        db.add_all(anomaly_scores)

    await db.commit()

    for r in readings:
        await db.refresh(r)

    logger.info(
        "Ingested %d readings for asset %s — scored %d, anomalies: %d",
        len(readings), asset_id, len(anomaly_scores), n_anomalies,
    )
    return readings
