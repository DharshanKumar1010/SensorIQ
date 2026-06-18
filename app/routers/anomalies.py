import json
import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import get_db
from app.models.sensor import Asset, Reading, AnomalyScore
from app.models.user import User
from app.schemas.sensor import (
    AnomalySummary, AnomalyScoreOut, ModelInfo, ReadingOut,
    RulBucketStat, ScoredReadingOut,
)
from app.services.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


async def _require_owned_asset(asset_id: UUID, user_id: UUID, db: AsyncSession) -> Asset:
    result = await db.execute(
        select(Asset).where(Asset.id == asset_id, Asset.user_id == user_id)
    )
    asset = result.scalar_one_or_none()
    if not asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    return asset


@router.get("/{asset_id}", response_model=list[ScoredReadingOut])
async def get_scored_readings(
    asset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await _require_owned_asset(asset_id, current_user.id, db)

    result = await db.execute(
        select(Reading)
        .where(Reading.asset_id == asset_id)
        .options(selectinload(Reading.anomaly_score))
        .order_by(Reading.timestamp.desc())
        .limit(500)
    )
    readings = result.scalars().all()
    return [
        ScoredReadingOut(
            reading=ReadingOut.model_validate(r),
            anomaly_score=AnomalyScoreOut.model_validate(r.anomaly_score) if r.anomaly_score else None,
        )
        for r in readings
    ]


@router.get("/{asset_id}/summary", response_model=AnomalySummary)
async def get_anomaly_summary(
    asset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await _require_owned_asset(asset_id, current_user.id, db)

    total_readings: int = (
        await db.execute(
            select(func.count()).select_from(Reading).where(Reading.asset_id == asset_id)
        )
    ).scalar_one()

    total_anomalies: int = (
        await db.execute(
            select(func.count())
            .select_from(AnomalyScore)
            .join(Reading, AnomalyScore.reading_id == Reading.id)
            .where(Reading.asset_id == asset_id, AnomalyScore.is_anomaly.is_(True))
        )
    ).scalar_one()

    recent_result = await db.execute(
        select(Reading)
        .where(Reading.asset_id == asset_id)
        .options(selectinload(Reading.anomaly_score))
        .order_by(Reading.timestamp.desc())
        .limit(10)
    )
    recent_readings = recent_result.scalars().all()
    recent_anomalies = [
        ScoredReadingOut(
            reading=ReadingOut.model_validate(r),
            anomaly_score=AnomalyScoreOut.model_validate(r.anomaly_score),
        )
        for r in recent_readings
        if r.anomaly_score and r.anomaly_score.is_anomaly
    ]

    return AnomalySummary(
        asset_id=asset_id,
        total_readings=total_readings,
        total_anomalies=total_anomalies,
        anomaly_rate=total_anomalies / total_readings if total_readings else 0.0,
        recent_anomalies=recent_anomalies,
    )


@router.get("/{asset_id}/model-info", response_model=ModelInfo)
async def get_model_info(
    asset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return active model type, version, threshold, and RUL bucket stats from the last evaluation."""
    await _require_owned_asset(asset_id, current_user.id, db)

    info = ModelInfo(
        model_type=settings.model_type,
        model_version=settings.model_version,
        threshold=settings.anomaly_threshold,
    )

    summary_path = Path("ml/artifacts/eval_summary.json")
    if summary_path.exists():
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
        model_data: dict = raw.get(settings.model_type, {})
        if model_data:
            info.pearson_corr = model_data.get("pearson_corr")
            bucket_list = model_data.get("rul_bucket_stats", [])
            info.rul_bucket_stats = [RulBucketStat(**s) for s in bucket_list]
    else:
        logger.debug("eval_summary.json not found — returning config-level info only")

    return info
