from datetime import datetime
from uuid import UUID
from pydantic import BaseModel


class AssetCreate(BaseModel):
    name: str
    asset_type: str


class AssetOut(BaseModel):
    id: UUID
    user_id: UUID
    name: str
    asset_type: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ReadingIn(BaseModel):
    timestamp: datetime
    cycle: int
    sensor_data: dict[str, float]


class ReadingBatch(BaseModel):
    readings: list[ReadingIn]


class ReadingOut(BaseModel):
    id: UUID
    asset_id: UUID
    timestamp: datetime
    cycle: int
    sensor_data: dict[str, float]
    created_at: datetime

    model_config = {"from_attributes": True}


class AnomalyScoreOut(BaseModel):
    id: UUID
    reading_id: UUID
    model_version: str
    score: float
    is_anomaly: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ScoredReadingOut(BaseModel):
    reading: ReadingOut
    anomaly_score: AnomalyScoreOut | None = None


class AnomalySummary(BaseModel):
    asset_id: UUID
    total_readings: int
    total_anomalies: int
    anomaly_rate: float
    recent_anomalies: list[ScoredReadingOut]


class RulBucketStat(BaseModel):
    bucket: str
    mean_score: float
    anomaly_rate: float
    n_windows: int


class ModelInfo(BaseModel):
    model_type: str
    model_version: str
    threshold: float
    pearson_corr: float | None = None
    rul_bucket_stats: list[RulBucketStat] | None = None
