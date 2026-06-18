from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    model_path: Path = Path("./ml/artifacts/autoencoder.pt")
    isolation_model_path: Path = Path("./ml/artifacts/isolation_forest.pkl")
    anomaly_threshold: float = 0.05
    mlflow_tracking_uri: str = "./mlruns"
    model_type: str = "isolation_forest"  # "isolation_forest" | "autoencoder"
    model_version: str = "lstm_autoencoder_v2"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
