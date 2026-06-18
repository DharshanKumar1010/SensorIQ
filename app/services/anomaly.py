import logging
import pickle
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
_WINDOW_SIZE = 30

# Reflects the active model so ingest.py can store a meaningful model_version string
MODEL_VERSION = (
    "lstm_autoencoder_v1"
    if settings.model_type == "autoencoder"
    else "isolation_forest_v1"
)

# Module-level cache — populated on first call, never reloaded
_isolation_model = None
_autoencoder_model = None
_scaler_params: dict[str, tuple[float, float]] | None = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_scaler() -> dict[str, tuple[float, float]]:
    global _scaler_params
    if _scaler_params is not None:
        return _scaler_params
    path = Path("data/processed/scaler_params.npy")
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler params not found at {path}. Run 'python -m ml.preprocess' first."
        )
    _scaler_params = np.load(path, allow_pickle=True).item()
    return _scaler_params


def _normalize(sensor_data: dict[str, float]) -> np.ndarray:
    """Return a float32 array (21,) of min-max-scaled sensor values."""
    scaler = _load_scaler()
    features = np.array(
        [sensor_data.get(col, 0.0) for col in _SENSOR_COLS], dtype=np.float32
    )
    for i, col in enumerate(_SENSOR_COLS):
        col_min, col_max = scaler[col]
        denom = col_max - col_min
        features[i] = (features[i] - col_min) / denom if denom > 0 else 0.0
    return features


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

def _load_isolation_model():
    global _isolation_model
    if _isolation_model is not None:
        return _isolation_model
    pkl_path = Path(settings.isolation_model_path)
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Isolation Forest artifact not found at {pkl_path}. "
            "Run 'python -m ml.train_isolation' first."
        )
    with open(pkl_path, "rb") as f:
        _isolation_model = pickle.load(f)
    logger.info("Loaded Isolation Forest from %s", pkl_path)
    return _isolation_model


def _score_isolation_forest(sensor_data: dict[str, float]) -> tuple[float, bool]:
    model = _load_isolation_model()
    features = np.array(
        [sensor_data.get(col, 0.0) for col in _SENSOR_COLS], dtype=np.float32
    )
    # Replicate to (1, 630) — approximation; proper windowing comes in Week 4
    window = np.tile(features, (_WINDOW_SIZE, 1)).flatten().reshape(1, -1)
    score = float(-model.score_samples(window)[0])
    return score, score > settings.anomaly_threshold


# ---------------------------------------------------------------------------
# LSTM Autoencoder
# ---------------------------------------------------------------------------

def _load_autoencoder():
    """Lazy-load the autoencoder.  torch is imported here to keep startup fast."""
    global _autoencoder_model
    if _autoencoder_model is not None:
        return _autoencoder_model

    import torch
    from ml.model import LSTMAutoencoder

    pt_path = Path(settings.model_path)
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Autoencoder artifact not found at {pt_path}. "
            "Run 'python -m ml.train_autoencoder' first."
        )
    model = LSTMAutoencoder()
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()
    _autoencoder_model = model
    logger.info("Loaded LSTM Autoencoder from %s", pt_path)
    return _autoencoder_model


def score_reading_autoencoder(sensor_data: dict[str, float]) -> tuple[float, bool]:
    """
    Score a single reading using the LSTM Autoencoder.
    Normalises sensor values, replicates to a (1, 30, 21) window,
    and returns (mse_reconstruction_error, is_anomaly).
    """
    import torch

    model = _load_autoencoder()
    features = _normalize(sensor_data)
    window = np.tile(features, (_WINDOW_SIZE, 1))[np.newaxis, ...]  # (1, 30, 21)
    x = torch.from_numpy(window)
    with torch.no_grad():
        recon = model(x)
    score = float(((x - recon) ** 2).mean().item())
    return score, score > settings.anomaly_threshold


# ---------------------------------------------------------------------------
# Public dispatcher — called by ingest.py
# ---------------------------------------------------------------------------

def score_reading(sensor_data: dict[str, float]) -> tuple[float, bool]:
    """
    Route to the active model (controlled by MODEL_TYPE in .env).
    Returns (score, is_anomaly).  Raises FileNotFoundError if the artifact
    is missing — ingest.py catches this and skips scoring gracefully.
    """
    if settings.model_type == "autoencoder":
        return score_reading_autoencoder(sensor_data)
    return _score_isolation_forest(sensor_data)
