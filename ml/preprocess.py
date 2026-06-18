"""
ETL pipeline for NASA C-MAPSS FD001.
Reads raw train/test files, computes RUL, filters healthy windows (RUL > 50),
min-max scales sensor columns, builds sliding windows, and saves processed arrays.

Run from project root:
    python -m ml.preprocess
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/cmapss")
PROCESSED_DIR = Path("data/processed")

WINDOW_SIZE = 30
STRIDE = 1
HEALTHY_RUL_CUTOFF = 50

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit_id", "cycle", "op1", "op2", "op3"] + SENSOR_COLS  # 26 total


def _read_cmapss(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python", on_bad_lines="skip")
    # Trailing whitespace creates a 27th empty column — drop it
    df = df.iloc[:, :26]
    df.columns = COLS
    return df


def _add_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycles, on="unit_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    return df.drop(columns=["max_cycle"])


def _fit_scaler(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Compute min/max for each sensor column from df (should be healthy training data)."""
    return {col: (float(df[col].min()), float(df[col].max())) for col in SENSOR_COLS}


def _apply_scaler(df: pd.DataFrame, params: dict[str, tuple[float, float]]) -> pd.DataFrame:
    df = df.copy()
    for col, (col_min, col_max) in params.items():
        denom = col_max - col_min
        df[col] = (df[col] - col_min) / denom if denom > 0 else 0.0
    return df


def _build_windows(
    df: pd.DataFrame, window_size: int = WINDOW_SIZE, stride: int = STRIDE
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows per unit sorted by cycle.
    Returns:
        X      — float32 array of shape (n_windows, window_size, n_sensors)
        y_rul  — float32 array of shape (n_windows,) — RUL at the last step of each window
    """
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for _, group in df.groupby("unit_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        sensors = group[SENSOR_COLS].to_numpy(dtype=np.float32)
        ruls = group["rul"].to_numpy(dtype=np.float32)
        n = len(group)
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            X_list.append(sensors[start:end])
            y_list.append(ruls[end - 1])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading train_FD001.txt …")
    train_df = _read_cmapss("train_FD001.txt")
    train_df = _add_rul(train_df)

    healthy_df = train_df[train_df["rul"] > HEALTHY_RUL_CUTOFF].copy()
    logger.info(
        "Healthy rows: %d / %d  (%.1f%%)",
        len(healthy_df), len(train_df),
        100 * len(healthy_df) / len(train_df),
    )

    # Fit scaler on healthy training data only — prevents data leakage from failure cycles
    scaler_params = _fit_scaler(healthy_df)
    healthy_scaled = _apply_scaler(healthy_df, scaler_params)

    X_train, y_rul = _build_windows(healthy_scaled)
    logger.info("X_train: %s  y_rul: %s", X_train.shape, y_rul.shape)

    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "y_rul.npy", y_rul)
    # Save as object array so we can load it back with allow_pickle=True
    np.save(PROCESSED_DIR / "scaler_params.npy", scaler_params)

    logger.info("Saved X_train.npy, y_rul.npy, scaler_params.npy → %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
