"""
Evaluate Isolation Forest on C-MAPSS test data.
Labels windows with RUL < 20 as anomalous (near-failure), finds the threshold
that maximises F1, plots the precision-recall curve, and logs everything to MLflow.

Run from project root:
    python -m ml.evaluate
"""
import logging
import os
import pickle
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on Windows without a display
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/cmapss")
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("ml/artifacts")
MLFLOW_TRACKING_URI = "./mlruns"

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit_id", "cycle", "op1", "op2", "op3"] + SENSOR_COLS
WINDOW_SIZE = 30
FAILURE_RUL_CUTOFF = 20


def _load_test_windows() -> tuple[np.ndarray, np.ndarray]:
    """
    Load test_FD001.txt, compute per-row RUL using RUL_FD001.txt ground truth,
    scale with the training scaler, build sliding windows, and label each window.
    Label = 1 if RUL at the last timestep of the window is < FAILURE_RUL_CUTOFF.
    """
    scaler_params: dict[str, tuple[float, float]] = np.load(
        PROCESSED_DIR / "scaler_params.npy", allow_pickle=True
    ).item()

    test_df = pd.read_csv(
        RAW_DIR / "test_FD001.txt",
        sep=r"\s+", header=None, engine="python", on_bad_lines="skip",
    )
    test_df = test_df.iloc[:, :26]
    test_df.columns = COLS

    # RUL_FD001.txt contains one value per test unit: the RUL at the *last* recorded cycle
    rul_gt = pd.read_csv(RAW_DIR / "RUL_FD001.txt", header=None, names=["rul_at_end"])
    rul_gt["unit_id"] = range(1, len(rul_gt) + 1)

    max_cycles = test_df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    test_df = test_df.join(max_cycles, on="unit_id")
    test_df = test_df.merge(rul_gt, on="unit_id")
    # rul = cycles remaining after the last recorded cycle + cycles left in the segment
    test_df["rul"] = (test_df["max_cycle"] - test_df["cycle"]) + test_df["rul_at_end"]

    # Apply training scaler (no re-fitting — prevents data leakage)
    for col, (col_min, col_max) in scaler_params.items():
        denom = col_max - col_min
        test_df[col] = (test_df[col] - col_min) / denom if denom > 0 else 0.0

    X_list: list[np.ndarray] = []
    label_list: list[int] = []
    for _, group in test_df.groupby("unit_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        sensors = group[SENSOR_COLS].to_numpy(dtype=np.float32)
        ruls = group["rul"].to_numpy(dtype=np.float32)
        n = len(group)
        for start in range(0, n - WINDOW_SIZE + 1):
            end = start + WINDOW_SIZE
            X_list.append(sensors[start:end])
            label_list.append(1 if ruls[end - 1] < FAILURE_RUL_CUTOFF else 0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(label_list, dtype=np.int32)
    return X, y


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    pkl_path = ARTIFACTS_DIR / "isolation_forest.pkl"
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    logger.info("Loading and building test windows …")
    X_test, y_true = _load_test_windows()
    X_flat = X_test.reshape(X_test.shape[0], -1)
    logger.info(
        "Test windows: %d  |  anomaly windows (RUL<%d): %d  (%.1f%%)",
        len(y_true), FAILURE_RUL_CUTOFF, int(y_true.sum()),
        100 * y_true.mean(),
    )

    # Negate score_samples: higher value = more anomalous
    scores = -model.score_samples(X_flat)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # F1 for every candidate threshold (precision/recall have one extra element)
    with np.errstate(invalid="ignore"):
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best_idx = int(np.argmax(f1))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1[best_idx])
    best_precision = float(precision[best_idx])
    best_recall = float(recall[best_idx])

    logger.info(
        "Best threshold: %.6f  |  F1: %.4f  |  Precision: %.4f  |  Recall: %.4f",
        best_threshold, best_f1, best_precision, best_recall,
    )

    # Precision-recall curve plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2, color="steelblue", label="PR curve")
    ax.scatter(
        best_recall, best_precision,
        s=120, color="crimson", zorder=5,
        label=f"Best  threshold={best_threshold:.4f}  F1={best_f1:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall — Isolation Forest (FD001)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plot_path = ARTIFACTS_DIR / "pr_curve.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curve saved → %s", plot_path)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name="isolation_forest_evaluate"):
        mlflow.log_metrics({
            "best_f1": best_f1,
            "best_threshold": best_threshold,
            "best_precision": best_precision,
            "best_recall": best_recall,
            "n_test_windows": len(y_true),
            "n_anomaly_windows": int(y_true.sum()),
        })
        mlflow.log_artifact(str(plot_path))

    logger.info("")
    logger.info("→ Update ANOMALY_THRESHOLD in .env to: %.6f", best_threshold)


if __name__ == "__main__":
    main()
