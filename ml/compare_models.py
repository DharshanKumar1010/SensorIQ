"""
Side-by-side evaluation of Isolation Forest vs. LSTM Autoencoder on C-MAPSS FD001.

For each model computes per-RUL-bucket anomaly rate + mean score and Pearson
correlation between score and RUL.  Results are saved to:
  ml/artifacts/model_comparison.png   — bar charts for both models
  ml/artifacts/eval_summary.json      — machine-readable stats (read by /model-info API)

Run from project root:
    python -m ml.compare_models
"""
import json
import logging
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.model import LSTMAutoencoder

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/cmapss")
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("ml/artifacts")
MLFLOW_TRACKING_URI = "./mlruns"

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit_id", "cycle", "op1", "op2", "op3"] + SENSOR_COLS
WINDOW_SIZE = 30
INPUT_SIZE = 21
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

RUL_BUCKETS = [
    (0,   20,   "0–20"),
    (21,  50,   "21–50"),
    (51,  100,  "51–100"),
    (101, 150,  "101–150"),
    (151, 9999, "150+"),
]
BUCKET_COLORS = ["crimson", "orangered", "darkorange", "yellowgreen", "forestgreen"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_test_windows() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load test_FD001.txt, compute per-row RUL, apply training scaler, build windows.
    Returns (X, ruls, unit_ids, end_cycles).
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

    rul_gt = pd.read_csv(RAW_DIR / "RUL_FD001.txt", header=None, names=["rul_at_end"])
    rul_gt["unit_id"] = range(1, len(rul_gt) + 1)

    max_cycles = test_df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    test_df = test_df.join(max_cycles, on="unit_id")
    test_df = test_df.merge(rul_gt, on="unit_id")
    test_df["rul"] = (test_df["max_cycle"] - test_df["cycle"]) + test_df["rul_at_end"]

    for col, (col_min, col_max) in scaler_params.items():
        denom = col_max - col_min
        test_df[col] = (test_df[col] - col_min) / denom if denom > 0 else 0.0

    X_list, rul_list, uid_list, cycle_list = [], [], [], []
    for unit_id, group in test_df.groupby("unit_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        sensors = group[SENSOR_COLS].to_numpy(dtype=np.float32)
        ruls = group["rul"].to_numpy(dtype=np.float32)
        cycles = group["cycle"].to_numpy(dtype=np.int32)
        n = len(group)
        for start in range(0, n - WINDOW_SIZE + 1):
            end = start + WINDOW_SIZE
            X_list.append(sensors[start:end])
            rul_list.append(ruls[end - 1])
            uid_list.append(unit_id)
            cycle_list.append(cycles[end - 1])

    return (
        np.array(X_list, dtype=np.float32),
        np.array(rul_list, dtype=np.float32),
        np.array(uid_list, dtype=np.int32),
        np.array(cycle_list, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_isolation_forest(
    model, X_flat: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (scores, is_anomaly) for the Isolation Forest."""
    scores = (-model.score_samples(X_flat)).astype(np.float32)
    return scores, scores > threshold


def _score_autoencoder(
    model: LSTMAutoencoder, X: np.ndarray, device: torch.device,
    threshold: float, batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mse_scores, is_anomaly) for the LSTM Autoencoder."""
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    errors: list[float] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            mse = ((batch - recon) ** 2).mean(dim=(1, 2))
            errors.extend(mse.cpu().numpy().tolist())
    scores = np.array(errors, dtype=np.float32)
    return scores, scores > threshold


# ---------------------------------------------------------------------------
# Bucket statistics
# ---------------------------------------------------------------------------

def _bucket_stats(
    scores: np.ndarray, is_anomaly: np.ndarray, ruls: np.ndarray
) -> list[dict]:
    stats = []
    for lo, hi, label in RUL_BUCKETS:
        mask = (ruls >= lo) & (ruls <= hi)
        if not mask.any():
            stats.append({"bucket": label, "mean_score": 0.0, "anomaly_rate": 0.0, "n_windows": 0})
        else:
            stats.append({
                "bucket": label,
                "mean_score": float(scores[mask].mean()),
                "anomaly_rate": float(is_anomaly[mask].mean()),
                "n_windows": int(mask.sum()),
            })
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(if_stats: list[dict], ae_stats: list[dict]) -> None:
    header = f"{'RUL Bucket':<12}  {'IF Score':>12}  {'IF Anom%':>9}  {'AE Score':>12}  {'AE Anom%':>9}"
    divider = "─" * len(header)
    logger.info(divider)
    logger.info(header)
    logger.info(divider)
    for i, (_, _, label) in enumerate(RUL_BUCKETS):
        i_s = if_stats[i]
        a_s = ae_stats[i]
        logger.info(
            "%-12s  %12.6f  %8.1f%%  %12.6f  %8.1f%%",
            label,
            i_s["mean_score"], 100 * i_s["anomaly_rate"],
            a_s["mean_score"], 100 * a_s["anomaly_rate"],
        )
    logger.info(divider)


def _plot(
    if_stats: list[dict], ae_stats: list[dict],
    if_threshold: float, ae_threshold: float,
    if_corr: float, ae_corr: float,
) -> Path:
    labels = [b[2] for b in RUL_BUCKETS]
    if_means = [s["mean_score"] for s in if_stats]
    ae_means = [s["mean_score"] for s in ae_stats]
    if_rates = [100 * s["anomaly_rate"] for s in if_stats]
    ae_rates = [100 * s["anomaly_rate"] for s in ae_stats]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: IF mean score per bucket
    ax = axes[0]
    bars = ax.bar(labels, if_means, color=BUCKET_COLORS, edgecolor="white")
    ax.axhline(if_threshold, color="black", linestyle="--", linewidth=1.2,
               label=f"Threshold (p95) = {if_threshold:.4f}")
    for bar, s in zip(bars, if_stats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(if_means) * 0.02,
                f"n={s['n_windows']}", ha="center", va="bottom", fontsize=7, color="dimgray")
    ax.set_title(f"Isolation Forest\nMean Anomaly Score per RUL Bucket\nPearson r = {if_corr:.3f}")
    ax.set_ylabel("Mean Anomaly Score (higher = more anomalous)")
    ax.set_xlabel("RUL Bucket")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Middle: AE mean score per bucket
    ax = axes[1]
    bars = ax.bar(labels, ae_means, color=BUCKET_COLORS, edgecolor="white")
    ax.axhline(ae_threshold, color="black", linestyle="--", linewidth=1.2,
               label=f"Threshold (p95) = {ae_threshold:.4f}")
    for bar, s in zip(bars, ae_stats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ae_means) * 0.02,
                f"n={s['n_windows']}", ha="center", va="bottom", fontsize=7, color="dimgray")
    ax.set_title(f"LSTM Autoencoder\nMean Reconstruction Error per RUL Bucket\nPearson r = {ae_corr:.3f}")
    ax.set_ylabel("Mean Reconstruction Error (MSE)")
    ax.set_xlabel("RUL Bucket")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Right: Anomaly rate comparison (grouped bars)
    ax = axes[2]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, if_rates, width, label="Isolation Forest", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, ae_rates, width, label="LSTM Autoencoder", color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Anomaly Rate (%) per RUL Bucket\nBoth Models")
    ax.set_ylabel("Anomaly Rate (%)")
    ax.set_xlabel("RUL Bucket")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Model Comparison: Isolation Forest vs. LSTM Autoencoder — C-MAPSS FD001\n"
        "Colour: red = near failure (RUL 0–20), green = healthy (RUL 150+)",
        y=1.02,
    )
    fig.tight_layout()
    out_path = ARTIFACTS_DIR / "model_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load artifacts
    logger.info("Loading models …")
    with open(ARTIFACTS_DIR / "isolation_forest.pkl", "rb") as f:
        if_model = pickle.load(f)

    ae_model = LSTMAutoencoder(
        input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, dropout=DROPOUT,
    )
    ae_model.load_state_dict(
        torch.load(ARTIFACTS_DIR / "autoencoder.pt", map_location=device)
    )
    ae_model.to(device)

    # Thresholds: p95 of each model's training-set scores
    logger.info("Computing thresholds from training data …")
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    if_threshold = float(np.percentile(-if_model.score_samples(X_train_flat), 95))
    logger.info("IF threshold (p95 train):  %.6f", if_threshold)

    ae_train_errors = np.load(ARTIFACTS_DIR / "train_errors.npy")
    ae_threshold = float(np.percentile(ae_train_errors, 95))
    logger.info("AE threshold (p95 train):  %.6f", ae_threshold)

    # Load test windows
    logger.info("Building test windows …")
    X_test, ruls, unit_ids, _ = _load_test_windows()
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    logger.info("Test windows: %d across %d engines", len(ruls), len(np.unique(unit_ids)))

    # Score both models
    logger.info("Scoring with Isolation Forest …")
    if_scores, if_anomaly = _score_isolation_forest(if_model, X_test_flat, if_threshold)

    logger.info("Scoring with LSTM Autoencoder …")
    ae_scores, ae_anomaly = _score_autoencoder(ae_model, X_test, device, ae_threshold)

    # Pearson correlations
    if_corr = float(np.corrcoef(ruls, if_scores)[0, 1])
    ae_corr = float(np.corrcoef(ruls, ae_scores)[0, 1])
    logger.info("Pearson corr (RUL vs score):  IF = %.4f   AE = %.4f", if_corr, ae_corr)
    logger.info("  — negative = model score rises as RUL falls (correct)")

    # Bucket stats
    if_stats = _bucket_stats(if_scores, if_anomaly, ruls)
    ae_stats = _bucket_stats(ae_scores, ae_anomaly, ruls)

    # Print comparison table
    logger.info("")
    _print_table(if_stats, ae_stats)
    logger.info(
        "Overall anomaly rate:  IF = %.1f%%   AE = %.1f%%",
        100 * if_anomaly.mean(), 100 * ae_anomaly.mean(),
    )
    logger.info("")

    # Plot
    plot_path = _plot(if_stats, ae_stats, if_threshold, ae_threshold, if_corr, ae_corr)
    logger.info("Saved: %s", plot_path)

    # eval_summary.json — read by the /model-info API endpoint
    summary = {
        "autoencoder": {
            "model_version": "lstm_autoencoder_v2",
            "threshold": ae_threshold,
            "pearson_corr": ae_corr,
            "rul_bucket_stats": ae_stats,
        },
        "isolation_forest": {
            "model_version": "isolation_forest_v1",
            "threshold": if_threshold,
            "pearson_corr": if_corr,
            "rul_bucket_stats": if_stats,
        },
    }
    summary_path = ARTIFACTS_DIR / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved: %s", summary_path)

    # MLflow — one run per model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run(run_name="isolation_forest_comparison"):
        mlflow.log_metric("pearson_corr_rul_score", if_corr)
        mlflow.log_metric("threshold_p95", if_threshold)
        mlflow.log_metric("overall_anomaly_rate", float(if_anomaly.mean()))
        for s in if_stats:
            safe = s["bucket"].replace("–", "_").replace("+", "plus")
            mlflow.log_metric(f"mean_score_rul_{safe}", s["mean_score"])
            mlflow.log_metric(f"anomaly_rate_rul_{safe}", s["anomaly_rate"])
        mlflow.log_artifact(str(plot_path))

    with mlflow.start_run(run_name="autoencoder_comparison"):
        mlflow.log_metric("pearson_corr_rul_score", ae_corr)
        mlflow.log_metric("threshold_p95", ae_threshold)
        mlflow.log_metric("overall_anomaly_rate", float(ae_anomaly.mean()))
        for s in ae_stats:
            safe = s["bucket"].replace("–", "_").replace("+", "plus")
            mlflow.log_metric(f"mean_score_rul_{safe}", s["mean_score"])
            mlflow.log_metric(f"anomaly_rate_rul_{safe}", s["anomaly_rate"])
        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(summary_path))

    logger.info("MLflow runs logged.")


if __name__ == "__main__":
    main()
