"""
Unsupervised evaluation of the LSTM Autoencoder on C-MAPSS FD001 test data.

Portfolio narrative: reconstruction error tracks engine degradation without
any failure labels — the model learned what "healthy" looks like and flags
deviations as the engine approaches failure.

Run from project root:
    python -m ml.evaluate_autoencoder
"""
import logging
import os
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
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# RUL buckets: (low_inclusive, high_inclusive, label, bar_colour)
RUL_BUCKETS = [
    (0,   20,   "0–20",    "crimson"),
    (21,  50,   "21–50",   "orangered"),
    (51,  100,  "51–100",  "darkorange"),
    (101, 150,  "101–150", "yellowgreen"),
    (151, 9999, "150+",    "forestgreen"),
]


def _load_test_windows() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load test_FD001.txt, compute per-row RUL from RUL_FD001.txt ground truth,
    apply the training scaler, and build sliding windows.

    Returns
    -------
    X          — (n, 30, 21) float32 — scaled sensor windows
    ruls       — (n,) float32        — RUL at the last timestep of each window
    unit_ids   — (n,) int32          — engine number for each window
    end_cycles — (n,) int32          — absolute cycle at the last timestep
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
    # rul = cycles remaining in segment + ground-truth RUL after last recorded cycle
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


def _compute_errors(
    model: LSTMAutoencoder, X: np.ndarray, device: torch.device, batch_size: int = 256
) -> np.ndarray:
    """Return per-window MSE reconstruction error, processed in batches."""
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    errors: list[float] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            mse = ((batch - recon) ** 2).mean(dim=(1, 2))
            errors.extend(mse.cpu().numpy().tolist())
    return np.array(errors, dtype=np.float32)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load model ─────────────────────────────────────────────────────────
    pt_path = ARTIFACTS_DIR / "autoencoder.pt"
    model = LSTMAutoencoder(input_size=INPUT_SIZE, hidden_size=128, num_layers=NUM_LAYERS, dropout=0.2)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.to(device)
    logger.info("Loaded autoencoder from %s", pt_path)

    # ── Load ALL test windows with RUL metadata ─────────────────────────────
    logger.info("Building test windows …")
    X_test, ruls, unit_ids, end_cycles = _load_test_windows()
    logger.info(
        "Test windows: %d across %d engines (RUL range: %.0f – %.0f)",
        len(ruls), len(np.unique(unit_ids)), ruls.min(), ruls.max(),
    )

    # ── Reconstruction errors for every window ──────────────────────────────
    errors = _compute_errors(model, X_test, device)

    # ── Threshold: 95th percentile of training reconstruction errors ─────────
    # Using training distribution ensures the threshold is label-free
    train_errors = np.load(ARTIFACTS_DIR / "train_errors.npy")
    threshold = float(np.percentile(train_errors, 95))
    n_flagged = int((errors > threshold).sum())
    logger.info("Threshold (p95 of train errors): %.6f", threshold)
    logger.info(
        "Windows above threshold: %d / %d  (%.1f%%)",
        n_flagged, len(errors), 100 * n_flagged / len(errors),
    )

    # ── Correlation between reconstruction error and RUL ────────────────────
    corr = float(np.corrcoef(ruls, errors)[0, 1])
    logger.info("Pearson correlation (RUL vs reconstruction error): %.4f", corr)
    logger.info("  — negative value expected: higher error when engine is near failure")

    # ── RUL bucket statistics ───────────────────────────────────────────────
    bucket_means: list[float] = []
    bucket_counts: list[int] = []
    for lo, hi, _, _ in RUL_BUCKETS:
        mask = (ruls >= lo) & (ruls <= hi)
        bucket_means.append(float(errors[mask].mean()) if mask.any() else 0.0)
        bucket_counts.append(int(mask.sum()))

    # ── Plot 1: Mean reconstruction error per RUL bucket ────────────────────
    bucket_labels = [b[2] for b in RUL_BUCKETS]
    bucket_colors = [b[3] for b in RUL_BUCKETS]

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    bars = ax1.bar(
        bucket_labels, bucket_means,
        color=bucket_colors, edgecolor="white", linewidth=0.8,
    )
    ax1.axhline(
        threshold, color="black", linestyle="--", linewidth=1.2,
        label=f"Threshold  (p95 train) = {threshold:.4f}",
    )
    # Annotate each bar with its window count
    y_pad = max(bucket_means) * 0.02
    for bar, count in zip(bars, bucket_counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_pad,
            f"n={count}", ha="center", va="bottom", fontsize=8, color="dimgray",
        )
    ax1.set_xlabel("RUL Bucket  (cycles remaining)")
    ax1.set_ylabel("Mean Reconstruction Error  (MSE)")
    ax1.set_title(
        "Reconstruction Error vs. RUL Bucket\n"
        "LSTM Autoencoder — C-MAPSS FD001  |  colour: red = near failure, green = healthy"
    )
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    plot1_path = ARTIFACTS_DIR / "error_by_rul_bucket.png"
    fig1.savefig(plot1_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    logger.info("Saved: %s", plot1_path)

    # ── Plot 2: Reconstruction error over time for 3 individual engines ─────
    # Pick engines with the lowest, median, and highest final RUL so the plot
    # spans the full degradation spectrum
    rul_gt = pd.read_csv(RAW_DIR / "RUL_FD001.txt", header=None, names=["rul_at_end"])
    rul_gt["unit_id"] = range(1, len(rul_gt) + 1)
    rul_gt_sorted = rul_gt.sort_values("rul_at_end").reset_index(drop=True)
    n_engines = len(rul_gt_sorted)
    pick_rows = [0, n_engines // 2, n_engines - 1]  # lowest / median / highest final RUL
    selected = [
        (int(rul_gt_sorted.iloc[i]["unit_id"]), int(rul_gt_sorted.iloc[i]["rul_at_end"]))
        for i in pick_rows
    ]

    fig2, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (unit_id, final_rul) in zip(axes, selected):
        mask = unit_ids == unit_id
        x = end_cycles[mask]
        y = errors[mask]
        order = np.argsort(x)
        x, y = x[order], y[order]

        ax.plot(x, y, linewidth=1.5, color="steelblue")
        ax.fill_between(x, y, alpha=0.15, color="steelblue")
        ax.axhline(
            threshold, color="crimson", linestyle="--", linewidth=1.1,
            label=f"Threshold = {threshold:.3f}",
        )
        ax.set_title(f"Engine {unit_id}\n(final RUL = {final_rul} cycles)")
        ax.set_xlabel("Cycle")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Reconstruction Error  (MSE)")
    axes[0].legend(fontsize=7)
    fig2.suptitle(
        "Reconstruction Error Over Time — 3 Individual Engines\n"
        "Model detects degradation without ever seeing failure labels",
        y=1.02,
    )
    fig2.tight_layout()
    plot2_path = ARTIFACTS_DIR / "degradation_over_time.png"
    fig2.savefig(plot2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Saved: %s", plot2_path)

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("── Evaluation Summary ──────────────────────────────────────")
    logger.info("Threshold (p95 train errors):   %.6f", threshold)
    logger.info("Pearson corr (RUL vs error):    %.4f  (negative = correct)", corr)
    logger.info("Windows flagged as anomalous:   %d / %d  (%.1f%%)",
                n_flagged, len(errors), 100 * n_flagged / len(errors))
    logger.info("Mean error by RUL bucket:")
    for (lo, hi, label, _), mean, count in zip(RUL_BUCKETS, bucket_means, bucket_counts):
        logger.info("  %-10s  %.6f   (n=%d)", label, mean, count)
    logger.info("────────────────────────────────────────────────────────────")
    logger.info("→ Update ANOMALY_THRESHOLD in .env to: %.6f", threshold)

    # ── MLflow ──────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name="lstm_autoencoder_evaluate"):
        mlflow.log_metrics({
            "threshold_p95_train": threshold,
            "pearson_corr_rul_error": corr,
            "n_test_windows": int(len(errors)),
            "n_windows_flagged": n_flagged,
            "pct_windows_flagged": float(100 * n_flagged / len(errors)),
        })
        for (_, _, label, _), mean in zip(RUL_BUCKETS, bucket_means):
            safe = label.replace("–", "_").replace("+", "plus")
            mlflow.log_metric(f"mean_error_rul_{safe}", mean)
        mlflow.log_artifact(str(plot1_path))
        mlflow.log_artifact(str(plot2_path))
    logger.info("MLflow run logged.")


if __name__ == "__main__":
    main()
