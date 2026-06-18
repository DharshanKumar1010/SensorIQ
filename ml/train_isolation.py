"""
Train an Isolation Forest on healthy C-MAPSS windows.
Logs parameters and the model artifact to MLflow, and saves a local pkl
to ml/artifacts/isolation_forest.pkl for API serving.

Run from project root:
    python -m ml.train_isolation
"""
import logging
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("ml/artifacts")
MLFLOW_TRACKING_URI = "./mlruns"
MODEL_VERSION = "isolation_forest_v1"
CONTAMINATION = 0.05
N_ESTIMATORS = 100
RANDOM_STATE = 42


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    # Flatten windows: (n_windows, 30, 21) → (n_windows, 630)
    X_flat = X_train.reshape(X_train.shape[0], -1)
    n_windows, n_features = X_flat.shape
    logger.info("Training on %d windows × %d features", n_windows, n_features)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    with mlflow.start_run(run_name=MODEL_VERSION):
        mlflow.log_params({
            "contamination": CONTAMINATION,
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "n_windows": n_windows,
            "window_size": X_train.shape[1],
            "n_sensors": X_train.shape[2],
        })

        model = IsolationForest(
            contamination=CONTAMINATION,
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
        )
        model.fit(X_flat)

        # score_samples returns lower values for anomalies; negate so higher = more anomalous
        train_scores = -model.score_samples(X_flat)
        mlflow.log_metrics({
            "train_score_mean": float(train_scores.mean()),
            "train_score_std": float(train_scores.std()),
            "train_score_p95": float(np.percentile(train_scores, 95)),
        })

        mlflow.sklearn.log_model(model, artifact_path="model")

        pkl_path = ARTIFACTS_DIR / "isolation_forest.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(str(pkl_path), artifact_path="pkl")

        run_id = mlflow.active_run().info.run_id
        logger.info("MLflow run_id: %s", run_id)

    logger.info("Saved isolation_forest.pkl → %s", pkl_path)
    logger.info("p95 train anomaly score: %.6f  (use as a starting ANOMALY_THRESHOLD)", float(np.percentile(train_scores, 95)))


if __name__ == "__main__":
    main()
