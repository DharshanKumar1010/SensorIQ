"""
Train the LSTM Autoencoder on healthy C-MAPSS windows.
Logs per-epoch loss to MLflow, saves the state dict to ml/artifacts/autoencoder.pt,
and saves per-sample reconstruction errors to ml/artifacts/train_errors.npy.

Run from project root:
    python -m ml.train_autoencoder
"""
import copy
import logging
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from ml.model import LSTMAutoencoder

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("ml/artifacts")
MLFLOW_TRACKING_URI = "./mlruns"
MODEL_VERSION = "lstm_autoencoder_v1"

# Hyperparameters
INPUT_SIZE = 21
HIDDEN_SIZE = 128
NUM_LAYERS = 2
SEQ_LEN = 30
DROPOUT = 0.2
EPOCHS = 200
BATCH_SIZE = 64
LR = 5e-4
EARLY_STOPPING_PATIENCE = 15
VAL_SPLIT = 0.1


def _compute_errors(model: LSTMAutoencoder, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Return per-sample MSE reconstruction error (shape: n_windows)."""
    model.eval()
    errors: list[float] = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            mse = ((batch - recon) ** 2).mean(dim=(1, 2))  # (batch_size,)
            errors.extend(mse.cpu().numpy().tolist())
    return np.array(errors, dtype=np.float32)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    X = np.load(PROCESSED_DIR / "X_train.npy")  # (n, 30, 21)
    logger.info("Loaded X_train: %s", X.shape)
    print(f"X_train shape: {X.shape}")

    full_dataset = TensorDataset(torch.from_numpy(X))
    n_val = int(VAL_SPLIT * len(full_dataset))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info("Split: %d train / %d val windows", n_train, n_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    print(f"train_loader batches: {len(train_loader)}")

    model = LSTMAutoencoder(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run(run_name=MODEL_VERSION):
        mlflow.log_params({
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "seq_len": SEQ_LEN,
            "dropout": DROPOUT,
            "epochs_max": EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "val_split": VAL_SPLIT,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "n_train_windows": n_train,
            "n_val_windows": n_val,
            "device": str(device),
        })

        best_val_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        stopped_epoch = EPOCHS

        logger.info("Starting training…")
        print("DEBUG: about to start training loop")
        for epoch in range(1, EPOCHS + 1):
            print(f"DEBUG: epoch {epoch} starting")
            # Training pass
            model.train()
            running_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon = model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(batch)
            train_loss = running_loss / n_train

            # Validation pass
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    recon = model(batch)
                    val_running += criterion(recon, batch).item() * len(batch)
            val_loss = val_running / n_val

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if epoch == 1:
                logger.info(
                    "Epoch %3d/%d — train: %.6f  val: %.6f",
                    epoch, EPOCHS, train_loss, val_loss,
                )
            elif epoch % 10 == 0:
                logger.info(
                    "Epoch %3d/%d — train: %.6f  val: %.6f",
                    epoch, EPOCHS, train_loss, val_loss,
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    stopped_epoch = epoch
                    logger.info(
                        "Early stopping at epoch %d — best val loss: %.6f",
                        epoch, best_val_loss,
                    )
                    break

        # Restore weights from the best epoch before saving
        model.load_state_dict(best_state)
        model.eval()
        logger.info("Restored best weights (epoch where val loss = %.6f)", best_val_loss)
        logger.info(
            "Training complete — %d epochs run, best val loss: %.6f",
            stopped_epoch, best_val_loss,
        )

        mlflow.log_param("stopped_epoch", stopped_epoch)
        mlflow.log_metric("best_val_loss", best_val_loss)

        pt_path = ARTIFACTS_DIR / "autoencoder.pt"
        torch.save(model.state_dict(), pt_path)
        mlflow.log_artifact(str(pt_path), artifact_path="weights")
        mlflow.pytorch.log_model(model, artifact_path="model")

        # Per-sample errors on the training split only (val excluded to avoid threshold leakage)
        train_errors = _compute_errors(model, train_loader, device)
        errors_path = ARTIFACTS_DIR / "train_errors.npy"
        np.save(errors_path, train_errors)
        mlflow.log_artifact(str(errors_path))

        mlflow.log_metrics({
            "train_error_mean": float(train_errors.mean()),
            "train_error_std": float(train_errors.std()),
            "train_error_p95": float(np.percentile(train_errors, 95)),
        })

        run_id = mlflow.active_run().info.run_id
        logger.info("MLflow run_id: %s", run_id)

    logger.info("Saved autoencoder.pt → %s", pt_path)
    logger.info(
        "p95 train reconstruction error: %.6f  (candidate ANOMALY_THRESHOLD for autoencoder)",
        float(np.percentile(train_errors, 95)),
    )


if __name__ == "__main__":
    main()
