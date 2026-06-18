# SensorIQ — Sensor Anomaly Detection Platform
## Project context
SensorIQ is a predictive-maintenance SaaS portfolio project. It monitors industrial/IoT sensor
telemetry (vibration, temperature, pressure) and flags anomalous readings before equipment failure
— without requiring labeled failure data.

**Developer:** Dharshan  
**OS:** Windows 11, Lenovo laptop  
**Project path:** `C:\Users\LENOVO\OneDrive\Pictures\Documents\SensorIQ`  
**Python:** 3.10+ (use `venv\Scripts\activate` to activate venv on Windows cmd.exe)  
**Related project:** ChurnIQ (FastAPI + Supabase + React — already built)

---

## Architecture

```
SensorIQ/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Settings via pydantic-settings
│   ├── database.py          # Async SQLAlchemy + Supabase
│   ├── models/
│   │   ├── sensor.py        # SQLAlchemy ORM models
│   │   └── user.py          # Auth user model
│   ├── schemas/
│   │   ├── sensor.py        # Pydantic request/response schemas
│   │   └── auth.py          # Auth schemas
│   ├── routers/
│   │   ├── auth.py          # /auth/register, /auth/login, /auth/refresh
│   │   ├── sensors.py       # /sensors CRUD
│   │   ├── ingest.py        # /ingest — POST telemetry batches
│   │   └── anomalies.py     # /anomalies — GET scored readings
│   └── services/
│       ├── anomaly.py       # Loads model, runs inference, returns score
│       └── auth.py          # JWT logic
├── ml/
│   ├── preprocess.py        # Load C-MAPSS, build sliding windows
│   ├── train_isolation.py   # Isolation Forest baseline
│   ├── train_autoencoder.py # LSTM Autoencoder (PyTorch)
│   ├── evaluate.py          # Isolation Forest precision/recall threshold tuning
│   ├── evaluate_autoencoder.py  # Autoencoder RUL-bucket evaluation + degradation plots
│   ├── compare_models.py    # Side-by-side IF vs AE evaluation; saves eval_summary.json
│   └── serve.py             # MLflow model loading helpers
├── data/
│   ├── raw/cmapss/          # Downloaded NASA C-MAPSS files
│   └── processed/           # Windowed numpy arrays
├── notebooks/
│   ├── 01_eda.ipynb         # Explore C-MAPSS
│   ├── 02_baseline.ipynb    # Isolation Forest experiments
│   └── 03_autoencoder.ipynb # LSTM Autoencoder experiments
├── alembic/                 # DB migrations
├── frontend/                # React dashboard (Week 6)
├── .env                     # Environment variables (never commit)
├── requirements.txt
└── CLAUDE.md                # This file
```

---

## ML approach

### Why unsupervised?
Real industrial datasets rarely have labeled failure events. SensorIQ trains only on **healthy
operating data** and learns what "normal" looks like. Anomaly score = reconstruction error.

### Model 1 — Isolation Forest (baseline, Week 3)
- Fast to train, no deep learning required
- Good for tabular/low-dimension sensor features
- Scikit-learn `IsolationForest`
- Output: anomaly score per reading (-1/+1 label + raw score)

### Model 2 — LSTM Autoencoder (core, Week 4) ✅
- Encoder: LSTM(hidden_size=128, num_layers=2, dropout=0.2) compresses a time window into a latent vector
- Decoder: LSTM reconstructs the original sequence; anomaly score = MSE between input and reconstruction
- Higher reconstruction error = more anomalous; trained only on healthy data (RUL > 50)
- Training: 200 epochs max, early stopping patience=15, 90/10 train/val split, LR=0.0005
- **Key result:** RUL 0–20 mean error = 0.00715 vs RUL 51–100 mean error = 0.00545 — model captures degradation without labels
- Anomaly flag rate on test set: 30.8% of windows flagged (concentrated in low-RUL windows)

### Threshold tuning
- Unsupervised: threshold = 95th percentile of training reconstruction errors (no labels needed)
- Current threshold: **0.005618** — set as `ANOMALY_THRESHOLD` in `.env`
- Evaluation plots: `ml/artifacts/error_by_rul_bucket.png`, `ml/artifacts/degradation_over_time.png`
- Isolation Forest baseline uses PR curve + F1 peak (see `ml/evaluate.py`)

---

## Dataset — NASA C-MAPSS

Files: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`

Columns (space-separated, no header):
```
unit_id  cycle  op1  op2  op3  s1..s21
```

**Healthy = RUL > 50.** Compute RUL per unit as `max_cycle - current_cycle`. Filter training
data to `RUL > 50` for autoencoder training.

Window size: 30 cycles. Stride: 1 cycle. Normalise each sensor column to [0,1] per training set.

---

## Database schema (Supabase/PostgreSQL)

```sql
-- Users (from ChurnIQ pattern — reuse the same auth logic)
users (id uuid PK, email text UNIQUE, hashed_password text, created_at timestamptz)

-- Sensor assets being monitored
assets (id uuid PK, user_id uuid FK, name text, asset_type text, created_at timestamptz)

-- Raw telemetry readings
readings (
  id          uuid PK,
  asset_id    uuid FK,
  timestamp   timestamptz,
  cycle       int,
  sensor_data jsonb,     -- {s1: 1.2, s2: 0.8, ...}
  created_at  timestamptz
)

-- Anomaly scores for each reading
anomaly_scores (
  id            uuid PK,
  reading_id    uuid FK,
  model_version text,
  score         float,      -- reconstruction error or isolation score
  is_anomaly    boolean,    -- score > threshold
  created_at    timestamptz
)
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /auth/register | Register user |
| POST | /auth/login | Login, return JWT |
| GET  | /auth/me | Current user |
| GET  | /assets | List user's monitored assets |
| POST | /assets | Create asset |
| POST | /ingest/{asset_id} | Ingest batch of sensor readings |
| GET  | /anomalies/{asset_id} | Get scored readings with flags |
| GET  | /anomalies/{asset_id}/summary | Stats: anomaly rate, recent alerts |
| GET  | /anomalies/{asset_id}/model-info | Active model type, version, threshold, RUL bucket stats |
| GET  | /health | Health check |

---

## Environment variables (.env)

```
DATABASE_URL=postgresql+asyncpg://postgres:<password>@db.<project>.supabase.co:5432/postgres
SECRET_KEY=<random-32-char-string>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
MODEL_PATH=./ml/artifacts/autoencoder.pt
ISOLATION_MODEL_PATH=./ml/artifacts/isolation_forest.pkl
ANOMALY_THRESHOLD=0.005618
MLFLOW_TRACKING_URI=./mlruns
MODEL_VERSION=lstm_autoencoder_v2
```

---

## Coding conventions

- **Python:** async/await throughout FastAPI. Type hints everywhere. Pydantic v2.
- **No print statements** — use Python `logging` module.
- **Error handling:** raise `HTTPException` with clear status codes and messages.
- **Never hardcode credentials** — always read from `.env` via `Settings`.
- **Windows paths:** use `pathlib.Path` not raw strings.
- **ML artifacts:** always save/load via MLflow. Never load a raw `.pt` file directly in the API
  without checking MLflow first.
- **Tests:** add a `tests/` folder with at least one test per router using `httpx.AsyncClient`.

---

## Week-by-week roadmap

| Week | Focus | Done? |
|------|-------|-------|
| 1 | FastAPI skeleton: auth endpoints, health check, Swagger working | ✅ |
| 2 | Supabase connection, Alembic migrations, asset + reading CRUD | ⬜ |
| 3 | ETL pipeline for C-MAPSS, Isolation Forest baseline, /ingest + /anomalies endpoints | ⬜ |
| 4 | LSTM Autoencoder training, MLflow tracking, swap model in API | ✅ |
| 5 | Threshold tuning, precision/recall analysis, model versioning | ✅ |
| 6 | React dashboard: sensor timeline, anomaly flags, metrics panel | ⬜ |

---

## Portfolio framing

- **Ather:** "Detects abnormal motor/battery sensor patterns before EV component failure"
- **GE Aerospace:** "Reconstruction-error anomaly scoring on turbofan sensor streams (NASA C-MAPSS)"
- **IBM:** "Unsupervised time-series anomaly detection pipeline with FastAPI serving layer"
- **Dentsu:** "End-to-end MLOps: data ingestion → model training → versioning → live API"

Key differentiator to highlight: **no labeled failure data needed.** Define normal, catch deviations.
