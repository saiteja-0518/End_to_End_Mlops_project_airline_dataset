# ✈️ Airline Passenger Satisfaction — End-to-End MLOps Pipeline & Production System

An enterprise-ready, reproducible Machine Learning Operations (MLOps) system built to predict **Airline Passenger Satisfaction** across **129,880 passenger records** and **22 operational & service features**. The project demonstrates the complete production ML lifecycle: DVC data & pipeline versioning, multi-model Bayesian hyperparameter search (Optuna), MLflow experiment tracking, multi-criteria champion selection, quarantined test evaluation, hardened non-root Docker serving, GitHub Actions CI/CD quality gates, and live Kolmogorov-Smirnov (KS) data drift monitoring with automated continuous retraining (CT).

---

## 🏗️ End-to-End Pipeline Flow

```text
[Raw CSV: 129,880 rows]
        │
        ▼
1. Data Ingestion (src/data_ingestion.py) ──► Schema validation & ID cleanup
        │
        ▼
2. Preprocessing (src/data_preprocessing.py) ──► Deduplication & Stratified 80/20 Split
        │                                          (Train: 103,904 | Test: 25,976)
        ▼
3. Feature Engineering (src/feature_engineering.py) ──► 4 Categorical + 18 Numerical Features
        │
        ▼
4. Model Building (src/model_building.py) ──► Optuna Tuning + MLflow + Multi-Criteria Selection
        │
        ▼
5. Model Evaluation (src/model_evaluation.py) ──► Independent evaluation on quarantined test set
        │
        ▼
[models/model.pkl] ──► Docker Container (Streamlit 3-Tab App on Port 8501)
                              │
                              ▼
                  [data/monitoring/inference_logs.csv]
                              │
                              ▼
6. KS Drift Monitor (src/monitor_drift.py)
                              │
                   (If Drift >= 50% Features)
                              ▼
             GitHub Issue / Webhook Alert + Auto-Retrain CT
```

---

## 📁 Repository Directory Structure

```text
├── .dvc/                                   # DVC internal configuration and cache tracking
├── .github/workflows/
│   ├── main.yaml                           # CI/CD: DVC Repro, Pytest, Quality Gate, Docker Smoke Test
│   └── scheduled_drift_retrain.yaml        # CT: Scheduled Drift Monitor, Alerts & Auto-Retrain
├── data/
│   ├── raw/                                # Source raw dataset (airline_passenger_satisfaction.csv)
│   ├── ingested/                           # Ingested and validated raw data
│   ├── processed/                          # Train/Test splits (train.csv, test.csv)
│   └── monitoring/                         # Append-only production inference logs (inference_logs.csv)
├── models/                                 # Trained candidate models & active champion model
│   └── model.pkl                           # Active champion Scikit-Learn Pipeline used in production
├── reports/                                # Generated audit, metrics, and monitoring reports
│   ├── drift_alert.md                      # Auto-generated Markdown incident report for alerts
│   ├── drift_report.html                   # Standalone visual HTML data drift dashboard
│   ├── drift_report.json                   # Structured Kolmogorov-Smirnov drift statistics
│   ├── evaluation_details.json             # Per-class classification reports and benchmarks
│   ├── metrics.json                        # Flat metrics tracked by DVC
│   ├── model_comparison.csv                # Benchmark metrics across candidate models
│   ├── model_comparison.md                 # Markdown benchmark table (published to CI summary)
│   └── model_selection.json                # Champion selection rationale and scoring weights
├── src/                                    # Modular pipeline source code
│   ├── data_ingestion.py                   # Ingests raw CSV and validates 23-column schema
│   ├── data_preprocessing.py               # Deduplication and stratified 80/20 train/test split
│   ├── feature_engineering.py              # Isolates 22 features (defers transforms to Pipeline)
│   ├── model_building.py                   # Optuna HPT, 5-fold CV, multi-criteria champion selection
│   ├── model_evaluation.py                 # Independent evaluation on quarantined test set
│   └── monitor_drift.py                    # Inference logging, KS drift test, and alert dispatcher
├── tests/                                  # Automated test suite (pytest)
│   ├── test_data.py                        # Data validation, schema, and split checks
│   ├── test_model.py                       # Model loading, inference, probabilities, determinism
│   └── test_monitoring.py                  # Payload logging, KS drift test, and alert tests
├── .dockerignore                           # Optimized Docker build context exclusions
├── .gitignore                              # Git exclusions
├── app.py                                  # 3-tab Streamlit production web application
├── Dockerfile                              # Python 3.14 non-root production container
├── dvc.lock                                # Cryptographic manifest of DVC stage executions
├── dvc.yaml                                # Declarative 5-stage DVC pipeline definition
├── params.yaml                             # Centralized configuration parameters
├── requirements-dvc.txt                    # Training, testing, DVC, Optuna, and MLflow dependencies
└── requirements.txt                        # Minimal serving dependencies for Docker runtime
```

---

## ⚡ Core MLOps Pillars in This Project

### 1. Data & Pipeline Versioning (`DVC`)
The 5-stage ML workflow is declared in `dvc.yaml` and locked via MD5 hashes in `dvc.lock`:
- **Zero-Redundancy Caching:** Running `dvc repro` hashes every input file, script, and `params.yaml` hyperparameter. Unchanged stages are skipped instantly.
- **Zero Data Leakage:** Missing value imputation (`Arrival Delay`), standard scaling (18 numerical columns), and one-hot encoding (4 categorical columns) are encapsulated inside a Scikit-Learn `ColumnTransformer` within `models/model.pkl`—ensuring training and live Docker inference apply identical transformations.
- **Quarantined Test Set:** The 25,976-row test split is completely untouched during cross-validation and hyperparameter tuning, used strictly once during `model_evaluation.py`.

### 2. Multi-Model Bayesian Search & Champion Selection
`src/model_building.py` tunes candidate models using **Optuna (TPE Sampler)** with **MLflow** experiment logging and selects the production champion using explicit engineering constraints (`params.yaml`):
1. **Overfitting Guardrail:** Any candidate with `Train Accuracy - Validation Accuracy >= 5.0%` is disqualified.
2. **Accuracy Tolerance Window:** All candidates within `1.0%` validation accuracy of the top model qualify as finalists.
3. **Production Cost Ranking:** Finalists are ranked by a normalized deployment cost combining model file size (`40%`), training time (`30%`), and single-row inference latency (`30%`).

### 3. Production Data Drift Monitoring (`Kolmogorov-Smirnov Test`)
Every prediction made through `app.py` is appended to `data/monitoring/inference_logs.csv` with a 26-column audit schema (UTC timestamp, 22 passenger inputs, predicted class, and class probabilities):
- **Two-Sample KS Test:** `src/monitor_drift.py` compares live production traffic against the training baseline (`data/processed/train.csv`) across all **18 numerical features** (`p < 0.05` flags feature drift).
- **Dataset Drift Threshold:** If `>= 50%` of features drift simultaneously, dataset-level drift is triggered, generating `reports/drift_report.json`, `reports/drift_report.html`, and an incident summary in `reports/drift_alert.md`.

### 4. Hardened Containerization (`Docker`)
The `Dockerfile` builds a lightweight, security-hardened production container:
- **Base Image:** `python:3.14-slim` with minimal runtime packages (`requirements.txt`).
- **Non-Root Security:** Runs under an unprivileged `appuser` (UID `1000`) instead of `root`.
- **Self-Healing Healthcheck:** Built-in Docker `HEALTHCHECK` polls `http://localhost:8501/_stcore/health` every 30 seconds so orchestrators automatically detect and restart unhealthy instances.

### 5. CI/CD & Continuous Training (`GitHub Actions`)
- **CI/CD Pipeline (`.github/workflows/main.yaml`):** Runs on every push/PR to `main`—executes `dvc repro`, runs all **11 `pytest` tests**, enforces automated accuracy & overfitting quality gates, builds the Docker image, and verifies container health.
- **Scheduled Continuous Training (`.github/workflows/scheduled_drift_retrain.yaml`):** Runs weekly to inspect live drift logs; if drift is detected, it opens a GitHub Incident Issue, dispatches webhook alerts, retrains the champion model via `dvc repro`, and commits updated artifacts.

---

## 📊 Dataset Features (22 Inputs + 1 Target)

| Category | Features | Count |
| :--- | :--- | :---: |
| **Categorical** | `Gender`, `Customer Type`, `Type of Travel`, `Class` | 4 |
| **Flight & Demographics** | `Age`, `Flight Distance`, `Departure Delay`, `Arrival Delay` | 4 |
| **Service Ratings (0–5)** | `Departure and Arrival Time Convenience`, `Ease of Online Booking`, `Check-in Service`, `Online Boarding`, `Gate Location`, `On-board Service`, `Seat Comfort`, `Leg Room Service`, `Cleanliness`, `Food and Drink`, `In-flight Service`, `In-flight Wifi Service`, `In-flight Entertainment`, `Baggage Handling` | 14 |
| **Target** | `satisfaction` (`Neutral or Dissatisfied` = `0`, `Satisfied` = `1`) | 1 |

---

## 🚀 Step-by-Step Quickstart Guide

### Step 1: Clone Repository & Set Up Virtual Environment
```bash
git clone [https://github.com/saiteja-0518/End_to_End_Mlops_project_airline_dataset.git](https://github.com/saiteja-0518/End_to_End_Mlops_project_airline_dataset.git)
cd End_to_End_Mlops_project_airline_dataset

python -m venv myenv
# Windows:
myenv\Scripts\activate
# Linux/macOS:
source myenv/bin/activate

pip install -r requirements-dvc.txt
```

### Step 2: Reproduce the Full DVC Pipeline
```bash
dvc repro
dvc metrics show
```

### Step 3: Run the Automated Pytest Suite (11 Tests)
```bash
pytest tests/ -v
```

### Step 4: Run Production Data Drift Analysis
```bash
# Run baseline drift check:
python src/monitor_drift.py --generate-sample-data

# Simulate distribution shift (triggers alert report):
python src/monitor_drift.py --simulate-drift
```

### Step 5: Launch the Streamlit Web App Locally
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser:
- **Live Inference:** Interactive 22-feature passenger form with real-time probability chart and automatic request logging.
- **Model Lineage & Metrics:** Champion performance summary and candidate model comparison table.
- **Data Drift & Monitoring:** One-click live Kolmogorov-Smirnov drift audit against `data/processed/train.csv`.

### Step 6: Build & Run with Docker
```bash
docker build -t airline-mlops-app:latest .
docker run -d -p 8501:8501 --name airline-app airline-mlops-app:latest
docker ps
```
Once `docker ps` shows `(healthy)`, open `http://localhost:8501` in your browser.
