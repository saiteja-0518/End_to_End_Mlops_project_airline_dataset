✈️ Airline Passenger Satisfaction — End-to-End MLOps & Self-Healing Production SystemAn enterprise-grade, reproducible Machine Learning Operations (MLOps) system engineered to predict Airline Passenger Satisfaction across 129,880 passenger journeys and 22 behavioral & operational features.Rather than stopping at a static Jupyter notebook, this repository implements a closed-loop production lifecycle: cryptographic data/pipeline versioning (DVC), Bayesian hyperparameter search (Optuna), experiment lineage (MLflow), multi-criteria champion selection, quarantined test evaluation, automated unit/integration testing (Pytest), hardened non-root containerization (Docker), CI/CD quality gates (GitHub Actions), and live Kolmogorov-Smirnov (KS) statistical drift detection with self-healing Continuous Training (CT).🏛️ End-to-End MLOps Architectureflowchart LR
    A[(Raw Dataset<br/>129,880 Rows)] --> B[Stage 1: Data Ingestion<br/>Schema Validation]
    B --> C[Stage 2: Preprocessing<br/>Deduplication & Stratified Split]
    C --> D[Stage 3: Feature Engineering<br/>4 Categorical + 18 Numerical]
    D --> E[Stage 4: Model Building<br/>Optuna HPT + MLflow + Cost Rank]
    E --> F[Stage 5: Model Evaluation<br/>Quarantined Test Set]
    F --> G[(models/model.pkl<br/>Champion Pipeline)]
    G --> H[Docker Container<br/>Streamlit 3-Tab Portal]
    H --> I[(inference_logs.csv<br/>26-Column Audit Log)]
    I --> J[KS Drift Monitor<br/>p < 0.05 Statistical Check]
    J -- "Drift >= 50%" --> K[GitHub Issue / Webhook Alert<br/>+ Auto-Retrain CT]
    K -.-> A
🎯 Why This Project Stands Out (Engineering Highlights)Zero Data Leakage via Serialized ColumnTransformer: Missing values (Arrival Delay) and categorical encodings (Gender, Customer Type, Type of Travel, Class) are never fitted on the full dataset. Instead, imputation, scaling, and one-hot encoding live inside the serialized Scikit-Learn Pipeline (models/model.pkl), guaranteeing identical transformations from offline training to live Docker inference.Production-Aware Champion Selection (Not Just Max Accuracy): High accuracy alone can hide bloated models or slow inference. src/model_building.py applies a hard overfitting filter ($\text{Train} - \text{Val Gap} < 5.0\%$), groups finalists within $1.0\%$ of top validation accuracy, and ranks them by a normalized deployment cost:
$$\text{Deployment Cost} = 0.4 \times \text{Size}_{\text{norm}} + 0.3 \times \text{TrainTime}_{\text{norm}} + 0.3 \times \text{Latency}_{\text{norm}}$$Quarantined Test Split Integrity: The test partition (25,976 rows) is strictly isolated until src/model_evaluation.py runs, preventing hyperparameter search from ever peeking at holdout data.Closed-Loop Observability: Every user click in the Streamlit portal writes a canonical 26-column audit record (data/monitoring/inference_logs.csv) that feeds directly into a two-sample Kolmogorov-Smirnov (KS) distribution monitor against the 103,904-row training reference.📊 Dataset & 22-Feature SchemaFeature GroupCountFeatures IncludedPipeline TransformationPassenger & Trip Profile4Gender, Customer Type, Type of Travel, ClassSimpleImputer(most_frequent) $\rightarrow$ OneHotEncoder(handle_unknown="ignore")Demographics & Flight Ops4Age, Flight Distance, Departure Delay, Arrival DelaySimpleImputer(median) $\rightarrow$ StandardScaler()In-Flight & Ground Experience (0–5 Ratings)14Departure and Arrival Time Convenience, Ease of Online Booking, Check-in Service, Online Boarding, Gate Location, On-board Service, Seat Comfort, Leg Room Service, Cleanliness, Food and Drink, In-flight Service, In-flight Wifi Service, In-flight Entertainment, Baggage HandlingSimpleImputer(median) $\rightarrow$ StandardScaler()Target Label1satisfaction (Satisfied = 1, Neutral or Dissatisfied = 0)Stratified 80/20 Train/Test Split📁 Repository ArchitectureEnd_to_End_Mlops_project_airline_dataset/
├── .github/workflows/
│   ├── main.yaml                       # CI/CD: DVC Repro, Pytest, Quality Gate & Docker Smoke Test
│   └── scheduled_drift_retrain.yaml    # CT: Weekly Drift Monitor, Issue Alerts & Auto-Retraining
├── data/
│   ├── raw/                            # Source dataset (airline_passenger_satisfaction.csv)
│   ├── ingested/                       # Schema-validated raw dataset
│   ├── processed/                      # Stratified splits (train.csv: 103,904 | test.csv: 25,976)
│   └── monitoring/                     # Append-only live inference logs (inference_logs.csv)
├── models/
│   └── model.pkl                       # Active champion Scikit-Learn Pipeline
├── reports/
│   ├── metrics.json                    # Flat champion test metrics tracked by DVC
│   ├── model_selection.json            # Champion selection rationale & cost breakdown
│   ├── model_comparison.csv / .md      # Benchmark table across candidate models
│   ├── drift_report.json / .html       # Interactive & structured KS drift diagnostics
│   └── drift_alert.md                  # Auto-generated incident report when drift triggers
├── src/
│   ├── data_ingestion.py               # Stage 1: Raw CSV ingestion & schema validation
│   ├── data_preprocessing.py           # Stage 2: Deduplication & stratified train/test split
│   ├── feature_engineering.py          # Stage 3: Feature isolation (4 categorical + 18 numerical)
│   ├── model_building.py               # Stage 4: Optuna Bayesian HPT, MLflow & champion selection
│   ├── model_evaluation.py             # Stage 5: Independent evaluation on quarantined test set
│   └── monitor_drift.py                # Stage 6: Two-sample KS drift engine & webhook dispatcher
├── tests/
│   ├── test_data.py                    # Data integrity, missing-value tolerance & split checks
│   ├── test_model.py                   # Pipeline structure, batch inference & determinism tests
│   └── test_monitoring.py              # 26-column logging, KS shift detection & alert tests
├── app.py                              # 3-tab Streamlit production web application
├── Dockerfile                          # Hardened Python 3.14 non-root container definition
├── .dockerignore                       # Lean Docker build context filter
├── dvc.yaml                            # Declarative 5-stage DVC pipeline graph
├── dvc.lock                            # Cryptographic MD5 state manifest of pipeline runs
├── params.yaml                         # Centralized hyperparameters, splits & drift thresholds
├── requirements-dvc.txt                # Full training, DVC, Optuna, MLflow & testing dependencies
└── requirements.txt                    # Minimal serving dependencies for Docker runtime
🔍 Deep Dive: Core MLOps Pillars1. Data & Pipeline Versioning (DVC + MLflow)DAG Orchestration (dvc.yaml): Chains all 5 training stages into a directed acyclic graph (data_ingestion $\rightarrow$ data_preprocessing $\rightarrow$ feature_engineering $\rightarrow$ model_building $\rightarrow$ model_evaluation).Hash-Based Caching (dvc.lock): Tracks cryptographic MD5 hashes of datasets, source code, and params.yaml. If you modify only model_evaluation.py, DVC skips data prep and model training in milliseconds.Experiment Tracking: Logs every Optuna hyperparameter trial, cross-validation accuracy, training duration, and model artifact size into MLflow for full auditability.2. Production Data Drift Monitoring (src/monitor_drift.py)26-Column Inference Telemetry: Every prediction in app.py logs timestamp, all 22 input features, predicted_class, prob_Neutral_or_Dissatisfied, and prob_Satisfied to data/monitoring/inference_logs.csv.Two-Sample Kolmogorov-Smirnov (KS) Test: Evaluates whether live production traffic diverges statistically from the 103,904-row training reference (data/processed/train.csv) across all 18 numerical features ($p < 0.05$).Dual-Format Diagnostics & Alerting: Generates reports/drift_report.json for automated CI/CD parsing and reports/drift_report.html for visual inspection. When $\ge 50\%$ of features drift, it drafts reports/drift_alert.md and dispatches notifications via Slack/Discord webhook (ALERT_WEBHOOK_URL) or a GitHub Issue.3. Hardened Production Containerization (Dockerfile)Python 3.14 Binary Compatibility: Built on python:3.14-slim so joblib deserialization of models/model.pkl matches the exact Python and Scikit-Learn runtime used in CI/CD.Split Dependency Architecture: Installs only lean runtime packages from requirements.txt (excluding Optuna, DVC, and Pytest), keeping image size and attack surface minimal.Self-Contained Drift & Lineage: Bundles models/model.pkl, reports/, and data/processed/train.csv inside /app so all 3 Streamlit tabs—including live KS drift checks—work out of the box inside Docker.Zero-Root Execution & Health Probes: Runs Streamlit as an unprivileged user (appuser, UID 1000) with a native HEALTHCHECK polling /_stcore/health every 30 seconds.🚀 Step-by-Step Execution GuideStep 1: Clone Repository & Set Up Virtual Environmentgit clone https://github.com/saiteja-0518/End_to_End_Mlops_project_airline_dataset.git
cd End_to_End_Mlops_project_airline_dataset

python -m venv myenv
# Windows:
myenv\Scripts\activate
# Linux/macOS:
source myenv/bin/activate

pip install -r requirements-dvc.txt
Step 2: Reproduce the 5-Stage DVC Pipelinedvc repro
dvc metrics show
Step 3: Run the Automated Pytest Suite (11 Checks)pytest tests/ -v
Step 4: Run Production Data Drift Diagnostics# Baseline healthy traffic check:
python src/monitor_drift.py --generate-sample-data

# Simulate covariate shift to trigger a drift alert:
python src/monitor_drift.py --simulate-drift
Step 5: Launch the 3-Tab Streamlit Portal Locallystreamlit run app.py
Open http://localhost:8501 to explore:Live Inference: Interactive 22-feature passenger simulator with confidence bar charts and automatic request logging.Model Lineage & Metrics: Active champion KPIs, overfitting gap analysis, and candidate model comparison table.Data Drift & Monitoring: Live production request table and one-click KS drift check against the training distribution.Step 6: Build & Run the Production Docker Containerdocker build -t airline-mlops-app:latest .
docker run -d -p 8501:8501 --name airline-app airline-mlops-app:latest
docker ps
Open http://localhost:8501 once docker ps reports (healthy).⚙️ CI/CD & Self-Healing Continuous Training (GitHub Actions)Workflow FileTriggerAutomated Actions.github/workflows/main.yamlEvery push / PR to main1. Installs dependencies (Python 3.14 + optuna-integration)2. Executes dvc repro to build splits & champion model3. Runs all 11 pytest checks (test_data, test_model, test_monitoring)4. Enforces Quality Gate ($\text{Test Accuracy} \ge 85\%$, $\text{Overfitting Gap} < 5\%$, no fallback)5. Builds Docker image & runs live /_stcore/health container smoke test.github/workflows/scheduled_drift_retrain.y
