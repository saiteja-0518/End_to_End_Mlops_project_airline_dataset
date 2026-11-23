import os
import json
import time
import warnings
import yaml
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import optuna
from optuna.integration.mlflow import MLflowCallback

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def build_preprocessor(num_cols: list, cat_cols: list, scaler_type: str = "standard") -> ColumnTransformer:
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])


def evaluate_cv_with_gap_penalty(pipe, X, y, cv_folds, random_state, sel_cfg):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_validate(
        pipe,
        X,
        y,
        scoring="accuracy",
        cv=skf,
        return_train_score=True,
        n_jobs=-1,
    )
    mean_train = float(np.mean(scores["train_score"]))
    mean_val = float(np.mean(scores["test_score"]))
    gap_pct = (mean_train - mean_val) * 100.0

    objective_val = mean_val
    if sel_cfg.get("gap_penalty_in_search", True):
        max_gap = sel_cfg["max_accuracy_gap_pct"]
        if gap_pct >= max_gap:
            excess = (gap_pct - max_gap) / 100.0
            objective_val -= sel_cfg.get("gap_penalty_weight", 1.0) * excess

    return objective_val, mean_train, mean_val, gap_pct


def build_classifier(model_name: str, model_params: dict, random_state: int):
    if model_name == "KNN":
        return KNeighborsClassifier(**model_params, n_jobs=-1)
    if model_name == "DecisionTree":
        return DecisionTreeClassifier(**model_params, random_state=random_state)
    if model_name == "SVM":
        return SVC(**model_params, max_iter=2000, cache_size=1000, random_state=random_state)
    if model_name == "GaussianNB":
        return GaussianNB(**model_params)
    if model_name == "RandomForest":
        return RandomForestClassifier(**model_params, random_state=random_state, n_jobs=-1)
    if model_name == "GradientBoosting":
        return GradientBoostingClassifier(**model_params, random_state=random_state)
    raise ValueError(f"Unsupported model: {model_name}")


def select_champion(candidates: list, sel_cfg: dict) -> dict:
    max_gap = sel_cfg["max_accuracy_gap_pct"]
    eligible = [c for c in candidates if c["gap_pct"] < max_gap]

    if not eligible:
        fallback = sel_cfg.get("fallback_if_none_eligible", "smallest_gap")
        if fallback == "error":
            raise RuntimeError(f"No model passed max_accuracy_gap_pct < {max_gap}%")
        print(f"WARNING: No model passed gap < {max_gap}%. Falling back to smallest_gap.")
        eligible = [min(candidates, key=lambda c: c["gap_pct"])]

    best_val_acc = max(c["cv_val_accuracy"] for c in eligible)
    tol = sel_cfg["accuracy_tolerance_pct"] / 100.0
    finalists = [c for c in eligible if (best_val_acc - c["cv_val_accuracy"]) <= tol]

    if len(finalists) == 1:
        return finalists[0]

    weights = sel_cfg["weights"]
    criteria = ["model_size_mb", "training_time_seconds", "prediction_time_ms"]
    norms = {k: [] for k in criteria}

    for k in criteria:
        vals = [f[k] for f in finalists]
        lo, hi = min(vals), max(vals)
        denom = (hi - lo) if hi > lo else 1.0
        norms[k] = [(v - lo) / denom for v in vals]

    best_idx = 0
    best_score = float("inf")
    for idx in range(len(finalists)):
        score = sum(weights[k] * norms[k][idx] for k in criteria)
        if score < best_score:
            best_score = score
            best_idx = idx

    return finalists[best_idx]


def main():
    params = load_params()
    sel_cfg = params["selection"]
    meas_cfg = params["measurement"]
    random_state = params["random_state"]
    cv_folds = params["cv_folds"]

    processed_dir = os.path.join("data", "processed")
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze("columns")

    # Fast stratified subset for O(N^2)/O(N^3) hyperparameter tuning (KNN & SVM)
    if len(X_train) > 15000:
        X_tune_heavy, _, y_tune_heavy, _ = train_test_split(
            X_train, y_train, train_size=15000, stratify=y_train, random_state=random_state
        )
    else:
        X_tune_heavy, y_tune_heavy = X_train, y_train

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    os.makedirs("models", exist_ok=True)
    metadata = {
        "feature_columns": X_train.columns.tolist(),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_options": {
            col: sorted(X_train[col].dropna().unique().tolist()) for col in cat_cols
        },
        "num_ranges": {
            col: {
                "min": float(X_train[col].min()),
                "max": float(X_train[col].max()),
                "median": float(X_train[col].median()),
            }
            for col in num_cols
        },
    }
    with open(os.path.join("models", "feature_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    def make_pipe(scaler_type: str, clf):
        return Pipeline([
            ("preprocessor", build_preprocessor(num_cols, cat_cols, scaler_type)),
            ("classifier", clf),
        ])

    def objective_knn(trial):
        st = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
        clf = KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 3, 21, step=2),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            p=trial.suggest_int("p", 1, 2),
        )
        obj, _, _, _ = evaluate_cv_with_gap_penalty(
            make_pipe(st, clf), X_tune_heavy, y_tune_heavy, cv_folds, random_state, sel_cfg
        )
        return obj

    def objective_dt(trial):
        st = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
        clf = DecisionTreeClassifier(
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            max_depth=trial.suggest_int("max_depth", 2, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            random_state=random_state,
        )
        obj, _, _, _ = evaluate_cv_with_gap_penalty(
            make_pipe(st, clf), X_train, y_train, cv_folds, random_state, sel_cfg
        )
        return obj

    def objective_svm(trial):
        st = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
        p = {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "kernel": kernel,
            "max_iter": 2000,
            "cache_size": 1000,
            "random_state": random_state,
        }
        if kernel in ["rbf", "poly", "sigmoid"]:
            p["gamma"] = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
        if kernel == "poly":
            p["degree"] = trial.suggest_int("degree", 2, 5)
        obj, _, _, _ = evaluate_cv_with_gap_penalty(
            make_pipe(st, SVC(**p)), X_tune_heavy, y_tune_heavy, 2, random_state, sel_cfg
        )
        return obj

    def objective_gnb(trial):
        st = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
        clf = GaussianNB(
            var_smoothing=trial.suggest_float("var_smoothing", 1e-11, 1e-7, log=True)
        )
        obj, _, _, _ = evaluate_cv_with_gap_penalty(
            make_pipe(st, clf), X_train, y_train, cv_folds, random_state, sel_cfg
        )
        return obj

    def objective_rf(trial):
        st = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 250, step=50),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            max_depth=trial.suggest_int("max_depth", 5, 25),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            random_state=random_state,
            n_jobs=1,
        )
        obj, _, _, _ = evaluate_cv_with_gap_penalty(
            make_pipe(st, clf), X_train, y_train, cv_folds, random_state, sel_cfg
        )
        return obj

    def objective_gb(trial):
        st = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
        clf = GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 200, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 6),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            random_state=random_state,
        )
        obj, _, _, _ = evaluate_cv_with_gap_penalty(
            make_pipe(st, clf), X_train, y_train, cv_folds, random_state, sel_cfg
        )
        return obj

    objectives = {
        "KNN": objective_knn,
        "DecisionTree": objective_dt,
        "SVM": objective_svm,
        "GaussianNB": objective_gnb,
        "RandomForest": objective_rf,
        "GradientBoosting": objective_gb,
    }

    tracking_uri = params.get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["experiment_name"])

    candidates = []

    for model_name, obj_fn in objectives.items():
        print(f"\n--- Optimizing {model_name} ---")
        is_heavy = model_name in ["KNN", "SVM"]
        trials_count = (
            params.get("n_trials_heavy", 10)
            if is_heavy
            else params["n_trials"]
        )
        X_use, y_use = (X_tune_heavy, y_tune_heavy) if is_heavy else (X_train, y_train)

        with mlflow.start_run(run_name=f"{model_name}_Parent"):
            mlflow_cb = MLflowCallback(
                tracking_uri=tracking_uri,
                metric_name="cv_objective",
                create_experiment=False,
                mlflow_kwargs={"nested": True},
            )
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=random_state),
            )
            study.optimize(obj_fn, n_trials=trials_count, callbacks=[mlflow_cb])

            best_params = study.best_params
            scaler_type = best_params["scaler_type"]
            model_params = {k: v for k, v in best_params.items() if k != "scaler_type"}

            best_pipe = make_pipe(
                scaler_type, build_classifier(model_name, model_params, random_state)
            )
            _, cv_train_acc, cv_val_acc, gap_pct = evaluate_cv_with_gap_penalty(
                best_pipe, X_use, y_use, cv_folds, random_state, sel_cfg
            )

            fit_times = []
            for _ in range(meas_cfg["training_time_repeats"]):
                t0 = time.perf_counter()
                best_pipe.fit(X_use, y_use)
                fit_times.append(time.perf_counter() - t0)
            median_fit_time = float(np.median(fit_times))

            sample_rows = X_train.iloc[: meas_cfg["prediction_n_samples"]]
            pred_times_ms = []
            for _ in range(meas_cfg["prediction_repeats"]):
                t0 = time.perf_counter()
                for idx in range(len(sample_rows)):
                    best_pipe.predict(sample_rows.iloc[[idx]])
                elapsed_ms = ((time.perf_counter() - t0) * 1000.0) / len(sample_rows)
                pred_times_ms.append(elapsed_ms)
            median_pred_ms = float(np.median(pred_times_ms))

            model_path = os.path.join("models", f"{model_name}_airline_model.pkl")
            joblib.dump(best_pipe, model_path)
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

            signature = infer_signature(X_train.head(5), best_pipe.predict(X_train.head(5)))
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_train_accuracy", cv_train_acc)
            mlflow.log_metric("cv_val_accuracy", cv_val_acc)
            mlflow.log_metric("accuracy_gap_pct", gap_pct)
            mlflow.log_metric("training_time_seconds", median_fit_time)
            mlflow.log_metric("prediction_time_ms", median_pred_ms)
            mlflow.log_metric("model_size_mb", model_size_mb)

            mlflow.sklearn.log_model(
                sk_model=best_pipe,
                artifact_path=f"{model_name}_airline_model",
                signature=signature,
                input_example=X_train.head(2),
                serialization_format="cloudpickle",
            )

            candidates.append({
                "model_name": model_name,
                "model_path": model_path,
                "best_params": best_params,
                "cv_train_accuracy": round(cv_train_acc, 4),
                "cv_val_accuracy": round(cv_val_acc, 4),
                "gap_pct": round(gap_pct, 2),
                "training_time_seconds": round(median_fit_time, 4),
                "prediction_time_ms": round(median_pred_ms, 4),
                "model_size_mb": round(model_size_mb, 4),
            })

            print(
                f"{model_name}: CV Val={cv_val_acc:.4f}, Gap={gap_pct:.2f}%, "
                f"Size={model_size_mb:.2f}MB, Fit={median_fit_time:.2f}s, "
                f"Pred={median_pred_ms:.3f}ms/row"
            )

    champion = select_champion(candidates, sel_cfg)
    champion_pipe = joblib.load(champion["model_path"])
    joblib.dump(champion_pipe, os.path.join("models", "model.pkl"))

    os.makedirs("reports", exist_ok=True)
    selection_payload = {
        "champion": champion,
        "selection_rules": sel_cfg,
        "candidates": candidates,
    }
    with open(os.path.join("reports", "model_selection.json"), "w") as f:
        json.dump(selection_payload, f, indent=2)

    comp_df = pd.DataFrame([
        {k: v for k, v in c.items() if k not in ["model_path", "best_params"]}
        for c in candidates
    ]).sort_values(by="cv_val_accuracy", ascending=False)

    comp_df.to_csv(os.path.join("reports", "model_comparison.csv"), index=False)
    with open(os.path.join("reports", "model_comparison.md"), "w") as f:
        f.write("# Model Comparison Report\n\n")
        f.write(f"**Champion Model:** `{champion['model_name']}`\n\n")
        f.write(comp_df.to_markdown(index=False))
        f.write("\n")

    print(
        f"\nChampion selected: {champion['model_name']} "
        f"(CV Val={champion['cv_val_accuracy']:.4f}, Gap={champion['gap_pct']:.2f}%) -> models/model.pkl"
    )


if __name__ == "__main__":
    main()