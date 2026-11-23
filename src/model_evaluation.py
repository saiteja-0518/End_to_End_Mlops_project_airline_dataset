import os
import json
import yaml
import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    processed_dir = os.path.join("data", "processed")

    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze("columns")

    model = joblib.load(os.path.join("models", "model.pkl"))
    with open(os.path.join("reports", "model_selection.json"), "r") as f:
        selection = json.load(f)
    champion = selection["champion"]

    y_pred = model.predict(X_test)
    roc_auc = (
        float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "champion_model": champion["model_name"],
        "cv_val_accuracy": champion["cv_val_accuracy"],
        "accuracy_gap_pct": champion["gap_pct"],
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "test_precision": round(float(precision_score(y_test, y_pred)), 4),
        "test_recall": round(float(recall_score(y_test, y_pred)), 4),
        "test_f1": round(float(f1_score(y_test, y_pred)), 4),
        "test_roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
    }

    details = {
        "champion": champion,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join("reports", "evaluation_details.json"), "w") as f:
        json.dump(details, f, indent=2)

    tracking_uri = params.get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["experiment_name"])
    with mlflow.start_run(run_name=f"Champion_Evaluation_{champion['model_name']}"):
        mlflow.log_metrics({
            k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None
        })

    print("Evaluation complete:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()  