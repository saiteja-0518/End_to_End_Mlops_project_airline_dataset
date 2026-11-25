import numpy as np
import pandas as pd

from src.monitor_drift import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    analyze_drift,
    generate_alert_markdown,
    load_inference_logs,
    log_inference,
    send_drift_alert,
)


def test_log_inference_writes_record(tmp_path):
    """Verify that logging inference appends a valid structured Airline record."""
    test_log = tmp_path / "test_logs.csv"
    all_cols = CATEGORICAL_COLUMNS + FEATURE_COLUMNS
    sample_values = [
        "Male",
        "Loyal Customer",
        "Business travel",
        "Business",
        35,
        1200,
        0,
        0.0,
        3,
        3,
        3,
        4,
        3,
        4,
        4,
        4,
        4,
        3,
        4,
        3,
        4,
        4,
    ]
    input_sample = pd.DataFrame([sample_values], columns=all_cols)

    log_inference(
        input_sample,
        1,
        {"Neutral or Dissatisfied": 0.1, "Satisfied": 0.9},
        log_path=test_log,
    )

    loaded = load_inference_logs(test_log)
    assert len(loaded) == 1
    assert int(loaded["predicted_class"].iloc[0]) == 1
    assert np.isclose(loaded["prob_Satisfied"].iloc[0], 0.9)


def test_analyze_drift_no_drift_when_identical():
    """Identical distributions should yield 0% drift."""
    rng = np.random.RandomState(42)
    data = {col: rng.normal(3.0, 1.0, 100) for col in FEATURE_COLUMNS}
    ref_df = pd.DataFrame(data)
    cur_df = ref_df.copy()

    report = analyze_drift(ref_df, cur_df)
    assert report["dataset_drift_detected"] is False
    assert report["drifted_features"] == 0


def test_analyze_drift_detects_distribution_shift():
    """Significantly shifted distributions should trigger dataset drift."""
    rng = np.random.RandomState(42)
    ref_df = pd.DataFrame({col: rng.normal(3.0, 0.5, 100) for col in FEATURE_COLUMNS})
    cur_df = pd.DataFrame({col: rng.normal(10.0, 0.5, 100) for col in FEATURE_COLUMNS})

    report = analyze_drift(ref_df, cur_df)
    assert report["dataset_drift_detected"] is True
    assert report["drifted_features"] == len(FEATURE_COLUMNS)


def test_generate_alert_markdown_creates_valid_report(tmp_path):
    """Verify Markdown alert report generation."""
    md_path = tmp_path / "drift_alert.md"
    dummy_report = {
        "timestamp": "2026-11-25T00:00:00Z",
        "reference_samples": 100,
        "current_samples": 50,
        "total_features": 18,
        "drifted_features": 10,
        "drift_share_pct": 55.56,
        "dataset_drift_detected": True,
        "dataset_drift_threshold_pct": 50.0,
        "features": {
            "Age": {
                "drift_detected": True,
                "p_value": 0.0001,
                "ks_statistic": 0.85,
                "reference_mean": 39.4,
                "reference_std": 14.0,
                "current_mean": 62.0,
                "current_std": 8.0,
                "mean_shift": 22.6,
            }
        },
    }
    md = generate_alert_markdown(dummy_report, output_path=md_path)
    assert md_path.exists()
    assert "DRIFT DETECTED" in md


def test_send_drift_alert_graceful_without_webhook(monkeypatch):
    """Ensure webhook sender exits cleanly when no webhook URL is set."""
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert send_drift_alert({"dataset_drift_detected": True}) is False