from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.monitor_drift import (
    FEATURE_COLUMNS,
    analyze_drift,
    generate_alert_markdown,
    log_inference,
    send_drift_alert,
)


def test_log_inference_writes_record(tmp_path):
    """Verify that logging inference appends a valid structured record."""
    test_log = tmp_path / "test_logs.csv"
    input_sample = pd.DataFrame([[5.5, 3.0, 4.0, 1.3]], columns=FEATURE_COLUMNS)

    log_inference(input_sample, "Iris-versicolor", {"Setosa": 0.05, "Versicolor": 0.90, "Virginica": 0.05}, log_path=test_log)

    assert test_log.exists()
    logged_df = pd.read_csv(test_log)
    assert len(logged_df) == 1
    assert logged_df.iloc[0]["predicted_class"] == "Iris-versicolor"
    assert "timestamp" in logged_df.columns
    assert logged_df.iloc[0]["prob_Versicolor"] == 0.90


def test_analyze_drift_no_drift_when_identical():
    """Verify that identical distributions report no drift."""
    rng = np.random.RandomState(42)
    data = {col: rng.normal(5.0, 1.0, 100) for col in FEATURE_COLUMNS}
    ref_df = pd.DataFrame(data)
    curr_df = pd.DataFrame(data)

    report = analyze_drift(ref_df, curr_df)
    assert report["dataset_drift_detected"] is False
    assert report["drifted_features"] == 0
    assert report["drift_share_pct"] == 0.0


def test_analyze_drift_detects_distribution_shift():
    """Verify that significant shifts are detected by the KS test."""
    rng = np.random.RandomState(42)
    ref_data = {col: rng.normal(5.0, 0.5, 150) for col in FEATURE_COLUMNS}
    # Current data shifted by 5 standard deviations
    curr_data = {col: rng.normal(15.0, 0.5, 150) for col in FEATURE_COLUMNS}

    ref_df = pd.DataFrame(ref_data)
    curr_df = pd.DataFrame(curr_data)

    report = analyze_drift(ref_df, curr_df)
    assert report["dataset_drift_detected"] is True
    assert report["drifted_features"] == len(FEATURE_COLUMNS)


def test_generate_alert_markdown_creates_valid_report(tmp_path):
    """Verify that alert markdown is written with all tables and headers."""
    alert_path = tmp_path / "test_alert.md"
    rng = np.random.RandomState(42)
    ref_df = pd.DataFrame({col: rng.normal(5.0, 0.5, 100) for col in FEATURE_COLUMNS})
    curr_df = pd.DataFrame({col: rng.normal(10.0, 0.5, 100) for col in FEATURE_COLUMNS})

    report = analyze_drift(ref_df, curr_df)
    md_text = generate_alert_markdown(report, output_path=alert_path)

    assert alert_path.exists()
    assert "DATA DRIFT DETECTED" in md_text
    assert "Continuous Training Required" in md_text
    assert "Kolmogorov-Smirnov" in md_text


def test_send_drift_alert_graceful_without_webhook():
    """Verify send_drift_alert handles missing webhook without crashing."""
    report = {"dataset_drift_detected": True, "drifted_features": 4, "total_features": 4, "drift_share_pct": 100.0}
    result = send_drift_alert(report, webhook_url=None)
    assert result is False
