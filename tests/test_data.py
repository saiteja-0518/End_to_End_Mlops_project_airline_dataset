from pathlib import Path
import pandas as pd
import yaml


def _load_params():
    if Path("params.yaml").exists():
        with open("params.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def test_raw_dataset_exists_and_valid():
    """Verify raw Airline dataset exists and is non-empty."""
    params = _load_params()
    raw_path = Path(params.get("data", {}).get("raw_path", "data/raw/airline_passenger_satisfaction.csv"))
    assert raw_path.exists(), f"Raw dataset not found at {raw_path}"

    raw_df = pd.read_csv(raw_path)
    assert len(raw_df) > 1000, f"Expected full airline dataset, got {len(raw_df)} rows"


def test_preprocessed_splits_exist_and_consistent():
    """Verify train and test splits exist, share identical schemas, and match expected dataset size."""
    train_path = Path("data/processed/train.csv")
    test_path = Path("data/processed/test.csv")

    assert train_path.exists(), f"Train set not found at {train_path}"
    assert test_path.exists(), f"Test set not found at {test_path}"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    assert len(train_df) > 0 and len(test_df) > 0
    assert list(train_df.columns) == list(test_df.columns)

    # Arrival Delay nulls are handled inside the Pipeline imputer to prevent data leakage
    non_delay_cols = [c for c in train_df.columns if c != "Arrival Delay"]
    assert train_df[non_delay_cols].isna().sum().sum() == 0, "Unexpected NaNs found in train.csv"
    assert test_df[non_delay_cols].isna().sum().sum() == 0, "Unexpected NaNs found in test.csv"

    total_rows = len(train_df) + len(test_df)
    assert total_rows == 129880, f"Expected 129880 total rows across splits, got {total_rows}"