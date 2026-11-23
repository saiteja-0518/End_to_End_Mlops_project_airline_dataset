import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    df = pd.read_csv(os.path.join("data", "ingested", "airline.csv"))

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df["Satisfaction"] = df["Satisfaction"].map({
        "Satisfied": 1,
        "Neutral or Dissatisfied": 0,
    })
    df = df.dropna(subset=["Satisfaction"]).copy()
    df["Satisfaction"] = df["Satisfaction"].astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df["Satisfaction"],
    )

    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    print(f"Preprocessing complete: train={train_df.shape}, test={test_df.shape}")


if __name__ == "__main__":
    main()