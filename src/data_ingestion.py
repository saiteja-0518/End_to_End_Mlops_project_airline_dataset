import os
import pandas as pd


def main():
    raw_path = os.path.join("data", "raw", "airline_passenger_satisfaction.csv")
    ingested_dir = os.path.join("data", "ingested")
    os.makedirs(ingested_dir, exist_ok=True)

    df = pd.read_csv(raw_path)
    df = df.drop_duplicates()
    df = df.dropna(subset=["Satisfaction"]).copy()
    df["Satisfaction"] = df["Satisfaction"].astype(str).str.strip()

    out_path = os.path.join(ingested_dir, "airline.csv")
    df.to_csv(out_path, index=False)
    print(f"Ingestion complete: {df.shape} -> {out_path}")


if __name__ == "__main__":
    main()