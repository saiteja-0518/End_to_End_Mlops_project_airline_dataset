import os
import pandas as pd


def main():
    processed_dir = os.path.join("data", "processed")
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    target_col = "Satisfaction"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[[target_col]]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[[target_col]]

    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    print(f"Feature split complete: X_train={X_train.shape}, X_test={X_test.shape}")


if __name__ == "__main__":
    main()