import os
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(raw_data_path: str, processed_dir: str):
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load data
    df = pd.read_csv(raw_data_path)

    # 2. Basic cleanup
    # TotalCharges contains blank strings (" ") for new customers; convert to numeric & fillna
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].str.strip(), errors="coerce"
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Drop customerID (not a feature)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Target encoding: 'Yes' -> 1, 'No' -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 3. Train-Test Split (80/20)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Churn"]
    )

    # 4. Save processed artifacts
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Data preprocessing complete.")
    print(f"Train set: {train_df.shape} -> {train_path}")
    print(f"Test set:  {test_df.shape} -> {test_path}")


if __name__ == "__main__":
    RAW_PATH = os.path.join("data", "raw", "raw_churn.csv")
    PROCESSED_PATH = os.path.join("data", "processed")
    preprocess_data(RAW_PATH, PROCESSED_PATH)