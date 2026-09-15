import os
import urllib.request
import pandas as pd
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from prefect import task, flow


@task(retries=2, retry_delay_seconds=5)
def ingest_data(url: str, raw_path: str) -> str:
    """Downloads the raw dataset if not already present."""
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    if not os.path.exists(raw_path):
        print(f"Downloading data from {url}...")
        urllib.request.urlretrieve(url, raw_path)
        print(f"Raw data saved to {raw_path}")
    else:
        print(f"Raw data already exists at {raw_path}")
    return raw_path


@task
def preprocess_data_task(raw_path: str, processed_dir: str) -> tuple[str, str]:
    """Cleans, encodes, and splits data into train and test sets."""
    os.makedirs(processed_dir, exist_ok=True)

    df = pd.read_csv(raw_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce").fillna(0.0)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Churn"])

    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Preprocessing completed. Train: {train_df.shape}, Test: {test_df.shape}")
    return train_path, test_path


@task
def train_and_evaluate_task(train_path: str, test_path: str, n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.05):
    """Trains the model, logs run to MLflow, and gates model registration."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Customer-Churn-Detection")
    client = MlflowClient(tracking_uri="sqlite:///mlflow.db")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = train_df.drop(columns=["Churn"]), train_df["Churn"]
    X_test, y_test = test_df.drop(columns=["Churn"]), test_df["Churn"]

    with mlflow.start_run(run_name="prefect_orchestrated_run") as run:
        params = {
            "model_type": "xgboost",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }
        mlflow.log_params(params)

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc,
        }
        mlflow.log_metrics(metrics)

        # Log model artifact and register
        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="churn-prediction-model",
        )

        new_version = client.get_latest_versions("churn-prediction-model")[-1].version
        print(f"Model logged as Version {new_version} with ROC-AUC: {roc_auc:.4f}")

        # Performance gate: promote to champion if ROC-AUC exceeds standard baseline (e.g., 0.80)
        if roc_auc >= 0.80:
            client.set_registered_model_alias("churn-prediction-model", "champion", new_version)
            print(f"Version {new_version} promoted to champion alias!")
        else:
            print(f"ROC-AUC {roc_auc:.4f} did not meet baseline threshold (0.80). Champion remains unchanged.")

    return metrics


@flow(name="churn_training_pipeline")
def churn_pipeline():
    raw_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    raw_file = os.path.join("data", "raw", "raw_churn.csv")
    processed_folder = os.path.join("data", "processed")

    # Orchestrated task dependencies
    raw_path = ingest_data(url=raw_url, raw_path=raw_file)
    train_path, test_path = preprocess_data_task(raw_path=raw_path, processed_dir=processed_folder)
    metrics = train_and_evaluate_task(train_path=train_path, test_path=test_path)
    print("Pipeline execution finished successfully:", metrics)


if __name__ == "__main__":
    churn_pipeline()