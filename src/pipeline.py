import os
import urllib.request
import mlflow
from mlflow import MlflowClient
import pandas as pd
from prefect import flow, task
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


@task(name="Ingest Data", retries=2, retry_delay_seconds=5)
def ingest_data(raw_url: str, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Downloading raw data from {raw_url}...")
    urllib.request.urlretrieve(raw_url, output_path)
    print(f"Raw data saved to {output_path}")
    return output_path


@task(name="Preprocess Data")
def preprocess_data(raw_path: str, processed_dir: str) -> tuple[str, str]:
    os.makedirs(processed_dir, exist_ok=True)
    df = pd.read_csv(raw_path)

    # Clean TotalCharges and drop customerID
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce").fillna(0.0)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Encode target and categorical columns
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 80/20 train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Churn"])

    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Preprocessing finished. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_path, test_path


@task(name="Train and Evaluate Model")
def train_and_evaluate(train_path: str, test_path: str, n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.05) -> dict:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Customer-Churn-Detection")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    with mlflow.start_run(run_name="prefect_pipeline_run") as run:
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

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }
        mlflow.log_metrics(metrics)

        # Register to MLflow model registry
        reg_model = mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="churn-prediction-model",
        )

        return {
            "run_id": run.info.run_id,
            "model_version": getattr(reg_model, "registered_model_version", None),
            "metrics": metrics,
        }


@task(name="Promote Champion Model")
def promote_champion(metrics: dict):
    client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
    model_name = "churn-prediction-model"

    # Get latest version registered
    latest_versions = client.get_latest_versions(model_name)
    if not latest_versions:
        print("No registered model version found.")
        return

    latest_version = max([int(v.version) for v in latest_versions])

    # Tag the latest version as champion if ROC-AUC satisfies baseline threshold (e.g., > 0.80)
    if metrics.get("roc_auc", 0.0) >= 0.80:
        client.set_registered_model_alias(model_name, "champion", str(latest_version))
        print(f"Model version {latest_version} promoted to 'champion' alias (ROC-AUC: {metrics['roc_auc']:.4f})")
    else:
        print(f"Model version {latest_version} did not meet promotion threshold (ROC-AUC: {metrics['roc_auc']:.4f})")


@flow(name="Churn-End-to-End-Pipeline", log_prints=True)
def churn_mlops_pipeline():
    raw_data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    raw_path = "data/raw/raw_churn.csv"
    processed_dir = "data/processed"

    # 1. Ingest
    ingested_file = ingest_data(raw_data_url, raw_path)

    # 2. Preprocess
    train_file, test_file = preprocess_data(ingested_file, processed_dir)

    # 3. Train & Evaluate
    result = train_and_evaluate(train_file, test_file)

    # 4. Check & Promote
    promote_champion(result["metrics"])


if __name__ == "__main__":
    churn_mlops_pipeline()