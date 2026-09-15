import argparse
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


def load_data(data_dir: str):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    return X_train, y_train, X_test, y_test


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_model(model_type: str, params: dict, data_dir: str = "data/processed"):
    # SQLite backend keeps tracking organized and supports the Model Registry
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Customer-Churn-Detection")

    X_train, y_train, X_test, y_test = load_data(data_dir)

    run_name = f"{model_type}_{params.get('run_suffix', 'default')}"

    with mlflow.start_run(run_name=run_name):
        # 1. Initialize Model
        if model_type == "logistic_regression":
            C = float(params.get("C", 1.0))
            max_iter = int(params.get("max_iter", 1000))
            model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
            mlflow.log_params({"model_type": model_type, "C": C, "max_iter": max_iter})

        elif model_type == "xgboost":
            n_estimators = int(params.get("n_estimators", 100))
            max_depth = int(params.get("max_depth", 3))
            learning_rate = float(params.get("learning_rate", 0.1))
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                eval_metric="logloss",
                random_state=42,
            )
            mlflow.log_params(
                {
                    "model_type": model_type,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                }
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # 2. Train
        model.fit(X_train, y_train)

        # 3. Predict & Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        # 4. Log Model Artifact
        if model_type == "logistic_regression":
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="churn-prediction-model",
            )
        else:
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                registered_model_name="churn-prediction-model",
            )

        print(f"Run '{run_name}' completed.")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "xgboost"],
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--run_suffix", type=str, default="run")

    args = parser.parse_args()

    param_dict = {
        "C": args.C,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "run_suffix": args.run_suffix,
    }

    train_model(args.model, param_dict)