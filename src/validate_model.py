import sys
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import roc_auc_score

MINIMUM_ROC_AUC = 0.80

def validate():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = "models:/churn-prediction-model@champion"

    print(f"Loading champion model: {model_uri}")
    try:
        # Load native flavor to access predict_proba
        model = mlflow.xgboost.load_model(model_uri)
    except Exception:
        try:
            model = mlflow.sklearn.load_model(model_uri)
        except Exception:
            model = mlflow.pyfunc.load_model(model_uri)

    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    # Retrieve continuous probabilities
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)[:, 1]
    else:
        preds = model.predict(X_test)
        probas = preds[:, 1] if len(preds.shape) > 1 and preds.shape[1] > 1 else preds

    score = float(roc_auc_score(y_test, probas))
    print(f"Champion Model ROC-AUC: {score:.4f} (Threshold: {MINIMUM_ROC_AUC})")

    if score < MINIMUM_ROC_AUC:
        print(f"Model validation FAILED! ROC-AUC {score:.4f} is below threshold {MINIMUM_ROC_AUC}.")
        sys.exit(1)

    print("Model quality check passed. Eligible for deployment.")

if __name__ == "__main__":
    validate()