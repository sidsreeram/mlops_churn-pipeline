import mlflow
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
exp = client.get_experiment_by_name("Customer-Churn-Detection")

runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.roc_auc DESC"],
)

print("\n--- Top Runs in Experiment ---")
for r in runs[:5]:
    roc = r.data.metrics.get("roc_auc", 0.0)
    name = r.data.tags.get("mlflow.runName", "unnamed")
    print(f"Run ID: {r.info.run_id[:8]} | ROC-AUC: {roc:.4f} | Name: {name}")

print("\n--- Registered Model Versions ---")
models = client.search_model_versions("name='churn-prediction-model'")
for m in models:
    print(f"Version: {m.version} | Aliases: {m.aliases} | Run ID: {m.run_id[:8]}")