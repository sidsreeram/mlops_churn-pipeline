Markdown# 🚀 End-to-End Customer Churn MLOps Pipeline

A production-grade, reproducible MLOps platform built to predict customer churn. This project covers the full machine learning lifecycle: data versioning, experiment tracking, pipeline orchestration, containerized real-time inference, automated CI/CD performance gates, and data drift monitoring.

---

## 🏗️ System Architecture

```text
[Raw Data] 
   │
   ▼
[DVC Data Versioning] 
   │
   ▼
[Prefect Orchestration Flow]
   ├── 1. Ingest Raw Data
   ├── 2. Modular Preprocessing & Feature Encoding
   ├── 3. Train & Evaluate (Logistic Regression / XGBoost)
   └── 4. Register & Promote Best Model (@champion)
   │
   ▼
[MLflow Experiment Tracking & SQLite Model Registry]
   │
   ▼
[FastAPI Serving Layer] ──► [Docker Containerization] ──► [GHCR & Cloud Deploy]
   │                                                             ▲
   │ (Inference / Live Traffic)                                  │ (Automated Gates)
   ▼                                                             │
[Evidently AI Monitoring] ──► [Drift Report & Alerts]    [GitHub Actions CI/CD]
🧰 Tech StackComponentTechnologyRoleData VersioningDVCTracks raw and processed dataset hashes alongside GitExperiment TrackingMLflowTracks parameters, ROC-AUC/F1 metrics, and model artifactsModel RegistryMLflow RegistryManages versioning and dynamic production aliases (@champion)Pipeline OrchestrationPrefect 3.xSchedules, retries, and executes DAG workflowsServing APIFastAPI + UvicornHigh-performance inference endpoints with Pydantic validationTesting & Quality GatesPytestValidates schema integrity, endpoints, and deployment thresholdsCI/CDGitHub ActionsAutomated testing, validation, and Docker container buildsContainerizationDockerContainerized deployment across local and cloud environmentsMonitoring & DriftEvidently AIStatistical distribution drift detection and HTML reports📁 Repository StructurePlaintextmlops-churn-pipeline/
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # GitHub Actions pipeline (test, validate, build)
├── data/
│   ├── raw/                    # Raw Telco customer churn data (.dvc tracked)
│   └── processed/              # One-hot encoded train.csv and test.csv
├── reports/
│   ├── drift_report.html       # Evidently AI visual drift dashboard
│   └── drift_summary.json      # Structured drift test results
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # Feature cleaning and train-test split
│   ├── train.py                # Hyperparameter tuning and MLflow logging
│   ├── pipeline.py             # Prefect end-to-end DAG flow
│   ├── app.py                  # FastAPI inference app with Pydantic schema
│   ├── validate_model.py       # CI/CD ROC-AUC gatekeeper script
│   └── monitor_drift.py        # Evidently AI drift detection script
├── tests/
│   └── test_pipeline.py        # Integration tests for data and API endpoints
├── .dockerignore
├── .gitignore
├── Dockerfile                  # Production container definition
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Locked Python dependencies
└── README.md
⚡ Quickstart Guide1. Prerequisites & Environment SetupEnsure Python 3.10 or 3.11 is installed.Bash# Clone the repository
git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/mlops-churn-pipeline.git
cd mlops-churn-pipeline

# Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
2. Run the Automated Pipeline (Prefect)Execute the full orchestration flow (download data, preprocess, train XGBoost model, evaluate metrics, and promote the @champion model alias):Bashpython src/pipeline.py
To view the execution graph and logs in the local Prefect dashboard:Bashprefect server start
# Access at [http://127.0.0.1:4200](http://127.0.0.1:4200)
3. Track Experiments (MLflow)Inspect hyperparameter tuning runs, compare models, and review registered versions:Bashmlflow ui --backend-store-uri sqlite:///mlflow.db
# Access at [http://127.0.0.1:5000](http://127.0.0.1:5000)
4. Serve Predictions (FastAPI)Launch the REST API server:Bashuvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
Interactive Swagger UI: Visit http://127.0.0.1:8000/docsHealth Check: GET http://127.0.0.1:8000/healthInference Endpoint: POST http://127.0.0.1:8000/predictSample Request Payload:JSON{
  "SeniorCitizen": 0,
  "tenure": 12,
  "MonthlyCharges": 70.35,
  "TotalCharges": 840.20,
  "gender_Male": 1,
  "Partner_Yes": 0,
  "Dependents_Yes": 0,
  "PhoneService_Yes": 1,
  "MultipleLines_No_phone_service": 0,
  "MultipleLines_Yes": 0,
  "InternetService_Fiber_optic": 1,
  "InternetService_No": 0,
  "OnlineSecurity_No_internet_service": 0,
  "OnlineSecurity_Yes": 0,
  "OnlineBackup_No_internet_service": 0,
  "OnlineBackup_Yes": 1,
  "DeviceProtection_No_internet_service": 0,
  "DeviceProtection_Yes": 0,
  "TechSupport_No_internet_service": 0,
  "TechSupport_Yes": 0,
  "StreamingTV_No_internet_service": 0,
  "StreamingTV_Yes": 1,
  "StreamingMovies_No_internet_service": 0,
  "StreamingMovies_Yes": 1,
  "Contract_One_year": 0,
  "Contract_Two_year": 0,
  "PaperlessBilling_Yes": 1,
  "PaymentMethod_Credit_card_automatic": 0,
  "PaymentMethod_Electronic_check": 1,
  "PaymentMethod_Mailed_check": 0
}
Sample Response:JSON{
  "churn_prediction": 1,
  "churn_probability": 0.6842,
  "risk_level": "High"
}
5. Automated Testing & Quality GatesRun integration test suite:Bashpython -m pytest tests/
Run model deployment gate (requires champion model ROC-AUC $\ge 0.80$):Bashpython src/validate_model.py
6. Production Drift Monitoring (Evidently AI)Evaluate production data shifts against baseline distributions and generate reports:Bashpython src/monitor_drift.py
Interactive Report: Open reports/drift_report.html in your browser.Test Summary: Review reports/drift_summary.json for drift alert triggers.7. Docker DeploymentBuild and run the container locally:Bash# Build Docker image
docker build -t churn-api:latest .

# Run container
docker run -d -p 8000:8000 --name churn-service churn-api:latest
🔄 CI/CD AutomationEvery push or pull request to main triggers .github/workflows/ci-cd.yml, which automatically:Sets up the Python runtime and installs dependencies.Executes the Prefect data and training pipeline.Runs Pytest integration and API schema unit tests.Executes src/validate_model.py to enforce minimum performance thresholds before release.Builds and pushes a Docker image to the GitHub Container Registry (GHCR).📈 Key ResultsBaseline Model: Logistic Regression (ROC-AUC: ~0.843)Champion Model: XGBoost Classifier (max_depth=3, n_estimators=100, learning_rate=0.05)Test ROC-AUC: 0.8464Inference Latency: < 15ms per single-record request
