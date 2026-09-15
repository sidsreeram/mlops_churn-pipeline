from contextlib import asynccontextmanager
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

model = None

# Mapping Pydantic variable names -> exact training DataFrame column names
COLUMN_MAPPING = {
    "MultipleLines_No_phone_service": "MultipleLines_No phone service",
    "InternetService_Fiber_optic": "InternetService_Fiber optic",
    "OnlineSecurity_No_internet_service": "OnlineSecurity_No internet service",
    "OnlineBackup_No_internet_service": "OnlineBackup_No internet service",
    "DeviceProtection_No_internet_service": "DeviceProtection_No internet service",
    "TechSupport_No_internet_service": "TechSupport_No internet service",
    "StreamingTV_No_internet_service": "StreamingTV_No internet service",
    "StreamingMovies_No_internet_service": "StreamingMovies_No internet service",
    "Contract_One_year": "Contract_One year",
    "Contract_Two_year": "Contract_Two year",
    "PaymentMethod_Credit_card_automatic": "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic_check": "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed_check": "PaymentMethod_Mailed check",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = "models:/churn-prediction-model@champion"
    try:
        print(f"Loading champion model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("Champion model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load model using alias ({e}). Falling back to latest version.")
        model = mlflow.pyfunc.load_model("models:/churn-prediction-model/latest")
    yield
    print("Shutting down prediction service.")


app = FastAPI(
    title="Customer Churn Prediction Service",
    description="MLOps Serving API using FastAPI and MLflow Champion Model",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    tenure: int = Field(..., ge=0, example=12)
    MonthlyCharges: float = Field(..., gt=0.0, example=70.35)
    TotalCharges: float = Field(..., ge=0.0, example=840.20)

    gender_Male: int = Field(0, ge=0, le=1)
    Partner_Yes: int = Field(0, ge=0, le=1)
    Dependents_Yes: int = Field(0, ge=0, le=1)
    PhoneService_Yes: int = Field(1, ge=0, le=1)
    MultipleLines_No_phone_service: int = Field(0, ge=0, le=1)
    MultipleLines_Yes: int = Field(0, ge=0, le=1)
    InternetService_Fiber_optic: int = Field(1, ge=0, le=1)
    InternetService_No: int = Field(0, ge=0, le=1)
    OnlineSecurity_No_internet_service: int = Field(0, ge=0, le=1)
    OnlineSecurity_Yes: int = Field(0, ge=0, le=1)
    OnlineBackup_No_internet_service: int = Field(0, ge=0, le=1)
    OnlineBackup_Yes: int = Field(1, ge=0, le=1)
    DeviceProtection_No_internet_service: int = Field(0, ge=0, le=1)
    DeviceProtection_Yes: int = Field(0, ge=0, le=1)
    TechSupport_No_internet_service: int = Field(0, ge=0, le=1)
    TechSupport_Yes: int = Field(0, ge=0, le=1)
    StreamingTV_No_internet_service: int = Field(0, ge=0, le=1)
    StreamingTV_Yes: int = Field(1, ge=0, le=1)
    StreamingMovies_No_internet_service: int = Field(0, ge=0, le=1)
    StreamingMovies_Yes: int = Field(1, ge=0, le=1)
    Contract_One_year: int = Field(0, ge=0, le=1)
    Contract_Two_year: int = Field(0, ge=0, le=1)
    PaperlessBilling_Yes: int = Field(1, ge=0, le=1)
    PaymentMethod_Credit_card_automatic: int = Field(0, ge=0, le=1)
    PaymentMethod_Electronic_check: int = Field(1, ge=0, le=1)
    PaymentMethod_Mailed_check: int = Field(0, ge=0, le=1)


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    # 1. Convert input to dict and rename keys to match model schema
    raw_dict = features.model_dump()
    aligned_dict = {COLUMN_MAPPING.get(k, k): v for k, v in raw_dict.items()}

    # 2. Build DataFrame
    input_data = pd.DataFrame([aligned_dict])

    try:
        raw_pred = model.predict(input_data)

        if hasattr(raw_pred, "shape") and len(raw_pred.shape) > 1 and raw_pred.shape[1] > 1:
            proba = float(raw_pred[0][1])
        else:
            proba = float(raw_pred[0])

        pred = int(proba >= 0.5)
        risk = "High" if proba > 0.65 else ("Medium" if proba >= 0.35 else "Low")

        return PredictionResponse(
            churn_prediction=pred,
            churn_probability=round(proba, 4),
            risk_level=risk,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")