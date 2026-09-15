import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from src.app import app


@pytest.fixture(scope="module")
def client():
    # Context manager triggers the lifespan startup event
    with TestClient(app) as test_client:
        yield test_client


def test_processed_data_exists():
    assert os.path.exists("data/processed/train.csv"), "train.csv does not exist"
    assert os.path.exists("data/processed/test.csv"), "test.csv does not exist"

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    assert not train_df.empty, "train.csv is empty"
    assert not test_df.empty, "test.csv is empty"
    assert "Churn" in train_df.columns, "Churn column missing from train set"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint_valid_payload(client):
    payload = {
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
        "PaymentMethod_Mailed_check": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]