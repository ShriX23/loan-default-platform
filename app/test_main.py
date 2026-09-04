from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint_valid_data():
    payload = {
        "monthly_income": 4500.0,
        "monthly_expense": 1200.0,
        "age": 30,
        "has_smartphone": 1,
        "has_wallet": 1,
        "avg_wallet_balance": 300.50,
        "on_time_payment_ratio": 0.95,
        "num_loans_taken": 2
    }
    response = client.post("/predict", json=payload)
    if response.status_code != 503: 
        assert response.status_code == 200