from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.xgboost
import pandas as pd

app = FastAPI(title="Financial Loan Default Prediction API")

class BorrowerData(BaseModel):
    monthly_income: float
    monthly_expense: float
    age: int
    has_smartphone: int
    has_wallet: int
    avg_wallet_balance: float
    on_time_payment_ratio: float
    num_loans_taken: int

try:
    model = mlflow.xgboost.load_model("runs:/<RUN_ID>/loan_default_model") 
except Exception:
    model = None

@app.post("/predict")
def predict_default(data: BorrowerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    
    return {
        "default_prediction": int(prediction[0]),
        "status": "High Risk" if int(prediction[0]) == 1 else "Low Risk"
    }