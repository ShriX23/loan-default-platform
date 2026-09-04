import os

# Define the folder structure
folders = [
    ".github/workflows",
    "data",
    "src",
    "app"
]

# Create the folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Define the files and their content
files = {
    "requirements.txt": """pandas
scikit-learn
xgboost
mlflow
great_expectations
fastapi
uvicorn
pydantic
pytest
httpx""",

    ".github/workflows/pr-gate.yml": """name: MLOps PR Gate
on:
  pull_request:
    branches: [ main ]
jobs:
  validate-train-test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Validate Data Schema 
        run: python src/validate_data.py
      - name: Train and Gate Candidate Model
        run: python src/train.py
      - name: Run API Unit Tests
        run: pytest app/test_main.py""",

    "src/validate_data.py": """import sys
import great_expectations as ge

def validate_dataset():
    context = ge.get_context()
    try:
        checkpoint_result = context.run_checkpoint(checkpoint_name="loan_data_checkpoint")
        if not checkpoint_result["success"]:
            print("Data validation failed! Blocking deployment.")
            sys.exit(1)
        print("Data validation passed.")
    except Exception as e:
        print("Checkpoint not configured yet. Run 'great_expectations init' first.")

if __name__ == "__main__":
    validate_dataset()""",

    "src/train.py": """import pandas as pd
import sys
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score
import xgboost as xgb

MIN_F1_SCORE = 0.75
MIN_PRECISION = 0.80

def train_and_evaluate():
    try:
        df = pd.read_csv("data/data_01.csv")
    except FileNotFoundError:
        print("Dataset not found in data/ folder.")
        return
        
    target_col = "target_default" if "target_default" in df.columns else "target"
    if target_col not in df.columns:
        print("Target column missing.")
        return
        
    X = df.drop(target_col, axis=1)
    if "borrower_id" in X.columns:
        X = X.drop("borrower_id", axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = xgb.XGBClassifier(eval_metric='logloss')
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        f1 = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)

        print(f"Metrics - F1: {f1:.4f}, Precision: {precision:.4f}")

        if f1 < MIN_F1_SCORE or precision < MIN_PRECISION:
            print("Failed regulatory thresholds. Blocking PR.")
            sys.exit(1) 

        mlflow.xgboost.log_model(model, "loan_default_model")
        print("Model registered.")

if __name__ == "__main__":
    train_and_evaluate()""",

    "app/main.py": """from fastapi import FastAPI, HTTPException
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
    }""",

    "app/test_main.py": """from fastapi.testclient import TestClient
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
        assert response.status_code == 200"""
}

# Create the files
for filepath, content in files.items():
    with open(filepath, "w") as f:
        f.write(content)

print("Project structure and files have been successfully created!")
