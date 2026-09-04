import pandas as pd
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
    train_and_evaluate()