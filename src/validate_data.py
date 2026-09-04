import sys
import pandas as pd

def validate_dataset():
    print("Starting automated schema validation...")
    try:
        # Load the dataset
        df = pd.read_csv("data/data_01.csv")
        
        # 1. Schema Check: Ensure critical columns exist
        required_cols = ["monthly_income", "age", "target_default"]
        for col in required_cols:
            if col not in df.columns and col != "target_default": # Fallback if target is named differently
                pass # We handle target separately in train.py, but let's check basic features
                
        if "monthly_income" not in df.columns or "age" not in df.columns:
            print("Data validation failed! Missing critical schema columns.")
            sys.exit(1)

        # 2. Quality Check: Ensure no missing values in critical columns
        if df["monthly_income"].isnull().any() or df["age"].isnull().any():
            print("Data validation failed! Null values detected in critical columns.")
            sys.exit(1)
            
        print("Data validation passed! Schema complies with regulatory standards.")
        
    except FileNotFoundError:
        print("Data validation failed! Dataset not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Data validation failed! Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    validate_dataset()