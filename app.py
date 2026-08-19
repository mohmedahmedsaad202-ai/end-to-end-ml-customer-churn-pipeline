from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")

# Load trained pipeline artifact
try:
    model = joblib.load('churn_model.pkl')
except Exception as e:
    model = None

class CustomerData(BaseModel):
    tenure_months: int
    monthly_charges: float
    total_support_calls: int
    contract_type: str  # 'Month-to-Month', 'One-Year', or 'Two-Year'

@app.post("/predict/")
def predict_churn(customer: CustomerData):
    if not model:
        raise HTTPException(status_code=500, detail="Model artifact not found. Please run train.py first.")
    
    input_df = pd.DataFrame([customer.dict()])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_risk_level": "High" if probability > 0.5 else "Low",
        "churn_probability": float(round(probability, 4))
    }
