# ===================================================================
# --- main.py ---
# --- Your Fraud Detection API ---
# ===================================================================
import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --- 1. Define the Application ---
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="An API to predict credit card fraud in real-time using a tuned XGBoost model.",
    version="1.0"
)

# --- 2. Load Our Trained Model ---
# This file must be in the same folder
print("--- Loading model... ---")
model = joblib.load('fraud_model.joblib')
print("--- Model loaded successfully. ---")


# --- 3. Define the Input Data Shape ---
# This tells FastAPI what the incoming JSON must look like.
# It's all our 30 features.
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    scaled_amount: float
    scaled_time: float
    
    # This provides a sample for the /docs page
    class Config:
        schema_extra = {
            "example": {
                "V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781, "V5": -0.3383,
                "V6": 0.4623, "V7": 0.2395, "V8": 0.0986, "V9": 0.3637, "V10": 0.0907,
                "V11": -0.5516, "V12": -0.6178, "V13": -0.9913, "V14": -0.3111, "V15": 1.4681,
                "V16": -0.4704, "V17": 0.2079, "V18": 0.0257, "V19": 0.4039, "V20": 0.2514,
                "V21": -0.0183, "V22": 0.2778, "V23": -0.1104, "V24": 0.0669, "V25": 0.1285,
                "V26": -0.1891, "V27": 0.1335, "V28": -0.021, "scaled_amount": 1.7832, "scaled_time": -0.9949
            }
        }


# --- 4. Create the /predict Endpoint ---
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    
    # 1. Convert the incoming transaction data into a DataFrame
    data_dict = transaction.dict()
    data_df = pd.DataFrame([data_dict])

    # 2. Get the prediction
    prediction = model.predict(data_df)
    
    # 3. Get the probability of fraud
    probability = model.predict_proba(data_df)[0][1] # Prob of being 1 (Fraud)

    # 4. Format the response
    prediction_int = int(prediction[0])
    
    if prediction_int == 0:
        status = "Transaction Approved"
    else:
        status = "Transaction BLOCKED (Fraud Detected)"

    # 5. Return the result as JSON
    return {
        "prediction": prediction_int,
        "status": status,
        # *** FIX IS HERE: Ensure probability is a standard Python float ***
        "fraud_probability": round(float(probability), 4) 
    }

# --- 5. Add a "root" endpoint for testing ---
@app.get("/")
def read_root():
    return {"message": "Credit Card Fraud Detection API is running!"}