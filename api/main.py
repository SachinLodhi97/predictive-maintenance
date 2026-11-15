
import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = "models/xgboost_model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run advanced_training.py first."
    )

artifact = joblib.load(MODEL_PATH)

model = artifact["pipeline"]
threshold = artifact["threshold"]
feature_columns = artifact["feature_columns"]


# ==========================================
# 2. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Predictive Maintenance API",
    description="Industrial Machine Failure Prediction System",
    version="1.0.0"
)


# ==========================================
# 3. INPUT VALIDATION
# ==========================================

class MachineData(BaseModel):

    Type: str = Field(pattern="^[LMH]$")

    Air_temperature: float = Field(
        alias="Air temperature",
        ge=250,
        le=350
    )

    Process_temperature: float = Field(
        alias="Process temperature",
        ge=250,
        le=350
    )

    Rotational_speed: float = Field(
        alias="Rotational speed",
        gt=0,
        le=10000
    )

    Torque: float = Field(
        gt=0,
        le=200
    )

    Tool_wear: float = Field(
        alias="Tool wear",
        ge=0,
        le=500
    )


# ==========================================
# 4. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Predictive Maintenance API is running",
        "model": artifact["model_name"],
        "version": "1.0.0"
    }


# ==========================================
# 5. HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ==========================================
# 6. MACHINE FAILURE PREDICTION
# ==========================================
@app.post("/predict")
def predict(data: MachineData):

    input_data = pd.DataFrame([{
        "Type": data.Type,
        "Air temperature [K]": data.Air_temperature,
        "Process temperature [K]": data.Process_temperature,
        "Rotational speed [rpm]": data.Rotational_speed,
        "Torque [Nm]": data.Torque,
        "Tool wear [min]": data.Tool_wear
    }])

    # Keep the exact order expected by the trained model
    input_data = input_data[feature_columns]

    probability = model.predict_proba(input_data)[0][1]

    prediction = int(probability >= threshold)

    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.30:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "success": True,
        "machine_failure": prediction,
        "prediction": (
            "Failure" if prediction == 1 else "No Failure"
        ),
        "failure_probability": round(float(probability), 4),
        "risk_level": risk,
        "decision_threshold": threshold
    }