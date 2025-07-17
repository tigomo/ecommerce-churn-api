
# 🚀 Extensions API FastAPI : batch prediction + exemples de tests

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
import joblib, pandas as pd
import uvicorn


# Charger le pipeline
MODEL_PATH = "churn_xgb_model.joblib"
pipeline = joblib.load(MODEL_PATH)

# Schéma Pydantic pour un client
class CustomerFeatures(BaseModel):
    Tenure: Optional[float] = Field(..., example=5)
    PreferredLoginDevice: str = Field(..., example="Phone")
    CityTier: int = Field(..., ge=0, le=5, example=2)
    WarehouseToHome: Optional[float] = Field(..., example=12)
    PreferredPaymentMode: str = Field(..., example="UPI")
    Gender: str = Field(..., example="Female")
    HourSpendOnApp: Optional[float] = Field(..., example=4)
    NumberOfDeviceRegistered: int = Field(..., example=3)
    PreferedOrderCat: str = Field(..., example="Laptop & Accessory")
    SatisfactionScore: int = Field(..., ge=1, le=5, example=3)
    MaritalStatus: str = Field(..., example="Single")
    NumberOfAddress: int = Field(..., example=4)
    Complain: int = Field(..., ge=0, le=1, example=0)
    OrderAmountHikeFromlastYear: Optional[float] = Field(..., example=10)
    CouponUsed: Optional[float] = Field(..., example=1)
    OrderCount: Optional[float] = Field(..., example=2)
    DaySinceLastOrder: Optional[float] = Field(..., example=7)
    CashbackAmount: Optional[float] = Field(..., example=120.0)

app = FastAPI(title="Customer Churn Prediction API (Batch)",
              description="API REST avec validation Pydantic, endpoints single & batch.",
              version="3.0.0" )
@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API de prédiction du churn !"}

@app.get("/health")
def health() -> dict:
    """Vérifie que l'API fonctionne"""
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: CustomerFeatures):
    try:
        df = pd.DataFrame([payload.dict()])
        proba = pipeline.predict_proba(df)[0, 1]
        return {"churn_probability": float(proba)}
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
def predict_batch(payloads: List[CustomerFeatures]):
    """Prend une liste de clients et renvoie une liste de probabilités"""
    try:
        df = pd.DataFrame([p.dict() for p in payloads])
        probs = pipeline.predict_proba(df)[:, 1].tolist()
        return {"churn_probabilities": probs}
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------- Exemple de test local (hors API) ----------
if __name__ == "__main__":
    # Simuler 3 clients pour test rapide
    samples = [
        CustomerFeatures(Tenure=6, PreferredLoginDevice="Phone", CityTier=2, WarehouseToHome=12, PreferredPaymentMode="UPI", Gender="Female", HourSpendOnApp=4, NumberOfDeviceRegistered=3, PreferedOrderCat="Laptop & Accessory", SatisfactionScore=4, MaritalStatus="Single", NumberOfAddress=2, Complain=0, OrderAmountHikeFromlastYear=12.5, CouponUsed=1.0, OrderCount=3.0, DaySinceLastOrder=5.0, CashbackAmount=125.50),
        CustomerFeatures(Tenure=1, PreferredLoginDevice="Mobile Phone", CityTier=3, WarehouseToHome=30, PreferredPaymentMode="Debit Card", Gender="Male", HourSpendOnApp=2, NumberOfDeviceRegistered=4, PreferedOrderCat="Mobile", SatisfactionScore=2, MaritalStatus="Married", NumberOfAddress=3, Complain=1, OrderAmountHikeFromlastYear=5, CouponUsed=0, OrderCount=1, DaySinceLastOrder=20, CashbackAmount=80),
        CustomerFeatures(Tenure=10, PreferredLoginDevice="Computer", CityTier=1, WarehouseToHome=6, PreferredPaymentMode="Credit Card", Gender="Male", HourSpendOnApp=5, NumberOfDeviceRegistered=2, PreferedOrderCat="Books", SatisfactionScore=5, MaritalStatus="Single", NumberOfAddress=1, Complain=0, OrderAmountHikeFromlastYear=15, CouponUsed=2, OrderCount=5, DaySinceLastOrder=2, CashbackAmount=200)
    ]
    df_test = pd.DataFrame([s.dict() for s in samples])
    probs = pipeline.predict_proba(df_test)[:, 1]
    print("Exemple batch - probabilités :", probs.tolist())

