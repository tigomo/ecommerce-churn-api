from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, ValidationError
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import io
import uvicorn

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API pour la prédiction du churn client (single, batch & JSON file)",
    version="3.0.0"
)

# Chargement du modèle pipeline
MODEL_PATH = "churn_xgb_model.joblib"
model = joblib.load(MODEL_PATH)

# Schéma d’entrée Pydantic
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

# Vérification de vie
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

# Prédiction single client
@app.post("/predict")
def predict(payload: CustomerFeatures):
    try:
        df = pd.DataFrame([payload.dict()])
        proba = model.predict_proba(df)[0, 1]
        return {"churn_probability": float(proba)}
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Prédiction batch
@app.post("/predict_batch")
def predict_batch(payloads: List[CustomerFeatures]):
    try:
        df = pd.DataFrame([p.dict() for p in payloads])
        probs = model.predict_proba(df)[:, 1].tolist()
        return {"churn_probabilities": probs}
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Prédiction via un fichier JSON
@app.post("/predict_json_file/")
async def predict_from_json_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".json"):
            raise HTTPException(status_code=400, detail="Veuillez envoyer un fichier .json")
        
        contents = await file.read()
        df = pd.read_json(io.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Le fichier JSON est vide ou mal formaté.")

        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        return {"error": str(e)}

# Lancer en local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
