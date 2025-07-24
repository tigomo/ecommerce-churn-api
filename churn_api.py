from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, ValidationError
from fastapi.responses import JSONResponse

from database import engine
from models import Prediction, Base

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
    client_id: Optional[str] = Field(None, example="CL_001")
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

# -----------------------------
# Routes supplémentaires (mockées)
# -----------------------------
@app.get("/user_dashboard/{client_id}")
def user_dashboard(client_id: str):
    return {"message": f"Dashboard pour le client {client_id}"}


@app.get("/recommend_products/{client_id}")
def recommend_products(client_id: str):
    return {"client_id": client_id, "products": ["Livre Audio A", "Livre Audio B"]}

@app.get("/churn_risk_insights/{client_id}")
def churn_risk_insights(client_id: str):
    return {"client_id": client_id, "insight": "Risque de churn élevé"}

@app.get("/churn_trends")
def churn_trends():
    return {"trend": "Taux de churn en augmentation"}

@app.get("/similar_customers/{client_id}")
def similar_customers(client_id: str):
    return {"client_id": client_id, "similar_customers": ["CL_101", "CL_202"]}

@app.get("/suggest_next_action/{client_id}")
def suggest_next_action(client_id: str):
    return {"client_id": client_id, "action": "Offrir un mois gratuit"}

@app.get("/feature-importance")
def feature_importance():
    return {"message": "Affiche les variables les plus importantes du modèle"}

@app.get("/monitoring")
def monitoring():
    return {"message": "État du modèle, du serveur et statistiques de prédiction"}

@app.get("/explain_prediction/{client_id}")
def explain_prediction(client_id: str):
    return {
        "client_id": client_id,
        "explanation": {
            "Tenure": "+0.3",
            "SupportTickets": "-0.2",
            "ListeningTime": "+0.1"
        }
    }

@app.get("/segment_clients")
def segment_clients():
    return {
        "segments": {
            "A": "Clients fidèles",
            "B": "Clients à risque",
            "C": "Nouveaux clients"
        }
    }

@app.get("/chat-insights/{client_id}")
def chat_insights(client_id: str):
    return {
        "client_id": client_id,
        "chat_response": "Ce client présente un risque de churn élevé."
    }

# Prédiction single client
@app.post("/predict")
def predict(payload: CustomerFeatures):
    try:
        df = pd.DataFrame([payload.dict(exclude={"client_id"})])
        proba = model.predict_proba(df)[0, 1]
        confidence = round(max(proba, 1 - proba) * 100, 2)
        return {
            "client_id": payload.client_id or "unknown",
            "churn_probability": round(float(proba), 4),
            "confidence_score": f"{confidence} %"
        }
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Prédiction batch
@app.post("/predict_batch")
def predict_batch(payloads: List[CustomerFeatures]):
    try:
        results = []
        for p in payloads:
            df = pd.DataFrame([p.dict(exclude={"client_id"})])
            proba = model.predict_proba(df)[0, 1]
            confidence = round(max(proba, 1 - proba) * 100, 2)
            results.append({
                "client_id": p.client_id or f"CL_{len(results)+1}",
                "churn_probability": round(float(proba), 4),
                "confidence_score": f"{confidence} %"
            })
        return {"predictions": results}
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Prédiction via fichier JSON
@app.post("/predict_json_file/")
async def predict_from_json_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".json"):
            raise HTTPException(status_code=400, detail="Veuillez envoyer un fichier .json")

        contents = await file.read()
        df = pd.read_json(io.BytesIO(contents))

        if df.empty:
            raise HTTPException(status_code=400, detail="Le fichier JSON est vide ou mal formaté.")

        predictions = []
        for i, row in df.iterrows():
            row_input = row.drop("client_id") if "client_id" in row else row
            input_df = pd.DataFrame([row_input])
            proba = model.predict_proba(input_df)[0, 1]
            confidence = round(max(proba, 1 - proba) * 100, 2)
            predictions.append({
                "client_id": row.get("client_id", f"CL_{i+1}"),
                "churn_probability": round(float(proba), 4),
                "confidence_score": f"{confidence} %"
            })

        return {"predictions": predictions}

    except Exception as e:
        return {"error": str(e)}

# Lancer en local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
