from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field, ValidationError
from fastapi.responses import JSONResponse

from database import SessionLocal
from models import Prediction

from sqlalchemy.orm import Session
from sqlalchemy import func 
from fastapi import Depends
from database import SessionLocal
from models import Prediction

import pandas as pd
import joblib
import io
import uvicorn

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API pour la prédiction du churn client (single, batch & JSON file)",
    version="3.0.0"
)

# Ajout du middleware CORS pour Flutter Web
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplace * par l'URL de ton front si nécessaire
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# Nouvelle route : Prédictions d'un client spécifique
@app.get("/get_predictions_by_client/{client_id}")
def get_predictions_by_client(client_id: str):
    db = SessionLocal()
    predictions = db.query(Prediction).filter(Prediction.client_id == client_id).all()
    db.close()
    return predictions

# Nouvelle route : N dernières prédictions
@app.get("/get_last_predictions")
def get_last_predictions(limit: int = Query(10, gt=0)):
    db = SessionLocal()
    predictions = db.query(Prediction).order_by(Prediction.timestamp.desc()).limit(limit).all()
    db.close()
    return predictions

# Nouvelle route : Statistiques globales
@app.get("/get_predictions_stats")
def get_predictions_stats():
    db = SessionLocal()

    try:
        total = db.query(func.count(Prediction.id)).scalar()
        avg_churn = db.query(func.avg(Prediction.churn_probability)).scalar()

        # Assurez-vous que prediction_type n'est pas None avant de grouper
        by_type = db.query(Prediction.prediction_type, func.count(Prediction.id))\
                    .filter(Prediction.prediction_type != None)\
                    .group_by(Prediction.prediction_type)\
                    .all()

        db.close()

        return {
            "total_predictions": total or 0,
            "average_churn_probability": round(float(avg_churn), 4) if avg_churn is not None else 0.0,
            "predictions_by_type": {ptype: count for ptype, count in by_type}
        }
    except Exception as e:
        db.close()
        raise HTTPException(status_code=500, detail=str(e))


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

# Prédiction single client plus database 
@app.post("/predict_db") 
def predict(payload: CustomerFeatures):
    try:
        df = pd.DataFrame([payload.dict(exclude={"client_id"})])
        proba = model.predict_proba(df)[0, 1]
        proba = float(proba)  # Convertir en float natif Python
        confidence = round(max(proba, 1 - proba) * 100, 2)

        db = SessionLocal()
        prediction = Prediction(
            client_id=payload.client_id or "unknown",
            churn_probability=round(proba, 4),
            confidence_score=f"{confidence} %",
            prediction_type="single"  # <-- ajouté ici
        )
        db.add(prediction)
        db.commit()
        db.close()

        return {
            "client_id": payload.client_id or "unknown",
            "churn_probability": round(proba, 4),
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
        
# Prédiction batch plus database
@app.post("/predict_batch_db")
def predict_batch(payloads: List[CustomerFeatures]):
    try:
        db = SessionLocal()
        results = []

        for idx, p in enumerate(payloads):
            df = pd.DataFrame([p.dict(exclude={"client_id"})])
            proba = model.predict_proba(df)[0, 1]
            proba = float(proba)  # Convertir NumPy float en float Python
            confidence = round(max(proba, 1 - proba) * 100, 2)

            client_id = p.client_id or f"CL_{idx+1}"
            prediction = Prediction(
                client_id=client_id,
                churn_probability=round(proba, 4),
                confidence_score=f"{confidence} %",
                prediction_type="batch"
            )
            db.add(prediction)
            results.append({
                "client_id": client_id,
                "churn_probability": round(proba, 4),
                "confidence_score": f"{confidence} %"
            })

        db.commit()
        db.close()

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

# Prédiction via fichier JSON
@app.post("/predict_json_file_db/")
async def predict_from_json_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".json"):
            raise HTTPException(status_code=400, detail="Veuillez envoyer un fichier .json")

        contents = await file.read()
        df = pd.read_json(io.BytesIO(contents))

        if df.empty:
            raise HTTPException(status_code=400, detail="Le fichier JSON est vide ou mal formaté.")

        db = SessionLocal()
        predictions = []

        for i, row in df.iterrows():
            row_input = row.drop("client_id") if "client_id" in row else row
            input_df = pd.DataFrame([row_input])
            proba = model.predict_proba(input_df)[0, 1]
            proba = float(proba)
            confidence = round(max(proba, 1 - proba) * 100, 2)

            client_id = row.get("client_id", f"CL_{i+1}")
            prediction = Prediction(
                client_id=client_id,
                churn_probability=round(proba, 4),
                confidence_score=f"{confidence} %",
                prediction_type="json_file"
            )
            db.add(prediction)
            predictions.append({
                "client_id": client_id,
                "churn_probability": round(proba, 4),
                "confidence_score": f"{confidence} %"
            })

        db.commit()
        db.close()

        return {"predictions": predictions}

    except Exception as e:
        return {"error": str(e)}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/get_predictions")
def get_predictions(db: Session = Depends(get_db)):
    predictions = db.query(Prediction).all()
    return [
        {
            "id": p.id,
            "client_id": p.client_id,
            "churn_probability": p.churn_probability,
            "confidence_score": p.confidence_score,
            "prediction_type": p.prediction_type,
            "timestamp": p.timestamp
        }
        for p in predictions
    ]


# Lancer en local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
