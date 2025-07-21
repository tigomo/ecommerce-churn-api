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

@app.get("/user_dashboard/{client_id}")
def user_dashboard(client_id: str):
    # Récupération des données + stats personnalisées
    return {...}

@app.get("/recommend_products/{client_id}")
def recommend_products(client_id: str):
    # Logique de recommandation
    return {...}

@app.get("/churn_risk_insights/{client_id}")
def churn_risk_insights(client_id: str):
    # Interprétation de la prédiction
    return {...}

@app.get("/churn_trends")
def churn_trends():
    # Statistiques globales
    return {...}

@app.get("/similar_customers/{client_id}")
def similar_customers(client_id: str):
    # Recherche des clients similaires
    return {...}

@app.get("/suggest_next_action/{client_id}")
def suggest_next_action(client_id: str):
    # Prochaine action recommandée
    return {...}


# Prédiction single client
@app.post("/predict")
def predict(customer: CustomerFeatures):
    try:
        input_df = pd.DataFrame([customer.dict()])
        prediction_proba = model.predict_proba(input_df)[0]
        churn_probability = round(float(prediction_proba[1]), 4)
        confidence_score = round(float(max(prediction_proba)) * 100, 2)  # en pourcentage

        return {
            "churn_probability": churn_probability,
            "confidence_score": f"{confidence_score} %"
        }
    except Exception as e:
        return {"error": str(e)}

# Prédiction batch
@app.post("/predict_batch")
def predict_batch(file: UploadFile = File(...)):
    try:
        content = file.file.read()
        df = pd.read_json(io.BytesIO(content))

        predictions_proba = model.predict_proba(df)
        predictions = []

        for proba in predictions_proba:
            churn_proba = round(float(proba[1]), 4)
            confidence = round(float(max(proba)) * 100, 2)

            predictions.append({
                "churn_probability": churn_proba,
                "confidence_score": f"{confidence} %"
            })

        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}


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

        probs = model.predict_proba(df)[:, 1]
        churn_probs = [round(p, 4) for p in probs]
        confidences = [f"{round(p * 100, 2)} %" for p in probs]

        return {
            "churn_probabilities": churn_probs,
            "confidence_scores": confidences
        }
    
    except Exception as e:
        return {"error": str(e)}

# Lancer en local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
