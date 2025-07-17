from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError
import joblib
import pandas as pd
import uvicorn
import os

# Charger le modèle
MODEL_PATH = "churn_xgb_model.joblib"
pipeline = joblib.load(MODEL_PATH)

# Créer l’application
app = FastAPI(
    title="Customer Churn Prediction API (Batch)",
    description="API REST avec formulaire HTML, endpoints single & batch.",
    version="3.0.0"
)

# Chargement du dossier templates/
templates = Jinja2Templates(directory="templates")

# ✅ Accueil
@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API de prédiction du churn !"}

# ✅ Vérification de santé
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

# ✅ Schéma d’entrée via Pydantic
class CustomerFeatures(BaseModel):
    Tenure: Optional[float]
    PreferredLoginDevice: str
    CityTier: int
    WarehouseToHome: Optional[float]
    PreferredPaymentMode: str
    Gender: str
    HourSpendOnApp: Optional[float]
    NumberOfDeviceRegistered: int
    PreferedOrderCat: str
    SatisfactionScore: int
    MaritalStatus: str
    NumberOfAddress: int
    Complain: int
    OrderAmountHikeFromlastYear: Optional[float]
    CouponUsed: Optional[float]
    OrderCount: Optional[float]
    DaySinceLastOrder: Optional[float]
    CashbackAmount: Optional[float]

# ✅ API : prédiction simple
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

# ✅ API : prédiction en lot
@app.post("/predict_batch")
def predict_batch(payloads: List[CustomerFeatures]):
    try:
        df = pd.DataFrame([p.dict() for p in payloads])
        probs = pipeline.predict_proba(df)[:, 1].tolist()
        return {"churn_probabilities": probs}
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ✅ Route formulaire HTML GET
@app.get("/form", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# ✅ Traitement du formulaire HTML POST
@app.post("/form_predict", response_class=HTMLResponse)
def form_predict(
    request: Request,
    Tenure: float = Form(...),
    PreferredLoginDevice: str = Form(...),
    CityTier: int = Form(...),
    WarehouseToHome: float = Form(...),
    PreferredPaymentMode: str = Form(...),
    Gender: str = Form(...),
    HourSpendOnApp: float = Form(...),
    NumberOfDeviceRegistered: int = Form(...),
    PreferedOrderCat: str = Form(...),
    SatisfactionScore: int = Form(...),
    MaritalStatus: str = Form(...),
    NumberOfAddress: int = Form(...),
    Complain: int = Form(...),
    OrderAmountHikeFromlastYear: float = Form(...),
    CouponUsed: float = Form(...),
    OrderCount: float = Form(...),
    DaySinceLastOrder: float = Form(...),
    CashbackAmount: float = Form(...)
):
    try:
        data = {
            "Tenure": Tenure,
            "PreferredLoginDevice": PreferredLoginDevice,
            "CityTier": CityTier,
            "WarehouseToHome": WarehouseToHome,
            "PreferredPaymentMode": PreferredPaymentMode,
            "Gender": Gender,
            "HourSpendOnApp": HourSpendOnApp,
            "NumberOfDeviceRegistered": NumberOfDeviceRegistered,
            "PreferedOrderCat": PreferedOrderCat,
            "SatisfactionScore": SatisfactionScore,
            "MaritalStatus": MaritalStatus,
            "NumberOfAddress": NumberOfAddress,
            "Complain": Complain,
            "OrderAmountHikeFromlastYear": OrderAmountHikeFromlastYear,
            "CouponUsed": CouponUsed,
            "OrderCount": OrderCount,
            "DaySinceLastOrder": DaySinceLastOrder,
            "CashbackAmount": CashbackAmount
        }

        df = pd.DataFrame([data])
        prob = pipeline.predict_proba(df)[0][1]
        result = f"{prob * 100:.2f}%"

        return templates.TemplateResponse("form.html", {
            "request": request,
            "result": result
        })
    except Exception as e:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "result": f"Erreur : {e}"
        })

# ✅ Test local
if __name__ == "__main__":
    uvicorn.run("churn_api:app", host="0.0.0.0", port=8000, reload=True)
