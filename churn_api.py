from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, Depends, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError
import joblib, pandas as pd

# Charger le pipeline
MODEL_PATH = "churn_xgb_model.joblib"
pipeline = joblib.load(MODEL_PATH)

# Clé API de sécurité simple
API_KEY = "supersecretkey"

# Fonction de vérification de la clé API
def verify_api_key(request: Request):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide")

# Pydantic Model
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

# FastAPI instance
app = FastAPI(title="Customer Churn Prediction API (Batch)",
              description="API REST avec validation Pydantic, endpoints single & batch.",
              version="3.0.0")

# Page d'accueil HTML
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>Bienvenue sur l'API de Prédiction de Churn</h2>
    <p>Utilisez <code>/form</code> pour un test manuel ou <code>/predict</code> pour une requête API.</p>
    """

# Vérification de santé
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

# Prédiction single (avec sécurité API)
@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(payload: CustomerFeatures):
    try:
        df = pd.DataFrame([payload.dict()])
        proba = pipeline.predict_proba(df)[0, 1]
        return {"churn_probability": float(proba)}
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Prédiction batch (avec sécurité API)
@app.post("/predict_batch", dependencies=[Depends(verify_api_key)])
def predict_batch(payloads: List[CustomerFeatures]):
    try:
        df = pd.DataFrame([p.dict() for p in payloads])
        probs = pipeline.predict_proba(df)[:, 1].tolist()
        return {"churn_probabilities": probs}
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Formulaire HTML
@app.get("/form", response_class=HTMLResponse)
def show_form():
    return """
    <h2>Formulaire de Prédiction du Churn</h2>
    <form action="/form_predict" method="post">
        <label>Tenure:</label><input type="number" step="any" name="Tenure"><br>
        <label>PreferredLoginDevice:</label><input type="text" name="PreferredLoginDevice"><br>
        <label>CityTier:</label><input type="number" name="CityTier"><br>
        <label>WarehouseToHome:</label><input type="number" step="any" name="WarehouseToHome"><br>
        <label>PreferredPaymentMode:</label><input type="text" name="PreferredPaymentMode"><br>
        <label>Gender:</label><input type="text" name="Gender"><br>
        <label>HourSpendOnApp:</label><input type="number" step="any" name="HourSpendOnApp"><br>
        <label>NumberOfDeviceRegistered:</label><input type="number" name="NumberOfDeviceRegistered"><br>
        <label>PreferedOrderCat:</label><input type="text" name="PreferedOrderCat"><br>
        <label>SatisfactionScore:</label><input type="number" name="SatisfactionScore"><br>
        <label>MaritalStatus:</label><input type="text" name="MaritalStatus"><br>
        <label>NumberOfAddress:</label><input type="number" name="NumberOfAddress"><br>
        <label>Complain:</label><input type="number" name="Complain"><br>
        <label>OrderAmountHikeFromlastYear:</label><input type="number" step="any" name="OrderAmountHikeFromlastYear"><br>
        <label>CouponUsed:</label><input type="number" step="any" name="CouponUsed"><br>
        <label>OrderCount:</label><input type="number" step="any" name="OrderCount"><br>
        <label>DaySinceLastOrder:</label><input type="number" step="any" name="DaySinceLastOrder"><br>
        <label>CashbackAmount:</label><input type="number" step="any" name="CashbackAmount"><br>
        <input type="submit" value="Prédire">
    </form>
    """

# Traitement du formulaire
@app.post("/form_predict", response_class=HTMLResponse)
def form_predict(
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
    CashbackAmount: float = Form(...),
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
            "CashbackAmount": CashbackAmount,
        }
        df = pd.DataFrame([data])
        proba = pipeline.predict_proba(df)[0, 1]
        return f"<h3>Probabilité de churn : {round(proba*100, 2)}%</h3><a href='/form'>⟵ Retour</a>"
    except Exception as e:
        return f"<p>Erreur : {str(e)}</p>"
