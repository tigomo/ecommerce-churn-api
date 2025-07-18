from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import io
import uvicorn

app = FastAPI(title="Customer Churn Prediction API (Batch)",
              description="API REST avec validation Pydantic, endpoints single & batch.",
              version="3.0.0" )

# Charger le modèle
model = joblib.load("model.pkl")  # Assurez-vous que model.pkl est bien dans le même dossier

# Pour exécuter FastAPI depuis Jupyter Notebook
nest_asyncio.apply()

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

# ----- PARTIE HTML COMMENTÉE -----
# from fastapi import Form, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# templates = Jinja2Templates(directory="templates")

# @app.get("/", response_class=HTMLResponse)
# def read_root(request: Request):
#     return templates.TemplateResponse("form.html", {"request": request})

# @app.post("/predict", response_class=HTMLResponse)
# async def predict_from_form(
#     request: Request,
#     Tenure: int = Form(...),
#     PreferredLoginDevice: str = Form(...),
#     CityTier: int = Form(...),
#     WarehouseToHome: int = Form(...),
#     PreferredPaymentMode: str = Form(...),
#     Gender: str = Form(...),
#     HourSpendOnApp: float = Form(...),
#     NumberOfDeviceRegistered: int = Form(...),
#     PreferedOrderCat: str = Form(...),
#     SatisfactionScore: int = Form(...),
#     MaritalStatus: str = Form(...),
#     NumberOfAddress: int = Form(...),
#     Complain: int = Form(...),
#     OrderAmountHikeFromlastYear: float = Form(...),
#     CouponUsed: float = Form(...),
#     OrderCount: float = Form(...),
#     DaySinceLastOrder: float = Form(...),
#     CashbackAmount: float = Form(...)
# ):
#     data = {
#         "Tenure": Tenure,
#         "PreferredLoginDevice": PreferredLoginDevice,
#         "CityTier": CityTier,
#         "WarehouseToHome": WarehouseToHome,
#         "PreferredPaymentMode": PreferredPaymentMode,
#         "Gender": Gender,
#         "HourSpendOnApp": HourSpendOnApp,
#         "NumberOfDeviceRegistered": NumberOfDeviceRegistered,
#         "PreferedOrderCat": PreferedOrderCat,
#         "SatisfactionScore": SatisfactionScore,
#         "MaritalStatus": MaritalStatus,
#         "NumberOfAddress": NumberOfAddress,
#         "Complain": Complain,
#         "OrderAmountHikeFromlastYear": OrderAmountHikeFromlastYear,
#         "CouponUsed": CouponUsed,
#         "OrderCount": OrderCount,
#         "DaySinceLastOrder": DaySinceLastOrder,
#         "CashbackAmount": CashbackAmount
#     }

#     df = pd.DataFrame([data])
#     probs = model.predict_proba(df)[:, 1]
#     prediction = int(probs[0] >= 0.5)

#     return templates.TemplateResponse("result.html", {
#         "request": request,
#         "prediction": prediction,
#         "probability": f"{probs[0]*100:.2f}%"
#     })

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


# ----- NOUVELLE ROUTE POUR LES FICHIERS JSON -----
@app.post("/predict_json_file/")
async def predict_from_json_file(file: UploadFile = File(...)):
    try:
        # Lire le contenu du fichier JSON
        contents = await file.read()
        
        # Charger le contenu dans un DataFrame
        df = pd.read_json(io.BytesIO(contents))

        # Faire les prédictions
        predictions = model.predict(df)

        # Retourner les résultats sous forme de liste
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        return {"error": str(e)}

# Lancer l'API localement
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
