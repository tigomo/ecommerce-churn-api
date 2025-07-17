from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import uvicorn


# Initialiser l’application FastAPI
app = FastAPI()

# Charger le modèle
model = joblib.load("model.pkl")  # Assurez-vous que model.pkl est bien dans le repo

# Configuration du dossier templates
templates = Jinja2Templates(directory="templates")

# Route racine
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# Route pour le formulaire
@app.post("/predict", response_class=HTMLResponse)
async def predict_from_form(
    request: Request,
    Tenure: int = Form(...),
    PreferredLoginDevice: str = Form(...),
    CityTier: int = Form(...),
    WarehouseToHome: int = Form(...),
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
    # Créer un dictionnaire avec les données du formulaire
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
    probs = model.predict_proba(df)[:, 1]
    prediction = int(probs[0] >= 0.5)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": prediction,
        "probability": f"{probs[0]*100:.2f}%"
    })

# Pour lancer localement si nécessaire
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
