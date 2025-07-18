from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import uvicorn

app = FastAPI()

# Charger le modèle
model = joblib.load("model.pkl")  # Assurez-vous que model.pkl est bien dans le même dossier

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

# ----- NOUVELLE ROUTE POUR LES FICHIERS JSON -----
@app.post("/predict_json")
async def predict_from_json(file: UploadFile = File(...)):
    try:
        # Lire le contenu du fichier JSON
        contents = await file.read()
        df = pd.read_json(contents)

        # Vérification rapide
        if df.empty:
            return JSONResponse(status_code=400, content={"error": "Fichier JSON vide ou invalide."})

        # Prédictions
        probs = model.predict_proba(df)[:, 1]
        predictions = (probs >= 0.5).astype(int)

        # Créer une réponse lisible
        results = []
        for i in range(len(df)):
            results.append({
                "id": i + 1,
                "prediction": int(predictions[i]),
                "probability": f"{probs[i]*100:.2f}%"
            })

        return {"results": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Lancer l'API localement
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
