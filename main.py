from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from preprocessor import CustomPreprocessor

import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Templates
templates = Jinja2Templates(directory="templates")

# Model yükleme
model = None
try:
    model = joblib.load("final_model.pkl")
    print("Model başarıyla yüklendi 🚀")
except Exception as e:
    print("Model yüklenemedi:", e)

# Ana sayfa (GET + HEAD desteği)
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Predict endpoint
@app.post("/predict")
def predict(data: dict):
    try:
        if model is None:
            return {"error": "Model yüklenmedi"}

        if not data:
            return {"error": "Boş veri gönderildi"}

        print("GELEN DATA:", data)

        df = pd.DataFrame([data])

        # 🔥 Pipeline varsa direkt çalışır
        prediction = model.predict(df)

        price = np.expm1(prediction[0])

        return {"predicted_price": float(price)}

    except Exception as e:
        print("HATA:", e)
        return {"error": str(e)}