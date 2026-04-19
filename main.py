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

# Ana sayfa
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Predict
@app.post("/predict")
def predict(data: dict):
    try:
        if model is None:
            return {"error": "Model yüklenmedi"}

        print("GELEN DATA:", data)

        df = pd.DataFrame([data])

        # ❌ feature_names_in_ KULLANMA
        prediction = model.predict(df)

        price = np.expm1(prediction[0])

        return {"predicted_price": float(price)}

    except Exception as e:
        print("HATA:", e)
        return {"error": str(e)}