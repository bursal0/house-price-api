from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# 🔥 SAFE MODEL LOAD
model = None
try:
    model = joblib.load("final_model.pkl")
    print("Model başarıyla yüklendi 🚀")
except Exception as e:
    print("Model yüklenemedi:", e)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(data: dict):
    try:
        if model is None:
            return {"error": "Model yüklenemedi"}

        df = pd.DataFrame([data])
        prediction = model.predict(df)
        price = np.expm1(prediction[0])

        return {"predicted_price": float(price)}

    except Exception as e:
        return {"error": str(e)}