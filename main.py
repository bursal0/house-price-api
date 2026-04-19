from fastapi import FastAPI
from fastapi.responses import FileResponse
from preprocessor import CustomPreprocessor

import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Model yükleme
model = None
try:
    model = joblib.load("final_model.pkl")
    print("Model başarıyla yüklendi 🚀")
except Exception as e:
    print("Model yüklenemedi:", e)

# ❗ JINJA YOK
@app.get("/")
def home():
    return FileResponse("templates/index.html")

@app.post("/predict")
def predict(data: dict):
    try:
        if model is None:
            return {"error": "Model yüklenmedi"}

        df = pd.DataFrame([data])

        prediction = model.predict(df)
        price = np.expm1(prediction[0])

        return {"predicted_price": float(price)}

    except Exception as e:
        print("HATA:", e)
        return {"error": str(e)}