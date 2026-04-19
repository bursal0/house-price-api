from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI()

try:
    model = joblib.load("final_model.pkl")
    print("Model başarıyla yüklendi 🚀")
except Exception as e:
    print("Model yüklenemedi:", e)


@app.get("/")
def read_root():
    return {"message": "API çalışıyor"}


@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        prediction = model.predict(df)
        price = np.expm1(prediction[0])

        return {"predicted_price": float(price)}

    except Exception as e:
        return {"error": str(e)}

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request}