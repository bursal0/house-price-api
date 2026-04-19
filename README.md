# 🏠 House Price Prediction API

A machine learning API that predicts house prices based on property features. Built using FastAPI and an XGBoost model with a custom preprocessing pipeline.

---

## 🚀 Features

* Trained on Kaggle House Prices dataset
* Custom preprocessing pipeline (handling missing values & encoding)
* Consistent feature transformation using sklearn Pipeline
* FastAPI-based REST API
* Deployed and accessible online

---

## 🧠 Model

* Model: XGBoost Regressor
* Target transformation: log1p
* Prediction transformation: expm1
* Includes preprocessing + model in a single pipeline

---

## 📦 Tech Stack

* Python
* FastAPI
* Scikit-learn
* XGBoost
* Pandas / NumPy

---

## 🔗 API Usage

### Endpoint

POST /predict

---

### Example Request

```json
{
  "OverallQual": 7,
  "GrLivArea": 1500,
  "Neighborhood": "CollgCr"
}
```

---

### Example Response

```json
{
  "predicted_price": 204371.82
}
```

---

## ⚙️ Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 🌐 Live Demo

https://your-render-link.onrender.com/docs

---

## 📁 Project Structure

```
├── main.py
├── preprocessor.py
├── train_pipeline.py
├── final_model.pkl
├── requirements.txt
```

---

## 📌 Notes

* The model expects structured input similar to training data
* Missing features are handled inside the preprocessing pipeline
* Feature consistency is maintained using column alignment

---
