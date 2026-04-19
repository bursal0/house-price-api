import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from preprocessor import CustomPreprocessor

train = pd.read_csv("train.csv")

X = train.drop("SalePrice", axis=1)
y = np.log1p(train["SalePrice"])

pipeline = Pipeline([
    ("prep", CustomPreprocessor()),
    ("model", XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

pipeline.fit(X, y)

joblib.dump(pipeline, "final_model.pkl")

print("Model başarıyla kaydedildi: final_model.pkl")