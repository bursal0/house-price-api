from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CustomPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X_processed = self._preprocess(X)
        self.columns = X_processed.columns
        return self

    def _preprocess(self, X):
        X = X.copy()

        expected_cols = [
            "LotFrontage", "Neighborhood", "Alley", "MasVnrType",
            "MasVnrArea", "TotalBsmtSF", "BsmtQual", "BsmtCond",
            "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
            "Electrical", "Fireplaces", "FireplaceQu",
            "GarageCars", "GarageType", "GarageFinish",
            "GarageQual", "GarageCond", "GarageArea",
            "GarageYrBlt", "PoolArea", "PoolQC", "Fence"
        ]

        for col in expected_cols:
            if col not in X.columns:
                X[col] = None

        X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"] \
            .transform(lambda x: x.fillna(x.mean()))

        X["LotFrontage"] = X["LotFrontage"].fillna(X["LotFrontage"].median())

        X.drop(["Id", "MiscFeature"], axis=1, inplace=True, errors="ignore")
        X["Alley"] = X["Alley"].fillna("None")
        X["Alley"] = X["Alley"].map({
            "Grvl": 1,
            "Pave": 2,
            "None": 0
        })
        X = pd.get_dummies(X, columns=["Alley"])

        X["MasVnrType"] = X["MasVnrType"].fillna("None")
        X.loc[X["MasVnrType"] == "None", "MasVnrArea"] = 0
        X["MasVnrArea"] = X["MasVnrArea"].fillna(X["MasVnrArea"].median())
        X = pd.get_dummies(X, columns=["MasVnrType"])

        cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
        for col in cols:
            X.loc[X["TotalBsmtSF"] == 0, col] = "None"
            X[col] = X[col].fillna("None")

        # ❗ BURAYI DÜZELT
        X["Electrical"] = X["Electrical"].fillna("SBrkr")

        X.loc[X["Fireplaces"] == 0, "FireplaceQu"] = "None"

        cols_cat = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
        X.loc[X["GarageCars"] == 0, cols_cat] = "None"

        cols_num = ["GarageArea", "GarageYrBlt"]
        X.loc[X["GarageCars"] == 0, cols_num] = 0

        X.loc[X["PoolArea"] == 0, "PoolQC"] = "None"
        X["Fence"] = X["Fence"].fillna("None")

        cat_cols = X.select_dtypes(include=["object"]).columns
        X = pd.get_dummies(X, columns=cat_cols)

        return X

    def transform(self, X):
        X_processed = self._preprocess(X)
        X_processed = X_processed.reindex(columns=self.columns, fill_value=0)
        return X_processed