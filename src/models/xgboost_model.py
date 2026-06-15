"""
xgboost_model.py
XGBoost regressor on engineered lag + calendar features.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


FEATURE_COLS = [
    # Cyclical calendar
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend",
    # Lag features
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168",
    # Rolling statistics
    "rolling_mean_6", "rolling_mean_12", "rolling_mean_24",
    "rolling_std_6",  "rolling_std_12",
]


class XGBoostForecaster:
    def __init__(self, params: dict | None = None):
        self.params = params or {
            "n_estimators":       500,
            "learning_rate":      0.05,
            "max_depth":          6,
            "subsample":          0.8,
            "colsample_bytree":   0.8,
            "early_stopping_rounds": 30,
            "eval_metric":        "rmse",
            "random_state":       42,
        }
        self.model_        = None
        self.scaler_       = StandardScaler()
        self.feature_cols_ = None

    def _available(self, df: pd.DataFrame) -> list[str]:
        return [c for c in FEATURE_COLS if c in df.columns]

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame | None = None,
        target_col: str = "y",
    ) -> "XGBoostForecaster":
        self.feature_cols_ = self._available(train_df)
        X_tr = self.scaler_.fit_transform(train_df[self.feature_cols_])
        y_tr = train_df[target_col].values

        eval_set = None
        if val_df is not None:
            X_v  = self.scaler_.transform(val_df[self.feature_cols_])
            eval_set = [(X_v, val_df[target_col].values)]

        print("Fitting XGBoost …")
        self.model_ = xgb.XGBRegressor(**self.params)
        self.model_.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
        print("XGBoost fitted ✓")
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = self.scaler_.transform(test_df[self.feature_cols_])
        return self.model_.predict(X)

    def save(self, path: str = "outputs/xgb_model.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(path)
        joblib.dump(self.scaler_, path.replace(".json", "_scaler.pkl"))
        print(f"Saved → {path}")

    @classmethod
    def load(cls, model_path: str, scaler_path: str) -> "XGBoostForecaster":
        obj         = cls()
        obj.model_  = xgb.XGBRegressor()
        obj.model_.load_model(model_path)
        obj.scaler_ = joblib.load(scaler_path)
        return obj
