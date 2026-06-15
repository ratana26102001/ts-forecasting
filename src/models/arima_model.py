"""
arima_model.py
SARIMA wrapper using statsmodels SARIMAX.

Default order (1,1,1)x(1,1,1,24) is tuned for hourly data with daily seasonality.
For quick runs, reduce to seasonal_order=(0,0,0,0).
"""

import warnings
import numpy as np
import joblib
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAModel:
    def __init__(
        self,
        order: tuple          = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 24),
    ):
        self.order          = order
        self.seasonal_order = seasonal_order
        self.result_        = None

    def fit(self, series: np.ndarray) -> "SARIMAModel":
        print(f"Fitting SARIMA{self.order}x{self.seasonal_order} …")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result_ = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        print("SARIMA fitted ✓")
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        return np.asarray(self.result_.forecast(steps=n_steps))

    def save(self, path: str = "outputs/sarima_model.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.result_, path)
        print(f"Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "SARIMAModel":
        obj       = cls()
        obj.result_ = joblib.load(path)
        return obj
