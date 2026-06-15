"""
evaluation.py
Regression metrics for time series: MAE, RMSE, MAPE.
"""

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def evaluate_model(name: str, y_true, y_pred) -> dict:
    return {
        "Model":    name,
        "MAE":      round(mae(y_true, y_pred),  4),
        "RMSE":     round(rmse(y_true, y_pred), 4),
        "MAPE (%)": round(mape(y_true, y_pred), 4),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results).set_index("Model")
    df["Rank"] = df["RMSE"].rank().astype(int)
    return df.sort_values("RMSE")
