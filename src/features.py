"""
features.py
Feature engineering for tree/linear models.
Produces lag features, rolling statistics, and cyclical calendar encodings.
"""

import numpy as np
import pandas as pd

DEFAULT_LAGS    = [1, 2, 3, 24, 48, 168]
DEFAULT_WINDOWS = [6, 12, 24]


def add_datetime_features(df: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    df = df.copy()
    dt = df[date_col]
    df["hour"]        = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"]       = dt.dt.month
    df["is_weekend"]  = (dt.dt.dayofweek >= 5).astype(int)
    # Cyclical encodings prevent ordinal leakage
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]        / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "y",
    lags: list[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "y",
    windows: list[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        shifted = df[target_col].shift(1)
        df[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
        df[f"rolling_std_{w}"]  = shifted.rolling(w).std()
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "y",
    date_col: str = "ds",
    lags: list[int] = DEFAULT_LAGS,
    windows: list[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Full feature pipeline. Drops NaN rows introduced by lags/rolling."""
    df = add_datetime_features(df, date_col)
    df = add_lag_features(df, target_col, lags)
    df = add_rolling_features(df, target_col, windows)
    df = df.dropna().reset_index(drop=True)
    print(f"Feature matrix: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df
