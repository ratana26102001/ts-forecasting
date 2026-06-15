"""
data_loader.py
Load CSV data and perform a time-aware train/test split.
"""

import pandas as pd


def load_data(filepath: str = "data/timeseries.csv") -> pd.DataFrame:
    """Load time series from CSV. Expects columns: ds (datetime), y (float)."""
    df = pd.read_csv(filepath, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    print(f"Loaded {len(df):,} rows  |  {df['ds'].min()} → {df['ds'].max()}")
    return df


def train_test_split_ts(
    df: pd.DataFrame,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split — no shuffling."""
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    print(f"Train: {len(train):,}  |  Test: {len(test):,}")
    return train, test
