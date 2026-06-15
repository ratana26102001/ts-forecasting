"""
generate_data.py
Generates a synthetic hourly time series with:
  - Linear trend
  - Daily seasonality
  - Weekly seasonality
  - Yearly seasonality
  - Gaussian noise
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_ts(
    n_days: int = 730,
    freq: str = "H",
    seed: int = 42,
    output_path: str = "data/timeseries.csv",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if freq == "H":
        n_periods = n_days * 24
        date_range = pd.date_range(start="2022-01-01", periods=n_periods, freq="h")
    else:
        n_periods = n_days
        date_range = pd.date_range(start="2022-01-01", periods=n_periods, freq=freq)

    t = np.arange(n_periods)

    # Components
    trend   = 0.005 * t
    daily   = (10 * np.sin(2 * np.pi * t / 24) + 5 * np.cos(2 * np.pi * t / 24)) if freq == "H" else 0
    weekly  = 3 * np.sin(2 * np.pi * t / (24 * 7 if freq == "H" else 7))
    yearly  = 15 * np.sin(2 * np.pi * t / (24 * 365 if freq == "H" else 365))
    noise   = rng.normal(0, 2, n_periods)

    values = np.clip(100 + trend + daily + weekly + yearly + noise, 0, None)

    df = pd.DataFrame({"ds": date_range, "y": values})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved → {output_path}  ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    df = generate_synthetic_ts()
    print(df.describe())
