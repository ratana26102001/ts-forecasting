"""
main.py
─────────────────────────────────────────────────
Time Series Forecasting  |  Full Pipeline
Models: SARIMA · XGBoost · LSTM
─────────────────────────────────────────────────

Usage:
    python main.py

Outputs (outputs/):
    01_raw_series.png
    02_forecast_comparison.png
    03_metrics_comparison.png
    04_residuals.png
    metrics.csv
    sarima_model.pkl
    xgb_model.json / xgb_model_scaler.pkl
    lstm_model.keras / lstm_model_scaler.pkl
"""

from pathlib import Path
import pandas as pd

# ── local imports ──────────────────────────────────────────────────────────
from data.generate_data import generate_synthetic_ts
from src.data_loader     import load_data, train_test_split_ts
from src.features        import build_feature_matrix
from src.evaluation      import evaluate_model, compare_models
from src.visualization   import (
    plot_raw_series,
    plot_forecast,
    plot_metrics,
    plot_residuals,
)
from src.models import SARIMAModel, XGBoostForecaster, LSTMForecaster

# ── config ─────────────────────────────────────────────────────────────────
DATA_PATH  = "data/timeseries.csv"
FREQ       = "H"       # hourly data
N_DAYS     = 365       # 1 year of synthetic data
TEST_SIZE  = 0.15      # 15 % held out for evaluation

# Tip: set SARIMA_SEASONAL_M = 0 (no seasonality) for speed during development
SARIMA_ORDER          = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 24)   # 24 → daily seasonality in hourly data

LSTM_WINDOW = 48       # look-back window (hours)
LSTM_EPOCHS = 30


def main():
    Path("outputs").mkdir(exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────
    if not Path(DATA_PATH).exists():
        print("Generating synthetic dataset …")
        generate_synthetic_ts(n_days=N_DAYS, freq=FREQ, output_path=DATA_PATH)

    df = load_data(DATA_PATH)
    plot_raw_series(df)

    train_raw, test_raw = train_test_split_ts(df, test_size=TEST_SIZE)
    y_test = test_raw["y"].values
    n_test = len(test_raw)

    # ── 2. Feature engineering (XGBoost) ─────────────────────────────────
    df_feat       = build_feature_matrix(df)
    n_feat_train  = int(len(df_feat) * (1 - TEST_SIZE))
    train_feat    = df_feat.iloc[:n_feat_train]
    test_feat     = df_feat.iloc[n_feat_train:]
    y_test_feat   = test_feat["y"].values

    # ── 3. SARIMA ─────────────────────────────────────────────────────────
    # Note: fitting a full SARIMA(1,1,1)x(1,1,1,24) on 8000+ rows can take
    # several minutes. Reduce n_days or set seasonal_order=(0,0,0,0) to test.
    sarima       = SARIMAModel(SARIMA_ORDER, SARIMA_SEASONAL_ORDER)
    sarima.fit(train_raw["y"].values)
    sarima_preds = sarima.predict(n_test)

    # ── 4. XGBoost ────────────────────────────────────────────────────────
    val_split   = int(len(train_feat) * 0.9)
    xgb_model   = XGBoostForecaster()
    xgb_model.fit(train_feat.iloc[:val_split], train_feat.iloc[val_split:])
    xgb_preds   = xgb_model.predict(test_feat)

    # ── 5. LSTM ───────────────────────────────────────────────────────────
    lstm_model  = LSTMForecaster(window_size=LSTM_WINDOW, epochs=LSTM_EPOCHS)
    lstm_model.fit(train_raw["y"].values)
    lstm_preds  = lstm_model.predict(train_raw["y"].values, n_steps=n_test)

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    results = [
        evaluate_model("SARIMA",  y_test,      sarima_preds),
        evaluate_model("XGBoost", y_test_feat, xgb_preds),
        evaluate_model("LSTM",    y_test,      lstm_preds),
    ]
    metrics_df = compare_models(results)

    print("\n── Model Leaderboard ──────────────────────────────")
    print(metrics_df.to_string())
    print("───────────────────────────────────────────────────\n")
    metrics_df.to_csv("outputs/metrics.csv")

    # ── 7. Visualise ──────────────────────────────────────────────────────
    predictions = {
        "SARIMA":  sarima_preds,
        "XGBoost": xgb_preds,
        "LSTM":    lstm_preds,
    }
    plot_forecast(train_raw, test_raw, predictions)
    plot_metrics(metrics_df)
    plot_residuals(test_raw, predictions)

    # ── 8. Save models ────────────────────────────────────────────────────
    sarima.save()
    xgb_model.save()
    lstm_model.save()

    print("\n✅  Pipeline complete — see outputs/ for all artefacts.")


if __name__ == "__main__":
    main()
