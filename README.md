# Time Series Forecasting

A modular ML project comparing three forecasting approaches on a synthetic hourly time series:

| Model | Type | Strengths |
|---|---|---|
| **SARIMA** | Statistical | Interpretable, handles seasonality |
| **XGBoost** | Tree-based ML | Fast, handles nonlinearity, lag features |
| **LSTM** | Deep Learning | Captures long-range temporal patterns |

---

## Project Structure

```
ts-forecasting/
├── main.py                  # Full pipeline entry point
├── requirements.txt
├── data/
│   └── generate_data.py     # Synthetic time series generator
├── src/
│   ├── data_loader.py       # Load + train/test split
│   ├── features.py          # Lag, rolling, and calendar features
│   ├── evaluation.py        # MAE, RMSE, MAPE + model comparison
│   ├── visualization.py     # Forecast, residual, and metric plots
│   └── models/
│       ├── arima_model.py   # SARIMA wrapper
│       ├── xgboost_model.py # XGBoost with feature engineering
│       └── lstm_model.py    # LSTM with sliding window
└── outputs/                 # Saved plots, metrics CSV, model files
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/ts-forecasting.git
cd ts-forecasting
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py
```

Outputs land in `outputs/`:
- `01_raw_series.png` — raw data overview
- `02_forecast_comparison.png` — all models vs actual
- `03_metrics_comparison.png` — MAE / RMSE / MAPE bar charts
- `04_residuals.png` — per-model residual plots
- `metrics.csv` — final model leaderboard

---

## Swap in Your Own Data

Edit `main.py` and point `DATA_PATH` to any CSV with columns `ds` (datetime) and `y` (numeric target). The pipeline handles the rest.

---

## Results (Synthetic Data)

> Actual numbers vary per run. SARIMA typically wins on short horizons; XGBoost excels with enough history for lag features; LSTM shines on longer sequences.

## Development Notes

This project was developed with the assistance of AI coding tools, including Claude. The project concept, model design, interpretation of results, testing, documentation, and project structure were reviewed and curated by the author.
---

## License

MIT
