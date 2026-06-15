"""
lstm_model.py
Univariate LSTM forecaster using a sliding-window approach.
Autoregressive inference: feeds its own predictions back as input.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler


class LSTMForecaster:
    def __init__(
        self,
        window_size: int  = 48,
        units:       int  = 64,
        dropout:     float = 0.2,
        epochs:      int  = 50,
        batch_size:  int  = 64,
    ):
        self.window_size = window_size
        self.units       = units
        self.dropout     = dropout
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.model_      = None
        self.scaler_     = MinMaxScaler()

    # ── internal helpers ───────────────────────────────────────────────────

    def _build(self) -> Sequential:
        return Sequential([
            LSTM(self.units, return_sequences=True,
                 input_shape=(self.window_size, 1)),
            Dropout(self.dropout),
            LSTM(self.units // 2),
            Dropout(self.dropout),
            Dense(32, activation="relu"),
            Dense(1),
        ], name="lstm_forecaster")

    def _sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i - self.window_size: i])
            y.append(data[i])
        return np.array(X), np.array(y)

    # ── public API ─────────────────────────────────────────────────────────

    def fit(
        self,
        train_series: np.ndarray,
        val_series:   np.ndarray | None = None,
    ) -> "LSTMForecaster":
        scaled = self.scaler_.fit_transform(train_series.reshape(-1, 1))
        X_tr, y_tr = self._sequences(scaled)

        val_data = None
        if val_series is not None:
            ctx   = np.concatenate([train_series[-self.window_size:], val_series])
            s_val = self.scaler_.transform(ctx.reshape(-1, 1))
            X_v, y_v = self._sequences(s_val)
            val_data = (X_v, y_v)

        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(patience=3, factor=0.5, verbose=0),
        ]

        print(f"Fitting LSTM  (window={self.window_size}, units={self.units}, epochs≤{self.epochs}) …")
        self.model_ = self._build()
        self.model_.compile(optimizer="adam", loss="mse", metrics=["mae"])
        self.model_.fit(
            X_tr, y_tr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=0,
        )
        print("LSTM fitted ✓")
        return self

    def predict(self, context_series: np.ndarray, n_steps: int) -> np.ndarray:
        """Autoregressive multi-step forecast from the last window of context."""
        window = self.scaler_.transform(
            context_series[-self.window_size:].reshape(-1, 1)
        ).flatten().tolist()

        preds = []
        for _ in range(n_steps):
            x   = np.array(window[-self.window_size:]).reshape(1, self.window_size, 1)
            p   = float(self.model_.predict(x, verbose=0)[0, 0])
            preds.append(p)
            window.append(p)

        return self.scaler_.inverse_transform(
            np.array(preds).reshape(-1, 1)
        ).flatten()

    def save(self, path: str = "outputs/lstm_model.keras"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model_.save(path)
        joblib.dump(self.scaler_, path.replace(".keras", "_scaler.pkl"))
        print(f"Saved → {path}")

    @classmethod
    def load(cls, model_path: str, scaler_path: str) -> "LSTMForecaster":
        obj         = cls()
        obj.model_  = tf.keras.models.load_model(model_path)
        obj.scaler_ = joblib.load(scaler_path)
        return obj
