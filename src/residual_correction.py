"""Residual correction utilities for regression models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

try:
    from model_evaluation import calculate_regression_metrics  # type: ignore
except ImportError:
    from .model_evaluation import calculate_regression_metrics  # type: ignore

try:
    from operational_metrics import tolerance_accuracy  # type: ignore
except ImportError:
    from .operational_metrics import tolerance_accuracy  # type: ignore


def train_residual_correction_model(
    base_model: Any,
    residual_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Any, Any]:
    """
    Entrena modelo base y luego modelo de residuos sobre el conjunto de entrenamiento.
    """
    fitted_base = clone(base_model)
    fitted_residual = clone(residual_model)

    fitted_base.fit(X_train, y_train)
    y_train_pred = fitted_base.predict(X_train)
    residuals = y_train.reset_index(drop=True) - pd.Series(y_train_pred)

    fitted_residual.fit(X_train, residuals)
    return fitted_base, fitted_residual


def predict_with_residual_correction(
    base_model: Any,
    residual_model: Any,
    X,
):
    """
    Predice y suma correccion de residuo.
    """
    base_pred = np.asarray(base_model.predict(X))
    residual_pred = np.asarray(residual_model.predict(X))
    return base_pred + residual_pred


def evaluate_residual_correction(
    base_model: Any,
    residual_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Evalua prediccion corregida con MAE, RMSE, R2 y tolerancias.
    """
    y_pred = predict_with_residual_correction(base_model, residual_model, X_test)
    metrics = calculate_regression_metrics(y_test, y_pred)

    error = pd.Series(y_test).reset_index(drop=True) - pd.Series(y_pred)
    abs_error = error.abs()

    metrics.update(
        {
            "mean_error": float(error.mean()),
            "median_absolute_error": float(abs_error.median()),
            "p90_absolute_error": float(abs_error.quantile(0.90)),
            "acc_5_min": float(tolerance_accuracy(y_test, y_pred, tolerance=5) * 100),
            "acc_10_min": float(tolerance_accuracy(y_test, y_pred, tolerance=10) * 100),
            "acc_15_min": float(tolerance_accuracy(y_test, y_pred, tolerance=15) * 100),
            "acc_20_min": float(tolerance_accuracy(y_test, y_pred, tolerance=20) * 100),
        }
    )
    return metrics
