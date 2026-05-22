"""Operational metrics utilities for regression tolerance-based evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer


def tolerance_accuracy(
    y_true,
    y_pred,
    tolerance: float = 10,
) -> float:
    """
    Calcula el porcentaje de predicciones cuyo error absoluto
    esta dentro de una tolerancia en minutos.
    """
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    absolute_errors = np.abs(y_true_array - y_pred_array)
    return float(np.mean(absolute_errors <= tolerance))


def tolerance_accuracy_10(y_true, y_pred) -> float:
    """Metrica especifica para acierto dentro de +/-10 minutos."""
    return tolerance_accuracy(y_true, y_pred, tolerance=10)


tolerance_accuracy_10_scorer = make_scorer(
    tolerance_accuracy_10,
    greater_is_better=True,
)


def evaluate_tolerance_accuracies(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tolerances: list[int] | None = None,
) -> pd.DataFrame:
    """Evalua acierto operacional para un modelo en varias tolerancias."""
    if tolerances is None:
        tolerances = [5, 10, 15, 20]

    y_pred = model.predict(X_test)
    rows: list[dict[str, float | int]] = []

    for tolerance in tolerances:
        accuracy = tolerance_accuracy(y_test, y_pred, tolerance)
        rows.append(
            {
                "tolerance_minutes": tolerance,
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100,
            }
        )

    return pd.DataFrame(rows)


def evaluate_tolerance_accuracies_for_models(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tolerances: list[int] | None = None,
) -> pd.DataFrame:
    """Evalua acierto por tolerancia para multiples modelos."""
    if tolerances is None:
        tolerances = [5, 10, 15, 20]

    rows = []

    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)

        for tolerance in tolerances:
            acc = tolerance_accuracy(y_test, y_pred, tolerance)
            rows.append(
                {
                    "model": model_name,
                    "tolerance_minutes": tolerance,
                    "accuracy": acc,
                    "accuracy_percent": acc * 100,
                }
            )

    return pd.DataFrame(rows)


def save_operational_accuracy(
    accuracy_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Guarda metricas operacionales por tolerancia."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    accuracy_df.to_csv(path, index=False)


def plot_tolerance_accuracy_comparison(
    accuracy_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Grafica el acierto por tolerancia para comparar modelos."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pivot_df = accuracy_df.pivot_table(
        index="tolerance_minutes",
        columns="model",
        values="accuracy_percent",
        aggfunc="mean",
    )

    plt.figure(figsize=(10, 6))
    for column in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[column], marker="o", label=column)

    plt.xlabel("Tolerancia en minutos")
    plt.ylabel("Acierto (%)")
    plt.title("Acierto operacional por tolerancia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
