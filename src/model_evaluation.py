"""Evaluation utilities for supervised and unsupervised workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate baseline regression metrics.

    Parameters
    ----------
    y_true:
        Real target values.
    y_pred:
        Predicted values.

    Returns
    -------
    dict[str, float]
        Dictionary with MAE, RMSE and R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def evaluate_regression_models(
    trained_models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate multiple regression models using MAE, RMSE and R2.

    Parameters
    ----------
    trained_models:
        Dictionary with trained models.
    X_test:
        Test predictors.
    y_test:
        Real target values from test set.

    Returns
    -------
    pd.DataFrame
        Metrics table sorted by RMSE ascending.
    """
    results: list[dict[str, float | str]] = []

    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        metrics = calculate_regression_metrics(y_test, y_pred)

        results.append(
            {
                "model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)

    return results_df


def get_best_regression_model_name(metrics_df: pd.DataFrame) -> str:
    """
    Return the best baseline model name using RMSE, MAE and R2.

    Main criterion is lower RMSE.
    """
    required_columns = {"model", "MAE", "RMSE", "R2"}

    if not required_columns.issubset(metrics_df.columns):
        raise ValueError(f"metrics_df debe contener columnas: {required_columns}")

    sorted_df = metrics_df.sort_values(
        by=["RMSE", "MAE", "R2"],
        ascending=[True, True, False],
    )

    return str(sorted_df.iloc[0]["model"])


def create_predictions_dataframe(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Create a DataFrame with real values, predictions and errors."""
    y_pred = model.predict(X_test)

    predictions_df = pd.DataFrame(
        {
            "real": y_test.reset_index(drop=True),
            "predicho": y_pred,
        }
    )

    predictions_df["error"] = predictions_df["real"] - predictions_df["predicho"]
    predictions_df["error_absoluto"] = predictions_df["error"].abs()

    return predictions_df


def save_metrics(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save evaluation metrics into a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(path, index=False)


def save_best_model_summary(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save a CSV summary row for the best baseline model."""
    best_model_name = get_best_regression_model_name(metrics_df)
    best_row = metrics_df[metrics_df["model"] == best_model_name].copy()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best_row.to_csv(path, index=False)


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
) -> None:
    """Generate and save a bar chart for a metric comparison."""
    if metric not in metrics_df.columns:
        raise ValueError(f"La metrica {metric} no existe en metrics_df.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df["model"], metrics_df[metric], color="#4C78A8")
    plt.xlabel("Modelo")
    plt.ylabel(metric)
    plt.title(f"Comparacion de modelos segun {metric}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_real_vs_predicted(
    predictions_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate and save real vs predicted scatter plot."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(predictions_df["real"], predictions_df["predicho"], alpha=0.7)

    min_value = min(predictions_df["real"].min(), predictions_df["predicho"].min())
    max_value = max(predictions_df["real"].max(), predictions_df["predicho"].max())

    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    plt.xlabel("Tiempo real")
    plt.ylabel("Tiempo predicho")
    plt.title("Tiempo real vs tiempo predicho - Mejor modelo base")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_residuals(
    predictions_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Generate and save residual plot for the best baseline model."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(predictions_df["predicho"], predictions_df["error"], alpha=0.7)
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Tiempo predicho")
    plt.ylabel("Error")
    plt.title("Residuos del mejor modelo base")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_regression_evaluation_workflow(
    trained_models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics_output_path: str | Path,
    best_summary_output_path: str | Path,
    rmse_plot_path: str | Path,
    mae_plot_path: str | Path,
    real_vs_pred_plot_path: str | Path,
    residuals_plot_path: str | Path,
) -> dict[str, Any]:
    """
    Execute full baseline regression evaluation workflow.

    This function:
    1. Calculates metrics.
    2. Selects the best baseline model.
    3. Saves metrics outputs.
    4. Generates comparison and diagnostics plots.
    """
    metrics_df = evaluate_regression_models(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test,
    )

    best_model_name = get_best_regression_model_name(metrics_df)
    best_model = trained_models[best_model_name]

    predictions_df = create_predictions_dataframe(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
    )

    save_metrics(metrics_df, metrics_output_path)
    save_best_model_summary(metrics_df, best_summary_output_path)

    plot_metric_comparison(metrics_df, "RMSE", rmse_plot_path)
    plot_metric_comparison(metrics_df, "MAE", mae_plot_path)
    plot_real_vs_predicted(predictions_df, real_vs_pred_plot_path)
    plot_residuals(predictions_df, residuals_plot_path)

    return {
        "metrics": metrics_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "predictions": predictions_df,
    }


def evaluate_regression_model(model, X_test, y_test):
    """Backward-compatible single-model regression evaluation helper."""
    y_pred = model.predict(X_test)
    return calculate_regression_metrics(pd.Series(y_test), np.asarray(y_pred))


def evaluate_classification_model(model, X_test, y_test):
    """Placeholder for future classification evaluation workflow."""
    raise NotImplementedError("Pendiente de implementacion")


def evaluate_clustering_model(model, X):
    """Placeholder for future clustering evaluation workflow."""
    raise NotImplementedError("Pendiente de implementacion")