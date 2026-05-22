"""Model persistence and final reporting helpers for regression workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def save_model(
    model: Any,
    output_path: str | Path,
) -> None:
    """
    Save a trained model using joblib.

    Parameters
    ----------
    model:
        Trained model or Pipeline instance.
    output_path:
        Destination path for the .joblib file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(
    model_path: str | Path,
) -> Any:
    """
    Load a model stored with joblib.

    Parameters
    ----------
    model_path:
        Path to the saved .joblib model.

    Returns
    -------
    Any
        Loaded model object.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontro el modelo en: {path}")

    return joblib.load(path)


def save_final_metrics(
    final_metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save final metrics of the selected model."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final_metrics_df.to_csv(path, index=False)


def create_final_summary_text(
    final_model_name: str,
    final_metrics_df: pd.DataFrame,
    base_vs_optimized_df: pd.DataFrame | None = None,
) -> str:
    """Create a Markdown summary for the final regression model."""
    metrics_row = final_metrics_df.iloc[0]

    mae = metrics_row.get("MAE", "N/A")
    rmse = metrics_row.get("RMSE", "N/A")
    r2 = metrics_row.get("R2", "N/A")

    summary = f"""# Resumen final - Modelo de Regresion Cesar

## Objetivo

Predecir el tiempo de entrega en el contexto Logistica 4.0.

## Tipo de modelo

Regresion supervisada.

## Modelo final

{final_model_name}

## Metricas finales

- MAE: {mae}
- RMSE: {rmse}
- R2: {r2}

## Interpretacion

El MAE representa el error absoluto promedio del modelo en minutos. El RMSE penaliza con mayor fuerza los errores grandes y permite analizar que tanto se alejan algunas predicciones del valor real. El R2 indica que proporcion de la variabilidad del tiempo de entrega logra explicar el modelo.

## Utilidad para el negocio

Este modelo puede ayudar a una empresa logistica a estimar tiempos de entrega antes de realizar la operacion, permitiendo promesas de entrega mas realistas y una mejor planificacion operativa.

## Limitaciones

El modelo depende de la calidad del dataset y de las variables disponibles. No considera necesariamente eventos externos no registrados, como accidentes, cortes de ruta, fallas del vehiculo o cambios operativos imprevistos.

## Proximos pasos recomendados

- Probar el modelo con mas datos reales.
- Comparar con nuevos algoritmos.
- Revisar importancia de variables.
- Integrar el modelo en un flujo de prediccion operativo.
"""

    if base_vs_optimized_df is not None:
        summary += """

## Comparacion base vs optimizado

Se comparo el modelo base con su version optimizada para revisar si la optimizacion de hiperparametros mejoro el rendimiento.
"""

    return summary


def save_final_summary_text(
    summary_text: str,
    output_path: str | Path,
) -> None:
    """Save final summary text in Markdown format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary_text, encoding="utf-8")


def verify_model_file(
    model_path: str | Path,
) -> bool:
    """Verify that a serialized model file exists."""
    return Path(model_path).exists()


def save_regression_final_artifacts(
    final_model: Any,
    final_model_name: str,
    final_metrics_df: pd.DataFrame,
    model_output_path: str | Path,
    final_metrics_output_path: str | Path,
    final_summary_output_path: str | Path,
    base_vs_optimized_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """
    Save all final artifacts for regression line closure.

    Artifacts:
    - Final model (.joblib)
    - Final metrics (.csv)
    - Final summary (.md)
    """
    model_path = Path(model_output_path)
    metrics_path = Path(final_metrics_output_path)
    summary_path = Path(final_summary_output_path)

    save_model(final_model, model_path)
    save_final_metrics(final_metrics_df, metrics_path)

    summary_text = create_final_summary_text(
        final_model_name=final_model_name,
        final_metrics_df=final_metrics_df,
        base_vs_optimized_df=base_vs_optimized_df,
    )
    save_final_summary_text(summary_text, summary_path)

    if not verify_model_file(model_path):
        raise FileNotFoundError(f"No se guardo correctamente el modelo en: {model_path}")

    return {
        "model_path": model_path,
        "metrics_path": metrics_path,
        "summary_path": summary_path,
    }