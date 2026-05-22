"""Funciones de evaluacion para modelos de clasificacion supervisada."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
import re

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
)


def compute_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    average: str = "weighted",
) -> dict[str, float]:
    """Calcula metricas principales de clasificacion."""
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
    }


def evaluate_classifier(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    average: str = "weighted",
) -> dict[str, float]:
    """Evalua un modelo entrenado sobre un set de prueba."""
    predictions = model.predict(X_test)
    return compute_classification_metrics(y_test, predictions, average=average)


def build_confusion_matrix_table(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> pd.DataFrame:
    """Construye una matriz de confusion en formato DataFrame."""
    matrix = confusion_matrix(y_true, y_pred)
    labels = pd.Index(sorted(set(y_true) | set(y_pred)))

    return pd.DataFrame(
        matrix,
        index=labels.map(lambda value: f"actual_{value}"),
        columns=labels.map(lambda value: f"pred_{value}"),
    )


def build_classification_report_table(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> pd.DataFrame:
    """Construye un reporte detallado por clase en DataFrame."""
    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report_dict).transpose()


def _sanitize_model_name(model_name: str) -> str:
    """Normaliza nombre de modelo para usarlo en nombres de archivos."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(model_name).strip().lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "modelo"


def evaluar_modelos_clasificacion(
    modelos: dict,
    X_test,
    y_test,
    output_dir: str = "results/metrics",
) -> pd.DataFrame:
    """Evalua modelos de clasificacion y guarda metricas/reportes en disco.

    Parametros:
    - modelos: diccionario con pares `nombre_modelo -> estimador/pipeline entrenado`.
    - X_test: features de prueba.
    - y_test: etiquetas verdaderas de prueba.
    - output_dir: ruta donde se guardan metricas y reportes.

    Retorna:
    - DataFrame comparativo con metricas por modelo.
    """
    if not isinstance(modelos, dict) or not modelos:
        raise ValueError("`modelos` debe ser un diccionario no vacio de modelos entrenados.")
    if X_test is None or y_test is None:
        raise ValueError("X_test e y_test no pueden ser None.")
    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("X_test e y_test deben contener datos para evaluar.")
    if len(X_test) != len(y_test):
        raise ValueError("X_test e y_test deben tener la misma cantidad de filas.")

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, float | str]] = []
    y_true = pd.Series(y_test).reset_index(drop=True)

    print("Iniciando evaluacion de modelos de clasificacion...")
    print(f"Modelos recibidos: {len(modelos)}")

    for model_name, model in modelos.items():
        if model is None or not hasattr(model, "predict"):
            raise TypeError(
                f"El modelo '{model_name}' no es valido. Debe implementar el metodo `predict`."
            )

        print(f"[Evaluando] {model_name}...")
        try:
            y_pred = pd.Series(model.predict(X_test)).reset_index(drop=True)
        except Exception as exc:
            raise RuntimeError(f"Error al predecir con el modelo '{model_name}': {exc}") from exc

        if len(y_pred) != len(y_true):
            raise ValueError(
                f"El modelo '{model_name}' retorno {len(y_pred)} predicciones, "
                f"pero se esperaban {len(y_true)}."
            )

        model_metrics = {
            "modelo": str(model_name),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        metrics_rows.append(model_metrics)

        report_text = classification_report(y_true, y_pred, zero_division=0)
        labels = sorted(set(y_true.astype(str)) | set(y_pred.astype(str)))
        matrix = confusion_matrix(y_true.astype(str), y_pred.astype(str), labels=labels)
        matrix_df = pd.DataFrame(
            matrix,
            index=[f"actual_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        )

        report_file = output_path / f"classification_report_{_sanitize_model_name(model_name)}.txt"
        report_file.write_text(
            "\n".join(
                [
                    f"Modelo: {model_name}",
                    "",
                    "Classification Report:",
                    report_text,
                    "",
                    "Confusion Matrix:",
                    matrix_df.to_string(),
                    "",
                ]
            ),
            encoding="utf-8",
        )

        print(f"[OK] {model_name} evaluado. Reporte guardado en: {report_file}")

    if not metrics_rows:
        raise ValueError("No fue posible calcular metricas para los modelos proporcionados.")

    metricas_df = pd.DataFrame(metrics_rows).sort_values(
        by="f1_macro", ascending=False
    ).reset_index(drop=True)
    metrics_file = output_path / "classification_metrics.csv"
    metricas_df.to_csv(metrics_file, index=False, encoding="utf-8")

    print(f"Metricas comparativas guardadas en: {metrics_file}")
    return metricas_df


def obtener_mejor_modelo(metricas_df: pd.DataFrame, metrica: str = "f1_macro") -> str:
    """Retorna el nombre del mejor modelo segun una metrica del DataFrame."""
    if not isinstance(metricas_df, pd.DataFrame) or metricas_df.empty:
        raise ValueError("`metricas_df` debe ser un DataFrame no vacio.")
    if "modelo" not in metricas_df.columns:
        raise ValueError("`metricas_df` debe contener la columna 'modelo'.")
    if metrica not in metricas_df.columns:
        raise ValueError(
            f"La metrica '{metrica}' no existe en metricas_df. "
            f"Metricas disponibles: {[col for col in metricas_df.columns if col != 'modelo']}"
        )

    metrica_values = pd.to_numeric(metricas_df[metrica], errors="coerce")
    if metrica_values.isna().all():
        raise ValueError(f"La metrica '{metrica}' no contiene valores numericos validos.")

    best_index = metrica_values.idxmax()
    best_model = str(metricas_df.loc[best_index, "modelo"])
    return best_model
