"""Funciones de visualizacion para modelos de clasificacion supervisada."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree


def _sanitize_model_name(model_name: str) -> str:
    """Normaliza nombre de modelo para nombres de archivos."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(model_name).strip().lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "modelo"


def _ensure_output_dir(output_dir: str | Path) -> Path:
    """Crea el directorio de salida si no existe."""
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _extract_pipeline_components(modelo: Any) -> tuple[Any, Any | None]:
    """Extrae estimador final y preprocesador desde un modelo o Pipeline."""
    if modelo is None:
        raise ValueError("El argumento `modelo` no puede ser None.")

    if isinstance(modelo, Pipeline):
        estimator = modelo.named_steps.get("model")
        if estimator is None and modelo.steps:
            estimator = modelo.steps[-1][1]
        preprocessor = modelo.named_steps.get("preprocessor")
        return estimator, preprocessor

    return modelo, None


def _to_feature_list(feature_names: Iterable[str] | pd.Index | None) -> list[str] | None:
    """Convierte nombres de variables a lista de strings."""
    if feature_names is None:
        return None
    if isinstance(feature_names, pd.Index):
        return [str(value) for value in feature_names.tolist()]
    return [str(value) for value in list(feature_names)]


def _resolve_transformed_feature_names(
    modelo: Any,
    feature_names: Iterable[str] | pd.Index | None,
    expected_size: int | None = None,
) -> list[str]:
    """Resuelve nombres post-preprocesamiento si el pipeline los expone."""
    _, preprocessor = _extract_pipeline_components(modelo)
    base_feature_names = _to_feature_list(feature_names)

    transformed_names: list[str] | None = None
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            if base_feature_names is not None:
                transformed_names = [
                    str(name) for name in preprocessor.get_feature_names_out(base_feature_names)
                ]
            else:
                transformed_names = [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception:
            transformed_names = None

    if transformed_names is None:
        transformed_names = base_feature_names

    if transformed_names is None:
        size = expected_size if expected_size is not None else 0
        return [f"feature_{idx}" for idx in range(size)]

    if expected_size is not None and len(transformed_names) != expected_size:
        return [f"feature_{idx}" for idx in range(expected_size)]

    return transformed_names


def graficar_matriz_confusion(
    modelo,
    X_test,
    y_test,
    model_name: str,
    output_dir: str = "results/plots",
) -> str:
    """Grafica y guarda la matriz de confusion para un modelo de clasificacion."""
    estimator, _ = _extract_pipeline_components(modelo)
    if estimator is None or not hasattr(modelo, "predict"):
        raise TypeError("El modelo debe estar entrenado y exponer el metodo `predict`.")
    if X_test is None or y_test is None:
        raise ValueError("X_test e y_test no pueden ser None.")
    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("X_test e y_test deben contener registros para graficar.")

    y_true = pd.Series(y_test).astype(str).reset_index(drop=True)
    y_pred = pd.Series(modelo.predict(X_test)).astype(str).reset_index(drop=True)
    labels = sorted(set(y_true) | set(y_pred))

    output_path = _ensure_output_dir(output_dir)
    file_path = output_path / f"confusion_matrix_{_sanitize_model_name(model_name)}.png"

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        cmap="Blues",
        xticks_rotation=25,
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Matriz de confusion - {model_name}")
    fig.tight_layout()
    fig.savefig(file_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return str(file_path)


def graficar_importancia_variables(
    modelo,
    feature_names,
    model_name: str,
    output_dir: str = "results/plots",
    top_n: int = 15,
) -> str | None:
    """Grafica top-N de importancia de variables para modelos tipo arbol."""
    if top_n <= 0:
        raise ValueError("`top_n` debe ser mayor a 0.")

    estimator, _ = _extract_pipeline_components(modelo)
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        print(f"El modelo '{model_name}' no expone feature_importances_. Se omite grafico.")
        return None

    importances = np.asarray(estimator.feature_importances_, dtype="float64")
    if importances.size == 0:
        print(f"El modelo '{model_name}' no tiene importancias disponibles. Se omite grafico.")
        return None

    resolved_feature_names = _resolve_transformed_feature_names(
        modelo=modelo,
        feature_names=feature_names,
        expected_size=int(importances.size),
    )

    importance_df = (
        pd.DataFrame(
            {
                "feature": resolved_feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )

    output_path = _ensure_output_dir(output_dir)
    file_path = output_path / f"feature_importance_{_sanitize_model_name(model_name)}.png"

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(importance_df))))
    ax.barh(importance_df["feature"], importance_df["importance"], color="#1f77b4")
    ax.set_title(f"Top {len(importance_df)} importancia de variables - {model_name}")
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variable")
    fig.tight_layout()
    fig.savefig(file_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return str(file_path)


def graficar_arbol_decision(
    modelo,
    feature_names,
    class_names,
    output_dir: str = "results/plots",
) -> str | None:
    """Grafica y guarda el arbol de decision si el estimador final es DecisionTreeClassifier."""
    estimator, _ = _extract_pipeline_components(modelo)
    if not isinstance(estimator, DecisionTreeClassifier):
        print("El modelo proporcionado no es un DecisionTreeClassifier. Se omite grafico.")
        return None

    n_features = int(getattr(estimator, "n_features_in_", 0))
    resolved_feature_names = _resolve_transformed_feature_names(
        modelo=modelo,
        feature_names=feature_names,
        expected_size=n_features,
    )

    if not class_names:
        class_name_values = [str(value) for value in getattr(estimator, "classes_", [])]
    else:
        class_name_values = [str(value) for value in class_names]

    output_path = _ensure_output_dir(output_dir)
    file_path = output_path / "decision_tree_plot.png"

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        estimator,
        feature_names=resolved_feature_names,
        class_names=class_name_values if class_name_values else None,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree - Clasificacion de entregas")
    fig.tight_layout()
    fig.savefig(file_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return str(file_path)
