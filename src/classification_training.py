"""Funciones base para entrenamiento de modelos de clasificacion supervisada."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import re

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from .classification_utils import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
except ImportError:  # pragma: no cover - soporte para ejecucion directa del script
    from classification_utils import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE


@dataclass(frozen=True)
class ClassificationSplit:
    """Contenedor tipado para el split de entrenamiento y prueba."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_dataset_for_classification(
    data: pd.DataFrame,
    target_column: str,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ClassificationSplit:
    """[LEGACY] Separa features/target y genera split estratificado.

    Nota:
    - Esta funcion se mantiene por compatibilidad con codigo previo del repositorio.
    - El flujo oficial del notebook de Lucas usa `preparar_datos_clasificacion`
      desde `src.classification_utils`.
    """
    if target_column not in data.columns:
        raise KeyError(f"La columna objetivo '{target_column}' no existe en el DataFrame")

    X = data.drop(columns=[target_column])
    y = data[target_column]
    stratify_labels = y if y.nunique(dropna=False) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    return ClassificationSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def build_baseline_classifiers(
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, BaseEstimator]:
    """[LEGACY] Construye un set base de clasificadores para pruebas iniciales.

    Nota:
    - Se conserva por compatibilidad interna.
    - El flujo oficial de Lucas usa `entrenar_modelos_clasificacion`, que define
      explicitamente los modelos objetivo del entregable.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }


def train_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    classifiers: Mapping[str, BaseEstimator] | None = None,
) -> dict[str, BaseEstimator]:
    """[LEGACY] Entrena clasificadores y retorna modelos ajustados.

    Nota:
    - Funcion mantenida para compatibilidad con rutas antiguas.
    - El notebook oficial de Lucas no la utiliza.
    """
    models_to_train = classifiers or build_baseline_classifiers()
    trained_models: dict[str, BaseEstimator] = {}

    for model_name, model in models_to_train.items():
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)
        trained_models[model_name] = fitted_model

    return trained_models


def _build_classification_model_registry(
    random_state: int,
) -> dict[str, BaseEstimator]:
    """Crea configuracion base de clasificadores para pipelines."""
    return {
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        ),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
    }


def entrenar_modelos_clasificacion(
    X_train,
    y_train,
    preprocessor,
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Entrena modelos base de clasificacion en pipelines de Scikit-learn.

    Cada modelo se entrena con un pipeline:
    `Pipeline([('preprocessor', preprocessor), ('model', estimador)])`.

    Retorna un diccionario `nombre_modelo -> pipeline_entrenado`.
    """
    if X_train is None or y_train is None:
        raise ValueError("X_train e y_train no pueden ser None.")
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("X_train e y_train deben contener datos para entrenar.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train e y_train deben tener la misma cantidad de filas.")
    if preprocessor is None:
        raise ValueError("Se requiere un preprocessor valido para construir los pipelines.")
    if not hasattr(preprocessor, "fit") or not hasattr(preprocessor, "transform"):
        raise TypeError(
            "El argumento `preprocessor` debe ser compatible con la API de Scikit-learn "
            "(fit/transform), por ejemplo ColumnTransformer."
        )

    model_registry = _build_classification_model_registry(random_state=random_state)
    trained_pipelines: dict[str, Pipeline] = {}

    print("Iniciando entrenamiento de modelos base de clasificacion...")
    print(f"Total de modelos a entrenar: {len(model_registry)}")

    for model_name, estimator in model_registry.items():
        print(f"[Entrenando] {model_name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline
        print(f"[OK] {model_name} entrenado.")

    print("Entrenamiento base completado.")
    return trained_pipelines


def _sanitize_model_name(model_name: str) -> str:
    """Normaliza nombre de modelo para usarlo en archivo."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(model_name).strip().lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "modelo"


def guardar_modelo(
    modelo,
    model_name: str,
    output_dir: str = "models/trained_models",
) -> str:
    """Guarda un modelo de clasificacion (idealmente Pipeline completo) usando joblib.

    Retorna la ruta absoluta del archivo guardado.
    """
    if modelo is None:
        raise ValueError("El argumento `modelo` no puede ser None.")
    if not model_name or not str(model_name).strip():
        raise ValueError("`model_name` debe ser un string no vacio.")

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    file_name = f"classification_{_sanitize_model_name(model_name)}.joblib"
    model_path = output_path / file_name

    try:
        joblib.dump(modelo, model_path)
    except Exception as exc:
        raise RuntimeError(f"No fue posible guardar el modelo en {model_path}: {exc}") from exc

    return str(model_path.resolve())


def cargar_modelo(model_path: str):
    """Carga un modelo serializado con joblib desde disco."""
    if not model_path or not str(model_path).strip():
        raise ValueError("`model_path` debe ser una ruta valida no vacia.")

    path = Path(model_path).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de modelo en: {path.resolve()}")

    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"No fue posible cargar el modelo desde {path.resolve()}: {exc}") from exc
