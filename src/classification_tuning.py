"""Funciones de ajuste de hiperparametros para clasificadores."""

from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Any, Mapping
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

try:
    from .classification_utils import DEFAULT_RANDOM_STATE
except ImportError:  # pragma: no cover - soporte para ejecucion directa del script
    from classification_utils import DEFAULT_RANDOM_STATE


def _sanitize_name(name: str) -> str:
    """Normaliza texto para usarlo en nombres de archivo."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name).strip().lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "modelo"


def _validate_search_method(method: str) -> str:
    """Valida y normaliza el metodo de busqueda."""
    normalized = str(method).strip().lower()
    if normalized not in {"grid", "random"}:
        raise ValueError("`metodo` debe ser 'grid' o 'random'.")
    return normalized


def _resolve_parallel_jobs(n_jobs: int | None = None) -> int:
    """Define n_jobs efectivo priorizando estabilidad en Windows.

    Regla por defecto:
    - Windows (`os.name == "nt"`): `n_jobs=1` para evitar ruido de `loky/resource_tracker`.
    - Otros sistemas: `n_jobs=-1` para usar todos los nucleos.

    Si se provee `n_jobs`, se respeta tras validacion basica.
    """
    if n_jobs is not None:
        if not isinstance(n_jobs, int):
            raise TypeError("`n_jobs` debe ser un entero o None.")
        if n_jobs == 0:
            raise ValueError("`n_jobs=0` no es valido en Scikit-learn/joblib.")
        return n_jobs

    return 1 if os.name == "nt" else -1


def _resolve_effective_cv(y_train, cv: int) -> int:
    """Ajusta el valor de CV segun la distribucion de clases disponible."""
    if cv < 2:
        raise ValueError("`cv` debe ser >= 2.")

    y_series = pd.Series(y_train)
    if y_series.empty:
        raise ValueError("`y_train` no puede estar vacio.")
    if y_series.nunique(dropna=False) < 2:
        raise ValueError("Se requieren al menos 2 clases para optimizar clasificadores.")

    min_class_count = int(y_series.value_counts(dropna=False).min())
    effective_cv = min(cv, min_class_count)
    if effective_cv < 2:
        raise ValueError(
            "No hay suficientes muestras por clase para validacion cruzada. "
            "Aumenta datos o reduce desbalance extremo."
        )
    return effective_cv


def _build_search_spaces(
    random_state: int,
) -> dict[str, tuple[BaseEstimator, dict[str, list[Any]]]]:
    """Define modelos base y espacios de hiperparametros para optimizacion."""
    return {
        "random_forest": (
            RandomForestClassifier(random_state=random_state),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [5, 10, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__criterion": ["gini", "entropy"],
            },
        ),
        "logistic_regression": (
            LogisticRegression(random_state=random_state),
            {
                "model__C": [0.01, 0.1, 1, 10, 100],
                # En scikit-learn >=1.8, liblinear no soporta multiclase directo.
                "model__solver": ["lbfgs"],
                "model__class_weight": [None, "balanced"],
                "model__max_iter": [1000, 2000],
            },
        ),
    }


def _to_serializable(value: Any) -> Any:
    """Convierte tipos de numpy/pandas a tipos JSON serializables."""
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def run_grid_search_for_classification(
    model: BaseEstimator,
    param_grid: Mapping[str, list[Any]],
    X_train,
    y_train,
    scoring: str = "f1_weighted",
    cv: int = 5,
    n_jobs: int | None = None,
) -> GridSearchCV:
    """Ejecuta GridSearchCV para un clasificador."""
    resolved_n_jobs = _resolve_parallel_jobs(n_jobs)
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=resolved_n_jobs,
        verbose=1,
    )

    search.fit(X_train, y_train)
    return search


def run_randomized_search_for_classification(
    model: BaseEstimator,
    param_distributions: Mapping[str, list[Any]],
    X_train,
    y_train,
    scoring: str = "f1_weighted",
    cv: int = 5,
    n_iter: int = 20,
    n_jobs: int | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> RandomizedSearchCV:
    """Ejecuta RandomizedSearchCV para un clasificador."""
    resolved_n_jobs = _resolve_parallel_jobs(n_jobs)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        scoring=scoring,
        cv=cv,
        n_iter=n_iter,
        random_state=random_state,
        n_jobs=resolved_n_jobs,
        verbose=1,
    )

    search.fit(X_train, y_train)
    return search


def extract_search_results(search_object: GridSearchCV | RandomizedSearchCV) -> pd.DataFrame:
    """Retorna los resultados de CV como DataFrame."""
    return pd.DataFrame(search_object.cv_results_)


def summarize_best_model(
    search_object: GridSearchCV | RandomizedSearchCV,
    model_name: str,
) -> dict[str, Any]:
    """Resume mejor configuracion encontrada por proceso de tuning."""
    return {
        "model_name": model_name,
        "best_score": float(search_object.best_score_),
        "best_params": search_object.best_params_,
    }


def optimizar_modelo_clasificacion(
    X_train,
    y_train,
    preprocessor,
    metodo: str = "grid",
    random_state: int = 42,
    scoring: str = "f1_macro",
    cv: int = 5,
) -> dict[str, Any]:
    """Optimiza clasificadores con validacion cruzada y scoring configurable.

    Retorna el mejor resultado global con:
    - best_estimator
    - best_params
    - best_score
    - search_object
    - model_name
    Tambien retorna resultados detallados por modelo en `optimized_models`.
    """
    if X_train is None or y_train is None:
        raise ValueError("X_train e y_train no pueden ser None.")
    if preprocessor is None:
        raise ValueError("Se requiere un preprocessor valido para construir el pipeline.")
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("X_train e y_train deben contener datos para optimizacion.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train e y_train deben tener la misma cantidad de filas.")
    if not hasattr(preprocessor, "fit") or not hasattr(preprocessor, "transform"):
        raise TypeError(
            "El argumento `preprocessor` debe ser compatible con la API de Scikit-learn "
            "(fit/transform), por ejemplo ColumnTransformer."
        )

    method = _validate_search_method(metodo)
    effective_cv = _resolve_effective_cv(y_train=y_train, cv=cv)
    search_spaces = _build_search_spaces(random_state=random_state)
    parallel_jobs = _resolve_parallel_jobs()

    print("Iniciando optimizacion de hiperparametros para clasificacion...")
    print(
        f"Metodo: {method} | scoring: {scoring} | cv: {effective_cv} | n_jobs: {parallel_jobs}"
    )

    all_model_results: list[dict[str, Any]] = []

    for model_name, (estimator, param_space) in search_spaces.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )

        if method == "grid":
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_space,
                scoring=scoring,
                cv=effective_cv,
                n_jobs=parallel_jobs,
                verbose=1,
            )
        else:
            total_combinations = int(prod(len(values) for values in param_space.values()))
            n_iter = min(25, total_combinations)
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_space,
                scoring=scoring,
                cv=effective_cv,
                n_iter=n_iter,
                random_state=random_state,
                n_jobs=parallel_jobs,
                verbose=1,
            )

        print(f"[Optimizando] {model_name}...")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*'penalty' was deprecated.*",
                category=FutureWarning,
            )
            search.fit(X_train, y_train)
        model_result = {
            "model_name": model_name,
            "best_estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "search_object": search,
        }
        all_model_results.append(model_result)
        print(f"[OK] {model_name} | best_score={model_result['best_score']:.4f}")

    best_result = max(all_model_results, key=lambda result: result["best_score"])
    summary_by_model = {
        result["model_name"]: {
            "best_score": result["best_score"],
            "best_params": result["best_params"],
        }
        for result in all_model_results
    }
    optimized_models = {
        result["model_name"]: {
            "best_estimator": result["best_estimator"],
            "best_params": result["best_params"],
            "best_score": result["best_score"],
            "search_object": result["search_object"],
        }
        for result in all_model_results
    }

    return {
        "best_estimator": best_result["best_estimator"],
        "best_params": best_result["best_params"],
        "best_score": best_result["best_score"],
        "search_object": best_result["search_object"],
        "model_name": best_result["model_name"],
        "method": method,
        "scoring": scoring,
        "cv": effective_cv,
        "n_jobs": parallel_jobs,
        "all_results": summary_by_model,
        "optimized_models": optimized_models,
    }


def guardar_resultados_optimizacion(
    resultado: dict,
    output_dir: str = "results/metrics",
) -> str:
    """Guarda en JSON los resultados clave de optimizacion y retorna la ruta."""
    required_keys = {
        "model_name",
        "best_params",
        "best_score",
        "method",
        "scoring",
    }
    if not isinstance(resultado, dict):
        raise TypeError("`resultado` debe ser un diccionario.")
    missing_keys = sorted(required_keys - set(resultado.keys()))
    if missing_keys:
        raise ValueError(
            f"El diccionario de resultado no contiene claves requeridas: {missing_keys}"
        )

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = _sanitize_name(str(resultado["model_name"]))
    method_name = _sanitize_name(str(resultado["method"]))
    file_path = output_path / f"optimization_result_{model_name}_{method_name}.json"

    payload = {
        "model_name": str(resultado["model_name"]),
        "best_score": float(resultado["best_score"]),
        "best_params": _to_serializable(resultado["best_params"]),
        "method": str(resultado["method"]),
        "scoring": str(resultado["scoring"]),
        "cv": int(resultado.get("cv", 0)),
        "all_results": _to_serializable(resultado.get("all_results", {})),
    }

    file_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Resultados de optimizacion guardados en: {file_path}")
    return str(file_path)
