"""Hyperparameter tuning utilities for ML workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

try:
    from model_evaluation import (  # type: ignore
        evaluate_regression_models,
        create_predictions_dataframe,
        calculate_regression_metrics,
        plot_real_vs_predicted,
        plot_residuals,
    )
except ImportError:
    try:
        from .model_evaluation import (  # type: ignore
            evaluate_regression_models,
            create_predictions_dataframe,
            calculate_regression_metrics,
            plot_real_vs_predicted,
            plot_residuals,
        )
    except ImportError:
        evaluate_regression_models = None
        create_predictions_dataframe = None
        calculate_regression_metrics = None
        plot_real_vs_predicted = None
        plot_residuals = None

try:
    from operational_metrics import tolerance_accuracy_10_scorer  # type: ignore
except ImportError:
    try:
        from .operational_metrics import tolerance_accuracy_10_scorer  # type: ignore
    except ImportError:
        tolerance_accuracy_10_scorer = None


RANDOM_STATE = 42


def get_param_grid_for_model(model_name: str) -> dict[str, list[Any]]:
    """
    Return a hyperparameter grid for a given regression model.

    Parameter names use the 'model__' prefix because the estimator is
    wrapped inside a Scikit-learn Pipeline.
    """
    param_grids: dict[str, dict[str, list[Any]]] = {
        "linear_regression": {
            "model__fit_intercept": [True, False],
        },
        "decision_tree_regressor": {
            "model__max_depth": [3, 5, 8, 10, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "random_forest_regressor": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 8, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "gradient_boosting_regressor": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 5],
            "model__min_samples_split": [2, 5, 10],
        },
        "extra_trees_regressor": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 8, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "hist_gradient_boosting_regressor": {
            "model__max_iter": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [10, 20, 30],
        },
        "knn_regressor": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
    }

    if model_name not in param_grids:
        raise ValueError(
            f"No hay grilla definida para el modelo: {model_name}. "
            f"Modelos disponibles: {list(param_grids.keys())}"
        )

    return param_grids[model_name]


def tune_regression_model_gridsearch(
    model: Any,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
) -> GridSearchCV:
    """
    Tune a regression pipeline using GridSearchCV.

    Parameters
    ----------
    model:
        Scikit-learn Pipeline.
    param_grid:
        Hyperparameter search grid.
    X_train:
        Training predictors.
    y_train:
        Training target.
    cv:
        Number of cross-validation folds.
    scoring:
        Optimization metric.
    n_jobs:
        Number of worker processes.

    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object.
    """
    if not param_grid:
        raise ValueError(
            "La grilla de parametros esta vacia. "
            "No corresponde optimizar este modelo con GridSearchCV."
        )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    return grid_search


def tune_regression_model_randomizedsearch(
    model: Any,
    param_distributions: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomizedSearchCV:
    """Tune a regression pipeline using RandomizedSearchCV."""
    if not param_distributions:
        raise ValueError(
            "La distribucion de parametros esta vacia. "
            "No corresponde optimizar este modelo con RandomizedSearchCV."
        )

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        return_train_score=True,
    )
    random_search.fit(X_train, y_train)

    return random_search


def tune_regression_model_for_tolerance_10(
    model: Any,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    n_jobs: int = -1,
) -> GridSearchCV:
    """
    Optimiza un modelo buscando maximizar el acierto dentro de +/-10 minutos.
    """
    if tolerance_accuracy_10_scorer is None:
        raise ImportError("No se pudo importar tolerance_accuracy_10_scorer.")

    if not param_grid:
        raise ValueError("La grilla de parametros esta vacia.")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=tolerance_accuracy_10_scorer,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    return grid_search


def get_tuning_results_dataframe(search_object: Any) -> pd.DataFrame:
    """Convert cv_results_ from a search object into a sorted DataFrame."""
    results_df = pd.DataFrame(search_object.cv_results_)
    results_df = results_df.sort_values(by="rank_test_score", ascending=True).reset_index(drop=True)
    return results_df


def evaluate_optimized_model(
    optimized_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    """Evaluate optimized model with MAE, RMSE and R2 metrics."""
    if calculate_regression_metrics is None:
        raise ImportError("No se pudo importar calculate_regression_metrics.")

    y_pred = optimized_model.predict(X_test)
    metrics = calculate_regression_metrics(y_test, y_pred)

    return pd.DataFrame(
        [
            {
                "model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        ]
    )


def compare_base_vs_optimized(
    base_metrics_df: pd.DataFrame,
    optimized_metrics_df: pd.DataFrame,
    base_model_name: str,
    optimized_model_name: str,
) -> pd.DataFrame:
    """Compare baseline metrics with optimized metrics."""
    base_row = base_metrics_df[base_metrics_df["model"] == base_model_name].copy()
    if base_row.empty:
        raise ValueError(f"No se encontro el modelo base: {base_model_name}")

    base_row["version"] = "base"

    optimized_row = optimized_metrics_df.copy()
    optimized_row["version"] = "optimized"
    optimized_row["model"] = optimized_model_name

    comparison = pd.concat([base_row, optimized_row], ignore_index=True)
    return comparison[["version", "model", "MAE", "RMSE", "R2"]]


def save_tuning_results(
    tuning_results_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save full hyperparameter tuning results in CSV format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tuning_results_df.to_csv(path, index=False)


def save_optimized_metrics(
    optimized_metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save optimized model metrics in CSV format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    optimized_metrics_df.to_csv(path, index=False)


def save_base_vs_optimized_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save base vs optimized metrics comparison in CSV format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(path, index=False)


def plot_base_vs_optimized_metric(
    comparison_df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
) -> None:
    """Generate a bar chart comparing base vs optimized for one metric."""
    if metric not in comparison_df.columns:
        raise ValueError(f"La metrica {metric} no existe.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.bar(comparison_df["version"], comparison_df[metric], color=["#4C78A8", "#F58518"])
    plt.xlabel("Version")
    plt.ylabel(metric)
    plt.title(f"Comparacion base vs optimizado - {metric}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_regression_tuning_workflow(
    best_model_name: str,
    trained_models: dict[str, Any],
    base_metrics_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    tuning_results_path: str | Path,
    optimized_metrics_path: str | Path,
    comparison_path: str | Path,
    rmse_plot_path: str | Path,
    mae_plot_path: str | Path,
    optimized_real_vs_pred_path: str | Path,
    optimized_residuals_path: str | Path,
) -> dict[str, Any]:
    """
    Execute full hyperparameter tuning workflow for regression.

    This function:
    1. Retrieves a parameter grid for best baseline model.
    2. Runs GridSearchCV.
    3. Evaluates optimized model.
    4. Compares base vs optimized.
    5. Saves outputs and charts.
    """
    if best_model_name not in trained_models:
        raise ValueError(f"No existe el modelo entrenado: {best_model_name}")

    base_model = trained_models[best_model_name]
    param_grid = get_param_grid_for_model(best_model_name)

    search = tune_regression_model_gridsearch(
        model=base_model,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
    )

    tuning_results_df = get_tuning_results_dataframe(search)
    optimized_model = search.best_estimator_
    optimized_model_name = f"{best_model_name}_optimized"

    optimized_metrics_df = evaluate_optimized_model(
        optimized_model=optimized_model,
        X_test=X_test,
        y_test=y_test,
        model_name=optimized_model_name,
    )

    comparison_df = compare_base_vs_optimized(
        base_metrics_df=base_metrics_df,
        optimized_metrics_df=optimized_metrics_df,
        base_model_name=best_model_name,
        optimized_model_name=optimized_model_name,
    )

    save_tuning_results(tuning_results_df, tuning_results_path)
    save_optimized_metrics(optimized_metrics_df, optimized_metrics_path)
    save_base_vs_optimized_comparison(comparison_df, comparison_path)

    plot_base_vs_optimized_metric(comparison_df, "RMSE", rmse_plot_path)
    plot_base_vs_optimized_metric(comparison_df, "MAE", mae_plot_path)

    if create_predictions_dataframe is None:
        raise ImportError("No se pudo importar create_predictions_dataframe.")

    optimized_predictions_df = create_predictions_dataframe(
        model=optimized_model,
        X_test=X_test,
        y_test=y_test,
    )

    if plot_real_vs_predicted is not None:
        plot_real_vs_predicted(
            predictions_df=optimized_predictions_df,
            output_path=optimized_real_vs_pred_path,
        )

    if plot_residuals is not None:
        plot_residuals(
            predictions_df=optimized_predictions_df,
            output_path=optimized_residuals_path,
        )

    return {
        "search": search,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "optimized_model": optimized_model,
        "optimized_model_name": optimized_model_name,
        "tuning_results": tuning_results_df,
        "optimized_metrics": optimized_metrics_df,
        "comparison": comparison_df,
        "optimized_predictions": optimized_predictions_df,
    }


# ---------------------------------------------------------------------------
# Backward-compatible helpers for previously existing workflows
# ---------------------------------------------------------------------------
def run_grid_search(model, param_grid, X_train, y_train, scoring, cv=5):
    """Backward-compatible GridSearch helper."""
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def run_randomized_search(
    model,
    param_distributions,
    X_train,
    y_train,
    scoring,
    cv=5,
    n_iter=20,
):
    """Backward-compatible RandomizedSearch helper."""
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        scoring=scoring,
        cv=cv,
        n_iter=n_iter,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    random_search.fit(X_train, y_train)
    return random_search


def evaluate_regression_model(model, X_test, y_test):
    """Backward-compatible regression evaluation helper."""
    if calculate_regression_metrics is None:
        raise ImportError("No se pudo importar calculate_regression_metrics.")

    y_pred = model.predict(X_test)
    return calculate_regression_metrics(pd.Series(y_test), y_pred)


def evaluate_classification_model(model, X_test, y_test):
    """Backward-compatible classification evaluation helper."""
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }


def save_best_model(model, output_path):
    """Backward-compatible model saver (not used in this branch)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)


def summarize_best_params(search_object, model_name, output_path):
    """Backward-compatible best-params summary saver."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "best_score": search_object.best_score_,
                "best_params": search_object.best_params_,
            }
        ]
    )
    summary.to_csv(output_path, index=False)

    return summary
