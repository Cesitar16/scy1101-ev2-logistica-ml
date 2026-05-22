"""Improved regression workflow for delivery-time prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from regression_setup import (  # type: ignore
        CATEGORICAL_FEATURES,
        FEATURE_COLUMNS,
        NUMERIC_FEATURES,
        TARGET_COLUMN,
    )
except ImportError:
    try:
        from .regression_setup import (  # type: ignore
            CATEGORICAL_FEATURES,
            FEATURE_COLUMNS,
            NUMERIC_FEATURES,
            TARGET_COLUMN,
        )
    except ImportError:
        TARGET_COLUMN = "target_tiempo_entrega"
        FEATURE_COLUMNS = [
            "distancia_km",
            "trafico_nivel",
            "clima",
            "tipo_vehiculo",
            "peso_carga_kg",
            "paradas_previas",
            "experiencia_chofer_anios",
            "hora_despacho",
            "costo_envio",
            "consumo_nafta",
            "id_bodega",
        ]
        NUMERIC_FEATURES = [
            "distancia_km",
            "peso_carga_kg",
            "paradas_previas",
            "experiencia_chofer_anios",
            "hora_despacho",
            "costo_envio",
            "consumo_nafta",
        ]
        CATEGORICAL_FEATURES = [
            "trafico_nivel",
            "clima",
            "tipo_vehiculo",
            "id_bodega",
        ]

try:
    from data_preprocessing import (  # type: ignore
        cap_outliers_iqr,
        clean_text_columns,
        convert_numeric_columns,
        convert_target_column,
        normalize_column_names,
        remove_duplicate_rows,
        remove_rows_without_target,
        validate_required_columns,
    )
except ImportError:
    from .data_preprocessing import (  # type: ignore
        cap_outliers_iqr,
        clean_text_columns,
        convert_numeric_columns,
        convert_target_column,
        normalize_column_names,
        remove_duplicate_rows,
        remove_rows_without_target,
        validate_required_columns,
    )

try:
    from model_evaluation import calculate_regression_metrics  # type: ignore
except ImportError:
    from .model_evaluation import calculate_regression_metrics  # type: ignore


NUMERIC_ENGINEERED_FEATURES = [
    "hora_sin",
    "hora_cos",
    "costo_por_km",
    "consumo_por_km",
    "carga_por_km",
    "paradas_por_km",
]

CATEGORICAL_ENGINEERED_FEATURES = [
    "trafico_clima",
    "trafico_vehiculo",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide safely handling zeros, nulls, and non-numeric values."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    result = np.where((den == 0) | den.isna(), np.nan, num / den)
    return pd.Series(result, index=numerator.index)


def add_regression_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical, ratio, and interaction features for regression."""
    required = [
        "hora_despacho",
        "distancia_km",
        "costo_envio",
        "consumo_nafta",
        "peso_carga_kg",
        "paradas_previas",
        "trafico_nivel",
        "clima",
        "tipo_vehiculo",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para feature engineering: {missing}")

    df_copy = df.copy()

    hour_values = pd.to_numeric(df_copy["hora_despacho"], errors="coerce")
    hour_values = np.mod(hour_values, 24)
    angle = (2 * np.pi * hour_values) / 24.0
    df_copy["hora_sin"] = np.sin(angle)
    df_copy["hora_cos"] = np.cos(angle)

    df_copy["costo_por_km"] = _safe_divide(df_copy["costo_envio"], df_copy["distancia_km"])
    df_copy["consumo_por_km"] = _safe_divide(df_copy["consumo_nafta"], df_copy["distancia_km"])
    df_copy["carga_por_km"] = _safe_divide(df_copy["peso_carga_kg"], df_copy["distancia_km"])
    df_copy["paradas_por_km"] = _safe_divide(df_copy["paradas_previas"], df_copy["distancia_km"])

    trafico = df_copy["trafico_nivel"].astype("string").fillna("desconocido")
    clima = df_copy["clima"].astype("string").fillna("desconocido")
    vehiculo = df_copy["tipo_vehiculo"].astype("string").fillna("desconocido")
    df_copy["trafico_clima"] = (trafico + "__" + clima).astype("string")
    df_copy["trafico_vehiculo"] = (trafico + "__" + vehiculo).astype("string")

    return df_copy


def preprocess_regression_for_improvement(
    df: pd.DataFrame,
    cap_numeric_outliers: bool = True,
) -> pd.DataFrame:
    """
    Prepare clean dataset for improved modeling.

    This version intentionally avoids clipping the target variable to preserve
    regression signal in y.
    """
    df_clean = normalize_column_names(df)
    validate_required_columns(df_clean, FEATURE_COLUMNS + [TARGET_COLUMN])

    df_clean = clean_text_columns(df_clean, CATEGORICAL_FEATURES)
    df_clean = convert_numeric_columns(df_clean, NUMERIC_FEATURES)
    df_clean = convert_target_column(df_clean, TARGET_COLUMN)
    df_clean = remove_duplicate_rows(df_clean)
    df_clean = remove_rows_without_target(df_clean, TARGET_COLUMN)

    # Target values <= 0 are operationally invalid for delivery time.
    df_clean = df_clean.loc[df_clean[TARGET_COLUMN] > 0].copy()

    if cap_numeric_outliers:
        df_clean = cap_outliers_iqr(df_clean, NUMERIC_FEATURES)

    df_clean = add_regression_engineered_features(df_clean)
    return df_clean.reset_index(drop=True)


def get_improved_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return numeric/categorical/all feature lists present in dataframe."""
    numeric_candidates = NUMERIC_FEATURES + NUMERIC_ENGINEERED_FEATURES
    categorical_candidates = CATEGORICAL_FEATURES + CATEGORICAL_ENGINEERED_FEATURES

    numeric = [column for column in numeric_candidates if column in df.columns]
    categorical = [column for column in categorical_candidates if column in df.columns]
    all_features = numeric + categorical

    return {
        "numeric": numeric,
        "categorical": categorical,
        "all_features": all_features,
    }


def build_improved_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build preprocessor compatible with linear and tree-based regressors."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def split_improved_regression_data(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create deterministic train/test split for improved regression workflow."""
    missing_columns = [column for column in feature_columns + [target_column] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas para split: {missing_columns}")

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def get_improved_regression_estimators(random_state: int = 42) -> dict[str, Any]:
    """Return a stronger and more diverse baseline estimator set."""
    return {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0, random_state=random_state),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=250,
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees_regressor": ExtraTreesRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(
            random_state=random_state,
        ),
        "hist_gradient_boosting_regressor": HistGradientBoostingRegressor(
            random_state=random_state,
            max_iter=250,
        ),
    }


def build_regression_pipelines(
    preprocessor: ColumnTransformer,
    estimators: dict[str, Any],
) -> dict[str, Pipeline]:
    """Build full pipelines using a shared preprocessor."""
    return {
        model_name: Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        for model_name, estimator in estimators.items()
    }


def evaluate_regression_pipelines(
    pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray]]:
    """Fit/evaluate all baseline pipelines and return sorted metrics table."""
    rows: list[dict[str, float | str]] = []
    trained_models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}

    for model_name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline

        y_pred = pipeline.predict(X_test)
        predictions[model_name] = y_pred
        metrics = calculate_regression_metrics(y_test, y_pred)

        rows.append(
            {
                "model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values(
        by=["RMSE", "MAE", "R2"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return metrics_df, trained_models, predictions


def select_best_model_name(metrics_df: pd.DataFrame) -> str:
    """Select best model using RMSE primary criterion."""
    required = {"model", "MAE", "RMSE", "R2"}
    if not required.issubset(metrics_df.columns):
        raise ValueError(f"metrics_df debe contener columnas: {required}")

    sorted_df = metrics_df.sort_values(
        by=["RMSE", "MAE", "R2"],
        ascending=[True, True, False],
    )
    return str(sorted_df.iloc[0]["model"])


def get_param_search_space(model_name: str) -> dict[str, list[Any]]:
    """Return hyperparameter search space for a selected model."""
    param_spaces: dict[str, dict[str, list[Any]]] = {
        "linear_regression": {
            "model__fit_intercept": [True, False],
        },
        "ridge_regression": {
            "model__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 25.0],
            "model__fit_intercept": [True, False],
        },
        "random_forest_regressor": {
            "model__n_estimators": [200, 300, 500, 700],
            "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.6, 0.8],
        },
        "extra_trees_regressor": {
            "model__n_estimators": [200, 300, 500, 700],
            "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.6, 0.8],
        },
        "gradient_boosting_regressor": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, 5],
            "model__min_samples_split": [2, 5, 10],
            "model__subsample": [0.8, 1.0],
        },
        "hist_gradient_boosting_regressor": {
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [None, 4, 6, 8, 10],
            "model__max_leaf_nodes": [15, 31, 63, 127],
            "model__min_samples_leaf": [10, 20, 30, 50],
            "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            "model__max_iter": [200, 300, 500],
        },
    }

    if model_name not in param_spaces:
        raise ValueError(f"No existe espacio de busqueda para: {model_name}")

    return param_spaces[model_name]


def _grid_size(param_grid: dict[str, list[Any]]) -> int:
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def tune_best_regression_pipeline(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Tune best model with GridSearchCV or RandomizedSearchCV automatically."""
    param_space = get_param_search_space(model_name)
    grid_size = _grid_size(param_space)

    if grid_size <= 120:
        search: GridSearchCV | RandomizedSearchCV = GridSearchCV(
            estimator=pipeline,
            param_grid=param_space,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
        )
        search_method = "grid"
    else:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_space,
            n_iter=60,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            return_train_score=True,
        )
        search_method = "randomized"

    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    metrics = calculate_regression_metrics(y_test, y_pred)

    metrics_df = pd.DataFrame(
        [
            {
                "model": f"{model_name}_improved",
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        ]
    )

    return {
        "search_method": search_method,
        "search_object": search,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "best_model_name": f"{model_name}_improved",
        "best_model": best_pipeline,
        "y_pred": y_pred,
        "metrics": metrics_df,
        "cv_results": pd.DataFrame(search.cv_results_).sort_values("rank_test_score"),
    }


def compare_base_vs_tuned(
    baseline_metrics_df: pd.DataFrame,
    tuned_metrics_df: pd.DataFrame,
    baseline_model_name: str,
) -> pd.DataFrame:
    """Create a side-by-side comparison for baseline vs tuned model."""
    base_row = baseline_metrics_df.loc[baseline_metrics_df["model"] == baseline_model_name].copy()
    if base_row.empty:
        raise ValueError(f"No se encontro modelo base: {baseline_model_name}")

    base_row["version"] = "base"
    tuned_row = tuned_metrics_df.copy()
    tuned_row["version"] = "tuned"

    comparison = pd.concat([base_row, tuned_row], ignore_index=True)
    return comparison[["version", "model", "MAE", "RMSE", "R2"]]


def calculate_tolerance_accuracy(
    y_true: pd.Series,
    y_pred: np.ndarray,
    tolerances: tuple[int, ...] = (5, 10, 15),
) -> pd.DataFrame:
    """Calculate hit rate where absolute error is within tolerance in minutes."""
    abs_error = np.abs(pd.Series(y_true).reset_index(drop=True) - pd.Series(y_pred))
    total = len(abs_error)

    rows = []
    for tolerance in tolerances:
        hits = int((abs_error <= tolerance).sum())
        accuracy = (hits / total) * 100 if total else 0.0
        rows.append(
            {
                "tolerance_minutes": tolerance,
                "hit_count": hits,
                "total_count": total,
                "accuracy_percent": round(accuracy, 4),
            }
        )

    return pd.DataFrame(rows)


def calculate_mean_baseline_metrics(
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Calculate a naive baseline (predicting train mean) for reference."""
    mean_value = float(pd.Series(y_train).mean())
    y_pred = np.full(shape=len(y_test), fill_value=mean_value, dtype=float)
    metrics = calculate_regression_metrics(y_test, y_pred)

    return pd.DataFrame(
        [
            {
                "model": "mean_baseline",
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        ]
    )


def _run_modeling_stage(
    df_prepared: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execute training, tuning, and evaluation from an already prepared dataframe."""
    feature_groups = get_improved_feature_groups(df_prepared)

    X_train, X_test, y_train, y_test = split_improved_regression_data(
        df=df_prepared,
        feature_columns=feature_groups["all_features"],
        target_column=TARGET_COLUMN,
        test_size=0.2,
        random_state=random_state,
    )

    preprocessor = build_improved_preprocessor(
        numeric_features=feature_groups["numeric"],
        categorical_features=feature_groups["categorical"],
    )

    estimators = get_improved_regression_estimators(random_state=random_state)
    pipelines = build_regression_pipelines(preprocessor=preprocessor, estimators=estimators)

    baseline_metrics, trained_models, baseline_predictions = evaluate_regression_pipelines(
        pipelines=pipelines,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    best_baseline_model = select_best_model_name(baseline_metrics)
    tuning = tune_best_regression_pipeline(
        model_name=best_baseline_model,
        pipeline=trained_models[best_baseline_model],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv=5,
        random_state=random_state,
    )

    comparison = compare_base_vs_tuned(
        baseline_metrics_df=baseline_metrics,
        tuned_metrics_df=tuning["metrics"],
        baseline_model_name=best_baseline_model,
    )

    base_metrics_row = baseline_metrics.loc[baseline_metrics["model"] == best_baseline_model].iloc[0]
    tuned_metrics_row = tuning["metrics"].iloc[0]

    # Main criterion: lower RMSE. Secondary: lower MAE. Third: higher R2.
    use_tuned = (
        (tuned_metrics_row["RMSE"] < base_metrics_row["RMSE"])
        or (
            tuned_metrics_row["RMSE"] == base_metrics_row["RMSE"]
            and tuned_metrics_row["MAE"] < base_metrics_row["MAE"]
        )
        or (
            tuned_metrics_row["RMSE"] == base_metrics_row["RMSE"]
            and tuned_metrics_row["MAE"] == base_metrics_row["MAE"]
            and tuned_metrics_row["R2"] > base_metrics_row["R2"]
        )
    )

    if use_tuned:
        recommended_model_name = str(tuning["best_model_name"])
        recommended_model = tuning["best_model"]
        recommended_metrics = tuning["metrics"].copy()
        recommended_predictions = tuning["y_pred"]
        recommended_source = "tuned"
    else:
        recommended_model_name = str(best_baseline_model)
        recommended_model = trained_models[best_baseline_model]
        recommended_metrics = baseline_metrics.loc[
            baseline_metrics["model"] == best_baseline_model
        ].copy()
        recommended_predictions = baseline_predictions[best_baseline_model]
        recommended_source = "baseline"

    tolerance_accuracy = calculate_tolerance_accuracy(
        y_true=y_test,
        y_pred=recommended_predictions,
    )
    tolerance_base = calculate_tolerance_accuracy(
        y_true=y_test,
        y_pred=baseline_predictions[best_baseline_model],
    )
    tolerance_tuned = calculate_tolerance_accuracy(
        y_true=y_test,
        y_pred=tuning["y_pred"],
    )
    mean_baseline_metrics = calculate_mean_baseline_metrics(y_train=y_train, y_test=y_test)

    predictions_df = pd.DataFrame(
        {
            "real": y_test.reset_index(drop=True),
            "predicho": recommended_predictions,
        }
    )
    predictions_df["error"] = predictions_df["real"] - predictions_df["predicho"]
    predictions_df["error_absoluto"] = predictions_df["error"].abs()

    return {
        "df_prepared": df_prepared,
        "feature_groups": feature_groups,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "baseline_metrics": baseline_metrics,
        "mean_baseline_metrics": mean_baseline_metrics,
        "best_baseline_model": best_baseline_model,
        "trained_models": trained_models,
        "baseline_predictions": baseline_predictions,
        "tuning": tuning,
        "recommended_model_name": recommended_model_name,
        "recommended_model": recommended_model,
        "recommended_metrics": recommended_metrics,
        "recommended_source": recommended_source,
        "comparison": comparison,
        "tolerance_accuracy": tolerance_accuracy,
        "tolerance_base": tolerance_base,
        "tolerance_tuned": tolerance_tuned,
        "predictions": predictions_df,
    }


def run_regression_improvement_workflow(
    df_raw: pd.DataFrame,
    random_state: int = 42,
    cap_numeric_outliers: bool = True,
) -> dict[str, Any]:
    """Execute end-to-end workflow starting from raw logistics data."""
    df_prepared = preprocess_regression_for_improvement(
        df_raw,
        cap_numeric_outliers=cap_numeric_outliers,
    )
    return _run_modeling_stage(df_prepared=df_prepared, random_state=random_state)


def run_regression_improvement_with_prepared_dataframe(
    df_prepared: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Execute end-to-end workflow from an already cleaned/prepared dataframe.

    This is useful when the project already has a processed dataset and we only
    want to add engineered features and improve modeling quality.
    """
    validate_required_columns(df_prepared, FEATURE_COLUMNS + [TARGET_COLUMN])

    df_copy = normalize_column_names(df_prepared)
    df_copy = convert_numeric_columns(df_copy, NUMERIC_FEATURES)
    df_copy = convert_target_column(df_copy, TARGET_COLUMN)
    df_copy = remove_rows_without_target(df_copy, TARGET_COLUMN)
    df_copy = df_copy.loc[df_copy[TARGET_COLUMN] > 0].copy()

    # Keep existing cleaned structure but enrich with new engineered variables.
    df_copy = add_regression_engineered_features(df_copy)
    df_copy = df_copy.reset_index(drop=True)

    return _run_modeling_stage(df_prepared=df_copy, random_state=random_state)


def save_regression_improvement_artifacts(
    workflow_results: dict[str, Any],
    project_root: str | Path,
) -> dict[str, Path]:
    """Persist metrics, reports, and improved model artifacts."""
    root = Path(project_root)
    metrics_dir = root / "results" / "metrics"
    reports_dir = root / "results" / "reports"
    model_dir = root / "models" / "trained_models"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics_path = metrics_dir / "cesar_regression_improved_baseline_metrics.csv"
    tuned_metrics_path = metrics_dir / "cesar_regression_improved_tuned_metrics.csv"
    comparison_path = metrics_dir / "cesar_regression_improved_base_vs_tuned.csv"
    tolerance_path = metrics_dir / "cesar_regression_improved_tolerance_accuracy.csv"
    mean_baseline_path = metrics_dir / "cesar_regression_improved_mean_baseline_metrics.csv"
    cv_results_path = metrics_dir / "cesar_regression_improved_tuning_results.csv"
    predictions_path = reports_dir / "cesar_regression_improved_predictions.csv"
    model_path = model_dir / "cesar_regression_model_improved.joblib"
    summary_path = reports_dir / "cesar_regression_improved_summary.md"

    workflow_results["baseline_metrics"].to_csv(baseline_metrics_path, index=False)
    workflow_results["tuning"]["metrics"].to_csv(tuned_metrics_path, index=False)
    workflow_results["comparison"].to_csv(comparison_path, index=False)
    workflow_results["tolerance_accuracy"].to_csv(tolerance_path, index=False)
    workflow_results["mean_baseline_metrics"].to_csv(mean_baseline_path, index=False)
    workflow_results["tuning"]["cv_results"].to_csv(cv_results_path, index=False)
    workflow_results["predictions"].to_csv(predictions_path, index=False)
    joblib.dump(workflow_results["recommended_model"], model_path)

    tuned_row = workflow_results["recommended_metrics"].iloc[0]
    best_base = workflow_results["best_baseline_model"]
    recommended_source = workflow_results["recommended_source"]
    recommended_model_name = workflow_results["recommended_model_name"]
    summary = f"""# Mejora de modelo de regresion - Cesar

## Mejor modelo base

{best_base}

## Modelo recomendado final

{recommended_model_name} ({recommended_source})

## Metodo de tuning

{workflow_results["tuning"]["search_method"]}

## Mejores hiperparametros

{workflow_results["tuning"]["best_params"]}

## Metricas modelo recomendado

- MAE: {tuned_row["MAE"]:.6f}
- RMSE: {tuned_row["RMSE"]:.6f}
- R2: {tuned_row["R2"]:.6f}

## Acierto operacional por tolerancia
"""

    for _, row in workflow_results["tolerance_accuracy"].iterrows():
        summary += (
            f"\n- +/-{int(row['tolerance_minutes'])} min: "
            f"{row['accuracy_percent']:.2f}% ({int(row['hit_count'])}/{int(row['total_count'])})"
        )

    summary_path.write_text(summary, encoding="utf-8")

    return {
        "baseline_metrics_path": baseline_metrics_path,
        "tuned_metrics_path": tuned_metrics_path,
        "comparison_path": comparison_path,
        "tolerance_path": tolerance_path,
        "mean_baseline_path": mean_baseline_path,
        "cv_results_path": cv_results_path,
        "predictions_path": predictions_path,
        "model_path": model_path,
        "summary_path": summary_path,
    }
