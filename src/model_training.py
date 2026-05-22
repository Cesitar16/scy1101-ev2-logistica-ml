"""Training utilities for supervised and unsupervised workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

try:
    from regression_setup import (  # type: ignore
        TARGET_COLUMN,
        FEATURE_COLUMNS,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
    )
except ImportError:
    try:
        from .regression_setup import (  # type: ignore
            TARGET_COLUMN,
            FEATURE_COLUMNS,
            NUMERIC_FEATURES,
            CATEGORICAL_FEATURES,
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
    from data_preprocessing import build_regression_preprocessor  # type: ignore
except ImportError:
    try:
        from .data_preprocessing import build_regression_preprocessor  # type: ignore
    except ImportError:
        build_regression_preprocessor = None


def split_regression_data(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a processed regression dataset into train and test sets.

    Parameters
    ----------
    df:
        Processed DataFrame.
    feature_columns:
        Predictor columns. If None, uses FEATURE_COLUMNS.
    target_column:
        Regression target variable.
    test_size:
        Proportion of rows assigned to test set.
    random_state:
        Seed used for reproducibility.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test.
    """
    selected_features = feature_columns or FEATURE_COLUMNS

    missing_columns = [
        column for column in selected_features + [target_column] if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

    X = df[selected_features].copy()
    y = df[target_column].copy()

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def get_regression_estimators(
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Return baseline regression estimators.

    Estimators are returned without preprocessing. The preprocessor is
    attached later via Pipeline.
    """
    return {
        "linear_regression": LinearRegression(),
        "decision_tree_regressor": DecisionTreeRegressor(
            max_depth=5,
            random_state=random_state,
        ),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(
            random_state=random_state,
        ),
        "extra_trees_regressor": ExtraTreesRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting_regressor": HistGradientBoostingRegressor(
            random_state=random_state,
        ),
        "knn_regressor": KNeighborsRegressor(
            n_neighbors=5,
        ),
    }


def build_regression_pipelines(
    preprocessor: Any,
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """
    Build full regression pipelines with a shared preprocessor.

    Each pipeline uses:
    preprocessor -> model
    """
    if preprocessor is None:
        raise ValueError("El preprocesador no puede ser None.")

    estimators = get_regression_estimators(random_state=random_state)

    pipelines: dict[str, Pipeline] = {}
    for model_name, estimator in estimators.items():
        pipelines[model_name] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

    return pipelines


def train_regression_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    """
    Train multiple regression pipelines.

    Parameters
    ----------
    models:
        Dictionary of regression pipelines.
    X_train:
        Training predictors.
    y_train:
        Training target values.

    Returns
    -------
    dict[str, Pipeline]
        Trained models.
    """
    trained_models: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model

    return trained_models


def generate_baseline_predictions(
    trained_models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Generate baseline predictions for each trained model.

    This function does not compute metrics. It only produces a table
    with real and predicted values for later analysis.
    """
    predictions = pd.DataFrame(
        {
            "real": y_test.reset_index(drop=True),
        }
    )

    for model_name, model in trained_models.items():
        predictions[f"pred_{model_name}"] = model.predict(X_test)

    return predictions


def save_baseline_predictions(
    predictions_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save baseline predictions in CSV format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(path, index=False)


def run_regression_training_workflow(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Execute the baseline training workflow for Cesar's regression line.

    This function:
    1. Splits data into train/test.
    2. Builds the preprocessor.
    3. Creates regression pipelines.
    4. Trains baseline models.
    5. Generates baseline predictions.

    It does not compute evaluation metrics, does not optimize hyperparameters,
    and does not save final model artifacts.
    """
    if build_regression_preprocessor is None:
        raise ImportError(
            "No se pudo importar build_regression_preprocessor "
            "desde data_preprocessing.py."
        )

    X_train, X_test, y_train, y_test = split_regression_data(
        df=df,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_regression_preprocessor(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    models = build_regression_pipelines(
        preprocessor=preprocessor,
        random_state=random_state,
    )

    trained_models = train_regression_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
    )

    predictions = generate_baseline_predictions(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "models": models,
        "trained_models": trained_models,
        "predictions": predictions,
    }


def train_classification_models(X_train, y_train):
    """Placeholder for future classification workflow."""
    raise NotImplementedError("Pendiente de implementación")


def train_clustering_model(X, n_clusters=3):
    """Placeholder for future clustering workflow."""
    raise NotImplementedError("Pendiente de implementación")
