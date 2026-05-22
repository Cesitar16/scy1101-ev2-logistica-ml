"""Preprocessing utilities for the Logistics 4.0 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


ID_BODEGA_COLUMN = "id_bodega"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
CATEGORICAL_COLUMNS = CATEGORICAL_FEATURES.copy()
TEXT_CATEGORICAL_COLUMNS = [column for column in CATEGORICAL_FEATURES if column != ID_BODEGA_COLUMN]
NUMERIC_COLUMNS = NUMERIC_FEATURES + [TARGET_COLUMN]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame column names.

    Converts names to lowercase, trims extra spaces, and replaces
    internal spaces with underscore characters.
    """
    df_clean = df.copy()
    df_clean.columns = (
        df_clean.columns.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df_clean


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """
    Validate that required columns exist in the DataFrame.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")


def clean_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Clean categorical/text columns.

    Converts to string, trims spaces, lowercases values, and maps null-like
    textual values to np.nan.
    """
    df_clean = df.copy()

    null_like_values = {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "sin dato",
        "sin_dato",
        "desconocido",
    }

    for column in columns:
        if column not in df_clean.columns:
            continue

        df_clean[column] = (
            df_clean[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        df_clean[column] = df_clean[column].replace(list(null_like_values), np.nan)

    return df_clean


def convert_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Convert numeric columns safely to numeric dtype.

    Supports decimal commas and coerces invalid values to NaN.
    """
    df_clean = df.copy()

    for column in columns:
        if column not in df_clean.columns:
            continue

        df_clean[column] = (
            df_clean[column]
            .astype("string")
            .str.replace(",", ".", regex=False)
        )
        df_clean[column] = pd.to_numeric(df_clean[column], errors="coerce")

    return df_clean


def convert_target_column(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Convert the target column to numeric values.

    Non-convertible values are transformed into NaN.
    """
    df_clean = df.copy()

    if target_column not in df_clean.columns:
        raise ValueError(f"No existe la variable objetivo: {target_column}")

    df_clean[target_column] = (
        df_clean[target_column]
        .astype("string")
        .str.replace(",", ".", regex=False)
    )
    df_clean[target_column] = pd.to_numeric(df_clean[target_column], errors="coerce")

    return df_clean


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows from the DataFrame."""
    return df.drop_duplicates().copy()


def remove_rows_without_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Remove rows with missing target values.

    In supervised learning, rows without y cannot be used to train or evaluate models.
    """
    if target_column not in df.columns:
        raise ValueError(f"No existe la variable objetivo: {target_column}")

    return df.dropna(subset=[target_column]).copy()


def get_missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing values count and percentage by column."""
    total_missing = df.isna().sum()
    base_rows = max(len(df), 1)
    percent_missing = (total_missing / base_rows) * 100

    report = pd.DataFrame(
        {
            "missing_count": total_missing,
            "missing_percent": percent_missing.round(2),
        }
    )
    return report.sort_values(by="missing_count", ascending=False)


def get_duplicate_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
) -> dict[str, int]:
    """Return a summary report for duplicate removal."""
    return {
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "rows_removed": len(df_before) - len(df_after),
    }


def get_iqr_bounds(df: pd.DataFrame, column: str) -> tuple[float, float]:
    """Compute lower and upper bounds using the IQR method."""
    if column not in df.columns:
        raise ValueError(f"La columna no existe en el DataFrame: {column}")

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return float("-inf"), float("inf")

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return lower_bound, upper_bound


def cap_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Cap extreme values using IQR bounds.

    Does not remove rows; it clips values below/above the computed limits.
    """
    df_clean = df.copy()

    for column in columns:
        if column not in df_clean.columns:
            continue

        numeric_series = pd.to_numeric(df_clean[column], errors="coerce")
        if numeric_series.dropna().empty:
            continue

        lower_bound, upper_bound = get_iqr_bounds(df_clean, column)
        df_clean[column] = numeric_series.clip(lower=lower_bound, upper=upper_bound)

    return df_clean


def build_regression_preprocessor(
    numeric_features: list[str] = NUMERIC_FEATURES,
    categorical_features: list[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """
    Build the Scikit-learn preprocessor for the regression workflow.

    Numeric features:
    - Median imputation
    - Standard scaling

    Categorical features:
    - Most frequent imputation
    - One-hot encoding
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def preprocess_regression_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the base preprocessing flow for Cesar's regression line.

    This prepares the data for next project stages but does not train models
    and does not perform train/test split.
    """
    df_clean = normalize_column_names(df)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    validate_required_columns(df_clean, required_columns)

    df_clean = clean_text_columns(df_clean, CATEGORICAL_FEATURES)
    df_clean = convert_numeric_columns(df_clean, NUMERIC_FEATURES)
    df_clean = convert_target_column(df_clean, TARGET_COLUMN)

    df_clean = remove_duplicate_rows(df_clean)
    df_clean = remove_rows_without_target(df_clean, TARGET_COLUMN)

    outlier_columns = NUMERIC_FEATURES + [TARGET_COLUMN]
    df_clean = cap_outliers_iqr(df_clean, outlier_columns)

    return df_clean


def save_processed_dataset(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save a processed DataFrame into CSV format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Backward-compatible helpers used by existing EDA modules/notebooks
# ---------------------------------------------------------------------------
def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return exact duplicate rows."""
    duplicate_mask = df.duplicated(keep="first")
    return df.loc[duplicate_mask].copy()


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing values report with legacy column names."""
    report = get_missing_values_report(df).reset_index().rename(columns={"index": "columna"})
    report = report.rename(
        columns={"missing_count": "cantidad_nulos", "missing_percent": "porcentaje_nulos"}
    )
    return report.sort_values("cantidad_nulos", ascending=False).reset_index(drop=True)


def convert_id_bodega_to_category(
    df: pd.DataFrame,
    id_column: str = ID_BODEGA_COLUMN,
) -> pd.DataFrame:
    """Convert id_bodega values to categorical string values."""
    if id_column not in df.columns:
        return df.copy()

    df_clean = df.copy()
    as_text = df_clean[id_column].astype("string").str.strip().str.lower()
    as_text = as_text.replace(["", "nan", "none", "null", "na", "n/a"], np.nan)
    df_clean[id_column] = as_text.astype("category")
    return df_clean


def validate_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Validate business rules and return a summary table."""
    validate_required_columns(df, REQUIRED_COLUMNS)

    total_rows = max(len(df), 1)
    business_rules = {
        "distancia_km > 0": pd.to_numeric(df["distancia_km"], errors="coerce") > 0,
        "peso_carga_kg > 0": pd.to_numeric(df["peso_carga_kg"], errors="coerce") > 0,
        "paradas_previas >= 0": pd.to_numeric(df["paradas_previas"], errors="coerce") >= 0,
        "experiencia_chofer_anios >= 0": pd.to_numeric(
            df["experiencia_chofer_anios"], errors="coerce"
        )
        >= 0,
        "hora_despacho entre 0 y 23": pd.to_numeric(df["hora_despacho"], errors="coerce").between(
            0, 23, inclusive="both"
        ),
        "costo_envio > 0": pd.to_numeric(df["costo_envio"], errors="coerce") > 0,
        "consumo_nafta > 0": pd.to_numeric(df["consumo_nafta"], errors="coerce") > 0,
        f"{TARGET_COLUMN} > 0": pd.to_numeric(df[TARGET_COLUMN], errors="coerce") > 0,
    }

    id_numeric = pd.to_numeric(df[ID_BODEGA_COLUMN].astype("string"), errors="coerce")
    id_min = id_numeric.min(skipna=True)
    id_max = id_numeric.max(skipna=True)
    if pd.notna(id_min) and pd.notna(id_max):
        business_rules["id_bodega dentro del rango observado"] = id_numeric.between(
            id_min, id_max, inclusive="both"
        )
    else:
        business_rules["id_bodega dentro del rango observado"] = id_numeric.notna()

    rows: list[dict[str, Any]] = []
    for rule_name, valid_mask in business_rules.items():
        invalid_count = int((~valid_mask.fillna(False)).sum())
        invalid_pct = round(invalid_count / total_rows * 100, 2)
        if invalid_count == 0:
            action = "Sin accion correctiva"
        elif TARGET_COLUMN in rule_name:
            action = "Eliminar registros invalidos para dataset limpio"
        else:
            action = "Revisar con criterio de negocio antes de eliminar"

        rows.append(
            {
                "regla": rule_name,
                "cantidad_incumplimientos": invalid_count,
                "porcentaje_incumplimientos": invalid_pct,
                "accion_recomendada": action,
            }
        )

    return pd.DataFrame(rows)


def remove_invalid_target_rows(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows where target is non-positive and return removed rows too."""
    if target_column not in df.columns:
        raise ValueError(f"No se encontro la columna objetivo: {target_column}")

    target_values = pd.to_numeric(df[target_column], errors="coerce")
    invalid_mask = target_values <= 0
    removed_rows = df.loc[invalid_mask].copy()
    cleaned_df = df.loc[~invalid_mask].copy()
    return cleaned_df, removed_rows


def add_suspicious_time_flag(
    df: pd.DataFrame,
    threshold: float = 10,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Add boolean flag for suspiciously short delivery times."""
    if target_column not in df.columns:
        raise ValueError(f"No se encontro la columna objetivo: {target_column}")

    df_copy = df.copy()
    target_values = pd.to_numeric(df_copy[target_column], errors="coerce")
    df_copy["tiempo_sospechoso"] = target_values < threshold
    return df_copy


def clean_logistics_dataset(
    df: pd.DataFrame,
    remove_invalid_target: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply exploratory cleaning flow used in the shared EDA.

    This function is kept for backward compatibility with previous work.
    """
    original_rows = len(df)
    df_clean = normalize_column_names(df)
    validate_required_columns(df_clean, REQUIRED_COLUMNS)

    df_clean = clean_text_columns(df_clean, TEXT_CATEGORICAL_COLUMNS)
    df_clean = convert_numeric_columns(df_clean, NUMERIC_FEATURES)
    df_clean = convert_target_column(df_clean, TARGET_COLUMN)
    df_clean = convert_id_bodega_to_category(df_clean)

    duplicate_rows = detect_duplicates(df_clean)
    if not duplicate_rows.empty:
        df_clean = remove_duplicate_rows(df_clean)

    business_rules_table = validate_business_rules(df_clean)

    removed_invalid_target = pd.DataFrame(columns=df_clean.columns)
    if remove_invalid_target:
        df_clean, removed_invalid_target = remove_invalid_target_rows(df_clean)

    df_clean = add_suspicious_time_flag(df_clean, threshold=10)
    df_clean = df_clean.reset_index(drop=True)

    summary: dict[str, Any] = {
        "rows_original": original_rows,
        "rows_after_duplicates": original_rows - len(duplicate_rows),
        "rows_final": len(df_clean),
        "duplicates_removed": int(len(duplicate_rows)),
        "invalid_target_removed": int(len(removed_invalid_target)),
        "suspicious_time_count": int(df_clean["tiempo_sospechoso"].sum()),
        "business_rules_report": business_rules_table,
        "invalid_target_rows": removed_invalid_target,
    }

    return df_clean, summary