"""Funciones de analisis exploratorio para el caso Logistica 4.0."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data_preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    validate_business_rules,
)


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna una vista general del DataFrame."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024**2), 4)

    overview = pd.DataFrame(
        [
            {
                "filas": len(df),
                "columnas": df.shape[1],
                "memoria_mb": memory_mb,
                "columnas_numericas": ", ".join(numeric_columns),
                "cantidad_numericas": len(numeric_columns),
                "columnas_categoricas": ", ".join(categorical_columns),
                "cantidad_categoricas": len(categorical_columns),
                "target_detectado": TARGET_COLUMN if TARGET_COLUMN in df.columns else None,
            }
        ]
    )
    return overview


def column_types_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resume tipo de dato y tipo semantico por columna."""
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        dtype_name = str(df[column].dtype)
        semantic_type = "numerica" if pd.api.types.is_numeric_dtype(df[column]) else "categorica"
        rows.append({"columna": column, "dtype": dtype_name, "tipo_semantico": semantic_type})
    return pd.DataFrame(rows)


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna cantidad y porcentaje de valores nulos."""
    missing_count = df.isna().sum()
    missing_pct = (missing_count / max(len(df), 1) * 100).round(2)
    table = pd.DataFrame(
        {
            "columna": missing_count.index,
            "cantidad_nulos": missing_count.values,
            "porcentaje_nulos": missing_pct.values,
        }
    )
    return table.sort_values(by=["cantidad_nulos", "columna"], ascending=[False, True]).reset_index(
        drop=True
    )


def duplicated_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna resumen de duplicados exactos."""
    duplicated_count = int(df.duplicated().sum())
    duplicated_pct = round(duplicated_count / max(len(df), 1) * 100, 2)
    return pd.DataFrame(
        [
            {
                "filas_totales": len(df),
                "duplicados_exactos": duplicated_count,
                "porcentaje_duplicados": duplicated_pct,
            }
        ]
    )


def categorical_value_counts(
    df: pd.DataFrame, categorical_columns: list[str]
) -> pd.DataFrame:
    """Retorna frecuencias y porcentajes para columnas categoricas."""
    rows: list[dict[str, Any]] = []
    total = max(len(df), 1)

    for column in categorical_columns:
        if column not in df.columns:
            continue
        counts = df[column].value_counts(dropna=False)
        for category, count in counts.items():
            rows.append(
                {
                    "columna": column,
                    "categoria": category,
                    "cantidad": int(count),
                    "porcentaje": round(count / total * 100, 2),
                }
            )

    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Retorna resumen estadistico de variables numericas."""
    available_columns = [column for column in numeric_columns if column in df.columns]
    if not available_columns:
        return pd.DataFrame()

    summary = df[available_columns].describe(percentiles=[0.25, 0.5, 0.75]).T
    summary = summary.reset_index().rename(columns={"index": "columna"})
    return summary


def target_summary(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Retorna estadisticos clave de la variable objetivo."""
    if target_column not in df.columns:
        raise ValueError(f"No se encontro la columna objetivo: {target_column}")

    target = pd.to_numeric(df[target_column], errors="coerce").dropna()
    q1 = float(target.quantile(0.25))
    q3 = float(target.quantile(0.75))
    iqr = q3 - q1
    summary = {
        "min": float(target.min()),
        "max": float(target.max()),
        "media": float(target.mean()),
        "mediana": float(target.median()),
        "desviacion_estandar": float(target.std(ddof=1)),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior_iqr": q1 - 1.5 * iqr,
        "limite_superior_iqr": q3 + 1.5 * iqr,
        "registros_target_menor_igual_0": int((target <= 0).sum()),
        "registros_target_menor_10": int((target < 10).sum()),
    }
    return pd.DataFrame(
        {"metrica": list(summary.keys()), "valor": list(summary.values())}
    )


def detect_outliers_iqr(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Detecta outliers por metodo IQR y retorna tabla resumen."""
    rows: list[dict[str, Any]] = []
    n_rows = max(len(df), 1)

    for column in numeric_columns:
        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        outlier_count = int(outliers.shape[0])

        rows.append(
            {
                "columna": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "limite_inferior": lower,
                "limite_superior": upper,
                "cantidad_outliers": outlier_count,
                "porcentaje_outliers": round(outlier_count / n_rows * 100, 2),
            }
        )

    return pd.DataFrame(rows)


def get_outlier_rows(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Retorna filas del DataFrame que son outliers en una columna."""
    if column not in df.columns:
        raise ValueError(f"La columna {column} no existe en el DataFrame.")

    series = pd.to_numeric(df[column], errors="coerce")
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df.loc[(series < lower) | (series > upper)].copy()


def correlation_with_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Calcula correlacion lineal de variables numericas con el target."""
    if target_column not in df.columns:
        raise ValueError(f"No se encontro la columna objetivo: {target_column}")

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target_column not in numeric_df.columns:
        raise ValueError(f"La columna objetivo {target_column} no es numerica.")

    correlations = numeric_df.corr(numeric_only=True)[target_column].drop(labels=[target_column])
    result = correlations.reset_index()
    result.columns = ["columna", "correlacion"]
    result["correlacion_absoluta"] = result["correlacion"].abs()
    return result.sort_values("correlacion_absoluta", ascending=False).reset_index(drop=True)


def business_rules_report(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna el reporte de reglas de negocio."""
    return validate_business_rules(df)


def suspicious_times_report(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    threshold: float = 10,
) -> pd.DataFrame:
    """Retorna registros con tiempos sospechosos para revision."""
    if target_column not in df.columns:
        raise ValueError(f"No se encontro la columna objetivo: {target_column}")

    target = pd.to_numeric(df[target_column], errors="coerce")
    suspicious_mask = target < threshold
    report = df.loc[suspicious_mask].copy()
    report["tiempo_sospechoso"] = True
    report["target_invalido"] = pd.to_numeric(report[target_column], errors="coerce") <= 0
    return report


def generate_eda_report_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Genera un diccionario con las tablas base del EDA."""
    available_numeric = [column for column in NUMERIC_COLUMNS if column in df.columns]
    available_categorical = [column for column in CATEGORICAL_COLUMNS if column in df.columns]

    return {
        "overview": dataset_overview(df),
        "column_types": column_types_summary(df),
        "missing_values": missing_values_table(df),
        "duplicates": duplicated_summary(df),
        "categorical_counts": categorical_value_counts(df, available_categorical),
        "numeric_summary": numeric_summary(df, available_numeric),
        "target_summary": target_summary(df, TARGET_COLUMN),
        "outliers_iqr": detect_outliers_iqr(df, available_numeric),
        "business_rules": business_rules_report(df),
        "suspicious_times": suspicious_times_report(df, TARGET_COLUMN, threshold=10),
    }
