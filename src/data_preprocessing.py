"""Funciones de limpieza y validacion para el dataset de logistica."""

from __future__ import annotations

from typing import Any

import pandas as pd

TARGET_COLUMN = "target_tiempo_entrega"
ID_BODEGA_COLUMN = "id_bodega"

REQUIRED_COLUMNS = [
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
    ID_BODEGA_COLUMN,
    TARGET_COLUMN,
]

CATEGORICAL_COLUMNS = ["trafico_nivel", "clima", "tipo_vehiculo", ID_BODEGA_COLUMN]
TEXT_CATEGORICAL_COLUMNS = ["trafico_nivel", "clima", "tipo_vehiculo"]

NUMERIC_COLUMNS = [
    "distancia_km",
    "peso_carga_kg",
    "paradas_previas",
    "experiencia_chofer_anios",
    "hora_despacho",
    "costo_envio",
    "consumo_nafta",
    TARGET_COLUMN,
]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas a minuscula con guiones bajos."""
    df_copy = df.copy()
    df_copy.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        for column in df_copy.columns
    ]
    return df_copy


def validate_required_columns(
    df: pd.DataFrame, required_columns: list[str] | None = None
) -> None:
    """Valida que el DataFrame tenga las columnas requeridas."""
    required = required_columns if required_columns is not None else REQUIRED_COLUMNS
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")


def clean_text_columns(df: pd.DataFrame, text_columns: list[str]) -> pd.DataFrame:
    """Limpia espacios y normaliza formato en columnas de texto."""
    df_copy = df.copy()
    for column in text_columns:
        if column not in df_copy.columns:
            continue
        cleaned = (
            df_copy[column]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        if column == "trafico_nivel":
            normalized = cleaned.str.lower().replace(
                {
                    "critico": "Critico",
                    "crítico": "Critico",
                    "alto": "Alto",
                    "medio": "Medio",
                    "bajo": "Bajo",
                }
            )
            df_copy[column] = normalized
        else:
            df_copy[column] = cleaned.str.title()
    return df_copy


def convert_id_bodega_to_category(
    df: pd.DataFrame, id_column: str = ID_BODEGA_COLUMN
) -> pd.DataFrame:
    """Convierte id_bodega a tipo categorico (texto)."""
    if id_column not in df.columns:
        return df.copy()

    df_copy = df.copy()
    numeric_values = pd.to_numeric(df_copy[id_column], errors="coerce")
    if (numeric_values.dropna() % 1 == 0).all():
        as_text = numeric_values.astype("Int64").astype("string")
    else:
        as_text = numeric_values.astype("string")
    df_copy[id_column] = as_text.astype("category")
    return df_copy


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna filas duplicadas exactas."""
    duplicate_mask = df.duplicated(keep="first")
    return df.loc[duplicate_mask].copy()


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna tabla de valores nulos por columna."""
    missing_count = df.isna().sum()
    missing_percentage = (missing_count / len(df) * 100).round(2)
    report = pd.DataFrame(
        {
            "columna": missing_count.index,
            "cantidad_nulos": missing_count.values,
            "porcentaje_nulos": missing_percentage.values,
        }
    )
    return report.sort_values("cantidad_nulos", ascending=False).reset_index(drop=True)


def validate_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Valida reglas de negocio y retorna un reporte tabular."""
    validate_required_columns(df)

    total_rows = max(len(df), 1)
    business_rules = {
        "distancia_km > 0": pd.to_numeric(df["distancia_km"], errors="coerce") > 0,
        "peso_carga_kg > 0": pd.to_numeric(df["peso_carga_kg"], errors="coerce") > 0,
        "paradas_previas >= 0": pd.to_numeric(df["paradas_previas"], errors="coerce")
        >= 0,
        "experiencia_chofer_anios >= 0": pd.to_numeric(
            df["experiencia_chofer_anios"], errors="coerce"
        )
        >= 0,
        "hora_despacho entre 0 y 23": pd.to_numeric(
            df["hora_despacho"], errors="coerce"
        ).between(0, 23, inclusive="both"),
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
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Elimina filas con target menor o igual a cero."""
    if target_column not in df.columns:
        raise ValueError(f"No se encontro la columna objetivo: {target_column}")

    target_values = pd.to_numeric(df[target_column], errors="coerce")
    invalid_mask = target_values <= 0
    removed_rows = df.loc[invalid_mask].copy()
    cleaned_df = df.loc[~invalid_mask].copy()
    return cleaned_df, removed_rows


def add_suspicious_time_flag(
    df: pd.DataFrame, threshold: float = 10, target_column: str = TARGET_COLUMN
) -> pd.DataFrame:
    """Agrega columna booleana para tiempos de entrega sospechosamente bajos."""
    if target_column not in df.columns:
        raise ValueError(f"No se encontro la columna objetivo: {target_column}")

    df_copy = df.copy()
    target_values = pd.to_numeric(df_copy[target_column], errors="coerce")
    df_copy["tiempo_sospechoso"] = target_values < threshold
    return df_copy


def clean_logistics_dataset(
    df: pd.DataFrame, remove_invalid_target: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aplica limpieza exploratoria del dataset de logistica."""
    original_rows = len(df)
    df_clean = normalize_column_names(df)
    validate_required_columns(df_clean)

    df_clean = clean_text_columns(df_clean, TEXT_CATEGORICAL_COLUMNS)
    df_clean = convert_id_bodega_to_category(df_clean)

    duplicate_rows = detect_duplicates(df_clean)
    if not duplicate_rows.empty:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

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
