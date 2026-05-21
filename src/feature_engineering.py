"""Funciones de feature engineering exploratorio para el EDA."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_preprocessing import ID_BODEGA_COLUMN


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide dos series manejando division por cero y nulos."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    result = np.where((den == 0) | den.isna(), np.nan, num / den)
    return pd.Series(result, index=numerator.index)


def add_time_cyclical_features(
    df: pd.DataFrame, hour_col: str = "hora_despacho"
) -> pd.DataFrame:
    """Agrega representacion ciclica de la hora de despacho."""
    if hour_col not in df.columns:
        raise ValueError(f"No se encontro la columna de hora: {hour_col}")

    df_copy = df.copy()
    hour_values = pd.to_numeric(df_copy[hour_col], errors="coerce")
    angle = 2 * np.pi * hour_values / 24.0
    df_copy["hora_sin"] = np.sin(angle)
    df_copy["hora_cos"] = np.cos(angle)
    return df_copy


def add_logistics_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features de razon logistica por kilometro."""
    required = ["distancia_km", "costo_envio", "consumo_nafta", "peso_carga_kg", "paradas_previas"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para crear ratios logisticos: {missing}")

    df_copy = df.copy()
    df_copy["costo_por_km"] = _safe_divide(df_copy["costo_envio"], df_copy["distancia_km"])
    df_copy["consumo_por_km"] = _safe_divide(df_copy["consumo_nafta"], df_copy["distancia_km"])
    df_copy["carga_por_km"] = _safe_divide(df_copy["peso_carga_kg"], df_copy["distancia_km"])
    df_copy["paradas_por_km"] = _safe_divide(df_copy["paradas_previas"], df_copy["distancia_km"])
    return df_copy


def prepare_exploratory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara un DataFrame con features exploratorias para analisis."""
    df_copy = df.copy()
    if ID_BODEGA_COLUMN in df_copy.columns:
        df_copy[ID_BODEGA_COLUMN] = df_copy[ID_BODEGA_COLUMN].astype("string").astype("category")
    df_copy = add_time_cyclical_features(df_copy, hour_col="hora_despacho")
    df_copy = add_logistics_ratio_features(df_copy)
    return df_copy


def get_feature_recommendations() -> pd.DataFrame:
    """Retorna recomendaciones de features para el modelado posterior."""
    rows = [
        {
            "feature": "hora_sin",
            "tipo_transformacion": "Ciclica",
            "recomendacion": "Usar junto a hora_cos para respetar periodicidad diaria.",
        },
        {
            "feature": "hora_cos",
            "tipo_transformacion": "Ciclica",
            "recomendacion": "Complementa hora_sin para capturar comportamiento horario.",
        },
        {
            "feature": "costo_por_km",
            "tipo_transformacion": "Ratio",
            "recomendacion": "Aproxima intensidad de costo por unidad de distancia.",
        },
        {
            "feature": "consumo_por_km",
            "tipo_transformacion": "Ratio",
            "recomendacion": "Puede capturar eficiencia de ruta y vehiculo.",
        },
        {
            "feature": "carga_por_km",
            "tipo_transformacion": "Ratio",
            "recomendacion": "Relaciona exigencia de carga con distancia recorrida.",
        },
        {
            "feature": "paradas_por_km",
            "tipo_transformacion": "Ratio",
            "recomendacion": "Representa densidad de paradas por trayecto.",
        },
        {
            "feature": "consumo_nafta",
            "tipo_transformacion": "Riesgo de leakage",
            "recomendacion": "Comparar modelos con y sin esta variable por posible fuga.",
        },
    ]
    return pd.DataFrame(rows)
