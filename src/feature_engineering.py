"""Feature engineering helpers for logistics regression workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from data_preprocessing import ID_BODEGA_COLUMN  # type: ignore
except ImportError:
    try:
        from .data_preprocessing import ID_BODEGA_COLUMN  # type: ignore
    except ImportError:
        ID_BODEGA_COLUMN = "id_bodega"


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series handling zeros, nulls and invalid values."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    result = np.where((den == 0) | den.isna(), np.nan, num / den)
    return pd.Series(result, index=numerator.index)


def add_time_cyclical_features(
    df: pd.DataFrame,
    hour_col: str = "hora_despacho",
) -> pd.DataFrame:
    """Add cyclical representation for dispatch hour."""
    if hour_col not in df.columns:
        raise ValueError(f"No se encontro la columna de hora: {hour_col}")

    df_copy = df.copy()
    hour_values = pd.to_numeric(df_copy[hour_col], errors="coerce")
    angle = 2 * np.pi * hour_values / 24.0
    df_copy["hora_sin"] = np.sin(angle)
    df_copy["hora_cos"] = np.cos(angle)
    return df_copy


def add_logistics_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create logistics ratio features per kilometer."""
    required = [
        "distancia_km",
        "costo_envio",
        "consumo_nafta",
        "peso_carga_kg",
        "paradas_previas",
    ]
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
    """Prepare exploratory engineered features for EDA usage."""
    df_copy = df.copy()
    if ID_BODEGA_COLUMN in df_copy.columns:
        df_copy[ID_BODEGA_COLUMN] = df_copy[ID_BODEGA_COLUMN].astype("string").astype("category")
    df_copy = add_time_cyclical_features(df_copy, hour_col="hora_despacho")
    df_copy = add_logistics_ratio_features(df_copy)
    return df_copy


def get_feature_recommendations() -> pd.DataFrame:
    """Return feature-engineering recommendations for regression experiments."""
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


def add_logistics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea nuevas variables para mejorar la prediccion del tiempo de entrega.

    Variables creadas:
    - distancia_por_parada
    - costo_por_km
    - consumo_por_km
    - hora_punta
    - muchas_paradas
    - carga_pesada
    - trafico_clima
    """
    df_features = df.copy()

    if "distancia_km" in df_features.columns and "paradas_previas" in df_features.columns:
        df_features["distancia_por_parada"] = (
            pd.to_numeric(df_features["distancia_km"], errors="coerce")
            / (pd.to_numeric(df_features["paradas_previas"], errors="coerce") + 1)
        )

    if "costo_envio" in df_features.columns and "distancia_km" in df_features.columns:
        distancia = pd.to_numeric(df_features["distancia_km"], errors="coerce").replace(0, np.nan)
        costo = pd.to_numeric(df_features["costo_envio"], errors="coerce")
        df_features["costo_por_km"] = costo / distancia

    if "consumo_nafta" in df_features.columns and "distancia_km" in df_features.columns:
        distancia = pd.to_numeric(df_features["distancia_km"], errors="coerce").replace(0, np.nan)
        consumo = pd.to_numeric(df_features["consumo_nafta"], errors="coerce")
        df_features["consumo_por_km"] = consumo / distancia

    if "hora_despacho" in df_features.columns:
        hora = pd.to_numeric(df_features["hora_despacho"], errors="coerce")
        df_features["hora_punta"] = (
            hora.between(7, 9, inclusive="both")
            | hora.between(18, 21, inclusive="both")
        ).astype(int)

    if "paradas_previas" in df_features.columns:
        paradas = pd.to_numeric(df_features["paradas_previas"], errors="coerce")
        df_features["muchas_paradas"] = (paradas >= paradas.median()).astype(int)

    if "peso_carga_kg" in df_features.columns:
        peso = pd.to_numeric(df_features["peso_carga_kg"], errors="coerce")
        df_features["carga_pesada"] = (peso >= peso.median()).astype(int)

    if "trafico_nivel" in df_features.columns and "clima" in df_features.columns:
        df_features["trafico_clima"] = (
            df_features["trafico_nivel"].astype(str)
            + "_"
            + df_features["clima"].astype(str)
        )

    return df_features


def get_enhanced_numeric_features() -> list[str]:
    """Retorna variables numericas adicionales de feature engineering."""
    return [
        "distancia_por_parada",
        "costo_por_km",
        "consumo_por_km",
        "hora_punta",
        "muchas_paradas",
        "carga_pesada",
    ]


def get_enhanced_categorical_features() -> list[str]:
    """Retorna variables categoricas adicionales de feature engineering."""
    return [
        "trafico_clima",
    ]


def build_enhanced_feature_lists(
    base_numeric_features: list[str],
    base_categorical_features: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Une las variables originales con las nuevas variables creadas.

    Returns
    -------
    tuple
        enhanced_numeric_features, enhanced_categorical_features, all_features
    """
    enhanced_numeric = list(
        dict.fromkeys(base_numeric_features + get_enhanced_numeric_features())
    )
    enhanced_categorical = list(
        dict.fromkeys(base_categorical_features + get_enhanced_categorical_features())
    )
    all_features = enhanced_numeric + enhanced_categorical

    return enhanced_numeric, enhanced_categorical, all_features
