"""Feature engineering focused on difficult error segments for regression."""

from __future__ import annotations

import pandas as pd


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def add_error_focused_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variables binarias y de interaccion enfocadas en segmentos dificiles.
    """
    df_features = df.copy()

    if "tipo_vehiculo" in df_features.columns:
        tipo = df_features["tipo_vehiculo"].astype("string").str.lower().str.strip()
        df_features["es_camion"] = (tipo == "camion").astype(int)
        df_features["es_moto"] = (tipo == "moto").astype(int)
    else:
        df_features["es_camion"] = 0
        df_features["es_moto"] = 0

    if "trafico_nivel" in df_features.columns:
        trafico = df_features["trafico_nivel"].astype("string").str.lower().str.strip()
        df_features["trafico_alto"] = (trafico == "alto").astype(int)
        df_features["trafico_bajo"] = (trafico == "bajo").astype(int)
    else:
        df_features["trafico_alto"] = 0
        df_features["trafico_bajo"] = 0

    if "hora_despacho" in df_features.columns:
        hora = _to_numeric(df_features["hora_despacho"])
        df_features["hora_punta"] = (
            hora.between(7, 9, inclusive="both")
            | hora.between(18, 21, inclusive="both")
        ).astype(int)
    else:
        df_features["hora_punta"] = 0

    if "distancia_km" in df_features.columns:
        distancia = _to_numeric(df_features["distancia_km"])
        q75_dist = distancia.quantile(0.75)
        q25_dist = distancia.quantile(0.25)
        df_features["distancia_larga"] = (distancia >= q75_dist).astype(int)
        df_features["distancia_corta"] = (distancia <= q25_dist).astype(int)
    else:
        df_features["distancia_larga"] = 0
        df_features["distancia_corta"] = 0

    if "peso_carga_kg" in df_features.columns:
        peso = _to_numeric(df_features["peso_carga_kg"])
        q75_peso = peso.quantile(0.75)
        df_features["carga_pesada"] = (peso >= q75_peso).astype(int)
    else:
        df_features["carga_pesada"] = 0

    if "paradas_previas" in df_features.columns:
        paradas = _to_numeric(df_features["paradas_previas"])
        q75_paradas = paradas.quantile(0.75)
        df_features["muchas_paradas"] = (paradas >= q75_paradas).astype(int)
    else:
        df_features["muchas_paradas"] = 0

    if "id_bodega" in df_features.columns:
        bodega = pd.to_numeric(df_features["id_bodega"], errors="coerce")
        df_features["bodega_1"] = (bodega == 1).astype(int)
        df_features["bodega_4"] = (bodega == 4).astype(int)
        df_features["bodega_problematico"] = bodega.isin([1, 4]).astype(int)
    else:
        df_features["bodega_1"] = 0
        df_features["bodega_4"] = 0
        df_features["bodega_problematico"] = 0

    df_features["camion_trafico_alto"] = df_features["es_camion"] * df_features["trafico_alto"]
    df_features["camion_carga_pesada"] = df_features["es_camion"] * df_features["carga_pesada"]
    df_features["trafico_alto_hora_punta"] = df_features["trafico_alto"] * df_features["hora_punta"]
    df_features["distancia_larga_muchas_paradas"] = (
        df_features["distancia_larga"] * df_features["muchas_paradas"]
    )
    df_features["bodega_problematico_trafico_alto"] = (
        df_features["bodega_problematico"] * df_features["trafico_alto"]
    )

    return df_features


def get_error_focused_numeric_features() -> list[str]:
    """
    Retorna lista de variables numericas creadas.
    """
    return [
        "es_camion",
        "es_moto",
        "trafico_alto",
        "trafico_bajo",
        "hora_punta",
        "distancia_larga",
        "distancia_corta",
        "carga_pesada",
        "muchas_paradas",
        "camion_trafico_alto",
        "camion_carga_pesada",
        "trafico_alto_hora_punta",
        "distancia_larga_muchas_paradas",
        "bodega_1",
        "bodega_4",
        "bodega_problematico",
        "bodega_problematico_trafico_alto",
    ]


def get_error_focused_categorical_features() -> list[str]:
    """
    Retorna lista de variables categoricas creadas, si aplica.
    """
    return []
