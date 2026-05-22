"""Advanced feature engineering utilities for precision-focused regression."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def add_basic_precision_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables adicionales que pueden ayudar a estimar mejor
    el tiempo exacto de entrega.
    """
    df_features = df.copy()

    if "hora_despacho" in df_features.columns:
        hour_values = pd.to_numeric(df_features["hora_despacho"], errors="coerce")
        df_features["hora_punta_manana"] = hour_values.between(7, 9, inclusive="both").astype(int)
        df_features["hora_punta_tarde"] = hour_values.between(18, 21, inclusive="both").astype(int)
        df_features["hora_nocturna"] = (
            (hour_values >= 22) | (hour_values <= 6)
        ).astype(int)

    if "distancia_km" in df_features.columns and "paradas_previas" in df_features.columns:
        distancia = pd.to_numeric(df_features["distancia_km"], errors="coerce")
        paradas = pd.to_numeric(df_features["paradas_previas"], errors="coerce")
        df_features["distancia_total_ajustada"] = distancia * (1 + 0.08 * paradas)
        df_features["paradas_por_km"] = paradas / distancia.replace(0, np.nan)

    if "peso_carga_kg" in df_features.columns and "distancia_km" in df_features.columns:
        peso = pd.to_numeric(df_features["peso_carga_kg"], errors="coerce")
        distancia = pd.to_numeric(df_features["distancia_km"], errors="coerce")
        df_features["peso_por_km"] = peso / distancia.replace(0, np.nan)
        df_features["carga_x_distancia"] = peso * distancia

    if "experiencia_chofer_anios" in df_features.columns:
        experiencia = pd.to_numeric(df_features["experiencia_chofer_anios"], errors="coerce")
        df_features["chofer_experto"] = (
            experiencia >= experiencia.median(skipna=True)
        ).astype(int)

    if "costo_envio" in df_features.columns and "peso_carga_kg" in df_features.columns:
        costo = pd.to_numeric(df_features["costo_envio"], errors="coerce")
        peso = pd.to_numeric(df_features["peso_carga_kg"], errors="coerce")
        df_features["costo_por_kg"] = costo / peso.replace(0, np.nan)

    if "consumo_nafta" in df_features.columns and "peso_carga_kg" in df_features.columns:
        consumo = pd.to_numeric(df_features["consumo_nafta"], errors="coerce")
        peso = pd.to_numeric(df_features["peso_carga_kg"], errors="coerce")
        df_features["consumo_por_kg"] = consumo / peso.replace(0, np.nan)

    if "trafico_nivel" in df_features.columns and "tipo_vehiculo" in df_features.columns:
        df_features["trafico_vehiculo"] = (
            df_features["trafico_nivel"].astype(str) + "_" + df_features["tipo_vehiculo"].astype(str)
        )

    if "clima" in df_features.columns and "tipo_vehiculo" in df_features.columns:
        df_features["clima_vehiculo"] = (
            df_features["clima"].astype(str) + "_" + df_features["tipo_vehiculo"].astype(str)
        )

    df_features = df_features.replace([np.inf, -np.inf], np.nan)

    return df_features


def get_precision_numeric_features() -> list[str]:
    return [
        "hora_punta_manana",
        "hora_punta_tarde",
        "hora_nocturna",
        "distancia_total_ajustada",
        "paradas_por_km",
        "peso_por_km",
        "carga_x_distancia",
        "chofer_experto",
        "costo_por_kg",
        "consumo_por_kg",
    ]


def get_precision_categorical_features() -> list[str]:
    return [
        "trafico_vehiculo",
        "clima_vehiculo",
    ]


def build_precision_feature_lists(
    base_numeric_features: list[str],
    base_categorical_features: list[str],
    previous_extra_numeric_features: list[str] | None = None,
    previous_extra_categorical_features: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Une variables originales, variables creadas anteriormente y nuevas variables de precision.
    """
    previous_extra_numeric_features = previous_extra_numeric_features or []
    previous_extra_categorical_features = previous_extra_categorical_features or []

    numeric_features = (
        base_numeric_features
        + previous_extra_numeric_features
        + get_precision_numeric_features()
    )

    categorical_features = (
        base_categorical_features
        + previous_extra_categorical_features
        + get_precision_categorical_features()
    )

    numeric_features = list(dict.fromkeys(numeric_features))
    categorical_features = list(dict.fromkeys(categorical_features))
    all_features = numeric_features + categorical_features

    return numeric_features, categorical_features, all_features


class GroupMeanTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Crea variables historicas usando el promedio del target por grupo.

    IMPORTANTE:
    Este transformer evita fuga de datos porque aprende las medias solo en fit()
    usando el conjunto de entrenamiento. Luego aplica esas medias en transform().
    """

    def __init__(
        self,
        group_columns: Iterable[str] | None = None,
        target_column: str = "target_tiempo_entrega",
    ) -> None:
        self.group_columns = list(group_columns) if group_columns is not None else [
            "id_bodega",
            "tipo_vehiculo",
            "trafico_nivel",
        ]
        self.target_column = target_column
        self.global_mean_: float | None = None
        self.group_maps_: dict[str, dict] = {}

    def fit(self, X, y):
        X_fit = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_fit = X_fit.reset_index(drop=True)
        y_series = pd.Series(y).reset_index(drop=True)
        X_fit[self.target_column] = y_series

        self.global_mean_ = float(y_series.mean())
        self.group_maps_ = {}

        for column in self.group_columns:
            if column in X_fit.columns:
                self.group_maps_[column] = (
                    X_fit.groupby(column)[self.target_column].mean().to_dict()
                )

        return self

    def transform(self, X):
        if self.global_mean_ is None:
            raise RuntimeError("GroupMeanTargetEncoder no ha sido ajustado (fit).")

        X_transform = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        for column, mapping in self.group_maps_.items():
            new_column = f"hist_mean_target_by_{column}"
            if column in X_transform.columns:
                X_transform[new_column] = X_transform[column].map(mapping).fillna(self.global_mean_)
            else:
                X_transform[new_column] = self.global_mean_

        return X_transform


def get_historical_numeric_features() -> list[str]:
    return [
        "hist_mean_target_by_id_bodega",
        "hist_mean_target_by_tipo_vehiculo",
        "hist_mean_target_by_trafico_nivel",
    ]
