"""Funciones de preprocesamiento de datos para el proyecto de logística predictiva."""

from __future__ import annotations


def load_data(file_path):
    """Carga datos desde un archivo y devuelve un DataFrame."""
    # TODO: Implementar carga robusta de datos según formato.
    raise NotImplementedError("Pendiente de implementación")


def clean_data(df):
    """Aplica limpieza básica sobre el DataFrame de entrada."""
    # TODO: Implementar tratamiento de nulos, duplicados y tipos.
    raise NotImplementedError("Pendiente de implementación")


def encode_features(df):
    """Codifica variables categóricas para modelado."""
    # TODO: Implementar estrategias de encoding según necesidad.
    raise NotImplementedError("Pendiente de implementación")


def scale_features(df, columns=None):
    """Escala variables numéricas en columnas seleccionadas."""
    # TODO: Implementar escalamiento (StandardScaler/MinMaxScaler, etc.).
    raise NotImplementedError("Pendiente de implementación")


def split_features_target(df, target_column):
    """Separa matriz de características y variable objetivo."""
    # TODO: Implementar separación X / y con validaciones.
    raise NotImplementedError("Pendiente de implementación")
