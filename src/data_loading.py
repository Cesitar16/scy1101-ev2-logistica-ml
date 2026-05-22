"""Funciones de carga de datos para el proyecto de Logistica 4.0."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_file_exists(path: str) -> None:
    """Valida que la ruta exista y sea un archivo."""
    dataset_path = Path(path).expanduser()
    if not dataset_path.exists() or not dataset_path.is_file():
        raise FileNotFoundError(
            f"No se encontro el archivo del dataset en: {dataset_path.resolve()}"
        )


def load_dataset(path: str) -> pd.DataFrame:
    """Carga un dataset CSV y retorna un DataFrame."""
    validate_file_exists(path)
    dataset_path = Path(path).expanduser()

    try:
        return pd.read_csv(dataset_path, encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError(f"No fue posible cargar el CSV desde {dataset_path}") from exc
