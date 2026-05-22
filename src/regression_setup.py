from pathlib import Path
from typing import Iterable

import pandas as pd


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


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Carga el dataset logistico desde un archivo CSV.

    Parameters
    ----------
    csv_path:
        Ruta del archivo CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos cargados.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    """
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo CSV en: {path}")

    return pd.read_csv(path)


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """
    Valida que el DataFrame contenga las columnas requeridas.

    Parameters
    ----------
    df:
        DataFrame a validar.
    required_columns:
        Columnas necesarias para el flujo de regresion.

    Raises
    ------
    ValueError
        Si faltan columnas requeridas.
    """
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")


def split_features_target(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa las variables de entrada X y la variable objetivo y.

    Esta funcion solo prepara X e y. No realiza train_test_split,
    entrenamiento, evaluacion ni optimizacion.

    Parameters
    ----------
    df:
        DataFrame con los datos.
    feature_columns:
        Lista de columnas predictoras. Si es None, usa FEATURE_COLUMNS.
    target_column:
        Nombre de la variable objetivo.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X con variables predictoras e y con la variable objetivo.
    """
    selected_features = feature_columns or FEATURE_COLUMNS
    required_columns = selected_features + [target_column]

    validate_required_columns(df, required_columns)

    X = df[selected_features].copy()
    y = df[target_column].copy()

    return X, y
