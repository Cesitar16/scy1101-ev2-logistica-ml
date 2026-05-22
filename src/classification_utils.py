"""Utilidades compartidas para el flujo de clasificacion supervisada."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2

TIME_TARGET_CANDIDATES = [
    "target_tiempo_entrega",
    "tiempo_entrega",
    "tiempo_entrega_min",
    "tiempo_entrega_minutos",
    "delivery_time",
    "delivery_time_minutes",
    "duracion_entrega",
    "duracion",
    "duration",
    "duration_minutes",
]

LEAKAGE_COLUMN_CANDIDATES = {
    "target_tiempo_entrega",
    "tiempo_entrega",
    "tiempo_entrega_min",
    "tiempo_entrega_minutos",
    "delivery_time",
    "delivery_time_minutes",
    "duracion_entrega",
    "duracion",
    "duration",
    "duration_minutes",
    "riesgo_logistico",
    "entrega_tardia",
    "alta_demora_estimada",
    "categoria_riesgo_entrega",
    "riesgo_entrega",
}

ID_COLUMN_HINTS = ("id", "uuid", "guid")


def ensure_directories(paths: Iterable[str | Path]) -> None:
    """[LEGACY/GENERICA] Crea directorios si no existen.

    Se mantiene como helper transversal. El flujo principal de Lucas usa las
    funciones de clasificacion del modulo (`crear_target_clasificacion`,
    `preparar_datos_clasificacion`) y los guardados especificos por modulo.
    """
    for path in paths:
        Path(path).expanduser().mkdir(parents=True, exist_ok=True)


def timestamp_tag(now: datetime | None = None) -> str:
    """Genera una etiqueta de tiempo util para versionar resultados."""
    current_time = now or datetime.now()
    return current_time.strftime("%Y%m%d_%H%M%S")


def save_metrics(metrics: Mapping[str, float], output_path: str | Path) -> Path:
    """[LEGACY/GENERICA] Guarda metricas en JSON y retorna ruta final.

    Nota:
    - Para clasificacion supervisada, el flujo oficial usa
      `evaluar_modelos_clasificacion`, que exporta CSV/TXT de forma estandar.
    """
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    serializable_metrics = {
        str(metric_name): float(metric_value)
        for metric_name, metric_value in metrics.items()
    }

    destination.write_text(
        json.dumps(serializable_metrics, indent=2),
        encoding="utf-8",
    )
    return destination


def save_dataframe(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    """[LEGACY/GENERICA] Persiste DataFrame en CSV y retorna la ruta final."""
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False)
    return destination


def save_model(model: Any, output_path: str | Path) -> Path:
    """[LEGACY/GENERICA] Serializa modelo en joblib.

    Nota:
    - Para la parte de Lucas se recomienda usar `guardar_modelo` /
      `cargar_modelo` en `src.classification_training` para mantener una API
      explicita del flujo de clasificacion.
    """
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def _normalize_text(value: Any) -> str:
    """Normaliza texto a minusculas sin acentos para mapeos categoricos."""
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    return without_accents.strip().lower()


def _first_available_column(columns: pd.Index, candidates: list[str]) -> str | None:
    """Retorna la primera columna disponible segun lista de candidatos."""
    existing = set(columns)
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


def _categorize_with_terciles(series: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    """Convierte una serie numerica en categorias rapida/normal/tardia usando terciles."""
    clean_series = pd.to_numeric(series, errors="coerce")
    valid_values = clean_series[clean_series.notna()]
    if valid_values.empty:
        raise ValueError(
            "No hay valores numericos validos para construir el target de clasificacion."
        )

    q33 = float(valid_values.quantile(1 / 3))
    q66 = float(valid_values.quantile(2 / 3))

    categories = pd.cut(
        clean_series,
        bins=[-np.inf, q33, q66, np.inf],
        labels=["rapida", "normal", "tardia"],
        include_lowest=True,
    )

    categories = categories.astype("string").fillna("normal")
    return categories, {"p33": q33, "p66": q66}


def _rank_risk_component(
    series: pd.Series,
    high_values_increase_risk: bool = True,
) -> pd.Series:
    """Escala una serie al rango [0, 1] con percentiles para componer riesgo."""
    numeric_values = pd.to_numeric(series, errors="coerce")
    ranked = numeric_values.rank(pct=True, method="average")
    if not high_values_increase_risk:
        ranked = 1 - ranked
    return ranked


def _detect_columns_to_exclude(
    df: pd.DataFrame,
    target_col: str,
) -> dict[str, str]:
    """Detecta columnas a excluir y el motivo."""
    drop_reasons: dict[str, str] = {}

    normalized_leakage = {_normalize_text(column) for column in LEAKAGE_COLUMN_CANDIDATES}
    normalized_leakage.update({_normalize_text(column) for column in TIME_TARGET_CANDIDATES})

    for column in df.columns:
        normalized_column = _normalize_text(column).replace(" ", "_")
        if column == target_col:
            drop_reasons[column] = "columna_target"
            continue

        if column in LEAKAGE_COLUMN_CANDIDATES or normalized_column in normalized_leakage:
            drop_reasons[column] = "posible_fuga_de_datos"
            continue

        series = df[column]
        non_null = series.dropna()
        unique_non_null = int(non_null.nunique(dropna=True))
        unique_ratio = unique_non_null / max(len(non_null), 1)

        is_id_like = (
            normalized_column == "id"
            or normalized_column.endswith("_id")
            or normalized_column.startswith("id_")
            or any(hint in normalized_column for hint in ID_COLUMN_HINTS)
        )
        if is_id_like and unique_ratio >= 0.9:
            drop_reasons[column] = "identificador_alta_unicidad"
            continue

        is_text = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        if is_text and not non_null.empty:
            avg_text_length = float(non_null.astype(str).str.len().mean())
            if unique_ratio >= 0.8 and avg_text_length >= 20:
                drop_reasons[column] = "texto_libre_irrelevante"
                continue

        if unique_non_null <= 1:
            drop_reasons[column] = "columna_constante"

    return drop_reasons


def crear_target_clasificacion(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `categoria_entrega` para clasificacion supervisada.

    Estrategia:
    1. Si existe una columna de tiempo/duracion de entrega, usa terciles (P33, P66)
       para etiquetar `rapida`, `normal`, `tardia`.
    2. Si no existe esa columna, construye un indice `riesgo_logistico` con columnas
       operativas disponibles (distancia, trafico, clima, paradas, experiencia, etc.)
       y luego lo discretiza en terciles para obtener `categoria_entrega`.

    Requisitos que cumple:
    - No modifica el DataFrame original.
    - Evita nulos en `categoria_entrega`.
    - Guarda un resumen en `df.attrs['target_clasificacion_info']`.
    - Imprime distribucion de clases para inspeccion rapida.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se esperaba un pandas.DataFrame como entrada.")
    if df.empty:
        raise ValueError("El DataFrame esta vacio; no es posible crear la variable objetivo.")

    df_copy = df.copy(deep=True)
    target_time_column = _first_available_column(df_copy.columns, TIME_TARGET_CANDIDATES)

    if target_time_column is not None:
        numeric_time = pd.to_numeric(df_copy[target_time_column], errors="coerce")
        positive_time = numeric_time.where(numeric_time > 0)
        categories, thresholds = _categorize_with_terciles(positive_time)
        df_copy["categoria_entrega"] = categories

        method_description = "terciles_sobre_tiempo_real"
        metadata: dict[str, Any] = {
            "metodo": method_description,
            "columna_base": target_time_column,
            "umbrales": thresholds,
            "registros_invalidos_o_nulos_tiempo": int((positive_time.isna()).sum()),
        }
    else:
        risk_components: list[pd.Series] = []
        used_columns: list[str] = []

        distance_column = _first_available_column(
            df_copy.columns,
            ["distancia_km", "distance_km", "distance"],
        )
        if distance_column:
            risk_components.append(_rank_risk_component(df_copy[distance_column], True))
            used_columns.append(distance_column)

        stop_column = _first_available_column(
            df_copy.columns,
            ["paradas_previas", "stops", "prior_stops", "num_stops"],
        )
        if stop_column:
            risk_components.append(_rank_risk_component(df_copy[stop_column], True))
            used_columns.append(stop_column)

        load_column = _first_available_column(
            df_copy.columns,
            ["peso_carga_kg", "peso_carga", "cargo_weight_kg", "load_weight_kg"],
        )
        if load_column:
            risk_components.append(_rank_risk_component(df_copy[load_column], True))
            used_columns.append(load_column)

        experience_column = _first_available_column(
            df_copy.columns,
            ["experiencia_chofer_anios", "driver_experience_years", "experiencia_conductor"],
        )
        if experience_column:
            risk_components.append(_rank_risk_component(df_copy[experience_column], False))
            used_columns.append(experience_column)

        cost_column = _first_available_column(
            df_copy.columns,
            ["costo_envio", "shipping_cost", "delivery_cost"],
        )
        if cost_column:
            risk_components.append(_rank_risk_component(df_copy[cost_column], True))
            used_columns.append(cost_column)

        fuel_column = _first_available_column(
            df_copy.columns,
            ["consumo_nafta", "fuel_consumption", "consumo_combustible"],
        )
        if fuel_column:
            risk_components.append(_rank_risk_component(df_copy[fuel_column], True))
            used_columns.append(fuel_column)

        hour_column = _first_available_column(
            df_copy.columns,
            ["hora_despacho", "hora_pedido", "order_hour", "delivery_hour"],
        )
        if hour_column:
            hour_values = pd.to_numeric(df_copy[hour_column], errors="coerce")
            hour_risk = pd.Series(0.4, index=df_copy.index, dtype="float64")
            peak_mask = hour_values.between(7, 10, inclusive="both") | hour_values.between(
                17, 20, inclusive="both"
            )
            night_mask = hour_values.between(0, 5, inclusive="both")
            hour_risk.loc[peak_mask] = 1.0
            hour_risk.loc[night_mask] = 0.6
            hour_risk.loc[hour_values.isna()] = np.nan
            risk_components.append(hour_risk)
            used_columns.append(hour_column)

        traffic_column = _first_available_column(
            df_copy.columns,
            ["trafico_nivel", "traffic_level", "traffic"],
        )
        if traffic_column:
            traffic_map = {
                "bajo": 0.1,
                "low": 0.1,
                "medio": 0.5,
                "medium": 0.5,
                "alto": 0.8,
                "high": 0.8,
                "critico": 1.0,
                "critical": 1.0,
            }
            traffic_risk = df_copy[traffic_column].map(
                lambda value: traffic_map.get(_normalize_text(value), np.nan)
            )
            risk_components.append(pd.Series(traffic_risk, index=df_copy.index))
            used_columns.append(traffic_column)

        weather_column = _first_available_column(
            df_copy.columns,
            ["clima", "weather", "weather_condition"],
        )
        if weather_column:
            weather_map = {
                "despejado": 0.1,
                "soleado": 0.1,
                "sunny": 0.1,
                "nublado": 0.4,
                "cloudy": 0.4,
                "parcialmente nublado": 0.35,
                "lluvia": 0.8,
                "lluvioso": 0.8,
                "rainy": 0.8,
                "tormenta": 1.0,
                "storm": 1.0,
                "niebla": 0.7,
                "fog": 0.7,
            }
            weather_risk = df_copy[weather_column].map(
                lambda value: weather_map.get(_normalize_text(value), np.nan)
            )
            risk_components.append(pd.Series(weather_risk, index=df_copy.index))
            used_columns.append(weather_column)

        vehicle_column = _first_available_column(
            df_copy.columns,
            ["tipo_vehiculo", "vehicle_type", "vehicle"],
        )
        if vehicle_column:
            vehicle_map = {
                "moto": 0.2,
                "motocicleta": 0.2,
                "bike": 0.2,
                "van": 0.5,
                "furgon": 0.5,
                "camioneta": 0.55,
                "auto": 0.35,
                "car": 0.35,
                "camion": 0.8,
                "truck": 0.8,
            }
            vehicle_risk = df_copy[vehicle_column].map(
                lambda value: vehicle_map.get(_normalize_text(value), np.nan)
            )
            risk_components.append(pd.Series(vehicle_risk, index=df_copy.index))
            used_columns.append(vehicle_column)

        if not risk_components:
            raise ValueError(
                "No se encontro columna de tiempo de entrega ni variables operativas "
                "suficientes para construir un target alternativo. "
                "Verifica columnas como distancia_km, trafico_nivel, clima, tipo_vehiculo, "
                "peso_carga_kg, paradas_previas, experiencia_chofer_anios, hora_despacho, "
                "costo_envio o consumo_nafta."
            )

        risk_df = pd.concat(risk_components, axis=1)
        risk_score = risk_df.mean(axis=1, skipna=True)
        risk_score = risk_score.fillna(float(risk_score.median(skipna=True)))
        risk_score = risk_score.fillna(0.5)

        categories, thresholds = _categorize_with_terciles(risk_score)
        df_copy["riesgo_logistico"] = risk_score.round(4)
        df_copy["categoria_entrega"] = categories

        method_description = "terciles_sobre_riesgo_estimado"
        metadata = {
            "metodo": method_description,
            "columna_base": "riesgo_logistico",
            "columnas_utilizadas": used_columns,
            "umbrales": thresholds,
            "registros_con_score_imputado": int(risk_df.isna().all(axis=1).sum()),
        }

    class_distribution = (
        df_copy["categoria_entrega"]
        .value_counts(dropna=False)
        .rename_axis("categoria_entrega")
        .reset_index(name="cantidad")
    )
    class_distribution["proporcion"] = (
        class_distribution["cantidad"] / class_distribution["cantidad"].sum()
    ).round(4)

    metadata["distribucion_clases"] = class_distribution.to_dict(orient="records")
    df_copy.attrs["target_clasificacion_info"] = metadata

    print("Distribucion de categoria_entrega:")
    print(class_distribution.to_string(index=False))
    print(f"Metodo de construccion: {metadata['metodo']}")
    print(f"Columna base: {metadata['columna_base']}")

    return df_copy


def calcular_umbrales_target_train(y_train_continuo: pd.Series) -> dict[str, float]:
    """Calcula umbrales de clasificacion (q33, q66) usando solo train.

    Parametros:
    - y_train_continuo: serie continua del tiempo/duracion de entrega en train.

    Retorna:
    - diccionario con llaves `q33` y `q66`.
    """
    if not isinstance(y_train_continuo, pd.Series):
        raise TypeError("`y_train_continuo` debe ser una serie de pandas.")

    numeric_target = pd.to_numeric(y_train_continuo, errors="coerce").where(
        lambda values: values > 0
    )
    valid_values = numeric_target.dropna()
    if valid_values.empty:
        raise ValueError(
            "No hay valores continuos validos en train para calcular umbrales del target."
        )
    if valid_values.nunique(dropna=True) < 3:
        raise ValueError(
            "Se requieren al menos 3 valores distintos en train para construir "
            "las clases rapida/normal/tardia con terciles."
        )

    q33 = float(valid_values.quantile(1 / 3))
    q66 = float(valid_values.quantile(2 / 3))
    if not np.isfinite(q33) or not np.isfinite(q66):
        raise ValueError("Los umbrales calculados no son numericamente validos.")
    if q33 >= q66:
        raise ValueError(
            f"Umbrales invalidos: q33 ({q33}) debe ser menor que q66 ({q66})."
        )

    return {"q33": q33, "q66": q66}


def aplicar_umbrales_categoria_entrega(
    y_continuo: pd.Series,
    umbrales: Mapping[str, float],
) -> pd.Series:
    """Aplica umbrales predefinidos para crear `categoria_entrega`.

    Etiquetas retornadas:
    - `rapida`: valores <= q33
    - `normal`: valores > q33 y <= q66
    - `tardia`: valores > q66
    """
    if not isinstance(y_continuo, pd.Series):
        raise TypeError("`y_continuo` debe ser una serie de pandas.")
    if not isinstance(umbrales, Mapping):
        raise TypeError("`umbrales` debe ser un diccionario/Mapping con q33 y q66.")
    if "q33" not in umbrales or "q66" not in umbrales:
        raise ValueError("`umbrales` debe contener las llaves 'q33' y 'q66'.")

    q33 = float(umbrales["q33"])
    q66 = float(umbrales["q66"])
    if not np.isfinite(q33) or not np.isfinite(q66):
        raise ValueError("Los valores de umbral deben ser numericos y finitos.")
    if q33 >= q66:
        raise ValueError(
            f"Umbrales invalidos: q33 ({q33}) debe ser menor que q66 ({q66})."
        )

    y_numeric = pd.to_numeric(y_continuo, errors="coerce")
    categories = pd.cut(
        y_numeric,
        bins=[-np.inf, q33, q66, np.inf],
        labels=["rapida", "normal", "tardia"],
        include_lowest=True,
    )

    # Se asigna "normal" a registros sin valor continuo valido para evitar nulos.
    target = categories.astype("string").fillna("normal")
    if target.isna().any():
        raise ValueError(
            "No fue posible construir categoria_entrega sin nulos tras aplicar umbrales."
        )
    return target


def preparar_datos_clasificacion_sin_leakage(
    df: pd.DataFrame,
    target_time_col: str = "target_tiempo_entrega",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    ColumnTransformer,
    dict[str, Any],
    dict[str, float],
]:
    """Prepara datos de clasificacion sin leakage de distribucion del target.

    Flujo:
    1. Divide train/test usando el dataset base (sin target categorico construido).
    2. Calcula umbrales q33/q66 solo con `target_time_col` de train.
    3. Aplica esos umbrales a train y test para crear `categoria_entrega`.
    4. Construye features y preprocesador listos para pipeline de Scikit-learn.

    Retorna:
    - X_train, X_test, y_train, y_test, preprocessor, feature_info, umbrales
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se esperaba un pandas.DataFrame como entrada.")
    if df.empty:
        raise ValueError("El DataFrame esta vacio; no es posible preparar datos.")
    if target_time_col not in df.columns:
        raise ValueError(
            f"No se encontro la columna continua '{target_time_col}' para construir "
            "el target sin leakage."
        )
    if not 0 < test_size < 1:
        raise ValueError("`test_size` debe ser un valor entre 0 y 1 (ejemplo: 0.2).")

    df_copy = df.copy(deep=True)

    train_index, test_index = train_test_split(
        df_copy.index,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_df = df_copy.loc[train_index].copy()
    test_df = df_copy.loc[test_index].copy()

    y_train_continuo = pd.to_numeric(train_df[target_time_col], errors="coerce").where(
        lambda values: values > 0
    )
    y_test_continuo = pd.to_numeric(test_df[target_time_col], errors="coerce").where(
        lambda values: values > 0
    )

    umbrales = calcular_umbrales_target_train(y_train_continuo=y_train_continuo)
    y_train = aplicar_umbrales_categoria_entrega(
        y_continuo=y_train_continuo,
        umbrales=umbrales,
    )
    y_test = aplicar_umbrales_categoria_entrega(
        y_continuo=y_test_continuo,
        umbrales=umbrales,
    )

    train_with_target = train_df.copy()
    test_with_target = test_df.copy()
    train_with_target["categoria_entrega"] = y_train.values
    test_with_target["categoria_entrega"] = y_test.values

    drop_reasons = _detect_columns_to_exclude(
        train_with_target,
        target_col="categoria_entrega",
    )
    dropped_columns = sorted(drop_reasons.keys())

    X_train = train_with_target.drop(columns=dropped_columns, errors="ignore")
    X_test = test_with_target.drop(columns=dropped_columns, errors="ignore")
    X_test = X_test.reindex(columns=X_train.columns)

    if X_train.empty:
        raise ValueError(
            "No quedaron columnas predictoras luego de aplicar exclusiones. "
            "Revisa la logica de limpieza o las columnas disponibles."
        )

    bool_columns = [
        column for column in X_train.columns if pd.api.types.is_bool_dtype(X_train[column])
    ]
    numeric_columns = [
        column
        for column in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[column]) and column not in bool_columns
    ]
    categorical_columns = [
        column for column in X_train.columns if column not in numeric_columns
    ]

    if not numeric_columns and not categorical_columns:
        raise ValueError("No se detectaron columnas numericas ni categoricas utilizables.")

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    feature_info: dict[str, Any] = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "dropped_columns": dropped_columns,
        "drop_reasons": drop_reasons,
        "stratify_used": False,
        "class_distribution_train": y_train.value_counts(normalize=True).round(4).to_dict(),
        "class_distribution_test": y_test.value_counts(normalize=True).round(4).to_dict(),
        "target_time_col": target_time_col,
        "split_method": "split_antes_de_target",
    }

    return X_train, X_test, y_train, y_test, preprocessor, feature_info, umbrales


def preparar_datos_clasificacion(
    df: pd.DataFrame,
    target_col: str = "categoria_entrega",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, dict[str, Any]]:
    """Prepara X/y y preprocesador para clasificacion supervisada.

    Pasos:
    1. Valida existencia de `target_col`.
    2. Excluye columnas no aptas para prediccion (target, posibles fugas, IDs unicos,
       texto libre irrelevante, columnas constantes).
    3. Detecta variables numericas y categoricas.
    4. Construye `ColumnTransformer`:
       - numericas: `SimpleImputer(strategy='median')` + `StandardScaler`
       - categoricas: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder`
    5. Aplica `train_test_split` y usa `stratify=y` cuando la distribucion lo permite.

    Retorna:
    `(X_train, X_test, y_train, y_test, preprocessor, feature_info)`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se esperaba un pandas.DataFrame como entrada.")
    if df.empty:
        raise ValueError("El DataFrame esta vacio; no es posible preparar datos.")
    if target_col not in df.columns:
        raise ValueError(
            f"No se encontro la columna objetivo '{target_col}'. "
            "Ejecuta crear_target_clasificacion(df) antes de preparar los datos."
        )
    if not 0 < test_size < 1:
        raise ValueError("`test_size` debe ser un valor entre 0 y 1 (ejemplo: 0.2).")

    df_copy = df.copy(deep=True)
    y = df_copy[target_col].copy()

    if y.isna().any():
        null_count = int(y.isna().sum())
        raise ValueError(
            f"La columna objetivo '{target_col}' contiene {null_count} valores nulos. "
            "Corrige el target antes de dividir los datos."
        )

    drop_reasons = _detect_columns_to_exclude(df_copy, target_col=target_col)
    dropped_columns = sorted(drop_reasons.keys())
    X = df_copy.drop(columns=dropped_columns, errors="ignore")

    if X.empty:
        raise ValueError(
            "No quedaron columnas predictoras luego de aplicar exclusiones. "
            "Revisa la logica de limpieza o las columnas disponibles."
        )

    bool_columns = [column for column in X.columns if pd.api.types.is_bool_dtype(X[column])]
    numeric_columns = [
        column
        for column in X.columns
        if pd.api.types.is_numeric_dtype(X[column]) and column not in bool_columns
    ]
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    if not numeric_columns and not categorical_columns:
        raise ValueError("No se detectaron columnas numericas ni categoricas utilizables.")

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    class_counts = y.value_counts(dropna=False)
    n_samples = len(y)
    n_classes = int(class_counts.shape[0])

    if isinstance(test_size, float):
        n_test = int(np.ceil(n_samples * test_size))
    else:
        n_test = int(test_size)
    n_train = n_samples - n_test

    stratify_is_possible = (
        n_classes > 1
        and int(class_counts.min()) >= 2
        and n_test >= n_classes
        and n_train >= n_classes
    )
    stratify_y = y if stratify_is_possible else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )

    feature_info: dict[str, Any] = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "dropped_columns": dropped_columns,
        "drop_reasons": drop_reasons,
        "stratify_used": stratify_y is not None,
        "class_distribution": y.value_counts(normalize=True).round(4).to_dict(),
    }

    return X_train, X_test, y_train, y_test, preprocessor, feature_info
