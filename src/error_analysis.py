"""Error analysis utilities for regression predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_prediction_error_dataframe(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred,
) -> pd.DataFrame:
    """
    Devuelve DataFrame con X_test + real + predicho + error + error_absoluto.
    """
    error_df = X_test.reset_index(drop=True).copy()
    error_df["real"] = y_test.reset_index(drop=True)
    error_df["predicho"] = pd.Series(y_pred).reset_index(drop=True)
    error_df["error"] = error_df["real"] - error_df["predicho"]
    error_df["error_absoluto"] = error_df["error"].abs()
    error_df["es_sobreestimacion"] = (error_df["predicho"] > error_df["real"]).astype(int)
    error_df["es_subestimacion"] = (error_df["predicho"] < error_df["real"]).astype(int)
    error_df["direccion_error"] = np.where(
        error_df["error"] > 0,
        "subestimacion",
        np.where(error_df["error"] < 0, "sobreestimacion", "sin_error"),
    )
    return error_df


def add_error_segment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas de segmento:
    - hora_punta
    - distancia_larga
    - carga_pesada
    - muchas_paradas
    """
    df_segment = df.copy()

    if "hora_despacho" in df_segment.columns:
        hora = pd.to_numeric(df_segment["hora_despacho"], errors="coerce")
        df_segment["hora_punta"] = (
            hora.between(7, 9, inclusive="both")
            | hora.between(18, 21, inclusive="both")
        ).astype(int)

    if "distancia_km" in df_segment.columns:
        distancia = pd.to_numeric(df_segment["distancia_km"], errors="coerce")
        p75_distancia = distancia.quantile(0.75)
        df_segment["distancia_larga"] = (distancia >= p75_distancia).astype(int)

    if "peso_carga_kg" in df_segment.columns:
        peso = pd.to_numeric(df_segment["peso_carga_kg"], errors="coerce")
        p75_peso = peso.quantile(0.75)
        df_segment["carga_pesada"] = (peso >= p75_peso).astype(int)

    if "paradas_previas" in df_segment.columns:
        paradas = pd.to_numeric(df_segment["paradas_previas"], errors="coerce")
        p75_paradas = paradas.quantile(0.75)
        df_segment["muchas_paradas"] = (paradas >= p75_paradas).astype(int)

    return df_segment


def calculate_segment_error_metrics(
    error_df: pd.DataFrame,
    segment_columns: list[str],
) -> pd.DataFrame:
    """
    Calcula MAE, RMSE, error medio, error mediano, n y max_error por segmento.
    """
    rows: list[dict[str, object]] = []

    for column in segment_columns:
        if column not in error_df.columns:
            continue

        grouped = error_df.groupby(column, dropna=False)
        for segment_value, group in grouped:
            if group.empty:
                continue

            rows.append(
                {
                    "segment_column": column,
                    "segment_value": segment_value,
                    "n": int(len(group)),
                    "MAE": float(group["error_absoluto"].mean()),
                    "RMSE": float(np.sqrt((group["error"] ** 2).mean())),
                    "mean_error": float(group["error"].mean()),
                    "median_error": float(group["error"].median()),
                    "median_absolute_error": float(group["error_absoluto"].median()),
                    "max_error": float(group["error_absoluto"].max()),
                    "overestimation_rate": float(group["es_sobreestimacion"].mean() * 100),
                    "underestimation_rate": float(group["es_subestimacion"].mean() * 100),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "segment_column",
                "segment_value",
                "n",
                "MAE",
                "RMSE",
                "mean_error",
                "median_error",
                "median_absolute_error",
                "max_error",
                "overestimation_rate",
                "underestimation_rate",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=["MAE", "RMSE"],
        ascending=[False, False],
    ).reset_index(drop=True)


def identify_problematic_segments(
    segment_metrics_df: pd.DataFrame,
    min_n: int = 5,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Identifica los segmentos con mayor MAE.
    """
    if segment_metrics_df.empty:
        return segment_metrics_df.copy()

    filtered = segment_metrics_df.copy()
    if "n" in filtered.columns:
        filtered = filtered[filtered["n"] >= min_n]

    return filtered.sort_values(by="MAE", ascending=False).head(top_n).reset_index(drop=True)


def save_segment_error_metrics(
    segment_metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Guarda metricas por segmento.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    segment_metrics_df.to_csv(path, index=False)


def plot_segment_errors(
    segment_metrics_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """
    Grafica los segmentos con mayor MAE.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if segment_metrics_df.empty:
        return

    plot_df = identify_problematic_segments(segment_metrics_df, min_n=5, top_n=top_n).copy()
    if plot_df.empty:
        plot_df = segment_metrics_df.head(top_n).copy()

    plot_df["segment_label"] = (
        plot_df["segment_column"].astype(str)
        + "="
        + plot_df["segment_value"].astype(str)
    )
    plot_df = plot_df.sort_values(by="MAE", ascending=True)

    plt.figure(figsize=(11, 8))
    plt.barh(plot_df["segment_label"], plot_df["MAE"], color="#B33A3A")
    plt.xlabel("MAE por segmento")
    plt.ylabel("Segmento")
    plt.title("Segmentos con mayor error absoluto medio")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------------------------------------------------------------------------
# Backward-compatible wrappers
# ---------------------------------------------------------------------------
def error_by_segment(
    error_df: pd.DataFrame,
    segment_columns: list[str],
) -> pd.DataFrame:
    """Backward-compatible segment report with legacy column names."""
    segment_metrics = calculate_segment_error_metrics(error_df, segment_columns)
    if segment_metrics.empty:
        return segment_metrics

    return segment_metrics.rename(
        columns={
            "MAE": "mean_absolute_error",
            "max_error": "max_absolute_error",
        }
    )[
        [
            "segment_column",
            "segment_value",
            "n",
            "mean_absolute_error",
            "median_absolute_error",
            "max_absolute_error",
        ]
    ]


def save_error_by_segment(
    segment_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Backward-compatible saver."""
    save_segment_error_metrics(segment_df, output_path)


def plot_top_error_segments(
    segment_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """Backward-compatible plotter."""
    if "mean_absolute_error" in segment_df.columns and "MAE" not in segment_df.columns:
        adapted = segment_df.rename(columns={"mean_absolute_error": "MAE"}).copy()
    else:
        adapted = segment_df.copy()
    plot_segment_errors(adapted, output_path, top_n=top_n)
