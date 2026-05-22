"""Helpers to compare segment-level errors between baseline and new models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_TARGET_SEGMENTS: list[tuple[str, object]] = [
    ("tipo_vehiculo", "camion"),
    ("trafico_nivel", "alto"),
    ("trafico_nivel", "bajo"),
    ("id_bodega", 1),
    ("id_bodega", 4),
    ("distancia_larga", 1),
    ("carga_pesada", 1),
    ("muchas_paradas", 1),
]


def _get_segment_row(
    segment_metrics_df: pd.DataFrame,
    segment_column: str,
    segment_value: object,
) -> pd.DataFrame:
    if segment_metrics_df.empty:
        return pd.DataFrame()

    candidates = segment_metrics_df[
        (segment_metrics_df["segment_column"].astype(str) == str(segment_column))
        & (segment_metrics_df["segment_value"].astype(str) == str(segment_value))
    ]
    return candidates


def build_segment_comparison(
    baseline_segment_metrics_df: pd.DataFrame,
    new_segment_metrics_df: pd.DataFrame,
    target_segments: list[tuple[str, object]] | None = None,
) -> pd.DataFrame:
    """Build segment-level MAE comparison baseline vs new model."""
    target_segments = target_segments or DEFAULT_TARGET_SEGMENTS

    rows: list[dict[str, object]] = []
    for segment_column, segment_value in target_segments:
        base_row = _get_segment_row(
            baseline_segment_metrics_df,
            segment_column,
            segment_value,
        )
        new_row = _get_segment_row(
            new_segment_metrics_df,
            segment_column,
            segment_value,
        )

        base_mae = float(base_row.iloc[0]["MAE"]) if not base_row.empty else float("nan")
        new_mae = float(new_row.iloc[0]["MAE"]) if not new_row.empty else float("nan")

        n_value = None
        if not new_row.empty:
            n_value = int(new_row.iloc[0]["n"])
        elif not base_row.empty:
            n_value = int(base_row.iloc[0]["n"])

        delta_mae = new_mae - base_mae
        improved = bool(pd.notna(delta_mae) and delta_mae < 0)

        rows.append(
            {
                "segment_column": segment_column,
                "segment_value": segment_value,
                "n": n_value,
                "baseline_MAE": base_mae,
                "new_MAE": new_mae,
                "delta_MAE": delta_mae,
                "improved": improved,
            }
        )

    return pd.DataFrame(rows)


def save_segment_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save segment comparison DataFrame to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(path, index=False)
