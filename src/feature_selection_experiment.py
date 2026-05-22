"""Feature selection helpers based on feature importance ranking."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def select_top_features(
    importance_df: pd.DataFrame,
    top_n: int,
) -> list[str]:
    """Selecciona las top N variables segun importance_mean."""
    return (
        importance_df.sort_values(by="importance_mean", ascending=False)
        .head(top_n)["feature"]
        .tolist()
    )


def save_feature_selection_results(
    results_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)
