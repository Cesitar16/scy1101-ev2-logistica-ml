"""Model interpretability helpers for regression workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance


def calculate_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str = "neg_root_mean_squared_error",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcula importancia de variables usando Permutation Importance.

    Esta tecnica sirve incluso cuando el modelo esta dentro de un Pipeline.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    return importance_df


def save_feature_importance(
    importance_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Guarda importancia de variables."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(path, index=False)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """Grafica las variables mas importantes."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    top_df = importance_df.head(top_n).sort_values(
        by="importance_mean",
        ascending=True,
    )

    plt.figure(figsize=(10, 7))
    plt.barh(top_df["feature"], top_df["importance_mean"], color="#2F4B7C")
    plt.xlabel("Importancia promedio")
    plt.ylabel("Variable")
    plt.title("Importancia de variables - Permutation Importance")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
