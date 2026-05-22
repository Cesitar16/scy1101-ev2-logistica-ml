"""Cross-validation helpers for regression model stability evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate


def cross_validate_regression_models(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> pd.DataFrame:
    """Evalua modelos con validacion cruzada usando MAE, RMSE y R2."""
    rows = []

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    for model_name, model in models.items():
        try:
            scores = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,
            )

            rows.append(
                {
                    "model": model_name,
                    "cv_mae_mean": -scores["test_mae"].mean(),
                    "cv_mae_std": scores["test_mae"].std(),
                    "cv_rmse_mean": -scores["test_rmse"].mean(),
                    "cv_rmse_std": scores["test_rmse"].std(),
                    "cv_r2_mean": scores["test_r2"].mean(),
                    "cv_r2_std": scores["test_r2"].std(),
                    "train_mae_mean": -scores["train_mae"].mean(),
                    "train_rmse_mean": -scores["train_rmse"].mean(),
                    "train_r2_mean": scores["train_r2"].mean(),
                    "cv_error": "",
                }
            )
        except Exception as exc:  # pragma: no cover
            rows.append(
                {
                    "model": model_name,
                    "cv_mae_mean": np.nan,
                    "cv_mae_std": np.nan,
                    "cv_rmse_mean": np.nan,
                    "cv_rmse_std": np.nan,
                    "cv_r2_mean": np.nan,
                    "cv_r2_std": np.nan,
                    "train_mae_mean": np.nan,
                    "train_rmse_mean": np.nan,
                    "train_r2_mean": np.nan,
                    "cv_error": str(exc),
                }
            )

    results_df = pd.DataFrame(rows)
    if "cv_rmse_mean" in results_df.columns:
        results_df = results_df.sort_values(
            by="cv_rmse_mean",
            ascending=True,
            na_position="last",
        )
    return results_df.reset_index(drop=True)


def save_cross_validation_results(
    cv_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(path, index=False)
