"""Target transformation experiments for regression models."""

from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor


def wrap_model_with_log_target(model):
    """Envuelve un modelo con transformacion log1p del target."""
    return TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def build_log_target_models(models: dict) -> dict:
    """Crea versiones con target logaritmico para cada modelo."""
    return {
        f"{name}_log_target": wrap_model_with_log_target(model)
        for name, model in models.items()
    }
