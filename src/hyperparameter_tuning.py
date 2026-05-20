"""Funciones de optimización de hiperparámetros para modelos de Machine Learning."""

from __future__ import annotations


def grid_search_model(model, param_grid, X_train, y_train, cv=5):
    """Ejecuta GridSearchCV sobre un modelo y devuelve el mejor estimador."""
    # TODO: Implementar búsqueda exhaustiva de hiperparámetros.
    raise NotImplementedError("Pendiente de implementación")


def randomized_search_model(model, param_distributions, X_train, y_train, cv=5, n_iter=20):
    """Ejecuta RandomizedSearchCV sobre un modelo y devuelve el mejor estimador."""
    # TODO: Implementar búsqueda aleatoria de hiperparámetros.
    raise NotImplementedError("Pendiente de implementación")
