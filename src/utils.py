"""Utilidades generales para persistencia y soporte del proyecto."""

from __future__ import annotations


def save_model(model, path):
    """Guarda un modelo entrenado en disco."""
    # TODO: Implementar serialización del modelo (joblib/pickle).
    raise NotImplementedError("Pendiente de implementación")


def load_model(path):
    """Carga un modelo serializado desde disco."""
    # TODO: Implementar carga del modelo serializado.
    raise NotImplementedError("Pendiente de implementación")


def ensure_directory(path):
    """Asegura que un directorio exista antes de usarlo."""
    # TODO: Implementar validación/creación de directorios.
    raise NotImplementedError("Pendiente de implementación")
