"""Tests minimos para guardado y carga de modelos de clasificacion."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.classification_training import cargar_modelo, guardar_modelo


class TestGuardarCargarModelo(unittest.TestCase):
    """Valida persistencia basica de modelos con joblib."""

    def test_guardar_y_cargar_modelo(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array(["a", "a", "b", "b"])

        modelo = Pipeline(
            steps=[
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        )
        modelo.fit(X, y)

        with TemporaryDirectory() as tmp_dir:
            model_path = guardar_modelo(
                modelo=modelo,
                model_name="dummy_test",
                output_dir=tmp_dir,
            )

            self.assertTrue(Path(model_path).exists())

            loaded_model = cargar_modelo(model_path)
            predictions = loaded_model.predict(X[:2])
            self.assertEqual(len(predictions), 2)


if __name__ == "__main__":
    unittest.main()
