"""Tests minimos para creacion de target de clasificacion."""

from __future__ import annotations

import unittest

import pandas as pd

from src.classification_utils import (
    aplicar_umbrales_categoria_entrega,
    calcular_umbrales_target_train,
    crear_target_clasificacion,
    preparar_datos_clasificacion_sin_leakage,
)


class TestCrearTargetClasificacion(unittest.TestCase):
    """Valida comportamiento basico de `crear_target_clasificacion`."""

    def test_crea_columna_objetivo_sin_nulos(self) -> None:
        df = pd.DataFrame(
            {
                "distancia_km": [2, 4, 8, 12, 18, 25],
                "target_tiempo_entrega": [20, 28, 33, 41, 55, 70],
            }
        )

        result = crear_target_clasificacion(df)

        self.assertIn("categoria_entrega", result.columns)
        self.assertFalse(result["categoria_entrega"].isna().any())

        clases_validas = {"rapida", "normal", "tardia"}
        clases_obtenidas = set(result["categoria_entrega"].astype(str).unique().tolist())
        self.assertTrue(clases_obtenidas.issubset(clases_validas))
        self.assertGreaterEqual(len(clases_obtenidas), 2)

    def test_no_modifica_dataframe_original(self) -> None:
        df = pd.DataFrame(
            {
                "distancia_km": [3, 5, 7],
                "target_tiempo_entrega": [24, 40, 65],
            }
        )
        original_columns = df.columns.tolist()

        _ = crear_target_clasificacion(df)

        self.assertEqual(df.columns.tolist(), original_columns)
        self.assertNotIn("categoria_entrega", df.columns)

    def test_calcular_umbrales_target_train(self) -> None:
        serie_train = pd.Series([20, 28, 33, 41, 55, 70], name="target_tiempo_entrega")
        umbrales = calcular_umbrales_target_train(serie_train)

        self.assertIn("q33", umbrales)
        self.assertIn("q66", umbrales)
        self.assertLess(umbrales["q33"], umbrales["q66"])

    def test_aplicar_umbrales_categoria_entrega_sin_nulos(self) -> None:
        y_continuo = pd.Series([18, 30, 42, 60, None, 0], name="target_tiempo_entrega")
        umbrales = {"q33": 30.0, "q66": 45.0}
        y_cat = aplicar_umbrales_categoria_entrega(y_continuo=y_continuo, umbrales=umbrales)

        self.assertFalse(y_cat.isna().any())
        clases_validas = {"rapida", "normal", "tardia"}
        self.assertTrue(set(y_cat.astype(str).unique()).issubset(clases_validas))

    def test_preparar_datos_sin_leakage(self) -> None:
        df = pd.DataFrame(
            {
                "distancia_km": [2, 4, 8, 12, 18, 25, 30, 35, 9, 14],
                "trafico_nivel": [
                    "Bajo",
                    "Medio",
                    "Alto",
                    "Medio",
                    "Alto",
                    "Bajo",
                    "Medio",
                    "Alto",
                    "Bajo",
                    "Medio",
                ],
                "clima": [
                    "Soleado",
                    "Nublado",
                    "Lluvia",
                    "Soleado",
                    "Lluvia",
                    "Nublado",
                    "Soleado",
                    "Lluvia",
                    "Nublado",
                    "Soleado",
                ],
                "target_tiempo_entrega": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
            }
        )

        X_train, X_test, y_train, y_test, preprocessor, feature_info, umbrales = (
            preparar_datos_clasificacion_sin_leakage(
                df=df,
                target_time_col="target_tiempo_entrega",
                test_size=0.2,
                random_state=42,
            )
        )

        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertIn("q33", umbrales)
        self.assertIn("q66", umbrales)
        self.assertLess(umbrales["q33"], umbrales["q66"])
        self.assertNotIn("target_tiempo_entrega", X_train.columns)
        self.assertNotIn("categoria_entrega", X_train.columns)
        self.assertIn("numeric_columns", feature_info)
        self.assertIn("categorical_columns", feature_info)
        self.assertIsNotNone(preprocessor)


if __name__ == "__main__":
    unittest.main()
