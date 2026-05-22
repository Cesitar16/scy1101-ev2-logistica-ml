# Pull Request - Lucas - Clasificación supervisada

## 1. Título sugerido del PR
`feat(classification): add supervised delivery classification workflow`

## 2. Resumen
Esta branch consolida la parte de Lucas para clasificación supervisada en logística de última milla: creación de target multiclase, preprocesamiento con `ColumnTransformer`, entrenamiento de modelos base y optimizados, evaluación en test, persistencia del modelo final y documentación técnica asociada.

## 3. Cambios principales
- Flujo completo de clasificación supervisada en `notebooks/02_supervised_classification.ipynb`.
- Construcción de target sin leakage de distribución (umbrales `q33/q66` calculados solo en train).
- Entrenamiento de modelos base: `DecisionTree`, `RandomForest`, `KNN`, `LogisticRegression`.
- Optimización acotada con validación cruzada para `RandomForest` y `LogisticRegression`.
- Evaluación comparativa con métricas multiclase (`accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `f1_weighted`).
- Exportación de métricas y reportes de clasificación.
- Persistencia del pipeline final en `models/trained_models/classification_best_model.joblib`.
- Tests mínimos para target y guardado/carga de modelo.
- Documentación académica actualizada para presentación y trazabilidad.

## 4. Resultados principales
Fuente: `results/metrics/final_classification_comparison.csv`

| Modelo | Tipo | Accuracy | Precision Macro | Recall Macro | F1 Macro | F1 Weighted | Observación |
|---|---|---:|---:|---:|---:|---:|---|
| optimized_logistic_regression | Optimizado | 0.3750 | 0.3746 | 0.3761 | 0.3738 | 0.3758 | Mejor resultado observado en test |
| optimized_random_forest | Optimizado | 0.3667 | 0.3651 | 0.3689 | 0.3659 | 0.3661 | Segundo en test |
| logistic_regression | Base | 0.3583 | 0.3616 | 0.3635 | 0.3583 | 0.3580 | Mejor baseline |
| random_forest | Base | 0.3417 | 0.3414 | 0.3423 | 0.3400 | 0.3425 | Baseline intermedio |
| decision_tree | Base | 0.3292 | 0.3469 | 0.3150 | 0.3096 | 0.3194 | Modelo interpretable |
| knn | Base | 0.3208 | 0.3089 | 0.3096 | 0.3002 | 0.3094 | Baseline de distancia |

Distinción metodológica:
- Mejor resultado observado en test: `optimized_logistic_regression` (`f1_macro=0.3738`).
- Modelo seleccionado por CV: `random_forest` (`best_score CV=0.3642`, ver `optimization_result_random_forest_random.json`).
- Modelo final guardado: `classification_best_model.joblib` (pipeline `optimized_random_forest`).

## 5. Decisiones metodológicas
- El target categórico se crea después del split para evitar leakage de distribución del test.
- La selección de modelo final se realiza por validación cruzada (train) para no sobreajustar decisiones al test.
- El conjunto de test se usa solo como reporte final comparativo.
- `f1_macro` es la métrica principal por ser multiclase y sensible a desempeño por clase.

## 6. Validaciones ejecutadas
Comandos ejecutados:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_supervised_classification.ipynb
uv run python -m unittest discover -s tests -p "test_*.py"
```

Resultado:
- Notebook: ejecución completa (`exit code 0`).
- Tests: `Ran 6 tests ... OK`.

## 7. Archivos importantes
- Notebook principal: `notebooks/02_supervised_classification.ipynb`
- Utilidades y preparación: `src/classification_utils.py`
- Entrenamiento y persistencia: `src/classification_training.py`
- Evaluación: `src/classification_evaluation.py`
- Visualización: `src/classification_visualization.py`
- Optimización: `src/classification_tuning.py`
- Documentación técnica: `docs/clasificacion_supervisada_lucas.md`
- Métricas finales: `results/metrics/final_classification_comparison.csv`
- Modelo final: `models/trained_models/classification_best_model.joblib`
- Tests: `tests/test_crear_target_clasificacion.py`, `tests/test_guardar_cargar_modelo.py`

## 8. Consideraciones de datos
- El CSV crudo no se versiona (`data/raw/*.csv` en `.gitignore`).
- Para ejecutar el flujo, el archivo debe estar disponible en `data/raw/5_logistica_40.csv`.
- Esta política evita subir datasets pesados/sensibles al repositorio.

## 9. Limitaciones conocidas
- Rendimiento moderado en test (F1 macro ~0.30–0.37).
- Target derivado por terciles (no basado en SLA/umbral de negocio real).
- Señal predictiva limitada en dataset actual.
- El modelo es válido para análisis académico, no para producción sin iteraciones adicionales.

## 10. Checklist antes de merge
- [x] Notebook principal ejecuta completo.
- [x] Tests mínimos pasan.
- [x] Dataset esperado documentado (`data/raw/5_logistica_40.csv`).
- [x] Modelo final guardado.
- [x] Métricas finales exportadas.
- [x] CSV crudo no se sube al repo.
- [x] `.gitignore` revisado.
- [x] `docs-personal/changes/` revisado según política de evidencias personales.

## 11. Mensaje de commit sugerido
```text
feat(classification): finalize Lucas supervised delivery classifier

- add leakage-safe target creation using train-derived thresholds
- train and evaluate baseline and tuned classifiers with sklearn pipelines
- select final model using cross-validation and report test metrics
- persist final classification pipeline and export metrics
- update documentation, README, and tests
```
