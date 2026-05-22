# RegressionModel Docs (Cesar)

Esta carpeta resume toda la linea de regresion supervisada para `target_tiempo_entrega`:

- [01_justificacion_modelo_recomendado.md](./01_justificacion_modelo_recomendado.md)
- [02_comparacion_modelos_y_resultados.md](./02_comparacion_modelos_y_resultados.md)
- [03_metodologia_conceptos_herramientas.md](./03_metodologia_conceptos_herramientas.md)
- [04_decisiones_hallazgos_fallas.md](./04_decisiones_hallazgos_fallas.md)

## Resumen ejecutivo

- **Mejor ranking estricto (ultimo experimento local):** `gradient_boosting_baseline_like`.
- **Modelo recomendado para mantener como equilibrio tecnico-operacional:** `GradientBoostingRegressor` (baseline equilibrado actual).
- **Motivo:** el experimento de errores dificiles mejoro ligeramente `RMSE` y `R2`, pero empeoro `MAE` y bajo `+/-10` y `+/-15` minutos.
