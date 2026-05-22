# Comparacion de Modelos y Resultados

## Etapa 1: Modelos base iniciales (rama de evaluacion)

| model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| linear_regression | 11.625754 | 14.805800 | -0.000779 |
| random_forest_regressor | 11.632439 | 14.861226 | -0.008286 |
| gradient_boosting_regressor | 11.700831 | 14.945633 | -0.019772 |
| decision_tree_regressor | 12.552965 | 16.644479 | -0.264779 |

Lectura rapida:
- `linear_regression` quedo primero en esa etapa temprana.
- Todos los `R2` eran cercanos a 0 o negativos: senal de baja capacidad explicativa con ese set/base.

## Etapa 2: Optimizacion clasica

- Mejor base detectado: `linear_regression`
- Version optimizada: `linear_regression_optimized`
- Resultado: sin mejora practica (`MAE`, `RMSE`, `R2` iguales en esa etapa).

Tabla base vs optimizado:

| version | model | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| base | linear_regression | 11.625754191635036 | 14.805799765183702 | -0.0007785940055418 |
| optimized | linear_regression_optimized | 11.625754191635036 | 14.805799765183703 | -0.000778594005542 |

## Etapa 3: Experimento orientado a acierto operacional

Comparacion guardada:

| version | MAE | RMSE | R2 | acc_5_min | acc_10_min | acc_15_min |
| --- | --- | --- | --- | --- | --- | --- |
| current_optimized | 11.59 | 14.58 | 0.029 | 28.33 | 49.58 | 70.0 |
| improved_experiment | 11.506151016288245 | 14.739136956736507 | 0.0082130878972738 | 29.166666666666668 | 55.00000000000001 | 70.41666666666667 |

Interpretacion:
- Mejoro `+/-10` (49.58 -> 55.00) y `+/-15` (70.00 -> 70.42).
- Pero empeoro tecnica (`RMSE` subio y `R2` bajo).

## Etapa 4: Experimento de precision tecnica

Comparacion entre tecnico previo, operacional previo y precision:

| version | MAE | RMSE | R2 | acc_5_min | acc_10_min | acc_15_min | decision_notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| current_technical_optimized | 11.59 | 14.58 | 0.029 | 28.33 | 49.58 | 70.0 | baseline |
| previous_operational_improved | 11.5062 | 14.7391 | 0.0082 | 29.17 | 55.0 | 70.42 | baseline |
| precision_experiment | 11.47106043328628 | 14.570939193705147 | 0.0307197673970444 | 29.166666666666668 | 52.5 | 70.41666666666667 | MAE mejor que tecnico | RMSE mejor que tecnico | R2 mejor que tecnico | pierde acc_10 operacional |

Interpretacion:
- `precision_experiment` mejora tecnica frente al tecnico anterior (`MAE`, `RMSE`, `R2`).
- Pierde frente al operacional en `+/-10`.

## Etapa 5: Experimento enfocado en errores dificiles (local)

Ranking de modelos probados:

| model | MAE | RMSE | R2 | acc_10_min | acc_15_min | p90_absolute_error |
| --- | --- | --- | --- | --- | --- | --- |
| gradient_boosting_baseline_like | 11.473306 | 14.518579 | 0.037673 | 51.250000 | 70.000000 | 24.103126 |
| random_forest_error_focused | 11.475592 | 14.639054 | 0.021636 | 55.416667 | 69.166667 | 25.847492 |
| gradient_boosting_low_lr | 11.495554 | 14.616883 | 0.024598 | 53.333333 | 70.000000 | 24.693981 |
| gradient_boosting_huber | 11.585644 | 14.701069 | 0.013330 | 50.833333 | 69.166667 | 23.747248 |
| gradient_boosting_baseline_like_residual_correction | 11.665901 | 14.700982 | 0.013341 | 50.416667 | 70.000000 | 25.679992 |
| hist_gradient_boosting_error_focused | 11.720827 | 15.007297 | -0.028204 | 50.833333 | 69.166667 | 25.151367 |
| extra_trees_error_focused | 12.116310 | 15.444687 | -0.089011 | 54.166667 | 67.083333 | 26.458451 |

Comparacion final baseline vs mejor nuevo:

| version | model | MAE | RMSE | R2 | acc_5_min | acc_10_min | acc_15_min | delta_MAE | delta_RMSE | delta_R2 | delta_acc_10_min | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_balanced_model | GradientBoostingRegressor | 11.471100 | 14.570900 | 0.030700 | 29.170000 | 52.500000 | 70.420000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | baseline |
| best_error_focused_model | gradient_boosting_baseline_like | 11.473306 | 14.518579 | 0.037673 | 26.666667 | 51.250000 | 70.000000 | 0.002206 | -0.052321 | 0.006973 | -1.250000 | mantener modelo actual |

Conclusion de esta etapa:
- Mejor ranking estricto: `gradient_boosting_baseline_like`.
- Recomendacion operativa: **mantener modelo actual equilibrado**.
