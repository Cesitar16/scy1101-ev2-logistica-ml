# Evaluacion del modelo de regresion - Cesar

## Objetivo

Evaluar y comparar los modelos base de regresion entrenados para predecir el tiempo de entrega.

## Rama

cesar/feature/regression-evaluation

## Rama base

cesar/feature/regression-training

## Variable objetivo

target_tiempo_entrega

## Dataset utilizado

data/processed/cesar_logistica_clean.csv

## Modelos evaluados

1. LinearRegression
2. DecisionTreeRegressor
3. RandomForestRegressor
4. GradientBoostingRegressor

## Metricas utilizadas

### MAE

Error absoluto promedio. Permite interpretar cuantos minutos, en promedio, se equivoca el modelo.

### RMSE

Raiz del error cuadratico medio. Penaliza mas los errores grandes, por lo que es util para detectar modelos que fallan mucho en algunos casos.

### R2

Indica la proporcion de variabilidad del tiempo de entrega que el modelo logra explicar.

## Archivos generados

Metricas:
results/metrics/cesar_regression_metrics.csv

Resumen del mejor modelo:
results/metrics/cesar_regression_best_model_summary.csv

Graficos:
results/plots/cesar_regression_rmse_comparison.png
results/plots/cesar_regression_mae_comparison.png
results/plots/cesar_real_vs_predicho_best_model.png
results/plots/cesar_residuals_best_model.png

## Criterio de seleccion del mejor modelo

El criterio principal fue menor RMSE, porque penaliza mas los errores grandes. Como criterio secundario se reviso MAE y luego R2.

## Alcance de esta rama

Esta rama evalua modelos base, pero no optimiza hiperparametros.

No incluye:
- GridSearchCV.
- RandomizedSearchCV.
- Guardado del modelo final.
- Seleccion definitiva despues de optimizacion.

## Proxima rama

cesar/feature/regression-tuning