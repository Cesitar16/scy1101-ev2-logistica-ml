# Optimizacion del modelo de regresion - Cesar

## Objetivo

Optimizar hiperparametros del mejor modelo base de regresion para mejorar la prediccion del tiempo de entrega.

## Rama

cesar/feature/regression-tuning

## Rama base

cesar/feature/regression-evaluation

## Variable objetivo

target_tiempo_entrega

## Metodo utilizado

GridSearchCV

## Metrica de optimizacion

neg_root_mean_squared_error

Se utiliza esta metrica porque Scikit-learn maximiza el score, por eso usa el valor negativo del RMSE. En la interpretacion final se considera el RMSE positivo como medida de error.

## Flujo implementado

1. Cargar dataset limpio.
2. Entrenar modelos base.
3. Evaluar modelos base.
4. Seleccionar mejor modelo segun RMSE.
5. Definir grilla de hiperparametros.
6. Ejecutar GridSearchCV.
7. Obtener mejores parametros.
8. Evaluar modelo optimizado.
9. Comparar base vs optimizado.
10. Guardar resultados y graficos.

## Archivos generados

Resultados de busqueda:
results/metrics/cesar_regression_tuning_results.csv

Metricas optimizadas:
results/metrics/cesar_regression_optimized_metrics.csv

Comparacion base vs optimizado:
results/metrics/cesar_regression_base_vs_optimized.csv

Graficos:
results/plots/cesar_base_vs_optimized_rmse.png
results/plots/cesar_base_vs_optimized_mae.png
results/plots/cesar_optimized_real_vs_predicho.png
results/plots/cesar_optimized_residuals.png

## Alcance de esta rama

Esta rama optimiza el mejor modelo base, pero todavia no guarda el modelo final serializado.

No incluye:
- Guardado .joblib del modelo final.
- Merge a main.
- Cierre final del notebook.

## Proxima rama

cesar/feature/regression-finalization