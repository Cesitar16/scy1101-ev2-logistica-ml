# Experimento enfocado en errores dificiles - Cesar

## Objetivo

Reducir el margen de error del modelo de regresion en los casos donde presenta peor rendimiento.

## Baseline

Modelo actual equilibrado:
- GradientBoostingRegressor
- MAE: 11.4711
- RMSE: 14.5709
- R2: 0.0307
- +/-10 min: 52.50%
- +/-15 min: 70.42%

## Estrategia

1. Analizar errores por segmento.
2. Identificar casos problematicos.
3. Crear variables especificas para esos casos.
4. Probar modelos robustos.
5. Probar correccion de residuos.
6. Comparar metricas globales y por segmento.
7. Decidir si conviene reemplazar el modelo actual.

## Segmentos analizados

- tipo_vehiculo
- trafico_nivel
- clima
- id_bodega
- hora punta
- distancia larga
- carga pesada
- muchas paradas

## Modelos probados

- GradientBoostingRegressor (baseline-like)
- GradientBoostingRegressor con loss="huber"
- ExtraTreesRegressor
- RandomForestRegressor
- HistGradientBoostingRegressor
- GradientBoostingRegressor con learning_rate menor
- Modelo con correccion de residuos

## Resultados

Mejor modelo nuevo del experimento:
- Modelo: gradient_boosting_baseline_like
- MAE: 11.473306
- RMSE: 14.518579
- R2: 0.037673
- +/-5 min: 26.67%
- +/-10 min: 51.25%
- +/-15 min: 70.00%

Comparacion con baseline actual:
- Delta MAE: 0.002206
- Delta RMSE: -0.052321
- Delta R2: 0.006973
- Delta +/-10 min: -1.25

Segmentos objetivo mejorados (MAE): 0/8

## Conclusion

Decision recomendada del experimento: **mantener modelo actual**.

El experimento logro mejoras parciales (especialmente en RMSE y R2), pero no cumplio de forma suficientemente equilibrada el criterio para reemplazar el baseline actual, principalmente por caida en acierto +/-10 minutos y +/-15 minutos.
