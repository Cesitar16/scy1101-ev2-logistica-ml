# Analisis de mejora del modelo de regresion - Cesar

## Objetivo

Intentar mejorar la precision del modelo de regresion para predecir el tiempo de entrega, especialmente el acierto dentro de +/-10 minutos.

## Resultados actuales

- MAE: 11.59
- RMSE: 14.58
- R2: 0.0290
- Acierto +/-5 min: 28.33%
- Acierto +/-10 min: 49.58%
- Acierto +/-15 min: 70.00%

## Mejoras implementadas

1. Ingenieria de caracteristicas.
2. Nuevos modelos de regresion.
3. Metricas operacionales por tolerancia.
4. Optimizacion orientada a +/-10 minutos.
5. Importancia de variables.

## Variables nuevas

- distancia_por_parada
- costo_por_km
- consumo_por_km
- hora_punta
- muchas_paradas
- carga_pesada
- trafico_clima

## Modelos adicionales probados

- ExtraTreesRegressor
- HistGradientBoostingRegressor
- KNeighborsRegressor

## Metricas usadas

- MAE
- RMSE
- R2
- Acierto +/-5 minutos
- Acierto +/-10 minutos
- Acierto +/-15 minutos
- Acierto +/-20 minutos

## Criterio principal

El criterio principal de mejora fue el acierto dentro de +/-10 minutos, porque representa mejor la utilidad operacional del modelo.

## Resultado

- Mejoro.
- Mejor modelo nuevo: random_forest_regressor (seleccionado sobre su version optimizada por criterio operacional).
- Metricas nuevas:
  - MAE: 11.5062
  - RMSE: 14.7391
  - R2: 0.0082
- Acierto nuevo:
  - +/-5 min: 29.17%
  - +/-10 min: 55.00%
  - +/-15 min: 70.42%
  - +/-20 min: 84.58%
- Comparacion contra modelo actual:
  - MAE: mejora (11.59 -> 11.51)
  - RMSE: empeora (14.58 -> 14.74)
  - R2: empeora (0.0290 -> 0.0082)
  - +/-10 min: mejora clara (49.58% -> 55.00%)

## Variables mas importantes (Permutation Importance)

1. peso_carga_kg
2. costo_envio
3. distancia_km
4. consumo_por_km
5. distancia_por_parada

## Conclusion

La mejora propuesta cumplio el objetivo operacional principal: aumentar el acierto dentro de +/-10 minutos. Aunque RMSE y R2 no mejoraron respecto al modelo actual, el criterio de negocio priorizado (acierto +/-10) si mejoro de forma relevante. Si el foco del curso/equipo es operacion real de promesa de entrega, este modelo mejorado es una alternativa valida.
