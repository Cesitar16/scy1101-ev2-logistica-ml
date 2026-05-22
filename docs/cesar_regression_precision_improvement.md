# Experimento de mejora de estimacion exacta - Cesar

## Objetivo

Mejorar la estimacion exacta del tiempo de entrega, priorizando MAE, RMSE y R2.

## Diferencia con el experimento operacional

El experimento operacional priorizaba el acierto dentro de +/-10 minutos.
Este experimento prioriza la precision tecnica de regresion:
- menor MAE
- menor RMSE
- mayor R2

## Mejoras aplicadas

1. Nuevas variables de ingenieria de caracteristicas.
2. Variables historicas sin fuga de datos (fit en entrenamiento y transform en prueba).
3. Validacion cruzada.
4. Modelos adicionales.
5. Transformacion logaritmica del target (versiones log-target de pipelines).
6. Analisis de errores por segmento.
7. Importancia de variables por permutation importance.
8. Seleccion de variables (all, top 5, top 8, top 10).

## Variables nuevas de precision

- hora_punta_manana
- hora_punta_tarde
- hora_nocturna
- distancia_total_ajustada
- paradas_por_km
- peso_por_km
- carga_x_distancia
- chofer_experto
- costo_por_kg
- consumo_por_kg
- trafico_vehiculo
- clima_vehiculo
- hist_mean_target_by_id_bodega
- hist_mean_target_by_tipo_vehiculo
- hist_mean_target_by_trafico_nivel

## Modelos probados

- RandomForestRegressor
- ExtraTreesRegressor
- GradientBoostingRegressor
- HistGradientBoostingRegressor
- KNeighborsRegressor
- Variantes log-target de los modelos anteriores
- Version ajustada del mejor modelo base por RMSE

## Resultados comparados

### Modelo tecnico anterior

- MAE: 11.59
- RMSE: 14.58
- R2: 0.0290
- Acierto +/-5 min: 28.33%
- Acierto +/-10 min: 49.58%
- Acierto +/-15 min: 70.00%

### Modelo operacional anterior

- MAE: 11.5062
- RMSE: 14.7391
- R2: 0.0082
- Acierto +/-5 min: 29.17%
- Acierto +/-10 min: 55.00%
- Acierto +/-15 min: 70.42%

### Nuevo modelo de precision (mejor del experimento)

- Modelo: gradient_boosting_regressor
- MAE: 11.4711
- RMSE: 14.5709
- R2: 0.0307
- Acierto +/-5 min: 29.17%
- Acierto +/-10 min: 52.50%
- Acierto +/-15 min: 70.42%

## Conclusiones

- Mejora tecnica: SI.
- Mejora operacional principal (+/-10 min) frente al modelo operacional anterior: NO.
- Frente al modelo tecnico anterior, mejora en MAE, RMSE y R2.
- Frente al modelo operacional anterior, mejora tecnica pero cae el acierto +/-10 min.

## Recomendacion

Si el objetivo principal del curso o negocio es precision exacta del tiempo, este nuevo modelo de precision es mejor candidato tecnico.
Si el objetivo principal es acierto operacional estricto en +/-10 minutos, conviene mantener el modelo operacional anterior.
