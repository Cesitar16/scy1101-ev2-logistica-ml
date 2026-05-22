# Resumen experimento enfocado en errores dificiles - Cesar

## Baseline actual

- Modelo: GradientBoostingRegressor
- MAE: 11.4711
- RMSE: 14.5709
- R2: 0.0307
- Acierto +/-10 min: 52.5%
- Acierto +/-15 min: 70.42%

## Mejor modelo nuevo

- Modelo: gradient_boosting_baseline_like
- MAE: 11.473306
- RMSE: 14.518579
- R2: 0.037673
- Acierto +/-10 min: 51.25%
- Acierto +/-15 min: 70.00%
- P90 error absoluto: 24.1031

## Modelo usado para correccion de residuos

- Base para residuos: gradient_boosting_baseline_like
- Variante residual: gradient_boosting_baseline_like_residual_correction

## Decision recomendada

mantener modelo actual

## Segmentos comparados

 segment_column segment_value  n  baseline_MAE   new_MAE  delta_MAE  improved
  tipo_vehiculo        camion 82     12.338845 12.338845        0.0     False
  trafico_nivel          alto 58     11.555559 11.555559        0.0     False
  trafico_nivel          bajo 56     11.925398 11.925398        0.0     False
      id_bodega             1 60     11.733073 11.733073        0.0     False
      id_bodega             4 74     12.143550 12.143550        0.0     False
distancia_larga             1 60     12.806547 12.806547        0.0     False
   carga_pesada             1 60     12.024281 12.024281        0.0     False
 muchas_paradas             1 66     11.437159 11.437159        0.0     False
