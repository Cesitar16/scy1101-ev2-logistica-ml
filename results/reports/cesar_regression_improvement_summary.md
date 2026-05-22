# Resumen de mejora del modelo de regresion - Cesar

## Modelo base seleccionado para experimento

random_forest_regressor

## Modelo optimizado por tolerancia +/-10

random_forest_regressor_tolerance10_optimized

## Modelo final recomendado

random_forest_regressor (base)

## Mejores hiperparametros

{'model__max_depth': 8, 'model__min_samples_leaf': 1, 'model__min_samples_split': 2, 'model__n_estimators': 50}

## Mejor score CV (tolerance +/-10)

0.490625

## Metricas del modelo final recomendado

- MAE: 11.506151
- RMSE: 14.739137
- R2: 0.008213

## Acierto operacional del modelo final recomendado

- +/-5 min: 29.17%
- +/-10 min: 55.00%
- +/-15 min: 70.42%
- +/-20 min: 83.33%

## Comparacion current vs improved

            version       MAE      RMSE       R2  acc_5_min  acc_10_min  acc_15_min
  current_optimized 11.590000 14.580000 0.029000  28.330000       49.58   70.000000
improved_experiment 11.506151 14.739137 0.008213  29.166667       55.00   70.416667

## Conclusion

El modelo mejoro operacionalmente porque aumento el acierto dentro de +/-10 minutos.
