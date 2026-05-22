# Resumen experimento de precision - Cesar

## Modelo final seleccionado

gradient_boosting_regressor

## Metodo de tuning aplicado

grid_search_rmse

## Mejores hiperparametros

{'model__learning_rate': 0.01, 'model__max_depth': 2, 'model__min_samples_split': 2, 'model__n_estimators': 50}

## Metricas del modelo final

- MAE: 11.471060
- RMSE: 14.570939
- R2: 0.030720
- Acierto +/-5 min: 29.17%
- Acierto +/-10 min: 52.50%
- Acierto +/-15 min: 70.42%

## Comparacion contra modelos previos

                      version      MAE      RMSE      R2  acc_5_min  acc_10_min  acc_15_min                                                                                    decision_notes
  current_technical_optimized 11.59000 14.580000 0.02900  28.330000       49.58   70.000000                                                                                          baseline
previous_operational_improved 11.50620 14.739100 0.00820  29.170000       55.00   70.420000                                                                                          baseline
         precision_experiment 11.47106 14.570939 0.03072  29.166667       52.50   70.416667 MAE mejor que tecnico | RMSE mejor que tecnico | R2 mejor que tecnico | pierde acc_10 operacional

## Conclusion

Mejoro la estimacion exacta del tiempo de entrega.

## Warnings del experimento

- Sin warnings
