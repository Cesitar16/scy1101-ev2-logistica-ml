# Mejora de modelo de regresion - Cesar

## Mejor modelo base

gradient_boosting_regressor

## Modelo recomendado final

gradient_boosting_regressor (baseline)

## Metodo de tuning

randomized

## Mejores hiperparametros

{'model__subsample': 0.8, 'model__n_estimators': 100, 'model__min_samples_split': 2, 'model__max_depth': 5, 'model__learning_rate': 0.01}

## Metricas modelo recomendado

- MAE: 11.591409
- RMSE: 14.583611
- R2: 0.029033

## Acierto operacional por tolerancia

- +/-5 min: 28.33% (68/240)
- +/-10 min: 49.58% (119/240)
- +/-15 min: 70.00% (168/240)