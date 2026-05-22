# Entrenamiento del modelo de regresion - Cesar

## Objetivo

Entrenar modelos base de regresion supervisada para predecir el tiempo de entrega.

## Rama

cesar/feature/regression-training

## Rama base

cesar/feature/regression-preprocessing

## Variable objetivo

target_tiempo_entrega

## Dataset utilizado

data/processed/cesar_logistica_clean.csv

## Modelos entrenados

1. LinearRegression
2. DecisionTreeRegressor
3. RandomForestRegressor
4. GradientBoostingRegressor

## Flujo implementado

1. Carga del dataset limpio.
2. Separacion de X e y.
3. Division train/test con test_size=0.2 y random_state=42.
4. Construccion de preprocesador.
5. Creacion de pipelines.
6. Entrenamiento con .fit().
7. Generacion de predicciones con .predict().
8. Guardado de predicciones base.

## Archivo de predicciones generado

results/reports/cesar_regression_baseline_predictions.csv

## Decisiones tecnicas

Se utilizo Pipeline para unir el preprocesamiento y el modelo. Esto evita errores comunes y asegura que las transformaciones se apliquen de forma consistente.

Se uso random_state=42 para asegurar reproducibilidad.

Se entrenaron varios modelos porque la evaluacion exige comparar distintas alternativas de modelado supervisado.

## Alcance de esta rama

Esta rama solo entrena modelos base.

No incluye:
- Evaluacion formal con MAE, RMSE y R².
- Comparacion final de desempeno.
- Optimizacion con GridSearchCV o RandomizedSearchCV.
- Guardado del modelo final.

## Proxima rama

cesar/feature/regression-evaluation