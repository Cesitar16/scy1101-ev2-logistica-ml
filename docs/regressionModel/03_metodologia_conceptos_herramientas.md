# Metodologia, Conceptos y Herramientas Aplicadas

## Stack y herramientas

- **Python**, **Pandas**, **NumPy**
- **Scikit-learn**:
  - `Pipeline`
  - `ColumnTransformer`
  - `SimpleImputer`
  - `OneHotEncoder`
  - `StandardScaler`
  - Modelos de regresion lineales y basados en arboles/boosting
  - `GridSearchCV` / `RandomizedSearchCV` en etapas de tuning
  - `permutation_importance` para interpretabilidad
- **Matplotlib** para visualizacion
- **Jupyter Notebook** para narrativa reproducible

## Conceptos aplicados

1. **Regresion supervisada** sobre `target_tiempo_entrega`.
2. **Separacion `X` e `y`** y particion `train/test` con semilla fija.
3. **Preprocesamiento robusto**:
   - limpieza de texto categorico,
   - conversion numerica segura,
   - imputacion,
   - escalado,
   - codificacion one-hot.
4. **Ingenieria de caracteristicas**:
   - ratios logisticos (`distancia_por_parada`, `paradas_por_km`, etc.),
   - interacciones de contexto (`trafico_clima`, `trafico_vehiculo`, etc.),
   - variables de segmento dificiles (`bodega_problematico`, `camion_trafico_alto`, etc.).
5. **Metricas tecnicas**: `MAE`, `RMSE`, `R2`.
6. **Metricas operacionales**: acierto por tolerancia (`+/-5`, `+/-10`, `+/-15`, `+/-20` min).
7. **Analisis de error segmentado** por tipo de vehiculo, trafico, bodega y segmentos derivados.
8. **Interpretabilidad** con importancia por permutacion.

## Que se intento especificamente por modelo

- **LinearRegression**: baseline simple e interpretable para tener piso de comparacion.
- **DecisionTreeRegressor**: capturar no linealidad con modelo interpretable en arbol.
- **RandomForestRegressor**: reducir varianza mediante ensamble bagging.
- **GradientBoostingRegressor**: modelar relaciones no lineales con boosting secuencial.
- **ExtraTreesRegressor**: mayor aleatoriedad para robustez/varianza.
- **HistGradientBoostingRegressor**: boosting eficiente para datos tabulares.
- **KNN Regressor** (etapas de precision): aproximacion local por vecinos.
- **GradientBoosting Huber / low learning-rate**: robustecer frente a errores grandes/outliers.
- **Residual correction**: segundo modelo para aprender residuo del modelo base.

## Que se obtuvo (interpretacion general)

- El problema muestra **senal moderada-baja**: `R2` global sigue bajo (~0.03 en los mejores escenarios equilibrados).
- Hubo trade-off claro entre:
  - mejorar precision tecnica (`MAE/RMSE/R2`) y
  - mantener acierto operacional en ventanas de minutos.
- Las variables mas influyentes fueron consistentemente de carga/distancia/paradas:

Top importancia (experimento de precision):

| feature | importance_mean | importance_std |
| --- | --- | --- |
| distancia_por_parada | 0.315681 | 0.093046 |
| paradas_por_km | 0.281448 | 0.142761 |
| carga_x_distancia | 0.221657 | 0.067440 |
| peso_por_km | 0.196546 | 0.085844 |
| peso_carga_kg | 0.195166 | 0.058053 |
| distancia_km | 0.194879 | 0.065823 |
| costo_por_kg | 0.131428 | 0.041373 |
| consumo_por_km | 0.113392 | 0.111883 |
| distancia_total_ajustada | 0.082396 | 0.059557 |
| consumo_por_kg | 0.071355 | 0.058411 |

Top importancia (errores dificiles):

| feature | importance_mean | importance_std |
| --- | --- | --- |
| distancia_por_parada | 0.327172 | 0.067225 |
| peso_carga_kg | 0.269562 | 0.042111 |
| carga_x_distancia | 0.195054 | 0.065837 |
| distancia_km | 0.177402 | 0.090668 |
| costo_por_kg | 0.145177 | 0.041322 |
| consumo_por_km | 0.144412 | 0.112486 |
| paradas_por_km | 0.120153 | 0.100715 |
| peso_por_km | 0.112611 | 0.076419 |
| experiencia_chofer_anios | 0.072260 | 0.048730 |
| trafico_vehiculo | 0.054745 | 0.044263 |
