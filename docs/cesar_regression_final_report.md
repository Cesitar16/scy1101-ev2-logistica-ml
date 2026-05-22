# Informe final de la linea de regresion - Cesar

## Objetivo

Desarrollar un modelo de regresion supervisada para predecir el tiempo de entrega en el contexto Logistica 4.0.

## Pregunta del modelo

Cuanto demorara una entrega?

## Variable objetivo

target_tiempo_entrega

## Variables de entrada

- distancia_km
- trafico_nivel
- clima
- tipo_vehiculo
- peso_carga_kg
- paradas_previas
- experiencia_chofer_anios
- hora_despacho
- costo_envio
- consumo_nafta
- id_bodega

## Flujo desarrollado

1. Setup inicial.
2. Preprocesamiento.
3. Entrenamiento de modelos base.
4. Evaluacion de modelos.
5. Optimizacion de hiperparametros.
6. Guardado del modelo final.

## Modelos entrenados

- LinearRegression
- DecisionTreeRegressor
- RandomForestRegressor
- GradientBoostingRegressor

## Metricas utilizadas

- MAE
- RMSE
- R2

## Modelo final

linear_regression_optimized

## Archivos finales generados

Modelo final:
models/trained_models/cesar_regression_model.joblib

Metricas finales:
results/metrics/cesar_regression_final_metrics.csv

Resumen final:
results/reports/cesar_regression_final_summary.md

Notebook principal:
notebooks/02_supervised_regression_cesar.ipynb

## Interpretacion general

El modelo final permite estimar el tiempo de entrega a partir de variables logisticas. Su utilidad esta en apoyar decisiones operativas, planificacion y promesas de entrega mas realistas.

## Limitaciones

- El modelo depende de la calidad de los datos.
- Puede no capturar eventos externos no registrados.
- La prediccion puede mejorar con mas datos historicos.
- Los resultados deben validarse con nuevos datos reales.

## Recomendaciones futuras

- Agregar nuevas variables externas.
- Validar el modelo con datos recientes.
- Analizar importancia de variables.
- Integrar el modelo en un sistema de prediccion operativo.
