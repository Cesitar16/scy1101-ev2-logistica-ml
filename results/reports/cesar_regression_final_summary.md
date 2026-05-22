# Resumen final - Modelo de Regresion Cesar

## Objetivo

Predecir el tiempo de entrega en el contexto Logistica 4.0.

## Tipo de modelo

Regresion supervisada.

## Modelo final

linear_regression_optimized

## Metricas finales

- MAE: 11.625754191635037
- RMSE: 14.805799765183705
- R2: -0.0007785940055420326

## Interpretacion

El MAE representa el error absoluto promedio del modelo en minutos. El RMSE penaliza con mayor fuerza los errores grandes y permite analizar que tanto se alejan algunas predicciones del valor real. El R2 indica que proporcion de la variabilidad del tiempo de entrega logra explicar el modelo.

## Utilidad para el negocio

Este modelo puede ayudar a una empresa logistica a estimar tiempos de entrega antes de realizar la operacion, permitiendo promesas de entrega mas realistas y una mejor planificacion operativa.

## Limitaciones

El modelo depende de la calidad del dataset y de las variables disponibles. No considera necesariamente eventos externos no registrados, como accidentes, cortes de ruta, fallas del vehiculo o cambios operativos imprevistos.

## Proximos pasos recomendados

- Probar el modelo con mas datos reales.
- Comparar con nuevos algoritmos.
- Revisar importancia de variables.
- Integrar el modelo en un flujo de prediccion operativo.


## Comparacion base vs optimizado

Se comparo el modelo base con su version optimizada para revisar si la optimizacion de hiperparametros mejoro el rendimiento.
