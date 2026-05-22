# Setup del modelo de regresion - Cesar

## Objetivo

Preparar la linea de regresion supervisada para predecir el tiempo de entrega dentro del proyecto Logistica 4.0.

## Pregunta del modelo

¿Cuanto demorara una entrega?

## Tipo de aprendizaje

Regresion supervisada.

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

## Justificacion

Este problema corresponde a regresion porque busca predecir un valor numerico continuo: el tiempo de entrega. No se busca clasificar una entrega como tardia o normal en esta linea de trabajo, sino estimar directamente los minutos de duracion.

## Alcance de esta rama

Esta rama solo prepara:
- Estructura inicial.
- Notebook base.
- Carga de datos.
- Definicion de X e y.
- Validacion de columnas.
- Identificacion inicial de variables numericas y categoricas.

No incluye:
- Limpieza profunda.
- Encoding.
- Escalado.
- Train/test split.
- Entrenamiento.
- Evaluacion.
- Optimizacion.
- Guardado de modelos.

## Ramas futuras

Las siguientes ramas deberian continuar en este orden:

1. cesar/feature/regression-preprocessing
2. cesar/feature/regression-training
3. cesar/feature/regression-evaluation
4. cesar/feature/regression-tuning
5. cesar/feature/regression-finalization

Todas esas ramas deberan mergearse primero hacia:

cesar/modelo-regresion

Y solo cuando este completa toda la linea de regresion, se hara merge de:

cesar/modelo-regresion

hacia:

main
