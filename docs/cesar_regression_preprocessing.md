# Preprocesamiento del modelo de regresion - Cesar

## Objetivo

Preparar los datos de la linea de regresion supervisada para predecir el tiempo de entrega.

## Rama

cesar/feature/regression-preprocessing

## Rama base

cesar/feature/regression-setup

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

## Transformaciones realizadas

1. Normalizacion de nombres de columnas.
2. Validacion de columnas requeridas.
3. Limpieza de variables categoricas.
4. Conversion segura de variables numericas.
5. Conversion de la variable objetivo.
6. Eliminacion de duplicados.
7. Eliminacion de filas sin variable objetivo.
8. Tratamiento de outliers con metodo IQR.
9. Separacion de X e y.
10. Construccion de preprocesador con Scikit-learn.

## Preprocesador creado

Para variables numericas:
- SimpleImputer con estrategia median.
- StandardScaler.

Para variables categoricas:
- SimpleImputer con estrategia most_frequent.
- OneHotEncoder con handle_unknown="ignore".

## Dataset procesado generado

data/processed/cesar_logistica_clean.csv

## Decisiones importantes

No se entrenaron modelos en esta rama, porque esta etapa solo busca preparar los datos. El entrenamiento se realizara en la siguiente rama:

cesar/feature/regression-training

## Justificacion

La limpieza y preprocesamiento son fundamentales porque los datos reales pueden contener valores faltantes, textos inconsistentes, duplicados o valores extremos. Preparar correctamente los datos mejora la calidad de los modelos posteriores.