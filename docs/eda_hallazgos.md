# Hallazgos del Analisis Exploratorio de Datos

## Contexto
El caso Logistica 4.0 analiza entregas de ultima milla para entender factores asociados al tiempo de entrega y preparar el dataset para modelado posterior.

## Estructura del dataset
- Filas: 1200
- Columnas: 12
- Variable objetivo: `target_tiempo_entrega` (continua)
- Variables numericas iniciales: 9 (incluyendo `id_bodega` antes de convertirla a categoria)
- Variables categoricas iniciales: 3

## Calidad de datos
- Valores nulos: 0
- Duplicados exactos: 0
- Categorias observadas:
  - `trafico_nivel`: Critico, Alto, Medio, Bajo
  - `clima`: Lluvia, Soleado, Nublado
  - `tipo_vehiculo`: Moto, Van, Camion
- `id_bodega` debe tratarse como categoria, no como magnitud continua.

## Problemas detectados
- Registros con `target_tiempo_entrega <= 0`: 2 (indices 806 y 1153 en el dataset original).
- Registros con `target_tiempo_entrega < 10`: 10 (incluye los 2 negativos).
- Outliers IQR en `target_tiempo_entrega`: 6.
- Limites IQR del target (aprox.): inferior 5.21, superior 85.81.

## Decisiones de limpieza
- Eliminar registros con `target_tiempo_entrega <= 0` para el dataset limpio.
- Conservar tiempos `< 10` y marcarlos como `tiempo_sospechoso` para revision posterior.
- Mantener outliers hasta validacion de negocio (no eliminar automaticamente).
- Normalizar texto categorico y convertir `id_bodega` a tipo categorico.

## Variables categoricas y encoding recomendado
- `clima`: One-Hot Encoding.
- `tipo_vehiculo`: One-Hot Encoding.
- `id_bodega`: One-Hot Encoding o tratamiento categorico equivalente.
- `trafico_nivel`: comparar dos enfoques:
  - ordinal: Bajo=0, Medio=1, Alto=2, Critico=3
  - One-Hot Encoding

## Posible fuga de informacion
`consumo_nafta` puede ser leakage si representa consumo real posterior a la entrega. Si es estimado previo al despacho, puede usarse. Se recomienda comparar modelado con y sin esta variable.

## Recomendaciones para modelado posterior
- Regresion supervisada para `target_tiempo_entrega`.
- Clasificacion derivada (por ejemplo `entrega_tardia` o `tiempo_alto`) en experimentos separados.
- Clustering para descubrir patrones logisticos.
- Escalado con `StandardScaler` para metodos sensibles a distancia (KNN, K-Means, PCA, SVR).
- Comparar pipelines con/sin `consumo_nafta`.
- Evaluar nuevas features: `hora_sin`, `hora_cos`, `costo_por_km`, `consumo_por_km`, `carga_por_km`, `paradas_por_km`.

## Conclusion
El dataset queda preparado para etapas posteriores de modelado supervisado y no supervisado, con limpieza trazable, riesgos documentados y recomendaciones claras de transformacion.
