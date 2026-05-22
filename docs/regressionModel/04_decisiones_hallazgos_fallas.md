# Decisiones, Hallazgos, Fallos y Lecciones Aprendidas

## Decisiones clave tomadas

1. **No quedarse con un unico criterio**:
   - se evaluo tecnica (`MAE`, `RMSE`, `R2`) y operacion (`+/-10`, `+/-15`).
2. **Mantener baseline equilibrado actual** si nuevas variantes no mejoraban de forma global.
3. **Explorar segmentos problematicos** antes de forzar tuning global.
4. **Priorizar reproducibilidad**: rutas de outputs, metricas versionadas, notebook + modulos `src/`.

## Hallazgos importantes

- Segmentos mas complejos (MAE alto) en la etapa de errores dificiles:
  - `distancia_larga=1`
  - `tipo_vehiculo=camion`
  - `id_bodega=4`
  - `carga_pesada=1`
  - `trafico_nivel=bajo`
- La familia de boosting se mantuvo competitiva en casi todas las etapas.

## En que fallamos / limitaciones observadas

1. **Mejorar todo al mismo tiempo fue dificil**:
   - cuando subio `+/-10`, tendio a empeorar `RMSE/R2`, y viceversa.
2. **Experimento de errores dificiles no mejoro segmentos objetivo**:

| segment_column | segment_value | n | baseline_MAE | new_MAE | delta_MAE | improved |
| --- | --- | --- | --- | --- | --- | --- |
| tipo_vehiculo | camion | 82 | 12.338845 | 12.338845 | 0.000000 | False |
| trafico_nivel | alto | 58 | 11.555559 | 11.555559 | 0.000000 | False |
| trafico_nivel | bajo | 56 | 11.925398 | 11.925398 | 0.000000 | False |
| id_bodega | 1 | 60 | 11.733073 | 11.733073 | 0.000000 | False |
| id_bodega | 4 | 74 | 12.143550 | 12.143550 | 0.000000 | False |
| distancia_larga | 1 | 60 | 12.806547 | 12.806547 | 0.000000 | False |
| carga_pesada | 1 | 60 | 12.024281 | 12.024281 | 0.000000 | False |
| muchas_paradas | 1 | 66 | 11.437159 | 11.437159 | 0.000000 | False |

3. **Capacidad explicativa aun baja**:
   - `R2` permanece lejos de valores altos, sugiriendo variables faltantes o ruido estructural.
4. **Algunas optimizaciones clasicas no aportaron** (ejemplo temprano de `linear_regression_optimized`).

## Que recomendamos a futuro

- Agregar variables exogenas de operacion real (incidentes, trafico granular por hora/zona, ventanas de despacho, evento-clima mas fino).
- Probar enfoques con validacion temporal si los datos tienen componente cronologica.
- Evaluar perdidas/costos customizados de negocio (penalizar tardanzas criticas mas que errores simetricos).
- Revisar segmentacion por bodega/vehiculo con modelos especializados por subpoblacion si el volumen lo permite.

## Decision vigente

- **Modelo recomendado para continuidad del proyecto academico:** `GradientBoostingRegressor` equilibrado actual.
- **Estado del mejor candidato alternativo:** util para analisis, no para reemplazo directo sin degradar KPI operacional.
