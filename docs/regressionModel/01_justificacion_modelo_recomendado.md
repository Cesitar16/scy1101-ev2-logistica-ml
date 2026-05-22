# Justificacion del Modelo Recomendado

## Pregunta de negocio

Cuanto demorara una entrega?

## Criterio de decision usado

Se usaron dos lentes de decision:

1. **Ranking estricto de regresion:** menor `MAE`, menor `RMSE`, mayor `R2`, mayor `acc_10_min`, menor `p90_absolute_error`.
2. **Criterio equilibrado final:** no sacrificar demasiado desempeno operacional (especialmente `+/-10` y `+/-15` minutos) por mejoras tecnicas marginales.

## Resultado del ranking estricto (ultimo experimento local)

- Mejor fila: **`gradient_boosting_baseline_like`**
- MAE: `11.473306`
- RMSE: `14.518579`
- R2: `0.037673`
- Acc +/-10: `51.25%`
- Acc +/-15: `70.00%`

## Comparacion contra baseline equilibrado actual

Baseline equilibrado actual (`GradientBoostingRegressor`):
- MAE: `11.471100`
- RMSE: `14.570900`
- R2: `0.030700`
- Acc +/-10: `52.50%`
- Acc +/-15: `70.42%`

Mejor nuevo de experimento de errores dificiles (`gradient_boosting_baseline_like`):
- MAE: `11.473306`
- RMSE: `14.518579`
- R2: `0.037673`
- Acc +/-10: `51.25%`
- Acc +/-15: `70.00%`

Deltas (nuevo - baseline actual):
- DeltaMAE: `0.002206`
- DeltaRMSE: `-0.052321`
- DeltaR2: `0.006973`
- DeltaAcc +/-10: `-1.250000 pts`

## Decision final recomendada

**Mantener el baseline equilibrado actual** (`GradientBoostingRegressor`), porque:

- La mejora tecnica no fue consistente en todas las metricas (MAE sube ligeramente).
- Se pierde acierto operacional en ventanas clave (`+/-10` y `+/-15`).
- No hubo mejora en segmentos criticos (0/8 segmentos objetivo mejorados en MAE en el experimento de errores dificiles).

## Cuando si usar el nuevo modelo

Usarlo como **linea de investigacion complementaria** cuando el foco sea exclusivamente reducir `RMSE` o mejorar `R2` en iteraciones futuras, pero no como reemplazo inmediato del modelo equilibrado actual.
