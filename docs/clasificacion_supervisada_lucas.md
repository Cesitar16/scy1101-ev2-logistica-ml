# Clasificacion supervisada - Lucas

## 1. Objetivo
El objetivo fue predecir la categoria de entrega en ultima milla (`rapida`, `normal`, `tardia`) usando variables operativas logisticas.

Es un problema de clasificacion porque la salida esperada es discreta (clase), no continua.

## 2. Mejora metodologica aplicada (target sin leakage)
### Problema original
La primera version construia `categoria_entrega` con terciles calculados sobre todo el dataset antes del split.

### Cambio aplicado
Se implemento un flujo mas riguroso:
1. Se hace `train_test_split` primero sobre el dataset base.
2. Se calculan umbrales `q33` y `q66` solo con `target_tiempo_entrega` de train.
3. Esos mismos umbrales se aplican a train y test para crear `categoria_entrega`.

Con esto, la distribucion del test no participa en la creacion del target y el test queda reservado para evaluacion final.

## 3. Preparacion de datos
Dataset usado:
- `data/raw/5_logistica_40.csv`

Funcion principal del nuevo flujo:
- `preparar_datos_clasificacion_sin_leakage(...)` en `src/classification_utils.py`

Preprocesamiento:
- Numericas: `SimpleImputer(strategy="median")` + `StandardScaler()`
- Categoricas: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`

Todo el preprocesamiento se integra en `Pipeline` junto al modelo.

## 4. Modelos base entrenados
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `KNeighborsClassifier`
- `LogisticRegression`

## 5. Metricas usadas
- `accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`
- `f1_weighted`
- `classification_report`
- `confusion_matrix`

Metrica principal para comparacion: `f1_macro`.

## 6. Resultados actuales (flujo sin leakage)
Fuente:
- `results/metrics/classification_metrics.csv`
- `results/metrics/optimized_eval/classification_metrics.csv`
- `results/metrics/final_classification_comparison.csv`

### Modelos base
| Modelo | accuracy | precision_macro | recall_macro | f1_macro | f1_weighted |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.3583 | 0.3616 | 0.3635 | 0.3583 | 0.3580 |
| random_forest | 0.3417 | 0.3414 | 0.3423 | 0.3400 | 0.3425 |
| decision_tree | 0.3292 | 0.3469 | 0.3150 | 0.3096 | 0.3194 |
| knn | 0.3208 | 0.3089 | 0.3096 | 0.3002 | 0.3094 |

### Modelos optimizados (CV sobre train)
| Modelo | accuracy | precision_macro | recall_macro | f1_macro | f1_weighted |
|---|---:|---:|---:|---:|---:|
| optimized_logistic_regression | 0.3750 | 0.3746 | 0.3761 | 0.3738 | 0.3758 |
| optimized_random_forest | 0.3667 | 0.3651 | 0.3689 | 0.3659 | 0.3661 |

## 7. Optimizacion de hiperparametros
Optimizacion acotada a:
- `RandomForestClassifier`
- `LogisticRegression`

Configuracion:
- `scoring="f1_macro"`
- `cv=5` (ajustado si la clase minoritaria lo exige)
- `n_jobs`: `1` en Windows, `-1` en Linux/macOS
- metodo: `grid` o `random` segun tamano de train

Resultado de tuning (ultima ejecucion):
- archivo: `results/metrics/optimization_result_random_forest_random.json`
- ganador por CV: `random_forest`
- best CV score (`f1_macro`): `0.36420917531314906`
- mejores parametros RF:
  - `model__n_estimators=300`
  - `model__min_samples_split=5`
  - `model__min_samples_leaf=4`
  - `model__max_depth=None`
  - `model__criterion="entropy"`

## 8. Criterio de seleccion del modelo final
Regla aplicada:
- Seleccion por validacion cruzada (no por test).
- Test se usa solo para reporte final comparativo.

Con esa regla, el modelo final seleccionado fue:
- `optimized_random_forest`

Persistencia:
- `models/trained_models/classification_best_model.joblib`

Nota tecnica:
- En test, `optimized_logistic_regression` obtuvo mayor `f1_macro` (`0.3738`) que `optimized_random_forest` (`0.3659`).
- Aun asi se mantiene la regla metodologica de seleccionar por CV para evitar decision post-hoc sobre test.

## 9. Visualizaciones e interpretacion
Archivos generados:
- `results/plots/confusion_matrix_*.png`
- `results/plots/feature_importance_decision_tree.png`
- `results/plots/feature_importance_random_forest.png`
- `results/plots/decision_tree_plot.png`

Variable raiz del arbol (ultima ejecucion):
- `cat__trafico_nivel_Bajo`

Interpretacion:
- El trafico aparece como condicion inicial relevante para separar clases de entrega.
- Las matrices de confusion muestran confusion cruzada entre `normal`, `rapida` y `tardia`, lo que sugiere senal predictiva moderada.

## 10. Impacto del cambio metodologico
Comparacion contra corrida previa (terciles globales):
- Antes (baseline top): `logistic_regression` con `f1_macro ≈ 0.3662`.
- Ahora (sin leakage de distribucion):
  - mejor baseline: `logistic_regression` con `f1_macro ≈ 0.3583`
  - mejor optimizado en test: `optimized_logistic_regression` con `f1_macro ≈ 0.3738`

Conclusion:
- El flujo nuevo es metodologicamente mas correcto.
- Las metricas pueden subir o bajar; eso no invalida el cambio.
- Lo importante es que el test deja de influir en la construccion del target.

## 11. Limitaciones restantes
- El target sigue derivado por terciles, no por umbrales de negocio/SLA.
- El desempeno global sigue moderado para un problema multiclase.
- Puede haber variables operativas no incluidas que mejoren separacion entre clases.
