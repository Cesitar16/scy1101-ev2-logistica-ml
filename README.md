# scy1101-ev2-logistica-ml

Proyecto ML para SCY1101 EV2 enfocado en logística predictiva, con análisis de datos, modelos supervisados/no supervisados, evaluación y optimización.

## Contexto académico
Este repositorio corresponde a la Evaluación Parcial N°2 de la asignatura **SCY1101 - Programación para la Ciencia de Datos**. El trabajo está orientado a construir un flujo reproducible de Machine Learning aplicado a logística predictiva.

## Objetivo general
Desarrollar una solución de Machine Learning para un problema de logística predictiva, cubriendo desde el análisis exploratorio hasta la evaluación y optimización de modelos.

## Objetivos específicos
- Organizar y documentar un entorno reproducible para el proyecto.
- Realizar análisis exploratorio de datos (EDA).
- Implementar preprocesamiento de datos y preparación de variables.
- Entrenar modelos supervisados según el tipo de variable objetivo.
- Aplicar técnicas de modelado no supervisado para descubrir patrones.
- Evaluar desempeño de modelos con métricas apropiadas.
- Optimizar hiperparámetros mediante búsqueda sistemática.
- Consolidar hallazgos en informe y presentación final.

## Estructura de carpetas
```text
scy1101-ev2-logistica-ml/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
├── models/
│   └── trained_models/
├── results/
│   ├── metrics/
│   ├── plots/
│   └── reports/
├── docs/
│   ├── informe/
│   └── presentacion/
├── tests/
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

## Tecnologías utilizadas
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

## Instalación del entorno
### 1) Crear entorno virtual
En Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Instalar dependencias
```bash
pip install -r requirements.txt
```

## Cómo ejecutar notebooks
1. Activar el entorno virtual.
2. Levantar Jupyter:
```bash
jupyter notebook
```
3. Abrir la carpeta `notebooks/` y ejecutar el cuaderno correspondiente.

## Flujo de trabajo esperado
1. Cargar y validar datos en `notebooks/01_exploratory_analysis.ipynb`.
2. Implementar preprocesamiento en `src/data_preprocessing.py`.
3. Entrenar modelos supervisados y no supervisados en notebooks y módulos de `src/`.
4. Evaluar resultados en `notebooks/04_model_evaluation.ipynb`.
5. Optimizar hiperparámetros en `notebooks/05_hyperparameter_optimization.ipynb`.
6. Consolidar conclusiones en `notebooks/06_final_analysis.ipynb` y `docs/`.

## Estrategia de ramas sugerida
- `main`
- `develop`
- `feature/eda`
- `feature/preprocessing`
- `feature/supervised-models`
- `feature/unsupervised-models`
- `feature/evaluation`
- `feature/optimization`
- `feature/report`

## Estado actual del proyecto
Estructura inicial creada. El proyecto está listo para comenzar el desarrollo incremental de la Evaluación Parcial N°2.

## Autor / Equipo
`[Nombre del autor o equipo]`

## Licencia
Este proyecto se distribuye bajo la licencia MIT. Ver archivo `LICENSE`.

## Linea de trabajo de Cesar: Modelo de Regresion

Esta linea de trabajo busca construir un modelo de regresion supervisada para predecir el tiempo de entrega en el contexto Logistica 4.0.

### Rama general

cesar/modelo-regresion

### Rama actual de setup

cesar/feature/regression-setup

### Objetivo inicial

Preparar el notebook base, cargar el dataset, definir la variable objetivo y separar las variables de entrada X y la variable objetivo y.

### Notebook principal

notebooks/02_supervised_regression_cesar.ipynb

### Variable objetivo

target_tiempo_entrega

### Proximos pasos

- Preprocesamiento de datos.
- Entrenamiento de modelos de regresion.
- Evaluacion con MAE, RMSE y R².
- Optimizacion de hiperparametros.
- Guardado del modelo final.

### Rama de preprocesamiento

Rama:
cesar/feature/regression-preprocessing

Objetivo:
Preparar el dataset para el entrenamiento posterior de modelos de regresion.

Archivo principal:
notebooks/02_supervised_regression_cesar.ipynb

Modulo principal:
src/data_preprocessing.py

Dataset limpio generado:
data/processed/cesar_logistica_clean.csv

Transformaciones:
- Normalizacion de columnas.
- Limpieza de texto.
- Conversion de numericos.
- Tratamiento de nulos.
- Eliminacion de duplicados.
- Tratamiento de outliers con IQR.
- Construccion de ColumnTransformer para Scikit-learn.

Siguiente rama:
cesar/feature/regression-training

Restricciones importantes:
- No entrenar modelos todavia.
- No evaluar modelos todavia.
- No optimizar hiperparametros todavia.

### Rama de entrenamiento

Rama:
cesar/feature/regression-training

Objetivo:
Entrenar modelos base de regresion supervisada para predecir target_tiempo_entrega.

Dataset utilizado:
data/processed/cesar_logistica_clean.csv

Modulo principal:
src/model_training.py

Notebook:
notebooks/02_supervised_regression_cesar.ipynb

Modelos entrenados:
- LinearRegression
- DecisionTreeRegressor
- RandomForestRegressor
- GradientBoostingRegressor

Archivo generado:
results/reports/cesar_regression_baseline_predictions.csv

Siguiente paso:
Evaluar los modelos con MAE, RMSE y R? en la rama:
cesar/feature/regression-evaluation

Restricciones:
No se realizo optimizacion de hiperparametros en esta rama.
No se guardo modelo final en esta rama.

