[English](README.md) | **Español**

# Interpretabilidad en riesgo de crédito

Comparación de modelos interpretables (EBM y GAMI-Net) con un modelo de caja negra (XGBoost) explicado post hoc, aplicada al riesgo de crédito. Es el Trabajo Final de Grado de Ingeniería Informática de la Universitat Oberta de Catalunya (UOC). En desarrollo.

## Dataset

FICO HELOC desde OpenML (ID 45553, versión FICO-HELOC-cleaned), con 9871 solicitudes de crédito. Se descarga automáticamente la primera vez que se ejecuta el código, así que el repositorio no guarda ficheros de datos.

## Requisitos

- Python 3.10, porque PiML, que se usa para GAMI-Net, solo publica paquetes hasta Python 3.10.
- [uv](https://docs.astral.sh/uv/) para gestionar el entorno (en macOS: `brew install uv`).

Probado en macOS con Apple Silicon.

## Instalación y ejecución

Desde la raíz del repositorio, crea el entorno desde cero e instala las dependencias:

```bash
uv python install 3.10
```

```bash
uv venv --python 3.10 .venv
```

```bash
uv pip install --python .venv -r requirements.txt
```

Opcionalmente, instala las herramientas de desarrollo (ruff, para formato y lint):

```bash
uv pip install --python .venv -r requirements-dev.txt
```

Comprueba el entorno:

```bash
.venv/bin/python smoke_test.py
```

`smoke_test.py` descarga HELOC, hace una partición estratificada 80/20 (semilla 42) y entrena regresión logística, XGBoost, EBM y GAMI-Net con hiperparámetros por defecto, mostrando el AUC de test y el tiempo de entrenamiento de cada modelo. Solo comprueba que el entorno funciona; no es una comparación de modelos.

## Ejecución del experimento

```bash
.venv/bin/python run.py
```

`run.py` es el único punto de entrada. Descarga el dataset, hace una partición estratificada 60/20/20 en entrenamiento, calibración y test (semilla 42), ejecuta el análisis exploratorio, entrena los modelos y los evalúa en el conjunto de test. El análisis exploratorio y el entrenamiento usan solo el conjunto de entrenamiento. Los códigos especiales (-7, -8, -9, y el valor fuera de rango 9 en `MaxDelq2PublicRecLast12M`) se tratan como ausentes en las figuras y se resumen en una tabla aparte. Durante la ejecución muestra cada paso y los resultados principales (tamaños de las particiones, discrepancias de monotonía, pares muy correlacionados y el código especial con más efecto). Genera:

- `results/figures/eda/fig_tasa_impago_por_variable.png`: tasa de impago por tramo de cada variable, con su dirección esperada.
- `results/figures/eda/fig_correlacion.png`: correlación de Spearman entre las variables.
- `results/tables/eda/tabla_monotonia_observada.csv`: dirección monótona observada frente a la esperada para cada variable.
- `results/tables/eda/tabla_codigos_especiales.csv`: número de casos y tasa de impago de cada código especial.

El paso de entrenamiento ajusta cuatro modelos (regresión logística, XGBoost, EBM y GAMI-Net) con validación cruzada estratificada de 5 bloques sobre el conjunto de entrenamiento. Cada modelo es un pipeline cuyo primer paso es el codificador de códigos especiales, así que las medianas de sustitución se calculan dentro de cada bloque. La mejor configuración se elige por log-loss y también se registra el AUC. El pipeline final se reentrena con todo el conjunto de entrenamiento. Genera:

- `results/models/{logistic_regression,xgboost,ebm,gaminet}.joblib`: cada pipeline ajustado con sus mejores hiperparámetros y sus métricas de validación cruzada.
- `results/tables/models/tabla_ajuste_hiperparametros.csv`: mejores hiperparámetros, AUC y log-loss de validación cruzada (media y desviación típica) y tiempo de búsqueda de cada modelo.

Los modelos guardados se cargan en lugar de reentrenarse, así que las siguientes ejecuciones tardan unos segundos. Para reentrenar todos los modelos:

```bash
.venv/bin/python run.py --retrain
```

El entrenamiento desde cero tarda unos 7-9 minutos en un portátil de 10 núcleos sin GPU, la mayor parte en GAMI-Net.

El paso de evaluación mide los cuatro modelos guardados solo en el conjunto de test (1975 solicitudes), con Bad como clase positiva. Los hiperparámetros están congelados y el conjunto de calibración no se usa. Calcula AUC, Gini, F1 con un umbral fijo de 0,5, Brier score, log-loss y ECE (10 tramos de igual anchura), cada uno con un intervalo de confianza al 95 % por percentiles a partir de 1000 remuestreos bootstrap (semilla 42). Los remuestreos son los mismos para todos los modelos, lo que da también un intervalo de confianza para la diferencia entre cada modelo y XGBoost; una diferencia es significativa cuando su intervalo no contiene el 0. Tarda unos 5 segundos. Genera:

- `results/tables/evaluation/tabla_metricas_o1.csv`: valor e intervalo de confianza de cada métrica para cada modelo.
- `results/tables/evaluation/tabla_metricas_o1_formateada.csv`: los mismos resultados con una fila por modelo y una columna por métrica, como texto con formato numérico español.
- `results/tables/evaluation/tabla_diferencias_xgboost.csv`: diferencia respecto a XGBoost para cada modelo y métrica, con su intervalo de confianza y si es significativa.
- `results/figures/evaluation/fig_diagramas_fiabilidad.png`: diagrama de fiabilidad de cada modelo, con el número de solicitudes por tramo.

Para formatear y revisar el código:

```bash
.venv/bin/ruff format . && .venv/bin/ruff check --fix .
```

## Nota sobre GAMI-Net

GAMI-Net se crea con `n_jobs=1`. Con más hilos el proceso hace segfault, porque torch, scikit-learn y XGBoost cargan runtimes de OpenMP distintos. Este ajuste no cambia el modelo. Por el mismo motivo, torch se limita a un hilo antes de predecir con un GAMI-Net guardado.

## Estructura del repositorio

```
.
├── .gitignore
├── .python-version         Versión de Python que usa uv (3.10)
├── LICENSE
├── README.md
├── README.es.md
├── requirements.txt        Dependencias de ejecución con versión fijada
├── requirements-dev.txt    Herramientas de desarrollo (ruff)
├── ruff.toml               Configuración de formato y lint
├── run.py                  Punto de entrada del experimento
├── smoke_test.py           Comprobación del entorno
└── src/
    ├── __init__.py
    ├── config.py           Constantes: semilla, códigos especiales, direcciones esperadas, rutas de salida
    ├── data.py             Descarga del dataset y partición en entrenamiento, calibración y test
    ├── preprocessing.py    Codificador de códigos especiales
    ├── eda.py              Análisis exploratorio
    ├── models.py           Búsqueda de hiperparámetros y entrenamiento
    └── evaluation.py       Métricas en test, intervalos bootstrap y diagramas de fiabilidad
```

## Licencia

MIT. Ver [LICENSE](LICENSE).
