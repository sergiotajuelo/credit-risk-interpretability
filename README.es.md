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

Ejecuta la comprobación del entorno:

```bash
.venv/bin/python smoke_test.py
```

`smoke_test.py` descarga HELOC, hace una partición estratificada 80/20 (semilla 42) y entrena regresión logística, XGBoost, EBM y GAMI-Net con hiperparámetros por defecto, mostrando el AUC de test y el tiempo de entrenamiento de cada modelo. Solo comprueba que el entorno funciona; no es una comparación de modelos.

## Nota sobre GAMI-Net

GAMI-Net se crea con `n_jobs=1`. Con más hilos el proceso hace segfault, porque torch, scikit-learn y XGBoost cargan runtimes de OpenMP distintos. Este ajuste no cambia el modelo.

## Estructura del repositorio

```
.
├── .gitignore
├── .python-version     Versión de Python que usa uv (3.10)
├── LICENSE
├── README.md
├── README.es.md
├── requirements.txt    Dependencias con versión fijada
└── smoke_test.py       Comprobación del entorno
```

## Licencia

MIT. Ver [LICENSE](LICENSE).
