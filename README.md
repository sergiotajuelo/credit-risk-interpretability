**English** | [Español](README.es.md)

# Interpretability in credit risk

Comparison of interpretable models (EBM and GAMI-Net) with a black-box model (XGBoost) explained post hoc, applied to credit risk. This is the Bachelor's thesis (Trabajo Final de Grado) in Computer Engineering at the Universitat Oberta de Catalunya (UOC). Work in progress.

## Dataset

FICO HELOC from OpenML (ID 45553, version FICO-HELOC-cleaned), with 9871 credit applications. It is downloaded automatically the first time the code runs, so no data files are stored in the repository.

## Requirements

- Python 3.10, because PiML, used for GAMI-Net, only publishes packages up to Python 3.10.
- [uv](https://docs.astral.sh/uv/) to manage the environment (on macOS: `brew install uv`).

Tested on macOS with Apple Silicon.

## Installation and usage

From the repository root, create the environment from scratch and install the dependencies:

```bash
uv python install 3.10
```

```bash
uv venv --python 3.10 .venv
```

```bash
uv pip install --python .venv -r requirements.txt
```

Optionally, install the development tools (ruff, for formatting and linting):

```bash
uv pip install --python .venv -r requirements-dev.txt
```

Check the environment:

```bash
.venv/bin/python smoke_test.py
```

`smoke_test.py` downloads HELOC, makes a stratified 80/20 split (seed 42) and trains logistic regression, XGBoost, EBM and GAMI-Net with default hyperparameters, printing each model's test AUC and training time. It only checks that the environment works; it is not a model comparison.

## Running the experiment

```bash
.venv/bin/python run.py
```

`run.py` is the single entry point. It downloads the dataset, makes a stratified 60/20/20 split into training, calibration and test sets (seed 42), runs the exploratory analysis, trains the models and evaluates them on the test set. The exploratory analysis and the training use the training set only. Special codes (-7, -8, -9, and the out-of-range value 9 in `MaxDelq2PublicRecLast12M`) are treated as missing in the figures and summarised in a separate table. While it runs, it prints each step and the main results (partition sizes, monotonicity mismatches, highly correlated pairs and the special code with the largest effect). It writes:

- `results/figures/eda/fig_tasa_impago_por_variable.png`: default rate by bin for each variable, with its expected direction.
- `results/figures/eda/fig_correlacion.png`: Spearman correlation between the variables.
- `results/tables/eda/tabla_monotonia_observada.csv`: observed versus expected monotonic direction for each variable.
- `results/tables/eda/tabla_codigos_especiales.csv`: number of cases and default rate for each special code.

The training step tunes four models (logistic regression, XGBoost, EBM and GAMI-Net) with 5-fold stratified cross-validation on the training set. Each model is a pipeline whose first step is the special-code encoder, so the replacement medians are computed inside each fold. The best configuration is chosen by log-loss, and AUC is recorded as well. The final pipeline is refitted on the whole training set. It writes:

- `results/models/{logistic_regression,xgboost,ebm,gaminet}.joblib`: each fitted pipeline with its best hyperparameters and cross-validation metrics.
- `results/tables/models/tabla_ajuste_hiperparametros.csv`: best hyperparameters, cross-validation AUC and log-loss (mean and standard deviation) and search time for each model.

Saved models are loaded instead of retrained, so later runs take a few seconds. To retrain all models:

```bash
.venv/bin/python run.py --retrain
```

Training from scratch takes about 7-9 minutes on a 10-core laptop without a GPU, most of it for GAMI-Net.

The evaluation step scores the four saved models on the test set only (1975 applications), with Bad as the positive class. The hyperparameters are frozen and the calibration set is not used. It computes AUC, Gini, F1 at a fixed threshold of 0.5, Brier score, log-loss and ECE (10 equal-width bins), each with a 95% percentile confidence interval from 1000 bootstrap resamples (seed 42). The resamples are the same for every model, which also gives a confidence interval for the difference between each model and XGBoost; a difference is significant when its interval does not contain 0. It takes about 5 seconds. It writes:

- `results/tables/evaluation/tabla_metricas_o1.csv`: value and confidence interval of each metric for each model.
- `results/tables/evaluation/tabla_metricas_o1_formateada.csv`: the same results with one row per model and one column per metric, as text in Spanish number format.
- `results/tables/evaluation/tabla_diferencias_xgboost.csv`: difference from XGBoost for each model and metric, with its confidence interval and whether it is significant.
- `results/figures/evaluation/fig_diagramas_fiabilidad.png`: reliability diagram of each model, with the number of applications per bin.

To format and lint the code:

```bash
.venv/bin/ruff format . && .venv/bin/ruff check --fix .
```

## Note on GAMI-Net

GAMI-Net is created with `n_jobs=1`. With more threads the process segfaults, because torch, scikit-learn and XGBoost load different OpenMP runtimes. This setting does not change the model. For the same reason, torch is limited to one thread before predicting with a saved GAMI-Net.

## Repository structure

```
.
├── .gitignore
├── .python-version         Python version used by uv (3.10)
├── LICENSE
├── README.md
├── README.es.md
├── requirements.txt        Pinned runtime dependencies
├── requirements-dev.txt    Development tools (ruff)
├── ruff.toml               Formatting and lint settings
├── run.py                  Entry point of the experiment
├── smoke_test.py           Environment check
└── src/
    ├── __init__.py
    ├── config.py           Constants: seed, special codes, expected directions, output paths
    ├── data.py             Dataset download and train/calibration/test split
    ├── preprocessing.py    Special-code encoder
    ├── eda.py              Exploratory analysis
    ├── models.py           Hyperparameter search and training
    └── evaluation.py       Test-set metrics, bootstrap intervals and reliability diagrams
```

## License

MIT. See [LICENSE](LICENSE).
