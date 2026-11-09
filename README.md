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

`run.py` is the single entry point. For now it downloads the dataset, makes a stratified 60/20/20 split into training, calibration and test sets (seed 42) and runs the exploratory analysis on the training set only. Special codes (-7, -8, -9, and the out-of-range value 9 in `MaxDelq2PublicRecLast12M`) are treated as missing in the figures and summarised in a separate table. While it runs, it prints each step and the main results (partition sizes, monotonicity mismatches, highly correlated pairs and the special code with the largest effect). It writes:

- `results/figures/eda/fig_tasa_impago_por_variable.png`: default rate by bin for each variable, with its expected direction.
- `results/figures/eda/fig_correlacion.png`: Spearman correlation between the variables.
- `results/tables/eda/tabla_monotonia_observada.csv`: observed versus expected monotonic direction for each variable.
- `results/tables/eda/tabla_codigos_especiales.csv`: number of cases and default rate for each special code.

Model training and evaluation will be added later.

To format and lint the code:

```bash
.venv/bin/ruff format . && .venv/bin/ruff check --fix .
```

## Note on GAMI-Net

GAMI-Net is created with `n_jobs=1`. With more threads the process segfaults, because torch, scikit-learn and XGBoost load different OpenMP runtimes. This setting does not change the model.

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
    └── eda.py              Exploratory analysis
```

## License

MIT. See [LICENSE](LICENSE).
