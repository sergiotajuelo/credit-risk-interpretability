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

Run the environment check:

```bash
.venv/bin/python smoke_test.py
```

`smoke_test.py` downloads HELOC, makes a stratified 80/20 split (seed 42) and trains logistic regression, XGBoost, EBM and GAMI-Net with default hyperparameters, printing each model's test AUC and training time. It only checks that the environment works; it is not a model comparison.

## Note on GAMI-Net

GAMI-Net is created with `n_jobs=1`. With more threads the process segfaults, because torch, scikit-learn and XGBoost load different OpenMP runtimes. This setting does not change the model.

## Repository structure

```
.
├── .gitignore
├── .python-version     Python version used by uv (3.10)
├── LICENSE
├── README.md
├── README.es.md
├── requirements.txt    Pinned dependencies
└── smoke_test.py       Environment check
```

## License

MIT. See [LICENSE](LICENSE).
