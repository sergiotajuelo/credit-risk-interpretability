import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from piml.models import GAMINetClassifier

SEED = 42

heloc = fetch_openml(data_id=45553, as_frame=True)
X, y_raw = heloc.data.copy(), heloc.target

print(f"Dataset: {heloc.details['name']} (OpenML {heloc.details['id']})")
print(f"Rows: {X.shape[0]}  Columns (variables): {X.shape[1]}")
print("Distribution of the target variable:")
print(y_raw.value_counts().to_string())
print(y_raw.value_counts(normalize=True).round(4).to_string())

for col in X.select_dtypes("category").columns:
    X[col] = X[col].astype(int)

n_all_missing = int((X == -9).all(axis=1).sum())
print(f"Rows with all variables to -9: {n_all_missing}")

y = (y_raw == "Bad").astype(int).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}\n")

models = {
    "Regresión logística": LogisticRegression(),
    "XGBoost": XGBClassifier(),
    "EBM": ExplainableBoostingClassifier(),
    # n_jobs=1: the default (10) segfaults on macOS because several OpenMP copies
    # are loaded at once (torch's and the Homebrew one used by XGBoost). The model is unchanged.
    "GAMI-Net": GAMINetClassifier(n_jobs=1),
}

results = []
for name, model in models.items():
    if name == "GAMI-Net":
        X_tr, X_te = X_train.to_numpy(dtype=float), X_test.to_numpy(dtype=float)
    else:
        X_tr, X_te = X_train, X_test

    t0 = time.perf_counter()
    model.fit(X_tr, y_train)
    elapsed = time.perf_counter() - t0

    proba = np.asarray(model.predict_proba(X_te))
    score = proba[:, 1] if proba.ndim == 2 else proba.ravel()
    auc = roc_auc_score(y_test, score)
    results.append((name, auc, elapsed))
    print(f"{name}: AUC={auc:.4f}  time={elapsed:.1f}s")

print("\nSummary")
print(f"{'Model':<22}{'AUC test':>10}{'Time (s)':>12}")
for name, auc, elapsed in results:
    print(f"{name:<22}{auc:>10.4f}{elapsed:>12.1f}")
