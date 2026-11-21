import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from piml.models import GAMINetClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    CV_N_SPLITS,
    LOGISTIC_MAX_ITER,
    MODEL_DISPLAY_NAMES,
    MODELS_DIR,
    MODELS_TABLES_DIR,
    PROJECT_ROOT,
    SEARCH_SPACES,
    SEED,
    XGB_N_ITER,
)
from src.data import DataSplit
from src.preprocessing import SpecialCodeEncoder

logger = logging.getLogger(__name__)

MODEL_STEP = "model"


@dataclass
class TrainedModel:
    name: str
    estimator: Pipeline
    best_params: dict
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    cv_log_loss_mean: float
    cv_log_loss_std: float
    search_time_s: float


class ModelTrainer:
    """Tunes and fits the four models on the training set only, selecting by log-loss."""

    def __init__(
        self,
        split: DataSplit,
        indicators: list[tuple[str, int]],
        models_dir: Path = MODELS_DIR,
        tables_dir: Path = MODELS_TABLES_DIR,
        retrain: bool = False,
    ) -> None:
        self.X_train = split.X_train
        self.y_train = split.y_train
        self.indicators = indicators
        self.models_dir = models_dir
        self.tables_dir = tables_dir
        self.retrain = retrain

    def train_all(self) -> list[TrainedModel]:
        models = [self._load_or_train(name) for name in MODEL_DISPLAY_NAMES]
        self._save_table(models)
        return models

    def _load_or_train(self, name: str) -> TrainedModel:
        path = self.models_dir / f"{name}.joblib"
        if path.exists() and not self.retrain:
            logger.info("Loaded cached model %s", path.relative_to(PROJECT_ROOT))
            return joblib.load(path)
        model = self._train(name)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info("Saved %s", path.relative_to(PROJECT_ROOT))
        return model

    def _train(self, name: str) -> TrainedModel:
        search = self._search(name)
        logger.info("Tuning %s on %d training rows", MODEL_DISPLAY_NAMES[name], len(self.y_train))
        start = time.perf_counter()
        search.fit(self.X_train, self.y_train)
        elapsed = time.perf_counter() - start

        results = search.cv_results_
        best = search.best_index_
        prefix = f"{MODEL_STEP}__"
        model = TrainedModel(
            name=name,
            estimator=search.best_estimator_,
            best_params={key.removeprefix(prefix): v for key, v in search.best_params_.items()},
            cv_roc_auc_mean=float(results["mean_test_roc_auc"][best]),
            cv_roc_auc_std=float(results["std_test_roc_auc"][best]),
            cv_log_loss_mean=float(-results["mean_test_neg_log_loss"][best]),
            cv_log_loss_std=float(results["std_test_neg_log_loss"][best]),
            search_time_s=elapsed,
        )
        logger.info(
            "  %d candidates x %d folds in %.0f s; best %s",
            len(results["params"]),
            CV_N_SPLITS,
            elapsed,
            model.best_params,
        )
        logger.info(
            "  CV AUC %.4f +/- %.4f, log-loss %.4f +/- %.4f",
            model.cv_roc_auc_mean,
            model.cv_roc_auc_std,
            model.cv_log_loss_mean,
            model.cv_log_loss_std,
        )
        return model

    def _search(self, name: str) -> BaseSearchCV:
        space = {f"{MODEL_STEP}__{key}": values for key, values in SEARCH_SPACES[name].items()}
        common = {
            "scoring": ["roc_auc", "neg_log_loss"],
            "refit": "neg_log_loss",
            "cv": StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=SEED),
            "error_score": "raise",
            "n_jobs": 1 if name == "gaminet" else -1,
        }
        if name == "xgboost":
            return RandomizedSearchCV(
                self._pipeline(name), space, n_iter=XGB_N_ITER, random_state=SEED, **common
            )
        return GridSearchCV(self._pipeline(name), space, **common)

    def _pipeline(self, name: str) -> Pipeline:
        encoder = ("encoder", SpecialCodeEncoder(self.indicators))
        # Model-level n_jobs=1 for XGBoost and EBM: the search already runs folds in parallel.
        if name == "logistic_regression":
            model = LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=SEED)
            return Pipeline([encoder, ("scaler", StandardScaler()), (MODEL_STEP, model)])
        if name == "xgboost":
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=SEED,
                n_jobs=1,
            )
        elif name == "ebm":
            model = ExplainableBoostingClassifier(random_state=SEED, n_jobs=1)
        else:
            # n_jobs=1: with more threads it segfaults because torch, scikit-learn and xgboost
            # load different OpenMP runtimes. No scaler: GAMI-Net normalizes inputs internally.
            model = GAMINetClassifier(n_jobs=1, random_state=SEED)
        return Pipeline([encoder, (MODEL_STEP, model)])

    def _save_table(self, models: list[TrainedModel]) -> Path:
        table = pd.DataFrame(
            [
                {
                    "modelo": MODEL_DISPLAY_NAMES[m.name],
                    "mejores_hiperparametros": json.dumps(m.best_params),
                    "auc_cv_media": round(m.cv_roc_auc_mean, 4),
                    "auc_cv_desv": round(m.cv_roc_auc_std, 4),
                    "logloss_cv_media": round(m.cv_log_loss_mean, 4),
                    "logloss_cv_desv": round(m.cv_log_loss_std, 4),
                    "tiempo_busqueda_s": round(m.search_time_s, 1),
                }
                for m in models
            ]
        )
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        path = self.tables_dir / "tabla_ajuste_hiperparametros.csv"
        table.to_csv(path, index=False)
        logger.info("Saved %s", path.relative_to(PROJECT_ROOT))
        return path
