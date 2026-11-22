import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import brier_score_loss, f1_score, log_loss, roc_auc_score

from src.config import (
    CI_PERCENTILES,
    DECISION_THRESHOLD,
    ECE_N_BINS,
    EVALUATION_FIGURES_DIR,
    EVALUATION_TABLES_DIR,
    FIGURE_DPI,
    MODEL_DISPLAY_NAMES,
    N_BOOTSTRAP,
    PROJECT_ROOT,
    REFERENCE_MODEL,
    SEED,
)
from src.data import DataSplit
from src.models import TrainedModel

METRICS = ["AUC", "Gini", "F1", "Brier", "Log-loss", "ECE"]
FORMATTED_DECIMALS = 3
CSV_FLOAT_FORMAT = "%.6f"
GRID_SHAPE = (2, 2)
HISTOGRAM_BOX_ASPECT = 0.3

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    metrics: pd.DataFrame
    differences: pd.DataFrame


def calibration_bins(
    y_true: np.ndarray, proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per equal-width bin of [0, 1]: number of cases, sum of predicted probabilities and
    number of observed defaults."""
    inner_edges = np.linspace(0, 1, ECE_N_BINS + 1)[1:-1]
    bins = np.digitize(proba, inner_edges)
    counts = np.bincount(bins, minlength=ECE_N_BINS)
    predicted = np.bincount(bins, weights=proba, minlength=ECE_N_BINS)
    observed = np.bincount(bins, weights=y_true, minlength=ECE_N_BINS)
    return counts, predicted, observed


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray) -> float:
    # (n_bin / n) * |observed rate - mean predicted| simplifies to |observed - predicted| / n
    # with the per-bin sums, so empty bins add nothing.
    _, predicted, observed = calibration_bins(y_true, proba)
    return float(np.abs(observed - predicted).sum() / len(y_true))


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y_true, proba)
    return {
        "AUC": auc,
        "Gini": 2 * auc - 1,
        "F1": f1_score(y_true, (proba >= DECISION_THRESHOLD).astype(int)),
        "Brier": brier_score_loss(y_true, proba),
        "Log-loss": log_loss(y_true, proba, labels=[0, 1]),
        "ECE": expected_calibration_error(y_true, proba),
    }


class ModelEvaluator:
    """Evaluates the frozen models on the test set only, with paired bootstrap confidence
    intervals. The positive class is Bad (y = 1)."""

    def __init__(
        self,
        models: list[TrainedModel],
        split: DataSplit,
        figures_dir: Path = EVALUATION_FIGURES_DIR,
        tables_dir: Path = EVALUATION_TABLES_DIR,
    ) -> None:
        #A GAMI-Net loaded from disk predicts with torch's default thread count and then hangs
        # or segfaults (the OpenMP conflict that forces n_jobs=1 in training). torch is imported
        # here because importing it before xgboost makes loading the XGBoost model segfault.
        import torch

        torch.set_num_threads(1)
        self.y_test = split.y_test.to_numpy()
        self.probabilities = {
            model.name: np.asarray(model.estimator.predict_proba(split.X_test))[:, 1]
            for model in models
        }
        self.figures_dir = figures_dir
        self.tables_dir = tables_dir

    def run(self) -> EvaluationResult:
        logger.info(
            "Evaluating %d models on %d test rows (default rate %.1f%%)",
            len(self.probabilities),
            len(self.y_test),
            100 * self.y_test.mean(),
        )
        point = pd.DataFrame(
            {
                name: compute_metrics(self.y_test, proba)
                for name, proba in self.probabilities.items()
            }
        ).T[METRICS]
        samples = self._bootstrap()
        result = EvaluationResult(
            metrics=self._metrics_table(point, samples),
            differences=self._differences_table(point, samples),
        )
        self._save_table(result.metrics, "tabla_metricas_o1.csv")
        self._save_table(self._formatted_table(result.metrics), "tabla_metricas_o1_formateada.csv")
        self._save_table(result.differences, "tabla_diferencias_xgboost.csv")
        self.plot_reliability_diagrams(point["ECE"])
        return result

    def _bootstrap(self) -> dict[str, pd.DataFrame]:
        """Metrics of every model on the same resampled test sets, which makes the
        comparison between models paired."""
        rng = np.random.default_rng(SEED)
        n = len(self.y_test)
        indices = rng.integers(0, n, size=(N_BOOTSTRAP, n))
        logger.info("Paired bootstrap: %d resamples of %d rows (seed %d)", N_BOOTSTRAP, n, SEED)
        return {
            name: pd.DataFrame([compute_metrics(self.y_test[idx], proba[idx]) for idx in indices])[
                METRICS
            ]
            for name, proba in self.probabilities.items()
        }

    def _metrics_table(self, point: pd.DataFrame, samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows = []
        for name in self.probabilities:
            lower, upper = np.percentile(samples[name], CI_PERCENTILES, axis=0)
            for i, metric in enumerate(METRICS):
                rows.append(
                    {
                        "modelo": MODEL_DISPLAY_NAMES[name],
                        "metrica": metric,
                        "valor": point.loc[name, metric],
                        "ic_inf": lower[i],
                        "ic_sup": upper[i],
                    }
                )
        table = pd.DataFrame(rows)
        for name in self.probabilities:
            logger.info(
                "  %-20s %s",
                MODEL_DISPLAY_NAMES[name],
                "  ".join(f"{metric} {point.loc[name, metric]:.4f}" for metric in METRICS),
            )
        return table

    def _differences_table(
        self, point: pd.DataFrame, samples: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        rows = []
        for name in self.probabilities:
            if name == REFERENCE_MODEL:
                continue
            differences = samples[name] - samples[REFERENCE_MODEL]
            lower, upper = np.percentile(differences, CI_PERCENTILES, axis=0)
            for i, metric in enumerate(METRICS):
                significant = lower[i] > 0 or upper[i] < 0
                rows.append(
                    {
                        "modelo": MODEL_DISPLAY_NAMES[name],
                        "metrica": metric,
                        "diferencia": point.loc[name, metric] - point.loc[REFERENCE_MODEL, metric],
                        "ic_inf": lower[i],
                        "ic_sup": upper[i],
                        "significativa": "Sí" if significant else "No",
                    }
                )
        table = pd.DataFrame(rows)
        significant = table[table["significativa"] == "Sí"]
        logger.info(
            "Differences from %s: %d of %d are significant",
            MODEL_DISPLAY_NAMES[REFERENCE_MODEL],
            len(significant),
            len(table),
        )
        for row in significant.itertuples():
            logger.info(
                "  %s, %s: %+.4f [%+.4f; %+.4f]",
                row.modelo,
                row.metrica,
                row.diferencia,
                row.ic_inf,
                row.ic_sup,
            )
        return table

    def _formatted_table(self, metrics: pd.DataFrame) -> pd.DataFrame:
        text = metrics.apply(
            lambda row: (
                f"{_spanish(row['valor'])} [{_spanish(row['ic_inf'])}; {_spanish(row['ic_sup'])}]"
            ),
            axis=1,
        )
        wide = metrics.assign(texto=text).pivot(index="modelo", columns="metrica", values="texto")
        order = [MODEL_DISPLAY_NAMES[name] for name in self.probabilities]
        return (
            wide.loc[order, METRICS].rename_axis("Modelo").rename_axis(None, axis=1).reset_index()
        )

    def plot_reliability_diagrams(self, ece: pd.Series) -> Path:
        n_rows, n_cols = GRID_SHAPE
        fig = plt.figure(figsize=(5.5 * n_cols, 7 * n_rows))
        outer = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.25)
        bin_width = 1 / ECE_N_BINS
        centers = np.arange(ECE_N_BINS) * bin_width + bin_width / 2
        comma = FuncFormatter(lambda value, _: _spanish(value, 1))
        for cell, (name, proba) in zip(outer, self.probabilities.items(), strict=False):
            counts, predicted, observed = calibration_bins(self.y_test, proba)
            filled = counts > 0
            inner = cell.subgridspec(2, 1, height_ratios=[1, HISTOGRAM_BOX_ASPECT], hspace=0.05)
            ax = fig.add_subplot(inner[0])
            ax_hist = fig.add_subplot(inner[1], sharex=ax)

            ax.plot([0, 1], [0, 1], color="#C44E52", linestyle="--", label="Calibración perfecta")
            ax.plot(
                predicted[filled] / counts[filled],
                observed[filled] / counts[filled],
                color="#4C72B0",
                marker="o",
                label="Modelo",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_box_aspect(1)
            ax.set_ylabel("Proporción observada de impagos")
            ax.set_title(f"{MODEL_DISPLAY_NAMES[name]} (ECE = {_spanish(ece[name])})")
            ax.yaxis.set_major_formatter(comma)
            ax.tick_params(labelbottom=False)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)

            ax_hist.bar(centers, counts, width=bin_width, color="#4C72B0", edgecolor="white")
            ax_hist.set_box_aspect(HISTOGRAM_BOX_ASPECT)
            ax_hist.set_xlabel("Probabilidad predicha de impago")
            ax_hist.set_ylabel("Solicitudes")
            ax_hist.xaxis.set_major_formatter(comma)
        fig.suptitle(
            "Diagramas de fiabilidad en el conjunto de prueba "
            f"({ECE_N_BINS} tramos de igual anchura)",
            y=0.93,
        )
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        path = self.figures_dir / "fig_diagramas_fiabilidad.png"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path.relative_to(PROJECT_ROOT))
        return path

    def _save_table(self, table: pd.DataFrame, name: str) -> Path:
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        path = self.tables_dir / name
        table.to_csv(path, index=False, float_format=CSV_FLOAT_FORMAT)
        logger.info("Saved %s", path.relative_to(PROJECT_ROOT))
        return path


def _spanish(value: float, decimals: int = FORMATTED_DECIMALS) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")
