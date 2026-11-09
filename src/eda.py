import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from src.config import (
    EDA_FIGURES_DIR,
    EDA_HIGH_CORRELATION_THRESHOLD,
    EDA_MAX_DISTINCT_FOR_VALUE_BINS,
    EDA_MONOTONICITY_RHO_THRESHOLD,
    EDA_N_QUANTILE_BINS,
    EDA_TABLES_DIR,
    EXPECTED_DIRECTION,
    FEATURES,
    FIGURE_DPI,
    PROJECT_ROOT,
)
from src.preprocessing import invalid_values

DIRECTION_LABELS = {
    "increasing": "creciente",
    "decreasing": "decreciente",
    "none": "sin restricción",
}
UNCLEAR_DIRECTION = "no clara"
GRID_COLUMNS = 5

logger = logging.getLogger(__name__)


class ExploratoryAnalysis:
    """EDA on the raw (not encoded) training set. Special codes are treated as missing in
    the figures and summarised separately in a table."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        figures_dir: Path = EDA_FIGURES_DIR,
        tables_dir: Path = EDA_TABLES_DIR,
    ) -> None:
        self.X = X
        self.y = y
        self.X_valid = pd.DataFrame(
            {feature: X[feature].mask(X[feature].isin(invalid_values(feature))) for feature in X}
        )
        self.figures_dir = figures_dir
        self.tables_dir = tables_dir

    def default_rate_by_bin(self, feature: str) -> pd.Series:
        values = self.X_valid[feature].dropna()
        if values.nunique() <= EDA_MAX_DISTINCT_FOR_VALUE_BINS:
            bins = values.astype(int)
        else:
            bins = pd.qcut(values, EDA_N_QUANTILE_BINS, duplicates="drop")
        return self.y.loc[values.index].groupby(bins, observed=True).mean().sort_index()

    def plot_default_rate_by_feature(self) -> Path:
        n_rows = math.ceil(len(FEATURES) / GRID_COLUMNS)
        fig, axes = plt.subplots(n_rows, GRID_COLUMNS, figsize=(4.2 * GRID_COLUMNS, 3.6 * n_rows))
        overall_rate = self.y.mean()
        for ax, feature in zip(axes.flat, FEATURES, strict=False):
            rates = self.default_rate_by_bin(feature)
            labels = [_bin_label(b) for b in rates.index]
            ax.bar(range(len(rates)), rates.to_numpy(), color="#4C72B0")
            ax.axhline(
                overall_rate, color="#C44E52", linestyle="--", linewidth=1, label="Tasa global"
            )
            ax.set_xticks(range(len(rates)))
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_title(
                f"{feature}\n({DIRECTION_LABELS[EXPECTED_DIRECTION[feature]]})", fontsize=9
            )
            ax.set_ylabel("Tasa de impago", fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
        for ax in axes.flat[len(FEATURES) :]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=7, loc="upper right")
        fig.suptitle("Tasa de impago por tramo de cada variable (conjunto de entrenamiento)")
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        return self._save_figure(fig, "fig_tasa_impago_por_variable.png")

    def monotonicity_table(self) -> pd.DataFrame:
        rows = []
        for feature in FEATURES:
            rates = self.default_rate_by_bin(feature)
            rho = spearmanr(range(len(rates)), rates.to_numpy()).correlation
            observed = _observed_direction(rho)
            expected = DIRECTION_LABELS[EXPECTED_DIRECTION[feature]]
            matches = "" if EXPECTED_DIRECTION[feature] == "none" else observed == expected
            rows.append(
                {
                    "variable": feature,
                    "n_tramos": len(rates),
                    "rho_spearman": round(rho, 3),
                    "direccion_observada": observed,
                    "direccion_esperada": expected,
                    "coincide": matches,
                }
            )
        table = pd.DataFrame(rows)
        mismatches = table[table["coincide"].eq(False)]
        logger.info(
            "Monotonicity: %d of %d constrained features match the expected direction",
            int(table["coincide"].eq(True).sum()),
            int(table["coincide"].ne("").sum()),
        )
        for row in mismatches.itertuples():
            logger.warning(
                "  %s: observed %s (rho=%.2f), expected %s",
                row.variable,
                row.direccion_observada,
                row.rho_spearman,
                row.direccion_esperada,
            )
        self._save_table(table, "tabla_monotonia_observada.csv")
        return table

    def plot_correlation(self) -> Path:
        corr = self.X_valid[FEATURES].corr(method="spearman")
        logger.info("Feature pairs with |Spearman rho| > %.1f:", EDA_HIGH_CORRELATION_THRESHOLD)
        for i, first in enumerate(FEATURES):
            for second in FEATURES[i + 1 :]:
                if abs(corr.loc[first, second]) > EDA_HIGH_CORRELATION_THRESHOLD:
                    logger.info("  %s ~ %s: %.2f", first, second, corr.loc[first, second])
        fig, ax = plt.subplots(figsize=(13, 11))
        image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(FEATURES)))
        ax.set_yticks(range(len(FEATURES)))
        ax.set_xticklabels(FEATURES, rotation=90, fontsize=8)
        ax.set_yticklabels(FEATURES, fontsize=8)
        for i in range(len(FEATURES)):
            for j in range(len(FEATURES)):
                value = corr.iat[i, j]
                color = "white" if abs(value) > 0.6 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5, color=color)
        fig.colorbar(image, ax=ax, label="Correlación de Spearman")
        ax.set_title(
            "Correlación de Spearman entre variables\n"
            "(entrenamiento, códigos especiales como ausentes)"
        )
        fig.tight_layout()
        return self._save_figure(fig, "fig_correlacion.png")

    def special_codes_table(self) -> pd.DataFrame:
        rows = []
        for feature in FEATURES:
            for code in invalid_values(feature):
                has_code = self.X[feature] == code
                if not has_code.any():
                    continue
                rate_code = self.y[has_code].mean()
                rate_rest = self.y[~has_code].mean()
                rows.append(
                    {
                        "variable": feature,
                        "codigo": code,
                        "n_casos": int(has_code.sum()),
                        "tasa_impago_codigo": round(rate_code, 4),
                        "tasa_impago_resto": round(rate_rest, 4),
                        "diferencia": round(rate_code - rate_rest, 4),
                    }
                )
        table = pd.DataFrame(rows)
        strongest = table.loc[table["diferencia"].abs().idxmax()]
        logger.info(
            "Special codes: %d (feature, code) pairs in train; largest effect %s = %d "
            "(default rate %.1f%% vs %.1f%% for the rest)",
            len(table),
            strongest["variable"],
            strongest["codigo"],
            100 * strongest["tasa_impago_codigo"],
            100 * strongest["tasa_impago_resto"],
        )
        self._save_table(table, "tabla_codigos_especiales.csv")
        return table

    def run(self) -> None:
        logger.info(
            "Exploratory analysis on %d training rows (overall default rate %.1f%%)",
            len(self.y),
            100 * self.y.mean(),
        )
        self.plot_default_rate_by_feature()
        self.monotonicity_table()
        self.plot_correlation()
        self.special_codes_table()

    def _save_figure(self, fig: plt.Figure, name: str) -> Path:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        path = self.figures_dir / name
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path.relative_to(PROJECT_ROOT))
        return path

    def _save_table(self, table: pd.DataFrame, name: str) -> Path:
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        path = self.tables_dir / name
        table.to_csv(path, index=False)
        logger.info("Saved %s", path.relative_to(PROJECT_ROOT))
        return path


def _bin_label(bin_value: int | pd.Interval) -> str:
    """Interval bins hold integers, so (a, b] is shown as the integer range a+1 to b."""
    if not isinstance(bin_value, pd.Interval):
        return str(bin_value)
    lowest = math.floor(bin_value.left) + 1
    highest = math.floor(bin_value.right)
    return str(highest) if lowest == highest else f"{lowest}–{highest}"


def _observed_direction(rho: float) -> str:
    if rho >= EDA_MONOTONICITY_RHO_THRESHOLD:
        return DIRECTION_LABELS["increasing"]
    if rho <= -EDA_MONOTONICITY_RHO_THRESHOLD:
        return DIRECTION_LABELS["decreasing"]
    return UNCLEAR_DIRECTION
