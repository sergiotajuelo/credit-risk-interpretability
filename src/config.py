from pathlib import Path

SEED = 42
OPENML_DATA_ID = 45553
TARGET = "RiskPerformance"
POSITIVE_CLASS = "Bad"
SPECIAL_CODES = (-7, -8, -9)

EXPECTED_DIRECTION = {
    "ExternalRiskEstimate": "decreasing",
    "MSinceOldestTradeOpen": "decreasing",
    "MSinceMostRecentTradeOpen": "decreasing",
    "AverageMInFile": "decreasing",
    "NumSatisfactoryTrades": "decreasing",
    "NumTrades60Ever2DerogPubRec": "decreasing",
    "NumTrades90Ever2DerogPubRec": "decreasing",
    "PercentTradesNeverDelq": "decreasing",
    "MSinceMostRecentDelq": "decreasing",
    "MaxDelq2PublicRecLast12M": "decreasing",
    "MaxDelqEver": "decreasing",
    "NumTotalTrades": "none",
    "NumTradesOpeninLast12M": "increasing",
    "PercentInstallTrades": "none",
    "MSinceMostRecentInqexcl7days": "decreasing",
    "NumInqLast6M": "increasing",
    "NumInqLast6Mexcl7days": "increasing",
    "NetFractionRevolvingBurden": "increasing",
    "NetFractionInstallBurden": "increasing",
    "NumRevolvingTradesWBalance": "none",
    "NumInstallTradesWBalance": "none",
    "NumBank2NatlTradesWHighUtilization": "increasing",
    "PercentTradesWBalance": "none",
}

MODIFIABLE = [
    "NetFractionRevolvingBurden",
    "NetFractionInstallBurden",
    "NumRevolvingTradesWBalance",
    "NumInstallTradesWBalance",
    "NumBank2NatlTradesWHighUtilization",
    "PercentTradesWBalance",
]

FEATURES = list(EXPECTED_DIRECTION)

# MaxDelq2PublicRecLast12M == 9 is outside the ordered range 0-7 of the data dictionary,
# so it is handled like a special code.
OUT_OF_RANGE_FEATURE = "MaxDelq2PublicRecLast12M"
OUT_OF_RANGE_VALUE = 9

EXPECTED_N_ROWS = 9871
EXPECTED_N_FEATURES = 23
EXPECTED_N_INDICATORS = 13

TEST_SIZE = 0.20
# Fraction of the remaining 80% that goes to calibration, giving a 60/20/20 split.
CALIBRATION_SIZE = 0.25

EDA_MAX_DISTINCT_FOR_VALUE_BINS = 12
EDA_N_QUANTILE_BINS = 10
EDA_MONOTONICITY_RHO_THRESHOLD = 0.5
EDA_HIGH_CORRELATION_THRESHOLD = 0.8
FIGURE_DPI = 200

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EDA_FIGURES_DIR = RESULTS_DIR / "figures" / "eda"
EDA_TABLES_DIR = RESULTS_DIR / "tables" / "eda"
MODELS_DIR = RESULTS_DIR / "models"
MODELS_TABLES_DIR = RESULTS_DIR / "tables" / "models"

CV_N_SPLITS = 5
XGB_N_ITER = 100
LOGISTIC_MAX_ITER = 2000

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Regresión logística",
    "xgboost": "XGBoost",
    "ebm": "EBM",
    "gaminet": "GAMI-Net",
}

SEARCH_SPACES = {
    "logistic_regression": {"C": [0.001, 0.003, 0.01, 0.03, 0.1, 1]},
    "xgboost": {
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
        "n_estimators": [200, 400, 800, 1200, 1600],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 5, 10, 20],
        "reg_lambda": [0, 1, 5, 10],
    },
    "ebm": {"interactions": [0, 5, 10, 20], "max_bins": [64, 128, 256]},
    # learning_rates rather than reg_clarity: reg_clarity has no effect without interactions.
    "gaminet": {
        "interact_num": [0, 10],
        "learning_rates": [(0.001, 0.001, 0.0001), (0.005, 0.005, 0.0005)],
    },
}
