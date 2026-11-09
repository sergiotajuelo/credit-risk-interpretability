import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.config import (
    CALIBRATION_SIZE,
    EXPECTED_N_FEATURES,
    EXPECTED_N_INDICATORS,
    EXPECTED_N_ROWS,
    FEATURES,
    OPENML_DATA_ID,
    OUT_OF_RANGE_FEATURE,
    OUT_OF_RANGE_VALUE,
    POSITIVE_CLASS,
    SEED,
    SPECIAL_CODES,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_cal: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_cal: pd.Series
    y_test: pd.Series


class HelocDataset:
    """FICO HELOC with integer features and target 1 = Bad, 0 = Good."""

    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = X
        self.y = y

    @classmethod
    def from_openml(cls) -> "HelocDataset":
        logger.info("Loading HELOC from OpenML (data_id=%d)", OPENML_DATA_ID)
        bunch = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
        X = bunch.data[FEATURES].astype(int)
        y = (bunch.target == POSITIVE_CLASS).astype(int)
        assert X.shape == (EXPECTED_N_ROWS, EXPECTED_N_FEATURES)
        logger.info("Loaded %d rows x %d features, default rate %.1f%%", *X.shape, 100 * y.mean())
        return cls(X, y)

    def split(self) -> DataSplit:
        """Stratified 60/20/20 split into train, calibration and test."""
        X_rest, X_test, y_rest, y_test = train_test_split(
            self.X, self.y, test_size=TEST_SIZE, stratify=self.y, random_state=SEED
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_rest, y_rest, test_size=CALIBRATION_SIZE, stratify=y_rest, random_state=SEED
        )
        logger.info("Split into train, calibration and test (stratified, seed %d):", SEED)
        for name, part in [("train", y_train), ("calibration", y_cal), ("test", y_test)]:
            logger.info(
                "  %-11s %5d rows (%.0f%%), default rate %.1f%%",
                name,
                len(part),
                100 * len(part) / len(self.y),
                100 * part.mean(),
            )
        return DataSplit(X_train, X_cal, X_test, y_train, y_cal, y_test)

    def special_value_indicators(self) -> list[tuple[str, int]]:
        """(feature, value) pairs that get an indicator column.

        Computed on the full dataset so that no indicator disappears when its code
        does not fall in the training set.
        """
        pairs = [
            (feature, code)
            for feature in FEATURES
            for code in SPECIAL_CODES
            if (self.X[feature] == code).any()
        ]
        pairs.append((OUT_OF_RANGE_FEATURE, OUT_OF_RANGE_VALUE))
        assert len(pairs) == EXPECTED_N_INDICATORS
        logger.info("Found %d special-value indicators", len(pairs))
        return pairs
