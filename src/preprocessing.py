import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.config import OUT_OF_RANGE_FEATURE, OUT_OF_RANGE_VALUE, SPECIAL_CODES

FEATURE_SEPARATOR = "__"


def invalid_values(feature: str) -> list[int]:
    if feature == OUT_OF_RANGE_FEATURE:
        return [*SPECIAL_CODES, OUT_OF_RANGE_VALUE]
    return list(SPECIAL_CODES)


def indicator_name(feature: str, value: int) -> str:
    suffix = f"code_m{-value}" if value in SPECIAL_CODES else f"value_{value}"
    return f"{feature}{FEATURE_SEPARATOR}{suffix}"


def original_feature(name: str) -> str:
    return name.split(FEATURE_SEPARATOR)[0]


class SpecialCodeEncoder(BaseEstimator, TransformerMixin):
    """Adds a 0/1 indicator per (feature, special value) pair and replaces those values
    with the median of the valid values seen in fit."""

    def __init__(self, indicators: list[tuple[str, int]]) -> None:
        self.indicators = indicators

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SpecialCodeEncoder":
        features = dict.fromkeys(feature for feature, _ in self.indicators)
        self.medians_ = {
            feature: X.loc[~X[feature].isin(invalid_values(feature)), feature].median()
            for feature in features
        }
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        indicators = pd.DataFrame(
            {
                indicator_name(feature, value): (X[feature] == value).astype(int)
                for feature, value in self.indicators
            },
            index=X.index,
        )
        X_out = X.copy()
        for feature, median in self.medians_.items():
            X_out[feature] = X_out[feature].mask(
                X_out[feature].isin(invalid_values(feature)), median
            )
        return pd.concat([X_out, indicators], axis=1)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        check_is_fitted(self)
        names = [*self.feature_names_in_, *(indicator_name(f, v) for f, v in self.indicators)]
        return np.array(names, dtype=object)
