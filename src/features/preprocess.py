"""Tier B.1 — Baesens preprocessing pipeline (configs/preprocessing_v1.md, sections B.1–B.3).

    numeric:      median imputation -> standardisation (mean 0, variance 1)
    categorical:  cast to string -> constant "__missing__" group -> one-hot (classic) / integer (gbdt)

Every statistic (medians, means, variances, category lists) is learned by `fit` on the training part
of a fold only, then applied to validation and test. No missing-value indicators are added.

impute="random" (leakage control, used for hmeq_blind): every missing value — numeric or categorical —
is replaced by a random draw from that column's observed training values, so a model cannot tell
which values were missing. Used when the missingness pattern itself reveals the label.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

MISSING = "__missing__"
FAMILIES = ("classic", "gbdt")  # classic = LR, RF, KNN, NB, SVM (table B.2)


class RandomDrawImputer(BaseEstimator, TransformerMixin):
    """Fill NaN with random draws from the column's observed training values (seeded)."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.pools_ = [X.iloc[:, j].dropna().to_numpy() for j in range(X.shape[1])]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        rng = np.random.default_rng(self.seed)
        for j, pool in enumerate(self.pools_):
            miss = X.iloc[:, j].isna().to_numpy()
            if miss.any() and len(pool):
                X.iloc[np.flatnonzero(miss), j] = rng.choice(pool, size=int(miss.sum()))
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _to_str(X: pd.DataFrame) -> pd.DataFrame:
    """Cast category codes to strings, keeping NaN so the imputer can put it in its own group."""
    X = pd.DataFrame(X)
    return X.apply(lambda s: s.map(lambda v: v if pd.isna(v) else str(v))).astype(object)


def build_preprocessor(num_cols: list, cat_cols: list, family: str, impute: str = "median") -> ColumnTransformer:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if impute not in ("median", "random"):
        raise ValueError("impute must be 'median' or 'random'")
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median") if impute == "median" else RandomDrawImputer()),
        ("scale", StandardScaler()),
    ])
    cat_encoder = (OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
                   if family == "classic"
                   else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                       dtype=np.float32))
    cat = Pipeline([
        ("to_str", FunctionTransformer(_to_str, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="constant", fill_value=MISSING) if impute == "median"
         else RandomDrawImputer()),
        ("encode", cat_encoder),
    ])
    return ColumnTransformer(
        [("num", num, num_cols), ("cat", cat, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=True,  # num__<col>, cat__<col>_<level>
    )
