"""Tier B.1 — Baesens preprocessing pipeline (configs/preprocessing_v1.md, sections B.1–B.3).

    numeric:      median imputation -> standardisation (mean 0, variance 1)
    categorical:  cast to string -> constant "__missing__" group -> one-hot (classic) / integer (gbdt)

Every statistic (medians, means, variances, category lists) is learned by `fit` on the training part
of a fold only, then applied to validation and test. No missing-value indicators are added.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

MISSING = "__missing__"
FAMILIES = ("classic", "gbdt")  # classic = LR, RF, KNN, NB, SVM (table B.2)


def _to_str(X: pd.DataFrame) -> pd.DataFrame:
    """Cast category codes to strings, keeping NaN so the imputer can put it in its own group."""
    X = pd.DataFrame(X)
    return X.apply(lambda s: s.map(lambda v: v if pd.isna(v) else str(v))).astype(object)


def build_preprocessor(num_cols: list, cat_cols: list, family: str) -> ColumnTransformer:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_encoder = (OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
                   if family == "classic"
                   else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                       dtype=np.float32))
    cat = Pipeline([
        ("to_str", FunctionTransformer(_to_str, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="constant", fill_value=MISSING)),
        ("encode", cat_encoder),
    ])
    return ColumnTransformer(
        [("num", num, num_cols), ("cat", cat, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=True,  # num__<col>, cat__<col>_<level>
    )
