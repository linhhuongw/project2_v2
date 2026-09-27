"""Load the splits written by src.data.make_splits.

    from src.data.splits import load_fold, xy
    train, val, test = load_fold("taiwan_default_credit", 1)
    X_train, y_train = xy(train, "taiwan_default_credit", 1)   # columns kept by tier B.0 of fold 1

Lending Club ("lending_club_B") has a single out-of-time fold (1).
"""

import json
from functools import lru_cache

import pandas as pd

from src.data.datasets import ROOT

SPLITS = ROOT / "data" / "splits"


@lru_cache(maxsize=None)
def load_meta(name: str) -> dict:
    return json.loads((SPLITS / name / "meta.json").read_text())


@lru_cache(maxsize=None)
def load_screening(name: str, fold: int) -> dict:
    """Tier B.0 result for one fold: dropped columns, numeric_kept, categorical_kept, warnings."""
    return json.loads((SPLITS / name / f"fold{fold}" / "screening.json").read_text())


def load_fold(name: str, fold: int, columns=None):
    """Return (train, val, test) DataFrames for a fold (tier A columns, before B.0)."""
    d = SPLITS / name / f"fold{fold}"
    return tuple(pd.read_parquet(d / f"{part}.parquet", columns=columns) for part in ("train", "val", "test"))


def xy(df: pd.DataFrame, name: str, fold: int):
    """Features kept by tier B.0 for this fold (numeric first, then categorical) and the label."""
    sc = load_screening(name, fold)
    return df[sc["numeric_kept"] + sc["categorical_kept"]], df[load_meta(name)["target"]]
