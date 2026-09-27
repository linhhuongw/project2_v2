"""Split every dataset and run tier B.0 column screening (configs/preprocessing_v1.md).

  * cross-sectional datasets: StratifiedGroupKFold(5, shuffle, seed 42), group = hash of all
    feature values, so identical rows never sit on both sides of a split; validation = first
    fold of an inner StratifiedGroupKFold(5) on the training part (~20%, same groups)
  * lending_club_B: one out-of-time split on issue_d (train 2013, val 2014, test 2015), no shuffling

Tier B.0 (fit on the training part of each fold only): drop columns that are 100% missing or
constant in train; keep but flag columns whose missing rate differs by > 30 points between train
and validation/test. Tier B itself (imputation, encoding, scaling) is NOT run here.

Output, per dataset, under data/splits/<name>/:
  data.parquet          prepared dataset (all rows that take part in the split, tier A applied)
  folds.parquet         row_id (+ group) + role_fold<k> ∈ {train, val, test, unused}
  fold<k>/{train,val,test}.parquet   materialised subsets (tier A columns, before B.0)
  fold<k>/screening.json             B.0 result: dropped columns, kept num/cat lists, warnings
  meta.json             method, sizes and bad rates per fold, column declarations, B.0 summary

Run from the project root:  python -m src.data.make_splits [name ...]
"""

import argparse
import json
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.datasets import ROOT, SPECS, DatasetSpec, categorical_columns

SPLITS = ROOT / "data" / "splits"
N_FOLDS = 5
SEED = 42
MISSING_SHIFT_THR = 0.30


# ------------------------------------------------------------------------------ splitting

def feature_groups(df: pd.DataFrame, features: list) -> np.ndarray:
    """Group id = hash of all feature values (rows with identical features share a group)."""
    h = pd.util.hash_pandas_object(df[features], index=False).to_numpy()
    return pd.factorize(h)[0]


def group_kfold_roles(df: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
    y = df["target"].to_numpy()
    roles = pd.DataFrame({"row_id": df["row_id"].to_numpy(), "group": groups})
    outer = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr_idx, te_idx) in enumerate(outer.split(np.zeros(len(y)), y, groups), start=1):
        inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fit_rel, val_rel = next(inner.split(np.zeros(len(tr_idx)), y[tr_idx], groups[tr_idx]))
        role = np.empty(len(y), dtype=object)
        role[tr_idx[fit_rel]], role[tr_idx[val_rel]], role[te_idx] = "train", "val", "test"
        roles[f"role_fold{k}"] = role
    return roles


def time_holdout_roles(df: pd.DataFrame, time_col: str, periods: dict) -> pd.DataFrame:
    year = df[time_col].dt.year.to_numpy()
    role = np.full(len(df), "unused", dtype=object)
    for part in ("train", "val", "test"):
        lo, hi = periods[part]
        role[(year >= lo) & (year <= hi)] = part
    return pd.DataFrame({"row_id": df["row_id"].to_numpy(), "role_fold1": role})


def n_folds(roles: pd.DataFrame) -> int:
    return sum(c.startswith("role_fold") for c in roles.columns)


def check(roles: pd.DataFrame, df: pd.DataFrame, spec: DatasetSpec):
    tests = []
    for k in range(1, n_folds(roles) + 1):
        r = roles[f"role_fold{k}"]
        assert {"train", "val", "test"} <= set(r), f"fold {k} missing a role"
        tests.append(set(roles.loc[r == "test", "row_id"]))
        if "group" in roles:
            g = {part: set(roles.loc[r == part, "group"]) for part in ("train", "val", "test")}
            assert not (g["train"] & g["test"] or g["val"] & g["test"] or g["train"] & g["val"]), \
                f"fold {k}: a duplicate group is split across parts"
        if spec.method == "time_holdout":
            t = df.set_index("row_id")[spec.time_col]
            tr, va, te = (t.loc[roles.loc[r == x, "row_id"]] for x in ("train", "val", "test"))
            assert tr.max() < va.min() and va.max() < te.min(), f"fold {k}: time order violated"
    if spec.method == "stratified_group_kfold":
        assert set().union(*tests) == set(df["row_id"]), "test folds do not cover all rows"
        assert sum(map(len, tests)) == len(df), "test folds overlap"


# ------------------------------------------------------------------------------ tier B.0

def screen_columns(X_train: pd.DataFrame, num_cols: list, cat_cols: list) -> dict:
    miss = X_train.isna().mean()
    dropped = {c: "100% missing in train" for c in miss.index[miss == 1.0]}
    for c in num_cols:
        if c not in dropped and X_train[c].nunique(dropna=True) <= 1:
            dropped[c] = "constant numeric in train"
    for c in cat_cols:
        if c not in dropped and X_train[c].nunique(dropna=False) <= 1:
            dropped[c] = "constant categorical in train (NaN counted as a level)"
    return dict(sorted(dropped.items()))


def missing_shift(X_train: pd.DataFrame, X_other: pd.DataFrame, thr: float = MISSING_SHIFT_THR) -> dict:
    tr, ot = X_train.isna().mean(), X_other.isna().mean()
    diff = (ot - tr)
    cols = diff[diff.abs() > thr].sort_values(key=abs, ascending=False).index
    return {c: {"train_missing_pct": round(100 * float(tr[c]), 1),
                "other_missing_pct": round(100 * float(ot[c]), 1)} for c in cols}


# ------------------------------------------------------------------------------ main

def fold_stats(roles, df, spec) -> list:
    y = df.set_index("row_id")["target"]
    t = df.set_index("row_id")[spec.time_col] if spec.time_col else None
    out = []
    for k in range(1, n_folds(roles) + 1):
        r = roles.set_index("row_id")[f"role_fold{k}"]
        entry = {"fold": k}
        for part in ("train", "val", "test"):
            ids = r.index[r == part]
            entry[part] = {"n": int(len(ids)), "bad_rate_pct": round(100 * float(y.loc[ids].mean()), 2)}
            if t is not None:
                entry[part]["period"] = [t.loc[ids].min().strftime("%Y-%m"), t.loc[ids].max().strftime("%Y-%m")]
        out.append(entry)
    return out


def make(spec: DatasetSpec):
    df = spec.load()
    features = [c for c in df.columns if c not in {"row_id", "target", *spec.non_features}]
    categorical = categorical_columns(spec, df)
    absent = set(categorical) - set(features)
    assert not absent, f"declared categorical but absent: {absent}"
    numeric = [c for c in features if c not in categorical]
    undeclared = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
    assert not undeclared, f"string columns not declared categorical: {undeclared}"

    if spec.method == "time_holdout":
        periods = spec.periods()
        roles = time_holdout_roles(df, spec.time_col, periods)
        dup_rows = None
    else:
        periods = None
        groups = feature_groups(df, features)
        roles = group_kfold_roles(df, groups)
        sizes = np.bincount(groups)
        dup_rows = int(sizes[sizes > 1].sum())
    check(roles, df, spec)

    out = SPLITS / spec.name
    if out.exists():
        shutil.rmtree(out)  # overwrite: never mix files from an older split
    out.mkdir(parents=True)
    df.to_parquet(out / "data.parquet", index=False)
    roles.to_parquet(out / "folds.parquet", index=False)

    indexed = df.set_index("row_id", drop=False)
    screening = []
    for k in range(1, n_folds(roles) + 1):
        fdir = out / f"fold{k}"
        fdir.mkdir()
        r = roles.set_index("row_id")[f"role_fold{k}"]
        parts = {p: indexed.loc[r.index[r == p]] for p in ("train", "val", "test")}
        for p, d in parts.items():
            d.to_parquet(fdir / f"{p}.parquet", index=False)

        dropped = screen_columns(parts["train"][features], numeric, categorical)  # B.0: train only
        kept = [c for c in features if c not in dropped]
        sc = {
            "fold": k,
            "dropped": dropped,
            "numeric_kept": [c for c in numeric if c not in dropped],
            "categorical_kept": [c for c in categorical if c not in dropped],
            "missing_shift_warnings": {
                "val": missing_shift(parts["train"][kept], parts["val"][kept]),
                "test": missing_shift(parts["train"][kept], parts["test"][kept]),
            },
        }
        (fdir / "screening.json").write_text(json.dumps(sc, ensure_ascii=False, indent=2))
        screening.append(sc)

    kfold = spec.method == "stratified_group_kfold"
    meta = {
        "name": spec.name,
        "method": spec.method,
        "n_folds": n_folds(roles),
        "validation": "first fold of inner StratifiedGroupKFold(5) (~20% of train part)" if kfold else "one year",
        "seed": SEED if kfold else None,
        "periods": periods,
        "rows": int(len(df)),
        "rows_in_duplicate_groups": dup_rows,
        "bad_rate_pct": round(100 * float(df["target"].mean()), 2),
        "key": "row_id",
        "target": "target",
        "non_features": spec.non_features,
        "features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "preparation_notes": spec.notes,
        "folds": fold_stats(roles, df, spec),
        "b0_dropped_per_fold": {s["fold"]: s["dropped"] for s in screening},
        "b0_missing_shift_warnings_per_fold": {
            s["fold"]: sorted(set(s["missing_shift_warnings"]["val"]) | set(s["missing_shift_warnings"]["test"]))
            for s in screening},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    extra = f"duplicate-group rows {dup_rows}" if dup_rows is not None else f"periods {periods}"
    print(f"[{spec.name}] {len(df):,} rows, {len(numeric)} num + {len(categorical)} cat, "
          f"bad {meta['bad_rate_pct']}%, {extra}")
    for f, s in zip(meta["folds"], screening):
        print("   fold {k}: train {a:>9,} val {b:>8,} test {c:>8,} | bad {ta}/{va}/{te}% | "
              "B.0 drop {nd}, shift warn {nw}".format(
                  k=f["fold"], a=f["train"]["n"], b=f["val"]["n"], c=f["test"]["n"],
                  ta=f["train"]["bad_rate_pct"], va=f["val"]["bad_rate_pct"], te=f["test"]["bad_rate_pct"],
                  nd=len(s["dropped"]), nw=len(meta["b0_missing_shift_warnings_per_fold"][f["fold"]])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help=f"datasets (default: all): {', '.join(SPECS)}")
    args = p.parse_args(argv)
    unknown = set(args.names) - set(SPECS)
    if unknown:
        p.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    for name in args.names or SPECS:
        make(SPECS[name])


if __name__ == "__main__":
    sys.exit(main())
