"""Apply tier B.1 to every fold and write model-ready matrices (configs/preprocessing_v1.md).

For each dataset and fold, the preprocessor is fitted on `train` only (columns kept by tier B.0 of
that fold) and applied to train / val / test. Lending Club additionally gets a `refit/` set
(rule 3 of exception 1): preprocessor fitted on train + val, applied to train + val and test.

Output under data/processed/<family>/<name>/fold<k>/:
  train.parquet, val.parquet, test.parquet   row_id, target, then float32 features
  preprocessor.joblib                        fitted ColumnTransformer
  info.json                                  feature names, sizes, unseen-category rates, checks

Run from the project root:  python -m src.features.make_processed [--family classic] [name ...]
"""

import argparse
import json
import shutil
import sys

import joblib
import numpy as np
import pandas as pd

from src.data.datasets import ROOT, SPECS
from src.data.splits import load_fold, load_meta, load_screening
from src.features.preprocess import FAMILIES, build_preprocessor

PROCESSED = ROOT / "data" / "processed"
TIME_SPLIT = {"time_holdout"}


def unseen_rate(train: pd.DataFrame, other: pd.DataFrame, cat_cols: list) -> dict:
    """Share of rows per categorical column whose level never appears in train (-> all-zero one-hot)."""
    out = {}
    for c in cat_cols:
        seen = set(train[c].dropna().astype(str)) | ({"<NA>"} if train[c].isna().any() else set())
        vals = other[c].astype(str).where(other[c].notna(), "<NA>")
        rate = float((~vals.isin(seen)).mean())
        if rate > 0:
            out[c] = round(100 * rate, 3)
    return out


def transform_parts(pre, parts: dict, cols: list) -> dict:
    out = {}
    names = list(pre.get_feature_names_out())
    for p, df in parts.items():
        X = pre.transform(df[cols]).astype(np.float32)
        frame = pd.DataFrame(X, columns=names, index=df.index)
        frame.insert(0, "target", df["target"].astype("int8").to_numpy())
        frame.insert(0, "row_id", df["row_id"].to_numpy())
        out[p] = frame
    return out


def checks(frames: dict, n_num: int) -> dict:
    tr = frames["train"].iloc[:, 2:]
    num = tr.iloc[:, :n_num]
    res = {
        "nan_or_inf": int(sum((~np.isfinite(f.iloc[:, 2:].to_numpy())).sum() for f in frames.values())),
        "train_num_mean_abs_max": round(float(num.mean().abs().max()), 6) if n_num else 0.0,
        "train_num_std_range": [round(float(num.std(ddof=0).min()), 4), round(float(num.std(ddof=0).max()), 4)]
        if n_num else None,
    }
    assert res["nan_or_inf"] == 0, "NaN/inf after preprocessing"
    assert res["train_num_mean_abs_max"] < 1e-3, "train numeric columns are not centred"
    return res


def process_fold(name: str, fold: int, family: str, out_dir, refit: bool = False) -> dict:
    sc = load_screening(name, fold)
    num_cols, cat_cols = sc["numeric_kept"], sc["categorical_kept"]
    cols = num_cols + cat_cols
    train, val, test = load_fold(name, fold)
    if refit:  # Lending Club rule 3: fit on train + val, evaluate on test
        train = pd.concat([train, val])
        parts = {"trainval": train, "test": test}
    else:
        parts = {"train": train, "val": val, "test": test}

    pre = build_preprocessor(num_cols, cat_cols, family).fit(train[cols])   # train only
    frames = transform_parts(pre, parts, cols)
    fit_key = "trainval" if refit else "train"
    chk = checks({"train": frames[fit_key], **{k: v for k, v in frames.items() if k != fit_key}}, len(num_cols))

    out_dir.mkdir(parents=True)
    for p, f in frames.items():
        f.to_parquet(out_dir / f"{p}.parquet", index=False)
    joblib.dump(pre, out_dir / "preprocessor.joblib")
    names = frames[fit_key].columns[2:].tolist()
    info = {
        "dataset": name, "fold": fold, "family": family, "fitted_on": fit_key,
        "n_rows": {p: int(len(f)) for p, f in frames.items()},
        "n_numeric_in": len(num_cols), "n_categorical_in": len(cat_cols),
        "n_features_out": len(names),
        "n_onehot_or_code_columns": len(names) - len(num_cols),
        "features": names,
        "unseen_category_pct": {p: unseen_rate(train, df, cat_cols) for p, df in parts.items() if p != fit_key},
        "checks": chk,
    }
    (out_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def make(name: str, family: str):
    meta = load_meta(name)
    base = PROCESSED / family / name
    if base.exists():
        shutil.rmtree(base)
    infos = [process_fold(name, k, family, base / f"fold{k}") for k in range(1, meta["n_folds"] + 1)]
    if meta["method"] in TIME_SPLIT:
        infos.append(process_fold(name, 1, family, base / "refit", refit=True))
    for i in infos:
        where = "refit" if i["fitted_on"] == "trainval" else f"fold{i['fold']}"
        unseen = {p: len(v) for p, v in i["unseen_category_pct"].items()}
        print(f"  [{name}] {where}: {i['n_numeric_in']} num + {i['n_categorical_in']} cat -> "
              f"{i['n_features_out']} features | rows {i['n_rows']} | "
              f"cat cols with unseen levels {unseen} | train num std {i['checks']['train_num_std_range']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", default="classic", choices=FAMILIES,
                   help="classic = LR/RF/KNN/NB/SVM (one-hot); gbdt = integer codes")
    p.add_argument("names", nargs="*", help=f"datasets (default: all): {', '.join(SPECS)}")
    args = p.parse_args(argv)
    unknown = set(args.names) - set(SPECS)
    if unknown:
        p.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    print(f"family = {args.family}")
    for name in args.names or SPECS:
        make(name, args.family)


if __name__ == "__main__":
    sys.exit(main())
