"""Leakage and overfitting checks for every dataset (fold 1 of each split, test parts only).

Leakage
  1. split leakage — median distance from test rows to their nearest training row vs. from training
     rows to their nearest other training row (same scale => test rows are not near-copies), and how
     often the nearest training row has the same label
  2. feature leakage — best single-feature test AUC (LightGBM on that feature alone, NaN kept)
  3. missingness leakage — test AUC of a model that sees only which values are missing
  4. detectability after tier B.1 — can a model recover "this row had missing values" from the
     preprocessed matrix? (checks whether a leakage control such as hmeq_blind works)
Overfitting
  5. AUC gaps train − test and validation − test for every model run (from reports/models/<run>/)

Output under reports/models/validity/: leakage.csv, overfit.csv, tables.md

Nearest neighbours are computed with plain numpy (chunked brute force): scikit-learn's multi-threaded
neighbour search and LightGBM load two different OpenMP runtimes, which crashes or deadlocks on macOS.
Each dataset also runs in its own Python process.

Run from the project root:  python -m src.models.validity_checks [name ...]
"""

import argparse
import json
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.data.datasets import ROOT, SPECS
from src.data.splits import load_fold, load_screening
from src.models.compare_models import NAMES, RUNS
from src.models.report_classic import LABELS

OUT = ROOT / "reports" / "models" / "validity"
PROCESSED = ROOT / "data" / "processed"
SEED = 42
MAX_TEST, MAX_TRAIN = 3_000, 60_000


def as_lgbm_frame(df, num, cat, cats=None):
    X = df[num + cat].copy()
    cats = cats or {c: pd.CategoricalDtype(sorted(df[c].dropna().astype(str).unique())) for c in cat}
    for c in cat:
        X[c] = X[c].astype(str).where(X[c].notna()).astype(cats[c])
    return X, cats


def nearest(train: np.ndarray, query: np.ndarray, k: int = 1, chunk: int = 256):
    """Euclidean k nearest training rows for each query row: (distances, indices), numpy only."""
    tr_sq = (train ** 2).sum(axis=1)
    dist, idx = [], []
    for i in range(0, len(query), chunk):
        q = query[i:i + chunk]
        d2 = (q ** 2).sum(axis=1)[:, None] + tr_sq[None, :] - 2 * q @ train.T
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(d2, part, axis=1).argsort(axis=1)
        part = np.take_along_axis(part, order, axis=1)
        idx.append(part)
        dist.append(np.sqrt(np.maximum(np.take_along_axis(d2, part, axis=1), 0)))
    return np.vstack(dist), np.vstack(idx)


def auc_sym(y, p):
    a = roc_auc_score(y, p)
    return max(a, 1 - a)


def leakage_row(name: str) -> dict:
    sc = load_screening(name, 1)
    num, cat = sc["numeric_kept"], sc["categorical_kept"]
    train, _, test = load_fold(name, 1)
    row = {"dataset": name}

    # 1. split leakage, on the GBDT matrix (standardised numbers + integer codes)
    tr = pd.read_parquet(PROCESSED / "gbdt" / name / "fold1" / "train.parquet")
    te = pd.read_parquet(PROCESSED / "gbdt" / name / "fold1" / "test.parquet")
    te = te.sample(min(MAX_TEST, len(te)), random_state=SEED)
    tr = tr.sample(min(MAX_TRAIN, len(tr)), random_state=SEED)
    Xtr_nn = tr.iloc[:, 2:].to_numpy(np.float64)
    d_te, i_te = nearest(Xtr_nn, te.iloc[:, 2:].to_numpy(np.float64), 1)
    q = tr.sample(min(MAX_TEST, len(tr)), random_state=SEED + 1)
    d_tr, _ = nearest(Xtr_nn, q.iloc[:, 2:].to_numpy(np.float64), 2)  # 2nd neighbour: 1st is the row itself
    bad = te.target.mean()
    row.update({"nn_dist_test_train": float(np.median(d_te)), "nn_dist_train_train": float(np.median(d_tr[:, 1])),
                "nn_label_agreement": float((te.target.to_numpy() == tr.target.to_numpy()[i_te[:, 0]]).mean()),
                "label_agreement_by_chance": float(1 - 2 * bad * (1 - bad))})

    # 2. single-feature leakage (raw tier-A columns, NaN kept)
    Xtr, cats = as_lgbm_frame(train, num, cat)
    Xte, _ = as_lgbm_frame(test, num, cat, cats)
    single = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in num + cat:
            m = LGBMClassifier(n_estimators=50, verbose=-1, random_state=SEED).fit(Xtr[[c]], train.target)
            single[c] = auc_sym(test.target, m.predict_proba(Xte[[c]])[:, 1])
    best = max(single, key=single.get)
    row.update({"best_single_feature": best, "best_single_feature_auc": single[best]})

    # 3. missingness-only model
    Mtr, Mte = Xtr.isna().astype(int), Xte.isna().astype(int)
    keep = Mtr.columns[Mtr.std() > 0]
    row["n_cols_with_missing"] = int(len(keep))
    row["missingness_only_auc"] = np.nan
    if len(keep):
        m = LGBMClassifier(n_estimators=100, verbose=-1, random_state=SEED).fit(Mtr[keep], train.target)
        row["missingness_only_auc"] = roc_auc_score(test.target, m.predict_proba(Mte[keep])[:, 1])

    # 4. detectability of incomplete rows after tier B.1 (GBDT matrix)
    row["missing_rows_detectable_auc"] = np.nan
    inc_tr = train[num + cat].isna().any(axis=1).to_numpy()
    inc_te = test[num + cat].isna().any(axis=1).to_numpy()
    if 0 < inc_tr.mean() < 1 and 0 < inc_te.mean() < 1:
        P = {p: pd.read_parquet(PROCESSED / "gbdt" / name / "fold1" / f"{p}.parquet").set_index("row_id")
             for p in ("train", "test")}
        A = P["train"].loc[train.row_id].iloc[:, 1:]
        B = P["test"].loc[test.row_id].iloc[:, 1:]
        m = LGBMClassifier(n_estimators=200, verbose=-1, random_state=SEED).fit(A, inc_tr)
        row["missing_rows_detectable_auc"] = roc_auc_score(inc_te, m.predict_proba(B)[:, 1])
    return row


def overfit_rows() -> pd.DataFrame:
    rows = []
    for r in RUNS:
        s = pd.read_csv(ROOT / "reports" / "models" / r / "summary.csv")
        for _, x in s.iterrows():
            rows.append({"model": NAMES[r], "dataset": x.dataset, "train_auc": x.train_auc, "val_auc": x.val_auc,
                         "test_auc": x.auc, "gap_train_test": x.train_auc - x.auc, "gap_val_test": x.val_auc - x.auc})
    return pd.DataFrame(rows)


def tables(leak: pd.DataFrame, of: pd.DataFrame) -> str:
    L = ["## Rò rỉ dữ liệu (fold 1, phần test)", "",
         "| Bộ dữ liệu | KC láng giềng test→train | KC train→train | Nhãn trùng láng giềng | Mức ngẫu nhiên | "
         "Đặc trưng đơn mạnh nhất (AUC) | AUC chỉ dùng mẫu thiếu | Nhận ra dòng thiếu sau B.1 (AUC) |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in leak.iterrows():
        f = lambda v: "—" if pd.isna(v) else f"{v:.3f}"  # noqa: E731
        L.append(f"| {LABELS[r.dataset]} | {r.nn_dist_test_train:.3f} | {r.nn_dist_train_train:.3f} | "
                 f"{r.nn_label_agreement:.3f} | {r.label_agreement_by_chance:.3f} | "
                 f"{r.best_single_feature} ({r.best_single_feature_auc:.3f}) | {f(r.missingness_only_auc)} | "
                 f"{f(r.missing_rows_detectable_auc)} |")
    for col, title in [("gap_train_test", "AUC train − AUC test"), ("gap_val_test", "AUC validation − AUC test")]:
        t = of.pivot(index="dataset", columns="model", values=col)
        t = t.loc[[d for d in SPECS if d in t.index], [NAMES[r] for r in RUNS]]
        L += ["", f"## {title}", "", "| Bộ dữ liệu | " + " | ".join(t.columns) + " |", "|---|" + "---|" * len(t.columns)]
        for d, r in t.iterrows():
            L.append(f"| {LABELS[d]} | " + " | ".join("—" if pd.isna(v) else f"{v:+.3f}" for v in r.values) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help=f"datasets (default: all): {', '.join(SPECS)}")
    p.add_argument("--one", help=argparse.SUPPRESS)  # internal: run one dataset, print JSON
    args = p.parse_args(argv)
    if args.one:
        print("JSON:" + json.dumps(leakage_row(args.one), default=float))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in args.names or SPECS:
        res = subprocess.run([sys.executable, "-m", "src.models.validity_checks", "--one", n],
                             cwd=ROOT, capture_output=True, text=True, timeout=1800)
        line = [x for x in res.stdout.splitlines() if x.startswith("JSON:")]
        if not line:
            raise RuntimeError(f"{n}: validity check failed\n{res.stderr[-2000:]}")
        rows.append(json.loads(line[0][5:]))
        print(f"[{n}] done", flush=True)
    leak = pd.DataFrame(rows)
    leak.to_csv(OUT / "leakage.csv", index=False)
    of = overfit_rows()
    of.to_csv(OUT / "overfit.csv", index=False)
    text = tables(leak, of)
    (OUT / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    sys.exit(main())
