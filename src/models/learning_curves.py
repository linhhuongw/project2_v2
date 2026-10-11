"""Learning curves for the classic-ML models: AUC as a function of the number of training rows.

For each dataset, fold and training fraction:
  * take a nested, stratified subsample of the fold's training part (fixed seed: the rows used at
    a smaller fraction are always contained in the larger ones);
  * fit tier B.1 (median, standardise, one-hot) on that subsample only, using the columns kept by
    tier B.0 of the fold, then fit the model with default hyperparameters;
  * score AUC on the subsample itself (training curve; not for KNN, whose training score is
    meaningless) and on the fold's validation part (validation curve).
The test parts are not used.

Folds: all 5 for the CV datasets, except fold 1 only when the training part exceeds 150,000 rows
(Home Credit), to bound KNN's cost; Lending Club has its single time split (train 2013, val 2014).

Output under reports/models/learning_curves/: curves.csv, tables.md, figures/learning_curves.png

Run from the project root:  python -m src.models.learning_curves [--runs ...] [name ...]
"""

import argparse
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.data.datasets import ROOT, SPECS  # noqa: E402
from src.data.splits import load_fold, load_meta, load_screening  # noqa: E402
from src.features.preprocess import build_preprocessor  # noqa: E402
from src.models.compare_models import CLASSIC_RUNS, COLORS, NAMES, RUNS  # noqa: E402
from src.models.report_classic import LABELS  # noqa: E402
from src.models.train_classic import FAMILY, MODELS, SKIP_TRAIN_SCORE  # noqa: E402

OUT = ROOT / "reports" / "models" / "learning_curves"
FRACTIONS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
MIN_ROWS = 50
MAX_TRAIN_ALL_FOLDS = 150_000
SEED = 42


def nested_order(y: np.ndarray, seed: int) -> np.ndarray:
    """Row order whose every prefix is stratified and nested: within each class rows are shuffled,
    then all rows are sorted by their relative position (rank / class size) inside their class."""
    rng = np.random.default_rng(seed)
    key = np.empty(len(y))
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        key[rng.permutation(rows)] = (np.arange(len(rows)) + rng.random()) / len(rows)
    return np.argsort(key, kind="stable")


def curve_for(name: str, runs: list) -> list:
    meta = load_meta(name)
    rows = []
    n_folds = meta["n_folds"]
    for k in range(1, n_folds + 1):
        sc = load_screening(name, k)
        cols = sc["numeric_kept"] + sc["categorical_kept"]
        train, val, _ = load_fold(name, k)
        if k > 1 and len(train) > MAX_TRAIN_ALL_FOLDS:
            break
        y = train["target"].to_numpy()
        order = nested_order(y, SEED + k)
        for frac in FRACTIONS:
            n = int(round(frac * len(train)))
            if n < MIN_ROWS:
                continue
            sub = train.iloc[order[:n]]
            if sub["target"].nunique() < 2:
                continue
            mats = {}
            for family in {FAMILY.get(r, "classic") for r in runs}:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pre = build_preprocessor(sc["numeric_kept"], sc["categorical_kept"], family,
                                             SPECS[name].impute).fit(sub[cols])
                    mats[family] = (pre.transform(sub[cols]).astype(np.float32),
                                    pre.transform(val[cols]).astype(np.float32))
            for run in runs:
                X_sub, X_val = mats[FAMILY.get(run, "classic")]
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model = MODELS[run]().fit(X_sub, sub["target"])
                val_auc = roc_auc_score(val["target"], model.predict_proba(X_val)[:, 1])
                train_auc = (np.nan if run in SKIP_TRAIN_SCORE
                             else roc_auc_score(sub["target"], model.predict_proba(X_sub)[:, 1]))
                rows.append({"dataset": name, "fold": k, "model": NAMES[run], "fraction": frac, "n_train": n,
                             "train_auc": train_auc, "val_auc": val_auc,
                             "seconds": round(time.perf_counter() - t0, 2)})
    return rows


def plot(curves: pd.DataFrame, path):
    names = list(dict.fromkeys(curves.dataset))
    ncols = 4
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        d = curves[curves.dataset == name]
        for model, g in d.groupby("model", sort=False):
            s = g.groupby("fraction").agg(n=("n_train", "mean"), val=("val_auc", "mean"), val_sd=("val_auc", "std"),
                                          tr=("train_auc", "mean"))
            ax.plot(s.n, s.val, "o-", ms=3, lw=1.4, color=COLORS[model], label=f"{model} (validation)")
            if s.val_sd.notna().any():
                ax.fill_between(s.n, s.val - s.val_sd.fillna(0), s.val + s.val_sd.fillna(0),
                                color=COLORS[model], alpha=0.12)
            if s.tr.notna().any():
                ax.plot(s.n, s.tr, "--", lw=1, color=COLORS[model], alpha=0.7)
        ax.set_xscale("log")
        ax.set_title(LABELS[name], fontsize=10)
        ax.set_xlabel("Số dòng train (thang log)", fontsize=8)
        ax.set_ylabel("AUC", fontsize=8)
        ax.set_ylim(0.45, 1.01)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="grey", ls="--"))
    labels.append("nét đứt = AUC trên chính tập train")
    axes.flat[-1].legend(handles, labels, loc="center", fontsize=9)
    fig.suptitle("Learning curves — AUC validation (nét liền, ± độ lệch chuẩn qua fold) và AUC train (nét đứt)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def tables(curves: pd.DataFrame) -> str:
    L = ["## AUC validation theo tỷ lệ dữ liệu train (trung bình qua các fold)", ""]
    for name in dict.fromkeys(curves.dataset):
        d = curves[curves.dataset == name]
        t = d.pivot_table(index="model", columns="fraction", values="val_auc", aggfunc="mean")
        n = d.groupby("fraction").n_train.mean().round().astype(int)
        t = t.reindex([NAMES[r] for r in RUNS if NAMES[r] in t.index])
        L += [f"**{LABELS[name]}** (số fold: {d.fold.nunique()})", "",
              "| Mô hình | " + " | ".join(f"{int(f * 100)}% (n≈{n[f]:,})".replace(",", ".") for f in t.columns) + " | Tăng từ 20% → 100% |",
              "|---|" + "---|" * (len(t.columns) + 1)]
        for m, r in t.iterrows():
            gain = r[1.0] - r[0.2] if 0.2 in r and 1.0 in r else np.nan
            L.append(f"| {m} | " + " | ".join(f"{v:.4f}" for v in r.values) + f" | {gain:+.4f} |")
        L.append("")
    L += ["## Khoảng cách AUC train − validation ở 100% dữ liệu train", "",
          "| Bộ dữ liệu | " + " | ".join(m for m in dict.fromkeys(curves.model) if m != "KNN") + " |",
          "|---|" + "---|" * (curves.model.nunique() - ("KNN" in set(curves.model)))]
    full = curves[curves.fraction == 1.0]
    for name in dict.fromkeys(curves.dataset):
        d = full[full.dataset == name].groupby("model")[["train_auc", "val_auc"]].mean()
        gap = (d.train_auc - d.val_auc)
        L.append(f"| {LABELS[name]} | " + " | ".join(f"{gap.get(m, np.nan):+.4f}" for m in dict.fromkeys(curves.model)
                                                     if m != "KNN") + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=CLASSIC_RUNS, choices=RUNS)
    p.add_argument("names", nargs="*", help=f"datasets (default: all): {', '.join(SPECS)}")
    args = p.parse_args(argv)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.names or SPECS:
        t0 = time.perf_counter()
        rows += curve_for(name, args.runs)
        print(f"[{name}] done in {time.perf_counter() - t0:.0f}s", flush=True)
    curves = pd.DataFrame(rows)
    curves.to_csv(OUT / "curves.csv", index=False)
    plot(curves, OUT / "figures" / "learning_curves.png")
    text = tables(curves)
    (OUT / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    sys.exit(main())
