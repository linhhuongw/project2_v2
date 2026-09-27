"""Shared EDA helpers used by the notebooks in notebooks/eda_*.ipynb."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "reports" / "figures" / "eda"
SUMMARY_DIR = ROOT / "reports" / "eda"

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {0: "#4C72B0", 1: "#DD5144"}


def savefig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=120, bbox_inches="tight")


def overview(df: pd.DataFrame, target: str = "target") -> pd.Series:
    dtypes = df.drop(columns=[target]).dtypes
    return pd.Series({
        "rows": len(df),
        "features": df.shape[1] - 1,
        "numeric_features": int(dtypes.map(pd.api.types.is_numeric_dtype).sum()),
        "categorical_features": int((~dtypes.map(pd.api.types.is_numeric_dtype)).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "cells_missing_pct": round(100 * df.isna().mean().mean(), 2),
        "features_with_missing": int(df.drop(columns=[target]).isna().any().sum()),
    })


def target_summary(y: pd.Series) -> dict:
    y = y.dropna().astype(int)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return {"n": int(len(y)), "n_bad": n1, "n_good": n0,
            "bad_rate_pct": round(100 * n1 / len(y), 2),
            "imbalance_ratio": round(n0 / n1, 2)}


def plot_target(y: pd.Series, name: str):
    y = y.dropna().astype(int)
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar(["0 (good)", "1 (bad)"], counts.values, color=[PALETTE[0], PALETTE[1]])
    for b, v in zip(bars, counts.values):
        ax.annotate(f"{v:,}\n({100 * v / len(y):.1f}%)", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{name}: phân phối nhãn")
    ax.set_ylabel("Số mẫu")
    ax.set_ylim(0, counts.max() * 1.2)
    savefig(fig, f"{name}_target")
    plt.show()


def missing_table(df: pd.DataFrame, top: int = 30) -> pd.DataFrame:
    miss = df.isna().mean().mul(100).round(2)
    miss = miss[miss > 0].sort_values(ascending=False)
    return miss.head(top).to_frame("missing_pct")


def missing_vs_target(df: pd.DataFrame, target: str = "target", top: int = 15,
                      min_missing: int = 30) -> pd.DataFrame:
    """Bad rate when a feature is missing vs present — shows informative missingness."""
    rows = []
    for c in df.columns.drop(target):
        m = df[c].isna()
        if min_missing <= m.sum() < len(m):
            rows.append({"feature": c, "missing_pct": round(100 * m.mean(), 2),
                         "bad_rate_if_missing": round(100 * df.loc[m, target].mean(), 2),
                         "bad_rate_if_present": round(100 * df.loc[~m, target].mean(), 2)})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["gap"] = (out.bad_rate_if_missing - out.bad_rate_if_present).round(2)
    return out.reindex(out.gap.abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


def numeric_profile(df: pd.DataFrame, target: str = "target", min_unique: int = 3) -> pd.DataFrame:
    """Per numeric feature: skewness, kurtosis, IQR outlier share, zero share, univariate AUC."""
    from sklearn.metrics import roc_auc_score

    rows = []
    y = df[target]
    for c in df.select_dtypes("number").columns.drop(target, errors="ignore"):
        x = df[c]
        ok = x.notna() & y.notna()
        xv = x[ok]
        if xv.nunique() < min_unique:
            continue
        q1, q3 = xv.quantile([0.25, 0.75])
        iqr = q3 - q1
        out = ((xv < q1 - 1.5 * iqr) | (xv > q3 + 1.5 * iqr)).mean() if iqr > 0 else np.nan
        auc = roc_auc_score(y[ok].astype(int), xv) if y[ok].nunique() == 2 else np.nan
        rows.append({
            "feature": c, "n_unique": xv.nunique(),
            "mean": xv.mean(), "median": xv.median(), "min": xv.min(), "max": xv.max(),
            "skew": stats.skew(xv), "kurtosis": stats.kurtosis(xv),
            "outlier_iqr_pct": 100 * out, "zero_pct": 100 * (xv == 0).mean(),
            "auc": max(auc, 1 - auc),
        })
    return pd.DataFrame(rows).set_index("feature").round(3)


def plot_numeric_by_target(df, cols, name, target="target", ncols=4, log=None):
    """Histograms of numeric features split by label (density-normalised)."""
    log = set(log or [])
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows), squeeze=False)
    for ax, c in zip(axes.flat, cols):
        d = df[[c, target]].dropna()
        x = d[c]
        if c in log:
            x = np.log1p(x.clip(lower=0))
        lo, hi = x.quantile([0.005, 0.995])
        bins = np.linspace(lo, hi, 40) if hi > lo else 20
        for k in (0, 1):
            ax.hist(x[d[target] == k], bins=bins, density=True, alpha=0.55, color=PALETTE[k],
                    label=f"{k}")
        ax.set_title(("log1p " if c in log else "") + c, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    axes.flat[0].legend(title="target", fontsize=7)
    fig.tight_layout()
    savefig(fig, f"{name}_numeric_by_target")
    plt.show()


def categorical_bad_rate(df, cols, target="target", min_count=1):
    """Bad rate and share per category level."""
    frames = []
    for c in cols:
        g = df.groupby(df[c].astype("object").fillna("<NA>"), observed=True)[target].agg(["count", "mean"])
        g = g[g["count"] >= min_count]
        g = g.rename(columns={"count": "n", "mean": "bad_rate"})
        g["share_pct"] = 100 * g["n"] / g["n"].sum()
        g["bad_rate"] = 100 * g["bad_rate"]
        g.index = pd.MultiIndex.from_product([[c], g.index.astype(str)], names=["feature", "level"])
        frames.append(g)
    return pd.concat(frames).round(2)


def plot_categorical_bad_rate(df, cols, name, target="target", ncols=3, max_levels=12):
    overall = df[target].mean() * 100
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    for ax, c in zip(axes.flat, cols):
        g = df.groupby(df[c].astype("object").fillna("<NA>"), observed=True)[target].agg(["count", "mean"])
        g = g.sort_values("count", ascending=False).head(max_levels)
        ax.barh(g.index.astype(str), 100 * g["mean"], color="#8172B3")
        ax.axvline(overall, color="k", ls="--", lw=1)
        ax.set_title(c, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("bad rate %", fontsize=8)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    savefig(fig, f"{name}_categorical_bad_rate")
    plt.show()


def correlation_heatmap(df, name, target="target", max_cols=25):
    num = df.select_dtypes("number")
    if num.shape[1] > max_cols:
        keep = num.corrwith(num[target], method="spearman").abs().sort_values(ascending=False)
        num = num[keep.head(max_cols).index]
    corr = num.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.45 * len(corr) + 3, 0.45 * len(corr) + 2))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"shrink": 0.6}, xticklabels=True, yticklabels=True)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{name}: tương quan Spearman")
    savefig(fig, f"{name}_corr")
    plt.show()
    return corr


def high_corr_pairs(corr: pd.DataFrame, thr: float = 0.9, target: str = "target") -> pd.DataFrame:
    c = corr.drop(index=target, columns=target, errors="ignore")
    mask = np.triu(np.ones(c.shape, dtype=bool), k=1)
    s = c.where(mask).stack()
    s = s[s.abs() >= thr].sort_values(key=abs, ascending=False)
    return s.rename("spearman").reset_index().rename(columns={"level_0": "a", "level_1": "b"})


def save_summary(name: str, summary: dict):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else round(float(o), 4)
        if isinstance(o, pd.Series):
            return o.to_dict()
        raise TypeError(type(o))

    with open(SUMMARY_DIR / f"{name}.json", "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=conv)
