"""Compare model runs (reports/models/<run>/) on the same folds: classic ML and GBDT.

  * test metrics per dataset x model (mean ± std over folds; Lending Club: refit on train + val)
  * ranks per dataset (1 = best) and mean rank per metric
  * Friedman test with the Iman–Davenport correction, then pairwise Wilcoxon signed-rank tests
    with Holm correction (configs/preprocessing_v1.md, "Tinh chỉnh siêu tham số")
  * per-dataset paired comparison: folds won vs the reference model; Lending Club: paired
    bootstrap of the test-AUC difference (same 282,787 test rows)

Datasets: the 7 main ones. --hmeq replaces HMEQ by a leakage-controlled variant (hmeq_blind or
hmeq_complete) and writes to reports/models/comparison_<variant>/ instead.

Output under reports/models/comparison/: tables.md, auc.csv, ranks.csv, stats.json, figures/*.png

Run from the project root:  python -m src.models.compare_models [--runs lr_default knn_default ...]
"""

import argparse
import itertools
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_auc_score  # noqa: E402

from src.data.datasets import ROOT  # noqa: E402
from src.models.report_classic import LABELS  # noqa: E402

REPORTS = ROOT / "reports" / "models"
CLASSIC_RUNS = ["lr_default", "knn_default", "rf_default", "nb_default"]
GBDT_RUNS = ["xgb_default", "lgbm_default", "cat_default"]
RUNS = CLASSIC_RUNS + GBDT_RUNS
NAMES = {"lr_default": "LR", "knn_default": "KNN", "rf_default": "RF", "nb_default": "NB",
         "xgb_default": "XGB", "lgbm_default": "LGBM", "cat_default": "CatBoost"}
COLORS = {"LR": "#4C72B0", "KNN": "#DD8452", "RF": "#55A868", "NB": "#C44E52",
          "XGB": "#8172B3", "LGBM": "#937860", "CatBoost": "#DA8BC3"}
HIGHER_IS_BETTER = {"auc": True, "gini": True, "ks": True, "ap": True, "f1": True,
                    "precision": True, "recall": True, "brier": False, "logloss": False}
THRESHOLD_METRICS = {"f1": "F1", "precision": "Precision", "recall": "Recall"}
SEED = 42
MAIN_DATASETS = ["give_me_some_credit", "taiwan_default_credit", "home_credit", "hmeq",
                 "south_german_credit", "fico_heloc", "lending_club_B"]


def load(runs, datasets=MAIN_DATASETS):
    summ = pd.concat([pd.read_csv(REPORTS / r / "summary.csv").assign(model=NAMES[r]) for r in runs])
    folds = pd.concat([pd.read_csv(REPORTS / r / "fold_metrics.csv").assign(model=NAMES[r]) for r in runs])
    summ, folds = summ[summ.dataset.isin(datasets)], folds[folds.dataset.isin(datasets)]
    order = {d: i for i, d in enumerate(datasets)}
    summ = summ.sort_values("dataset", key=lambda s: s.map(order), kind="stable")
    return summ, folds


def ranks(summ, metric):
    t = summ.pivot(index="dataset", columns="model", values=metric)
    return t.rank(axis=1, ascending=not HIGHER_IS_BETTER[metric], method="average")


def friedman(table: pd.DataFrame) -> dict:
    """table: datasets x models (higher = better). Friedman + Iman–Davenport F."""
    n, k = table.shape
    chi2, p = stats.friedmanchisquare(*[table[c] for c in table.columns])
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) != chi2 else np.inf
    p_ff = 1 - stats.f.cdf(ff, k - 1, (k - 1) * (n - 1))
    return {"n_datasets": n, "n_models": k, "chi2": chi2, "p_chi2": p, "F_iman_davenport": ff, "p_F": p_ff}


def wilcoxon_holm(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(table.columns, 2):
        d = table[a] - table[b]
        stat, p = stats.wilcoxon(table[a], table[b], zero_method="wilcox", alternative="two-sided")
        rows.append({"a": a, "b": b, "wins_a": int((d > 0).sum()), "wins_b": int((d < 0).sum()),
                     "mean_diff": d.mean(), "W": stat, "p": p})
    df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(df)
    adj, running = [], 0.0
    for i, p in enumerate(df.p):
        running = max(running, min(1.0, (m - i) * p))
        adj.append(running)
    df["p_holm"] = adj
    return df


def _weighted_auc_factory(y: np.ndarray, p: np.ndarray):
    """Return f(w) = AUC with per-row weights w (bootstrap counts); ties count one half.
    Scores are grouped once, so each call is a few vectorised passes over the data."""
    uniq, inv = np.unique(p, return_inverse=True)
    pos, neg = y.astype(float), 1.0 - y

    def auc(w):
        pw = np.bincount(inv, weights=w * pos, minlength=len(uniq))
        nw = np.bincount(inv, weights=w * neg, minlength=len(uniq))
        below = np.cumsum(nw) - nw
        return float((pw * (below + 0.5 * nw)).sum() / (pw.sum() * nw.sum()))
    return auc


# Studentized range q_0.05 / sqrt(2) for the Nemenyi test (Demšar 2006, table 5), k = 2..10
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def rank_posthoc(mean_ranks: pd.Series, n: int) -> dict:
    """Post-hoc tests on mean ranks (Demšar 2006): Nemenyi critical difference for all pairs, and
    every model vs the best-ranked one (z-test on ranks, Holm correction) — fewer comparisons,
    so more power than all-pairs tests when N is small."""
    k = len(mean_ranks)
    se = np.sqrt(k * (k + 1) / (6 * n))
    cd = NEMENYI_Q05[k] * se
    best = mean_ranks.idxmin()
    rows = []
    for m, r in mean_ranks.items():
        if m == best:
            continue
        z = (r - mean_ranks[best]) / se
        rows.append({"model": m, "rank_diff": r - mean_ranks[best], "z": z, "p": 2 * (1 - stats.norm.cdf(abs(z)))})
    df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    adj, running = [], 0.0
    for i, p in enumerate(df.p):
        running = max(running, min(1.0, (len(df) - i) * p))
        adj.append(running)
    df["p_holm"] = adj
    df["significant"] = df.p_holm < 0.05
    return {"critical_difference": cd, "control": best, "vs_control": df}


def cd_diagram(mean_ranks: pd.Series, cd: float, path):
    mr = mean_ranks.sort_values()
    k = len(mr)
    fig, ax = plt.subplots(figsize=(9, 0.45 * k + 1.6))
    ax.set_xlim(1, k)
    ax.set_ylim(0, k + 1.5)
    ax.invert_xaxis()
    ax.axhline(k + 0.8, color="k", lw=1)
    ax.set_yticks([])
    for i, (m, r) in enumerate(mr.items()):
        y = k - i
        ax.plot([r, r], [y, k + 0.8], color=COLORS.get(m, "k"), lw=1)
        ax.plot(r, k + 0.8, "o", color=COLORS.get(m, "k"))
        ax.text(r, y - 0.1, f"{m} ({r:.2f})", ha="center", va="top", fontsize=9)
    ax.plot([mr.iloc[0], mr.iloc[0] + cd], [k + 1.25] * 2, color="red", lw=3)
    ax.text(mr.iloc[0] + cd / 2, k + 1.35, f"CD = {cd:.2f}", ha="center", va="bottom", color="red", fontsize=9)
    ax.set_xlabel("Hạng trung bình theo AUC (1 = tốt nhất). Hai mô hình cách nhau ít hơn CD: chưa khác biệt có ý nghĩa (Nemenyi, α = 0,05)")
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def paired_bootstrap_lc(runs, ref="lr_default", n=1000):
    """Test-AUC difference model − reference on the same Lending Club test rows."""
    preds = {r: pd.read_parquet(REPORTS / r / "predictions" / "lending_club_B.parquet")
             .query("fit == 'train+val'").set_index("row_id").sort_index() for r in runs}
    y = preds[ref]["target"].to_numpy()
    rng = np.random.default_rng(SEED)
    weights = [np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(float) for _ in range(n)]
    f = {r: _weighted_auc_factory(y, preds[r]["proba"].to_numpy()) for r in runs}
    ref_aucs = np.array([f[ref](w) for w in weights])
    out = {}
    for r in runs:
        if r == ref:
            continue
        diffs = np.array([f[r](w) for w in weights]) - ref_aucs
        full = np.ones(len(y))
        out[NAMES[r]] = {"diff": f[r](full) - f[ref](full),
                         "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                         "p_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))}
    return out


def fold_wins(folds, ref="LR"):
    t = folds[(folds.part == "test") & (folds.fit == "train")].pivot_table(
        index=["dataset", "fold"], columns="model", values="auc")
    res = {}
    for name, g in t.groupby(level=0, sort=False):
        res[name] = {m: f"{int((g[m] > g[ref]).sum())}/{len(g)}" for m in g.columns if m != ref}
    return res


def figures(out, summ, rank_auc):
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = list(dict.fromkeys(summ.dataset))
    models = list(dict.fromkeys(summ.model))
    colors = COLORS

    fig, ax = plt.subplots(figsize=(12, 4.4))
    w = 0.8 / len(models)
    x = np.arange(len(order))
    for i, m in enumerate(models):
        s = summ[summ.model == m].set_index("dataset").loc[order]
        ax.bar(x + (i - (len(models) - 1) / 2) * w, s.auc, w, yerr=s.auc_std.fillna(0), capsize=2,
               label=m, color=colors.get(m))
    ax.set_xticks(x, [LABELS[d] for d in order], fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC test")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("So sánh mô hình (tham số mặc định)")
    ax.legend(ncol=len(models), fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "auc_by_model.png", dpi=130)
    plt.close(fig)

    pr_curves(fig_dir, summ, order, models, colors)

    mr = rank_auc.mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.scatter(mr.values, [0] * len(mr), s=60, color=[colors.get(m) for m in mr.index], zorder=3)
    for i, (m, v) in enumerate(mr.items()):
        ax.annotate(f"{m}\n{v:.2f}", (v, 0), xytext=(0, 10 if i % 2 == 0 else -28), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_xlim(1, len(mr))
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Hạng trung bình theo AUC (1 = tốt nhất), 7 bộ dữ liệu")
    ax.grid(axis="x", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "mean_rank.png", dpi=130)
    plt.close(fig)


def pr_curves(fig_dir, summ, order, models, colors):
    """Precision–recall curves on test (5 folds pooled; Lending Club: refit), with the operating
    point at the validation-F1 threshold (mean precision / recall over folds) marked."""
    runs = {NAMES[r]: r for r in RUNS if NAMES[r] in models}
    ncols = 4
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, order):
        base = summ[summ.dataset == name].bad_rate.iloc[0]
        for m in models:
            pr = pd.read_parquet(REPORTS / runs[m] / "predictions" / f"{name}.parquet").query("part == 'test'")
            pr = pr[pr.fit == "train+val"] if (pr.fit == "train+val").any() else pr
            prec, rec, _ = precision_recall_curve(pr.target, pr.proba)
            ax.plot(rec, prec, lw=1.2, color=colors.get(m), label=m)
            s = summ[(summ.dataset == name) & (summ.model == m)].iloc[0]
            ax.plot(s.recall, s.precision, "o", ms=6, color=colors.get(m), markeredgecolor="k")
        ax.axhline(base, color="grey", ls=":", lw=1)
        ax.set_title(LABELS[name], fontsize=10)
        ax.set_xlabel("Recall", fontsize=8)
        ax.set_ylabel("Precision", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(order):]:
        ax.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    h += [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markeredgecolor="k", ms=7),
          plt.Line2D([0], [0], color="grey", ls=":")]
    lab += ["điểm vận hành (ngưỡng F1 từ validation)", "mức cơ sở = tỷ lệ xấu"]
    axes.flat[-1].legend(h, lab, loc="center", fontsize=9)
    fig.suptitle("Đường precision–recall trên test")
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curves.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), sharey=True)
    w = 0.8 / len(models)
    x = np.arange(len(order))
    for ax, (met, title) in zip(axes, THRESHOLD_METRICS.items()):
        for i, m in enumerate(models):
            s = summ[summ.model == m].set_index("dataset").loc[order]
            ax.bar(x + (i - (len(models) - 1) / 2) * w, s[met], w, yerr=s[f"{met}_std"].fillna(0),
                   capsize=2, color=colors.get(m), label=m)
        ax.set_xticks(x, [LABELS[d] for d in order], rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(ncol=len(models), fontsize=8)
    fig.suptitle("Độ đo theo ngưỡng trên test (ngưỡng tối đa F1 trên validation)")
    fig.tight_layout()
    fig.savefig(fig_dir / "precision_recall_f1.png", dpi=130)
    plt.close(fig)


def fmt(m, s):
    return f"{m:.4f}" if np.isnan(s) else f"{m:.4f} ± {s:.4f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=RUNS, choices=RUNS)
    p.add_argument("--hmeq", default="hmeq", choices=["hmeq", "hmeq_blind", "hmeq_complete"])
    args = p.parse_args(argv)
    out = REPORTS / ("comparison" if args.hmeq == "hmeq" else f"comparison_{args.hmeq}")
    out.mkdir(parents=True, exist_ok=True)
    datasets = [args.hmeq if d == "hmeq" else d for d in MAIN_DATASETS]

    summ, folds = load(args.runs, datasets)
    models = [NAMES[r] for r in args.runs]
    order = list(dict.fromkeys(summ.dataset))

    auc = summ.pivot(index="dataset", columns="model", values="auc").loc[order, models]
    auc.to_csv(out / "auc.csv")
    r_auc = ranks(summ, "auc").loc[order, models]
    r_auc.to_csv(out / "ranks.csv")
    mean_ranks = {m: ranks(summ, m)[models].mean().round(3).to_dict() for m in HIGHER_IS_BETTER}
    fr = friedman(auc)
    wx = wilcoxon_holm(auc)
    ph = rank_posthoc(r_auc.mean(), len(order))
    lc = paired_bootstrap_lc(args.runs)
    wins = fold_wins(folds)
    figures(out, summ, r_auc)
    cd_diagram(r_auc.mean(), ph["critical_difference"], out / "figures" / "cd_diagram.png")

    L = ["## Bảng 1 — AUC test (trung bình ± độ lệch chuẩn qua 5 fold; Lending Club: fit lại train + val)", "",
         "| Bộ dữ liệu | " + " | ".join(models) + " | Tốt nhất |", "|---|" + "---|" * (len(models) + 1)]
    for d in order:
        s = summ[summ.dataset == d].set_index("model")
        best = s.auc.idxmax()
        cells = [("**" + fmt(s.auc[m], s.auc_std[m]) + "**") if m == best else fmt(s.auc[m], s.auc_std[m])
                 for m in models]
        L.append(f"| {LABELS[d]} | " + " | ".join(cells) + f" | {best} |")
    L.append("| **Hạng trung bình (AUC)** | " + " | ".join(f"{r_auc[m].mean():.2f}" for m in models) + " | |")
    L += ["", "## Bảng 2 — Hạng trung bình theo từng độ đo (1 = tốt nhất)", "",
          "| Độ đo | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for met, r in mean_ranks.items():
        L.append(f"| {met.upper() if met != 'logloss' else 'LogLoss'} | " + " | ".join(f"{r[m]:.2f}" for m in models) + " |")
    L += ["", "## Bảng 3 — Kiểm định Friedman (AUC)", "",
          f"- Số bộ dữ liệu N = {fr['n_datasets']}, số mô hình k = {fr['n_models']}",
          f"- χ²_F = {fr['chi2']:.3f}, p = {fr['p_chi2']:.4f}",
          f"- Iman–Davenport F = {fr['F_iman_davenport']:.3f}, p = {fr['p_F']:.4f}", "",
          f"## Bảng 3b — So sánh với mô hình hạng tốt nhất ({ph['control']}), kiểm định z trên hạng, hiệu chỉnh Holm "
          f"(Demšar 2006); Nemenyi CD = {ph['critical_difference']:.3f}", "",
          "| Mô hình | Chênh hạng so với " + ph["control"] + " | z | p | p (Holm) | Khác biệt có ý nghĩa |", "|---|---|---|---|---|---|"]
    for _, r in ph["vs_control"].iterrows():
        L.append(f"| {r.model} | {r.rank_diff:+.2f} | {r.z:.2f} | {r.p:.4f} | {r.p_holm:.4f} | {'có' if r.significant else 'không'} |")
    L += ["",
          "## Bảng 4 — Wilcoxon từng cặp (AUC, hiệu chỉnh Holm)", "",
          "| Cặp | Số bộ A thắng / B thắng | Chênh AUC trung bình (A − B) | p | p (Holm) |", "|---|---|---|---|---|"]
    for _, r in wx.iterrows():
        L.append(f"| {r.a} vs {r.b} | {r.wins_a} / {r.wins_b} | {r.mean_diff:+.4f} | {r.p:.4f} | {r.p_holm:.4f} |")
    L += ["", "## Bảng 5 — Số fold thắng LR (AUC test)", "",
          "| Bộ dữ liệu | " + " | ".join(m for m in models if m != "LR") + " |", "|---|" + "---|" * (len(models) - 1)]
    for d, w in wins.items():
        L.append(f"| {LABELS[d]} | " + " | ".join(w[m] for m in models if m != "LR") + " |")
    L += ["", "## Bảng 6 — Lending Club: chênh AUC so với LR trên cùng tập test (bootstrap ghép cặp, 1.000 lần)", "",
          "| Mô hình | Chênh AUC | KTC 95% | p |", "|---|---|---|---|"]
    for m, v in lc.items():
        L.append(f"| {m} − LR | {v['diff']:+.4f} | {v['ci95'][0]:+.4f} đến {v['ci95'][1]:+.4f} | {v['p_two_sided']:.3f} |")
    L += ["", "## Bảng 7 — F1, precision, recall trên test (ngưỡng tối đa F1 trên validation)", ""]
    for met, title in THRESHOLD_METRICS.items():
        t = summ.pivot(index="dataset", columns="model", values=met).loc[order, models]
        sd = summ.pivot(index="dataset", columns="model", values=f"{met}_std").loc[order, models]
        r = ranks(summ, met).loc[order, models]
        L += [f"**{title}**", "", "| Bộ dữ liệu | " + " | ".join(models) + " | Tốt nhất |", "|---|" + "---|" * (len(models) + 1)]
        for d in order:
            best = t.loc[d].idxmax()
            cells = [("**" + fmt(t.loc[d, m], sd.loc[d, m]) + "**") if m == best else fmt(t.loc[d, m], sd.loc[d, m])
                     for m in models]
            L.append(f"| {LABELS[d]} | " + " | ".join(cells) + f" | {best} |")
        L += ["| **Hạng trung bình** | " + " | ".join(f"{r[m].mean():.2f}" for m in models) + " | |", ""]
    thr = folds[(folds.part == "test")].groupby(["dataset", "model"]).threshold.mean().unstack().loc[order, models]
    L += ["**Ngưỡng quyết định** (trung bình qua các fold; chọn trên validation để tối đa F1)", "",
          "| Bộ dữ liệu | " + " | ".join(models) + " | Tỷ lệ xấu |", "|---|" + "---|" * (len(models) + 1)]
    for d in order:
        L.append(f"| {LABELS[d]} | " + " | ".join(f"{thr.loc[d, m]:.3f}" for m in models)
                 + f" | {summ[summ.dataset == d].bad_rate.iloc[0]:.3f} |")
    L += ["", "## Bảng 8 — Các độ đo khác trên test", ""]
    for met in ["ks", "brier", "ap"]:
        t = summ.pivot(index="dataset", columns="model", values=met).loc[order, models]
        L += [f"**{met.upper()}**", "", "| Bộ dữ liệu | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for d in order:
            L.append(f"| {LABELS[d]} | " + " | ".join(f"{t.loc[d, m]:.4f}" for m in models) + " |")
        L.append("")
    text = "\n".join(L) + "\n"
    (out / "tables.md").write_text(text)
    (out / "stats.json").write_text(json.dumps(
        {"friedman": fr, "wilcoxon_holm": wx.to_dict("records"), "mean_ranks": mean_ranks,
         "rank_posthoc": {"critical_difference": ph["critical_difference"], "control": ph["control"],
                          "vs_control": ph["vs_control"].to_dict("records")},
         "lending_club_paired_bootstrap_vs_lr": lc, "fold_wins_vs_lr": wins}, indent=2, default=float))
    print(text)


if __name__ == "__main__":
    sys.exit(main())
