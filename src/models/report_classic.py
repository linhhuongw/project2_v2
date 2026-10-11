"""Build tables and figures for a classic-ML run (reports/models/<run>/).

Writes figures/*.png and tables.md (markdown tables pasted into / referenced by report.md).
report.md itself holds the written analysis and is not overwritten.

Run from the project root:  python -m src.models.report_classic [--run lr_default]
"""

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from src.data.datasets import ROOT  # noqa: E402

REPORTS = ROOT / "reports" / "models"
LABELS = {
    "give_me_some_credit": "GMSC", "taiwan_default_credit": "Taiwan", "home_credit": "Home Credit",
    "hmeq": "HMEQ", "hmeq_complete": "HMEQ (đủ dữ liệu)", "hmeq_blind": "HMEQ (che giá trị thiếu)",
    "south_german_credit": "South German", "fico_heloc": "HELOC",
    "lending_club_B": "Lending Club",
}


def fmt(m, s=None, d=4):
    return f"{m:.{d}f}" if s is None or np.isnan(s) else f"{m:.{d}f} ± {s:.{d}f}"


def tables(out, summary, fm, run) -> str:
    lines = ["## Bảng 1 — Kết quả trên test", "",
             "| Dataset | Đánh giá | AUC | Gini | KS | Brier | LogLoss | AP |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {LABELS[r.dataset]} | {r['eval']} | {fmt(r.auc, r.auc_std)} | {fmt(r.gini, r.gini_std)} | "
                     f"{fmt(r.ks, r.ks_std)} | {fmt(r.brier, r.brier_std)} | {fmt(r.logloss, r.logloss_std)} | "
                     f"{fmt(r.ap, r.ap_std)} |")
    lines += ["", "## Bảng 2 — Độ đo theo ngưỡng (ngưỡng tối đa F1 trên validation)", "",
              "| Dataset | Precision | Recall | F1 | Accuracy |", "|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {LABELS[r.dataset]} | {fmt(r.precision, r.precision_std, 3)} | "
                     f"{fmt(r.recall, r.recall_std, 3)} | {fmt(r.f1, r.f1_std, 3)} | {fmt(r.accuracy, r.accuracy_std, 3)} |")
    lines += ["", "## Bảng 3 — AUC train / validation / test và calibration trung bình", "",
              "| Dataset | AUC train | AUC val | AUC test | Chênh train − test | Tỷ lệ vỡ nợ test | Xác suất dự đoán TB |",
              "|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        tr = "—" if np.isnan(r.train_auc) else f"{r.train_auc:.4f}"
        gap = "—" if np.isnan(r.train_auc) else f"{r.train_auc - r.auc:+.4f}"
        lines.append(f"| {LABELS[r.dataset]} | {tr} | {r.val_auc:.4f} | {r.auc:.4f} | "
                     f"{gap} | {100 * r.bad_rate:.2f}% | {100 * r.mean_pred:.2f}% |")
    lines += ["", "## Bảng 4 — AUC test từng fold", "",
              "| Dataset | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |", "|---|---|---|---|---|---|"]
    t = fm[(fm.part == "test") & (fm.fit == "train")]
    for name, g in t.groupby("dataset", sort=False):
        vals = [f"{v:.4f}" for v in g.sort_values("fold").auc]
        lines.append(f"| {LABELS[name]} | " + " | ".join(vals + [""] * (5 - len(vals))) + " |")
    lines += ["", "## Bảng 5 — Huấn luyện", "",
              "| Dataset | Lần fit | Số vòng lặp tối đa | Hội tụ | Thời gian fit (s) | Thời gian dự đoán (s) |",
              "|---|---|---|---|---|---|"]
    for name, d in run["datasets"].items():
        f = d["fits"]
        iters = [v["n_iter"] for v in f.values() if v.get("n_iter") is not None]
        lines.append(f"| {LABELS[name]} | {len(f)} | {max(iters) if iters else '—'} | "
                     f"{'có' if all(v['converged'] for v in f.values()) else 'KHÔNG'} | "
                     f"{sum(v['fit_seconds'] for v in f.values()):.1f} | "
                     f"{sum(v.get('predict_seconds', 0) for v in f.values()):.1f} |")
    text = "\n".join(lines) + "\n"
    (out / "tables.md").write_text(text)
    return text


def figures(out, summary, preds):
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    s = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh([LABELS[n] for n in s.dataset], s.auc, xerr=s.auc_std.fillna(0), color="#4C72B0", capsize=3)
    for y, (a, sd) in enumerate(zip(s.auc, s.auc_std)):
        ax.text(a + 0.012, y, f"{a:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.5, 0.9)
    ax.set_xlabel("AUC test (thanh lỗi = độ lệch chuẩn qua 5 fold)")
    ax.set_title(f"{out.name}: tham số mặc định")
    fig.tight_layout()
    fig.savefig(fig_dir / "auc_by_dataset.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, g in preds.groupby("dataset", sort=False, observed=True):
        test = g[g.part == "test"]
        test = test[test.fit == "train+val"] if (test.fit == "train+val").any() else test
        fpr, tpr, _ = roc_curve(test.target, test.proba)
        axes[0].plot(fpr, tpr, lw=1.2, label=LABELS[name])
        pt, pp = calibration_curve(test.target, test.proba, n_bins=10, strategy="quantile")
        axes[1].plot(pp, pt, "o-", ms=3, lw=1, label=LABELS[name])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC trên test (gộp 5 fold)")
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set(xlabel="Xác suất dự đoán", ylabel="Tỷ lệ vỡ nợ thực tế", title="Calibration trên test (10 nhóm)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_calibration.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="lr_default")
    args = p.parse_args(argv)
    out = REPORTS / args.run
    summary = pd.read_csv(out / "summary.csv")
    fm = pd.read_csv(out / "fold_metrics.csv")
    run = json.loads((out / "run.json").read_text())
    preds = pd.concat([pd.read_parquet(f).assign(dataset=f.stem)
                       for f in sorted((out / "predictions").glob("*.parquet"))])
    preds["dataset"] = pd.Categorical(preds["dataset"], categories=list(summary.dataset), ordered=True)
    figures(out, summary, preds.sort_values("dataset"))
    print(tables(out, summary, fm, run))


if __name__ == "__main__":
    sys.exit(main())
