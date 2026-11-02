"""Build every figure used by report/sections_new into report/hinh_ve/.

    python scripts/build_report_figures.py

New figures (diagrams, overfit heatmap, runtime, 4-model AUC, data quality) are drawn here;
figures already produced by the pipeline are copied in under their report names.
"""

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.compare_models import COLORS, MAIN_DATASETS  # noqa: E402
from src.models.report_classic import LABELS  # noqa: E402

OUT = ROOT / "report" / "hinh_ve"
REPORTS = ROOT / "reports"
DPI = 150

# Existing pipeline figures, copied under the name the report uses.
COPIES = {
    "lending_club_theo_nam.png": "figures/eda/lending_club_vintage.png",
    "hmeq_thieu_ty_le_xau.png": "figures/eda/hmeq_missing_bad_rate.png",
    "hmeq_thieu_tuong_quan.png": "figures/eda/hmeq_missing_corr.png",
    "gmsc_bien_so_theo_nhan.png": "figures/eda/give_me_some_credit_numeric_by_target.png",
    "lr_auc_theo_bo.png": "models/lr_default/figures/auc_by_dataset.png",
    "lr_roc_calibration.png": "models/lr_default/figures/roc_calibration.png",
    "cat_roc_calibration.png": "models/cat_default/figures/roc_calibration.png",
    "prf1_7_mo_hinh.png": "models/comparison/figures/precision_recall_f1.png",
    "pr_curves.png": "models/comparison/figures/pr_curves.png",
    "learning_curves.png": "models/learning_curves/figures/learning_curves.png",
    "auc_7_mo_hinh.png": "models/comparison/figures/auc_by_model.png",
    "hang_trung_binh_7_mo_hinh.png": "models/comparison/figures/mean_rank.png",
    "cd_diagram.png": "models/comparison/figures/cd_diagram.png",
    "cd_diagram_hmeq_blind.png": "models/comparison_hmeq_blind/figures/cd_diagram.png",
    "hmeq_ngan_ro_ri_auc.png": "figures/eda/hmeq_leakage_control_auc.png",
}

C_TRAIN, C_VAL, C_TEST = "#4C72B0", "#DD8452", "#C44E52"
C_BOX, C_FIT = "#EEF2F7", "#FCE9D9"


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc=C_BOX, ec="#4A4A4A", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#4A4A4A"))


def pipeline():
    fig, ax = plt.subplots(figsize=(9, 9.6))
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.4, 10.8)
    ax.axis("off")

    # Main column: one-off steps (grey) then per-fold steps (orange, fit on train only).
    cx = 4.5
    steps = [
        (10.0, "Dữ liệu gốc\n7 bộ dữ liệu tín dụng", C_BOX, False),
        (8.8, "Tầng A — Khai báo dữ liệu\nquy tắc cố định, không học từ dữ liệu", C_BOX, True),
        (7.5, "Chia dữ liệu\n5-fold CV theo nhóm dòng trùng lặp\nLending Club: theo thời gian", C_BOX, True),
        (6.25, "Tầng B.0 — Sàng lọc cột", C_FIT, True),
        (5.1, "Tầng B.1 — Tiền xử lý\nđiền thiếu, chuẩn hóa, mã hóa", C_FIT, True),
    ]
    for y, t, fc, b in steps:
        box(ax, cx, y, 4.6, 0.85 if "\n" in t else 0.6, t, fc=fc, bold=b)
    for (y0, *_), (y1, *_) in zip(steps, steps[1:]):
        arrow(ax, cx, y0 - 0.45, cx, y1 + 0.45)

    # Two preprocessing branches, each feeding its own model family, then evaluation.
    for x, branch, models in [(2.3, "Nhánh one-hot\n+ chuẩn hóa", "Classic ML\nLR · KNN · RF · NB"),
                              (6.7, "Nhánh mã số nguyên", "GBDT\nXGBoost · LightGBM · CatBoost")]:
        box(ax, x, 3.7, 3.0, 0.75, branch, fc=C_FIT)
        arrow(ax, cx + (x - cx) * 0.35, 4.65, x, 4.1)
        box(ax, x, 2.45, 3.4, 0.8, models, fc="#E3F0E3")
        arrow(ax, x, 3.3, x, 2.88)
        arrow(ax, x, 2.02, cx + (x - cx) * 0.4, 1.45)
    box(ax, cx, 0.75, 5.6, 1.35,
        "Đánh giá trên test\nAUC, KS, Brier, LogLoss, AP, precision, recall, F1\n"
        "hạng trung bình, Friedman, Wilcoxon–Holm\nkiểm tra rò rỉ dữ liệu, overfit",
        fc="#F3EAF7")

    # Legend for the fit-on-train-only rule.
    for y, fc, t in [(-0.05, C_BOX, "Chạy một lần trên toàn bộ dữ liệu"),
                     (-0.35, C_FIT, "Fit chỉ trên train của từng fold, áp sang validation và test")]:
        ax.add_patch(Rectangle((0.6, y - 0.1), 0.35, 0.2, fc=fc, ec="#4A4A4A"))
        ax.text(1.05, y, t, va="center", fontsize=9)
    ax.set_ylim(-0.6, 10.6)
    save(fig, "quy_trinh.png")


def split_scheme():
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5.6), gridspec_kw={"height_ratios": [5, 1.6]})

    # 5-fold CV: fold k tests on part k; 20% of the other four parts becomes validation
    # (drawn as one hatched block for readability; the real validation rows are a stratified sample).
    for k in range(5):
        y = 4 - k
        others = [p for p in range(5) if p != k]
        a.add_patch(Rectangle((k, y), 1, 0.7, fc=C_TEST, ec="white"))
        for p in others:
            a.add_patch(Rectangle((p, y), 1, 0.7, fc=C_TRAIN, ec="white"))
        a.add_patch(Rectangle((others[-1] + 1 - 0.8, y), 0.8, 0.7, fc=C_VAL, ec="white", hatch="//"))
        a.text(-0.15, y + 0.35, f"Fold {k + 1}", ha="right", va="center", fontsize=9)
    a.set_xlim(-0.9, 5.05)
    a.set_ylim(-0.2, 5)
    a.axis("off")
    a.set_title("Sáu bộ dữ liệu: 5-fold CV phân tầng, nhóm theo dòng trùng lặp", fontsize=10)
    handles = [Rectangle((0, 0), 1, 1, fc=C_TRAIN), Rectangle((0, 0), 1, 1, fc=C_VAL, hatch="//"),
               Rectangle((0, 0), 1, 1, fc=C_TEST)]
    a.legend(handles, ["Train", "Validation (20% của train)", "Test"], ncol=3, loc="upper center",
             bbox_to_anchor=(0.5, -0.0), fontsize=9, frameon=False)

    # Lending Club: chronological split by issue year.
    parts = [("Train\n2013\n100.422 khoản vay", C_TRAIN, 100_422), ("Validation\n2014\n162.570", C_VAL, 162_570),
             ("Test\n2015\n282.787", C_TEST, 282_787)]
    total = sum(p[2] for p in parts)
    x = 0
    for t, c, n in parts:
        w = 5 * n / total
        b.add_patch(Rectangle((x, 0), w, 0.8, fc=c, ec="white", hatch="//" if c == C_VAL else None))
        b.text(x + w / 2, 0.4, t, ha="center", va="center", fontsize=8.5,
               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        x += w
    b.annotate("", xy=(5.05, -0.15), xytext=(0, -0.15), arrowprops=dict(arrowstyle="-|>", color="#4A4A4A"))
    b.text(2.5, -0.35, "năm phát hành khoản vay", ha="center", va="top", fontsize=9)
    b.text(-0.15, 0.4, "Lending Club", ha="right", va="center", fontsize=9)
    b.set_xlim(-0.9, 5.05)
    b.set_ylim(-0.7, 1.2)
    b.axis("off")
    b.set_title("Lending Club: chia theo thời gian (chiều rộng tỷ lệ với số khoản vay)", fontsize=10)
    fig.tight_layout()
    save(fig, "chia_du_lieu.png")


def overfit_heatmap():
    df = pd.read_csv(REPORTS / "models" / "validity" / "overfit.csv")
    df = df[df.dataset.isin(MAIN_DATASETS)]
    models = ["LR", "NB", "KNN", "RF", "XGB", "LGBM", "CatBoost"]
    tab = df.pivot(index="model", columns="dataset", values="gap_train_test").reindex(models)[MAIN_DATASETS]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.imshow(tab.values, cmap="Reds", vmin=0, vmax=0.35, aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#777777")
            else:
                ax.text(j, i, f"{v:+.3f}".replace(".", ","), ha="center", va="center", fontsize=8,
                        color="white" if v > 0.2 else "black")
    ax.set_xticks(range(tab.shape[1]), [LABELS[d] for d in tab.columns], fontsize=9)
    ax.set_yticks(range(tab.shape[0]), tab.index, fontsize=9)
    fig.colorbar(im, ax=ax, label="AUC train − AUC test")
    ax.set_title("Mức overfit theo mô hình và bộ dữ liệu (KNN không chấm điểm trên train)", fontsize=10)
    save(fig, "overfit_heatmap.png")


def runtime():
    # Totals from the runtime table in section 4.2.3 (seconds, CPU 10 cores). The run.json files
    # hold timings of later re-runs (e.g. CatBoost after the HMEQ variants), so the table is the reference.
    t = pd.Series({"NB": 3.2, "XGB": 8.3, "LR": 13.6, "LGBM": 19.8, "RF": 68.2, "CatBoost": 124.4, "KNN": 339.9})
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(t.index, t.values, color=[COLORS[m] for m in t.index])
    for y, v in enumerate(t.values):
        ax.text(v * 1.08, y, f"{v:g}".replace(".", ",") + " s", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_xlabel("Tổng thời gian fit + dự đoán trên 7 bộ dữ liệu (giây, thang log)")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "thoi_gian_chay.png")


def auc_classic():
    s = pd.concat([pd.read_csv(REPORTS / "models" / r / "summary.csv").assign(model=m)
                   for r, m in [("lr_default", "LR"), ("knn_default", "KNN"), ("rf_default", "RF"),
                                ("nb_default", "NB")]])
    s = s[s.dataset.isin(MAIN_DATASETS)]
    models = ["LR", "KNN", "RF", "NB"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.8 / len(models)
    x = np.arange(len(MAIN_DATASETS))
    for i, m in enumerate(models):
        r = s[s.model == m].set_index("dataset").loc[MAIN_DATASETS]
        ax.bar(x + (i - (len(models) - 1) / 2) * w, r.auc, w, yerr=r.auc_std.fillna(0), capsize=2, label=m,
               color=COLORS[m])
    ax.set_xticks(x, [LABELS[d] for d in MAIN_DATASETS], fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC test")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    save(fig, "auc_4_mo_hinh_classic.png")


# EDA summaries of the 7 datasets in the report (the EDA notebooks also cover Statlog German Credit,
# which the report replaced with South German).
EDA_NAMES = {"give_me_some_credit": "GMSC", "taiwan_default_credit": "Taiwan", "home_credit": "Home Credit",
             "hmeq": "HMEQ", "south_german_credit": "South German", "fico_heloc": "FICO HELOC",
             "lending_club": "Lending Club"}


def eda_table():
    rows = []
    for f, n in EDA_NAMES.items():
        s = json.loads((REPORTS / "eda" / f"{f}.json").read_text())
        rows.append((n, s["overview"]["cells_missing_pct"], s["max_abs_skew"], s["overview"]["rows"],
                     s["target"]["bad_rate_pct"], max(s["top_auc"].values())))
    return pd.DataFrame(rows, columns=["name", "missing", "max_skew", "rows", "bad", "top_auc"])


def eda_overview():
    df = eda_table().sort_values("bad")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    axes[0].barh(df.name, df.bad, color="#C44E52")
    for y, v in enumerate(df.bad):
        axes[0].text(v + 0.8, y, f"{v:.1f}".replace(".", ","), va="center", fontsize=8)
    axes[0].set_title("Tỷ lệ xấu (%)", fontsize=10)
    axes[0].set_xlim(0, 60)
    axes[1].barh(df.name, np.log10(df.rows), color="#4C72B0")
    axes[1].set_title("log10(số dòng có nhãn)", fontsize=10)
    axes[2].barh(df.name, df.top_auc, color="#55A868")
    axes[2].set_xlim(0.5, 1)
    axes[2].set_title("AUC của biến đơn mạnh nhất", fontsize=10)
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "eda_tong_quan.png")


def data_quality():
    df = eda_table()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sc = ax.scatter(df.missing, np.log10(df.max_skew), s=np.log10(df.rows) * 45, c=df.bad, cmap="Reds",
                    edgecolor="k", vmin=0, vmax=55)
    for _, r in df.iterrows():
        right = r.missing > 20  # keep the label of the right-most point inside the axes
        ax.annotate(r["name"], (r.missing, np.log10(r.max_skew)), fontsize=9, xytext=(-8 if right else 6, 4),
                    textcoords="offset points", ha="right" if right else "left")
    ax.set_xlabel("Tỷ lệ ô bị thiếu (%)")
    ax.set_ylabel("log10(max |skew| của biến số)")
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Tỷ lệ xấu (%)")
    save(fig, "chat_luong_du_lieu.png")


def lr_sigmoid():
    z = np.linspace(-6, 6, 300)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(z, 1 / (1 + np.exp(-z)), color=COLORS["LR"], lw=2)
    ax.axhline(0.5, color="#999999", ls="--", lw=0.8)
    ax.axvline(0, color="#999999", ls="--", lw=0.8)
    ax.set_xlabel("z = β₀ + βᵀx (log-odds)")
    ax.set_ylabel("p(x) = P(y = 1 | x)")
    ax.grid(alpha=0.3)
    save(fig, "lr_sigmoid.png")


def _tree(ax, x0, y0, levels, mask=None, labels=None, dx0=1.6, dy=0.75, color="#4C72B0"):
    """Draw a binary tree; mask(level, index) decides whether a node is split further."""
    nodes = [(x0, y0, 0, 0)]
    i = 0
    while i < len(nodes):
        x, y, lv, idx = nodes[i]
        ax.plot(x, y, "o", ms=9, color=color, zorder=3)
        if labels and (lv, idx) in labels:
            ax.text(x + 0.12, y + 0.12, labels[(lv, idx)], fontsize=7, color="#333333")
        if lv < levels and (mask is None or mask(lv, idx)):
            dx = dx0 / (2 ** lv)
            for j, cx in enumerate((x - dx, x + dx)):
                ax.plot([x, cx], [y, y - dy], color="#888888", lw=1, zorder=1)
                nodes.append((cx, y - dy, lv + 1, 2 * idx + j))
        i += 1


def tree_growth():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    _tree(axes[0], 0, 0, 3, dx0=1.2)
    axes[0].set_title("XGBoost: theo độ sâu (level-wise)", fontsize=10)
    # Leaf-wise: only the leaf with the largest loss reduction is split, so one branch grows deep.
    deep = {(0, 0), (1, 0), (2, 1), (3, 2)}
    _tree(axes[1], 0, 0, 4, mask=lambda lv, idx: (lv, idx) in deep, dx0=1.4, color="#937860")
    axes[1].set_title("LightGBM: theo lá (leaf-wise)", fontsize=10)
    labels = {(0, 0): "x₁ < a", (1, 0): "x₂ < b", (1, 1): "x₂ < b", (2, 0): "x₃ < c", (2, 1): "x₃ < c",
              (2, 2): "x₃ < c", (2, 3): "x₃ < c"}
    _tree(axes[2], 0, 0, 3, labels=labels, dx0=1.2, color="#DA8BC3")
    axes[2].set_title("CatBoost: cây đối xứng (oblivious)", fontsize=10)
    for ax in axes:
        ax.set_xlim(-2.4, 2.6)
        ax.set_ylim(-3.3, 0.5)
        ax.axis("off")
    save(fig, "cach_phat_trien_cay.png")


def bagging_boosting():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax in (a, b):
        ax.set_xlim(-0.2, 10.6)
        ax.set_ylim(0, 5)
        ax.axis("off")

    # Bagging: independent trees on bootstrap samples, averaged.
    box(a, 1.1, 2.5, 2.1, 0.8, "Dữ liệu train", fs=8.5)
    for k, y in enumerate((4.0, 2.5, 1.0)):
        box(a, 4.4, y, 2.8, 0.7, f"Mẫu bootstrap {k + 1}", fs=8.5)
        box(a, 7.3, y, 1.5, 0.7, f"Cây {k + 1}", fc="#E3F0E3", fs=8.5)
        arrow(a, 2.2, 2.5, 2.95, y)
        arrow(a, 5.85, y, 6.5, y)
        arrow(a, 8.1, y, 8.95, 2.5)
    box(a, 9.75, 2.5, 1.4, 0.8, "Trung\nbình", fc="#F3EAF7", fs=8.5)
    a.set_title("Random Forest (bagging): các cây học song song, độc lập", fontsize=10)

    # Boosting: trees learned in sequence, each on the gradient of the current model.
    xs = (1.3, 4.1, 6.9)
    for k, x in enumerate(xs):
        box(b, x, 3.4, 1.7, 0.7, f"Cây {k + 1}", fc="#E3F0E3", fs=8.5)
        box(b, x, 1.6, 2.6, 0.8, "Gradient của\nmô hình hiện tại" if k else "Dữ liệu train", fs=8.5)
        arrow(b, x, 2.0, x, 3.05)
        if k < len(xs) - 1:
            arrow(b, x + 0.85, 3.4, xs[k + 1] - 1.2, 2.0)
    b.text(8.05, 3.4, "…", fontsize=14, va="center")
    box(b, 9.7, 3.4, 1.6, 0.8, "F = Σ η·hₘ", fc="#F3EAF7", fs=9)
    arrow(b, 8.45, 3.4, 8.85, 3.4)
    b.set_title("Gradient boosting: mỗi cây sửa sai số của các cây trước", fontsize=10)
    save(fig, "bagging_boosting.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, src in COPIES.items():
        shutil.copy2(REPORTS / src, OUT / name)
    for f in (pipeline, split_scheme, eda_overview, overfit_heatmap, runtime, auc_classic, data_quality, lr_sigmoid,
              tree_growth, bagging_boosting):
        f()
    print(f"{len(list(OUT.glob('*.png')))} figures in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
