"""Generate (and optionally execute) the EDA notebooks in notebooks/.

    python scripts/build_eda_notebooks.py            # write notebooks only
    python scripts/build_eda_notebooks.py --execute  # write + run them in place
"""

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = """\
import sys, warnings
from pathlib import Path

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from src.visualization import eda

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
INTERIM = ROOT / "data" / "interim"
"""

COMMON_TARGET = """\
tgt = eda.target_summary(df["target"])
print(tgt)
eda.plot_target(df["target"], NAME)
"""

COMMON_OVERVIEW = """\
ov = eda.overview(df)
display(ov.to_frame("value"))
display(df.head())
"""

COMMON_MISSING = """\
miss = eda.missing_table(df)
display(miss if len(miss) else "Không có giá trị thiếu (NaN).")
mvt = eda.missing_vs_target(df)
display(mvt if len(mvt) else "Không có cột nào đủ lượng thiếu để so sánh bad rate.")
"""

COMMON_PROFILE = """\
prof = eda.numeric_profile(df)
cols_show = ["n_unique", "min", "median", "max", "skew", "kurtosis", "outlier_iqr_pct", "zero_pct", "auc"]
display(prof.sort_values("auc", ascending=False)[cols_show])
print("Số cột số có |skew| > 2:", int((prof["skew"].abs() > 2).sum()), "/", len(prof))
"""

COMMON_CORR = """\
corr = eda.correlation_heatmap(df, NAME)
display(eda.high_corr_pairs(corr, thr=0.9).head(20))
"""

COMMON_SAVE = """\
eda.save_summary(NAME, {
    "name": NAME,
    "overview": ov.to_dict(),
    "target": tgt,
    "missing_top": miss["missing_pct"].head(10).to_dict() if len(miss) else {},
    "n_numeric_abs_skew_gt2": int((prof["skew"].abs() > 2).sum()),
    "n_numeric": int(len(prof)),
    "max_abs_skew": float(prof["skew"].abs().max()),
    "top_auc": prof["auc"].sort_values(ascending=False).head(5).to_dict(),
    **EXTRA,
})
"""


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def build(spec):
    cells = [
        md(spec["title"]),
        code(SETUP),
        md("## 1. Nạp dữ liệu"),
        code(spec["load"]),
        md("## 2. Tổng quan"),
        code(COMMON_OVERVIEW),
        md("## 3. Nhãn (target: 1 = xấu / vỡ nợ, 0 = tốt)"),
        code(COMMON_TARGET),
        md("## 4. Giá trị thiếu\n\nBảng thứ hai so sánh tỉ lệ nợ xấu khi cột bị thiếu và khi có giá trị: "
           "chênh lệch lớn nghĩa là *bản thân việc thiếu* đã mang thông tin về nhãn."),
        code(COMMON_MISSING),
        md("## 5. Hồ sơ các biến số\n\n`skew`/`kurtosis`: độ lệch và độ nhọn; `outlier_iqr_pct`: % điểm nằm ngoài "
           "1.5×IQR; `zero_pct`: % giá trị 0; `auc`: AUC đơn biến (max(AUC, 1−AUC)) — thước đo nhanh sức "
           "phân biệt của từng biến."),
        code(COMMON_PROFILE),
        md("## 6. Phân phối theo nhãn"),
        code(spec["dist"]),
    ]
    if spec.get("cat"):
        cells += [md("## 7. Biến phân loại: tỉ lệ nợ xấu theo nhóm\n\nĐường nét đứt = bad rate chung."),
                  code(spec["cat"])]
    cells += [md("## 8. Tương quan"), code(COMMON_CORR)]
    if spec.get("checks"):
        cells += [md("## 9. Kiểm tra riêng cho bộ dữ liệu")]
        for c in spec["checks"]:
            cells.append(md(c) if c.lstrip().startswith("###") else code(c))
    cells += [md("## 10. Nhận xét"), md(spec["notes"]),
              code(f"EXTRA = {spec.get('extra', '{}')}\n" + COMMON_SAVE)]
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


# ---------------------------------------------------------------------------------------------
SPECS = {}

SPECS["eda_01_hmeq"] = dict(
    title="""
# EDA 01 — HMEQ (Home Equity Loans)

Nguồn: bộ dữ liệu HMEQ (SAS / Kaggle) — 5.960 khoản vay thế chấp giá trị nhà (home equity line of credit).
Nhãn gốc `BAD` (1 = vỡ nợ hoặc nợ quá hạn nghiêm trọng) → `target`.
File đầu vào: `data/interim/hmeq.csv` (tạo bởi `src/data/make_interim.py`).
""",
    load='NAME = "hmeq"\ndf = pd.read_csv(INTERIM / "hmeq.csv")\ndf.shape',
    dist="""
top = prof.sort_values("auc", ascending=False).index[:10].tolist()
eda.plot_numeric_by_target(df, top, NAME, log=["LOAN", "MORTDUE", "VALUE", "DEBTINC", "CLAGE"])
""",
    cat="""
cats = ["REASON", "JOB"]
display(eda.categorical_bad_rate(df, cats))
eda.plot_categorical_bad_rate(df, cats, NAME, ncols=2)
""",
    checks=[
        "### Số dòng bị thiếu nhiều cột cùng lúc",
        """
n_miss = df.drop(columns="target").isna().sum(axis=1)
display(df.groupby(n_miss.clip(upper=6))["target"].agg(["size", "mean"]).rename(
    columns={"size": "n", "mean": "bad_rate"}).rename_axis("số cột thiếu (≥6 gộp)"))
""",
        """
# DEROG/DELINQ gần như là biến đếm rời rạc với phần lớn = 0
for c in ["DEROG", "DELINQ", "NINQ"]:
    display(df.groupby(df[c].clip(upper=5))["target"].agg(["size", "mean"]).rename_axis(f"{c} (≥5 gộp)").T)
""",
    ],
    notes="""
- **Nhãn:** bad rate ≈ **20%** (1.189/5.960), mất cân bằng ở mức vừa (≈ 4:1). Không cần xử lý đặc biệt ngoài việc dùng stratified split và các metric không phụ thuộc ngưỡng (AUC, PR-AUC).
- **Thiếu dữ liệu có tính thông tin rất mạnh (bất thường):** khi thiếu giá trị tài sản (`VALUE`) thì bad rate ≈ **94%**; khi thiếu tỉ lệ nợ/thu nhập (`DEBTINC`, thiếu ~21%) bad rate ≈ **62%** so với ~9% khi có giá trị. Điều này gợi ý việc thiếu gắn liền với quy trình xét duyệt/thu hồi nợ — một dạng *rò rỉ thông tin* (leakage) tiềm tàng. Mô hình cây sẽ khai thác rất mạnh đặc điểm này; cần cân nhắc khi diễn giải kết quả.
- Ngược lại, thiếu `DEROG`, `DELINQ`, `YOJ`, `JOB` lại đi kèm bad rate *thấp hơn* trung bình.
- **Độ lệch:** các biến tiền tệ (`LOAN`, `MORTDUE`, `VALUE`) lệch phải (skew 2–3); `DEBTINC` có đuôi rất dài (kurtosis ≈ 50, max > 200%). Các biến đếm (`DEROG`, `DELINQ`) tập trung ở 0 (78–86% bằng 0) nhưng là các biến phân biệt tốt nhất.
- Không có dòng trùng lặp. Kích thước nhỏ → phương sai đánh giá cao, nên dùng cross-validation.
""",
    extra='{"informative_missing": mvt.head(3).to_dict("records")}',
)

SPECS["eda_02_fico_heloc"] = dict(
    title="""
# EDA 02 — FICO HELOC (Explainable ML Challenge)

Nguồn: FICO xAI Challenge — hồ sơ tín dụng của người xin hạn mức vay thế chấp (HELOC), bản *cleaned* (đã loại
các dòng không có hồ sơ tín dụng). Nhãn gốc `RiskPerformance` (`Bad` = từng quá hạn ≥ 90 ngày trong 24 tháng) → `target`.

Lưu ý: dữ liệu dùng **mã âm đặc biệt** thay cho NaN: `-9` = không có bản ghi/không tìm thấy,
`-8` = không có giao dịch hợp lệ, `-7` = điều kiện không xảy ra (ví dụ chưa từng quá hạn).
""",
    load='NAME = "fico_heloc"\ndf = pd.read_csv(INTERIM / "fico_heloc.csv")\ndf.shape',
    dist="""
top = prof.sort_values("auc", ascending=False).index[:12].tolist()
eda.plot_numeric_by_target(df.replace({-7: np.nan, -8: np.nan, -9: np.nan}), top, NAME)
""",
    checks=[
        "### Mã đặc biệt -7 / -8 / -9 (missing “ẩn”)",
        """
X = df.drop(columns="target")
special = pd.DataFrame({k: (X == k).mean().mul(100).round(2) for k in (-7, -8, -9)})
display(special[special.sum(axis=1) > 0].sort_values(-7, ascending=False))
for k in (-7, -8):
    m = (X == k).any(axis=1)
    print(f"Dòng có ít nhất một giá trị {k}: {m.mean():.1%} — bad rate {df.loc[m, 'target'].mean():.1%} "
          f"vs {df.loc[~m, 'target'].mean():.1%}")
""",
        """
# Biến mạnh nhất: ExternalRiskEstimate (điểm rủi ro tổng hợp)
bins = pd.cut(df["ExternalRiskEstimate"].where(df["ExternalRiskEstimate"] > 0), [0, 60, 65, 70, 75, 80, 85, 100])
display(df.groupby(bins, observed=True)["target"].agg(["size", "mean"]))
""",
    ],
    notes="""
- **Nhãn:** bad rate ≈ **52%** — gần như **cân bằng**, rất khác các bộ dữ liệu tín dụng thực tế (thường 5–20%). Đây là mẫu đã được FICO chọn lọc/cân bằng cho cuộc thi, nên xác suất dự đoán **không phản ánh tỉ lệ vỡ nợ thật** và cần hiệu chỉnh (calibration / prior correction) nếu dùng ngoài thực tế.
- **Không có NaN nhưng có missing ẩn:** mã `-7` xuất hiện ở ~58% số dòng, `-8` ở ~43%. Nếu coi chúng là số thực, các thống kê min/skew bị sai lệch → cần mã hoá lại (NaN + cờ chỉ báo).
- **Phân phối:** nhiều biến là tỉ lệ phần trăm bị chặn [0,100] hoặc biến đếm; độ lệch chủ yếu ở các biến đếm số lần hỏi tín dụng (`NumInqLast6M`, skew > 6) và số giao dịch xấu. Nhìn chung ít lệch hơn GMSC/Taiwan.
- **Tín hiệu:** `ExternalRiskEstimate` (AUC đơn biến ≈ 0,77) chiếm ưu thế; bad rate giảm đơn điệu theo điểm này. Nhiều cặp biến gần như trùng lặp (vd. `NumInqLast6M` ↔ `NumInqLast6Mexcl7days`).
- Chỉ 1 dòng trùng lặp.
""",
)

SPECS["eda_03_taiwan_default_credit"] = dict(
    title="""
# EDA 03 — Taiwan Default of Credit Card Clients

Nguồn: UCI (Yeh & Lien, 2009) — 30.000 chủ thẻ tín dụng tại Đài Loan, lịch sử 6 tháng (04–09/2005).
Nhãn gốc `default.payment.next.month` → `target`.
""",
    load='NAME = "taiwan_default_credit"\ndf = pd.read_csv(INTERIM / "taiwan_default_credit.csv")\ndf.shape',
    dist="""
num = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4"]
eda.plot_numeric_by_target(df, num, NAME, log=["LIMIT_BAL", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4"])
""",
    cat="""
cats = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_6"]
display(eda.categorical_bad_rate(df, cats))
eda.plot_categorical_bad_rate(df, cats, NAME)
""",
    checks=[
        "### Mã không có trong tài liệu gốc",
        """
print("EDUCATION:", df["EDUCATION"].value_counts().sort_index().to_dict(), "(tài liệu chỉ có 1–4)")
print("MARRIAGE :", df["MARRIAGE"].value_counts().sort_index().to_dict(), "(tài liệu chỉ có 1–3)")
print("PAY_0    :", df["PAY_0"].value_counts().sort_index().to_dict(), "(tài liệu không định nghĩa -2 và 0)")
neg = (df.filter(like="BILL_AMT") < 0).any(axis=1)
print(f"Dòng có BILL_AMT âm (trả thừa): {neg.mean():.1%}")
print("Cột ID có AUC ~0.5 — chỉ là khoá, cần loại khỏi đặc trưng.")
""",
    ],
    notes="""
- **Nhãn:** bad rate ≈ **22%** (6.636/30.000), tỉ lệ ≈ 3,5:1 — mất cân bằng vừa phải.
- **Nhãn phụ thuộc rất mạnh vào trạng thái trả nợ gần nhất `PAY_0`:** bad rate nhảy từ ~13% (trả đúng hạn) lên ~34% (trễ 1 tháng) và ~70% (trễ ≥ 2 tháng). Bài toán gần như “dự đoán tháng sau dựa trên tháng này” — một biến chiếm phần lớn tín hiệu.
- **Dữ liệu bất thường về mã hoá:** `EDUCATION` có giá trị 0, 5, 6 và `MARRIAGE` có 0 không được mô tả; `PAY_*` có -2 và 0 không có trong tài liệu gốc. Cần gộp/diễn giải lại.
- **Độ lệch cực lớn** ở các khoản thanh toán `PAY_AMT*` (skew 10–30, kurtosis tới >1.600) và dư nợ `BILL_AMT*` (skew ≈ 3, có giá trị âm ở ~6% dòng). Nên dùng log hoặc mô hình cây.
- `BILL_AMT1..6` tương quan rất cao với nhau (ρ > 0,9) → đa cộng tuyến. Không có missing, không có dòng trùng.
""",
)

SPECS["eda_04_give_me_some_credit"] = dict(
    title="""
# EDA 04 — Give Me Some Credit (Kaggle 2011)

Nguồn: cuộc thi Kaggle “Give Me Some Credit” — 150.000 khách hàng (tập train).
Nhãn gốc `SeriousDlqin2yrs` (quá hạn ≥ 90 ngày trong 2 năm tới) → `target`.
""",
    load='NAME = "give_me_some_credit"\ndf = pd.read_csv(INTERIM / "give_me_some_credit.csv")\ndf.shape',
    dist="""
cols = ["RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio", "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines", "NumberOfDependents"]
clip = df.copy()
clip["RevolvingUtilizationOfUnsecuredLines"] = clip["RevolvingUtilizationOfUnsecuredLines"].clip(upper=1.5)
eda.plot_numeric_by_target(clip, cols, NAME, log=["DebtRatio", "MonthlyIncome"])
""",
    cat="""
late = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]
capped = df[late + ["target"]].copy()
for c in late:
    capped[c] = capped[c].where(capped[c] < 90, 96).clip(upper=96).map(lambda v: "96/98 (mã)" if v >= 96 else str(min(v, 5)) + ("+" if v >= 5 else ""))
display(eda.categorical_bad_rate(capped, late))
eda.plot_categorical_bad_rate(capped, late, NAME)
""",
    checks=[
        "### Giá trị bất thường",
        """
late = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]
code = df[late].ge(96).any(axis=1)
print(f"Dòng có mã 96/98 ở các cột trễ hạn: {code.sum()} — bad rate {df.loc[code, 'target'].mean():.1%}")
ru = df["RevolvingUtilizationOfUnsecuredLines"]
print(f"Tỉ lệ sử dụng hạn mức > 1: {(ru > 1).mean():.2%} (max = {ru.max():,.0f}); bad rate {df.loc[ru > 1, 'target'].mean():.1%}")
print(f"age = 0: {(df['age'] == 0).sum()} dòng; age < 21: {(df['age'] < 21).sum()} dòng")
dr = df["DebtRatio"]
print(f"DebtRatio > 1: {(dr > 1).mean():.1%}; median DebtRatio khi MonthlyIncome NaN = {dr[df.MonthlyIncome.isna()].median():,.0f} "
      f"vs khi có thu nhập = {dr[df.MonthlyIncome.notna()].median():.2f}")
""",
        """
X = df.drop(columns="target")
dup = X.duplicated(keep=False)
conflict = df[dup].groupby(list(X.columns), dropna=False)["target"].nunique().gt(1).sum()
print(f"Dòng trùng hoàn toàn: {df.duplicated().sum()}; trùng đặc trưng: {X.duplicated().sum()}; nhóm trùng nhưng khác nhãn: {conflict}")
""",
    ],
    notes="""
- **Nhãn: mất cân bằng nặng** — bad rate chỉ ≈ **6,7%** (≈ 14:1). Cần stratified split, đánh giá bằng AUC/PR-AUC, cân nhắc `class_weight`/`scale_pos_weight`.
- **Phân phối lệch cực đoan (bất thường nhất trong nhóm dữ liệu nhỏ):** `RevolvingUtilizationOfUnsecuredLines` (lẽ ra ∈ [0,1]) có max ≈ 50.708, skew ≈ 98; `DebtRatio` skew ≈ 95; `MonthlyIncome` skew ≈ 114. Kurtosis lên tới > 10⁴.
- **Mã đặc biệt 96/98** trong các cột đếm số lần trễ hạn (269 dòng, bad rate ≈ 55%) — không phải số đếm thực mà là mã lỗi/không xác định.
- **Lỗi dữ liệu:** có người `age = 0`; khi `MonthlyIncome` bị thiếu (~20%) thì `DebtRatio` có trung vị ≈ 1.159 thay vì ≈ 0,3 → DebtRatio lúc đó thực chất là tổng nợ tuyệt đối, không phải tỉ lệ.
- **Trùng lặp:** 609 dòng trùng hoàn toàn và 37 nhóm trùng đặc trưng nhưng khác nhãn (nhiễu nhãn).
- Thiếu dữ liệu ở `MonthlyIncome` và `NumberOfDependents` đi kèm bad rate hơi *thấp* hơn trung bình — tính thông tin yếu.
""",
)

SPECS["eda_05_german_credit"] = dict(
    title="""
# EDA 05 — German Credit (Statlog, UCI)

Nguồn: UCI Statlog German Credit (Hofmann, 1994), bản ARFF với nhãn thuộc tính dạng chữ.
1.000 người vay; nhãn gốc `class` (`bad` → 1) → `target`.
""",
    load='NAME = "german_credit"\ndf = pd.read_csv(INTERIM / "german_credit.csv")\ndf.shape',
    dist="""
eda.plot_numeric_by_target(df, ["duration", "credit_amount", "age"], NAME, ncols=3, log=["credit_amount"])
""",
    cat="""
cats = ["checking_status", "credit_history", "savings_status", "purpose", "employment",
        "property_magnitude", "housing", "personal_status", "foreign_worker"]
display(eda.categorical_bad_rate(df, cats))
eda.plot_categorical_bad_rate(df, cats, NAME)
""",
    checks=[
        "### Nhóm nhỏ và quan hệ phản trực giác",
        """
g = df.groupby("credit_history")["target"].agg(["size", "mean"]).sort_values("mean")
display(g)
print("→ Nhóm 'critical/other existing credit' lại có bad rate THẤP NHẤT, còn 'no credits/all paid' CAO NHẤT.")
display(df.groupby("foreign_worker")["target"].agg(["size", "mean"]))
""",
    ],
    notes="""
- **Nhãn:** bad rate = **30%** chính xác (300/700) — tỉ lệ cố định do *thiết kế lấy mẫu* (oversample người xấu), không phải tỉ lệ thực tế của ngân hàng (thực tế thấp hơn nhiều). Xác suất dự đoán cần hiệu chỉnh nếu dùng thực tế. Bộ dữ liệu cũng gắn với ma trận chi phí 5:1 (dự đoán nhầm bad→good đắt gấp 5).
- **Rất nhỏ (1.000 dòng)** và đa số biến là phân loại → kết quả đánh giá dao động mạnh giữa các lần chia; nhiều nhóm có < 50 mẫu.
- **Quan hệ phản trực giác / nghi lỗi mã hoá:** nhóm lịch sử tín dụng “critical account” có bad rate thấp nhất (~17%), còn “no credits / all paid” lại cao nhất (~62%). Grömping (2019) đã chỉ ra đây là lỗi mã hoá của bản Statlog (xem notebook South German Credit).
- Biến `foreign_worker` gần như hằng (96% “yes”).
- Không có missing, không trùng lặp; biến số ít lệch (skew ≤ 2).
""",
)

SPECS["eda_06_south_german_credit"] = dict(
    title="""
# EDA 06 — South German Credit (bản hiệu chỉnh, Grömping 2019)

Nguồn: UCI “South German Credit” — bản sửa lỗi của German Credit (Statlog) do Grömping (2019) công bố,
với bảng mã (`codetable.txt`) đã được hiệu chỉnh. Nhãn gốc `credit_risk` (0 = bad, 1 = good) → `target = 1 - credit_risk`.
Mọi biến được lưu dạng mã số nguyên.
""",
    load='NAME = "south_german_credit"\ndf = pd.read_csv(INTERIM / "south_german_credit.csv")\ndf.shape',
    dist="""
eda.plot_numeric_by_target(df, ["duration", "amount", "age"], NAME, ncols=3, log=["amount"])
""",
    cat="""
cats = ["status", "credit_history", "savings", "purpose", "employment_duration",
        "property", "housing", "personal_status_sex", "foreign_worker"]
display(eda.categorical_bad_rate(df, cats))
eda.plot_categorical_bad_rate(df, cats, NAME)
""",
    checks=[
        "### So sánh với bản Statlog",
        """
ge = pd.read_csv(INTERIM / "german_credit.csv")
cmp = pd.DataFrame({
    "statlog": [len(ge), ge.target.mean(), ge.credit_amount.sum(), ge.duration.mean(), ge.age.mean()],
    "south_german": [len(df), df.target.mean(), df.amount.sum(), df.duration.mean(), df.age.mean()],
}, index=["rows", "bad_rate", "sum_amount", "mean_duration", "mean_age"])
display(cmp)
print("Hai bộ có cùng thống kê biên nhưng thứ tự dòng khác nhau.")
display(pd.concat({
    "statlog": ge.groupby("credit_history")["target"].agg(["size", "mean"]).sort_values("size"),
    "south_german (mã 0–4)": df.groupby("credit_history")["target"].agg(["size", "mean"]).sort_values("size"),
}, axis=1))
""",
        """
# Trong bảng mã hiệu chỉnh: 0 = trễ hạn trong quá khứ, 1 = critical account, 2 = không vay/đã trả hết,
# 3 = đang trả đúng hạn, 4 = đã trả hết tại ngân hàng này. Với nhãn đúng, bad rate giảm dần theo mã → hợp lý.
""",
    ],
    notes="""
- **Nhãn:** giống hệt Statlog — **30%** bad, cùng 1.000 người vay (cùng phân phối biên, chỉ khác thứ tự dòng và cách mã hoá).
- **Điểm khác duy nhất đáng kể là ý nghĩa mã hoá:** với bảng mã đã hiệu chỉnh, nhóm “trễ hạn trong quá khứ” có bad rate cao nhất (~62%) và nhóm “đã trả hết tại ngân hàng này” thấp nhất (~17%) — quan hệ giờ **hợp lý về mặt nghiệp vụ**, giải quyết điểm phản trực giác ở bản Statlog. Hiệu năng mô hình không đổi (thông tin như nhau), nhưng **diễn giải** thì đúng hơn.
- Tất cả biến là mã số nguyên; nếu đưa thẳng vào mô hình tuyến tính sẽ ngầm giả định thứ tự → nên one-hot hoặc dùng mô hình cây.
- Nhỏ, không missing, không trùng lặp, ít lệch.
""",
)

SPECS["eda_07_home_credit"] = dict(
    title="""
# EDA 07 — Home Credit Default Risk (Kaggle 2018)

Nguồn: Kaggle “Home Credit Default Risk” — khách hàng ít/không có lịch sử tín dụng (unbanked).
Bảng chính `application_train` (307.511 hồ sơ); nhãn gốc `TARGET` (1 = gặp khó khăn thanh toán) → `target`.
Kèm 6 bảng phụ (bureau, bureau_balance, previous_application, POS_CASH_balance, credit_card_balance,
installments_payments) nối qua `SK_ID_CURR` / `SK_ID_PREV`, và `application_test` (không nhãn).
File: `data/interim/home_credit/*.parquet`.
""",
    load="""
NAME = "home_credit"
HC = INTERIM / "home_credit"
df = pd.read_parquet(HC / "application_train.parquet")
df.shape
""",
    dist="""
top = prof.sort_values("auc", ascending=False).index[:12].tolist()
plot_df = df.copy()
plot_df["DAYS_EMPLOYED"] = plot_df["DAYS_EMPLOYED"].replace(365243, np.nan)
eda.plot_numeric_by_target(plot_df, top, NAME)
""",
    cat="""
cats = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS", "OCCUPATION_TYPE"]
display(eda.categorical_bad_rate(df, cats).query("n >= 100"))
eda.plot_categorical_bad_rate(df, cats, NAME)
""",
    checks=[
        "### Bảng phụ",
        """
import pyarrow.parquet as pq
rows = []
for p in sorted(HC.glob("*.parquet")):
    meta = pq.ParquetFile(p).metadata
    rows.append({"table": p.stem, "rows": meta.num_rows, "cols": meta.num_columns})
display(pd.DataFrame(rows))
b = pd.read_parquet(HC / "bureau.parquet", columns=["SK_ID_CURR"])
pa_ = pd.read_parquet(HC / "previous_application.parquet", columns=["SK_ID_CURR"])
print(f"Khách có bản ghi bureau: {df.SK_ID_CURR.isin(b.SK_ID_CURR).mean():.1%}; "
      f"có hồ sơ vay trước tại Home Credit: {df.SK_ID_CURR.isin(pa_.SK_ID_CURR).mean():.1%}")
""",
        "### Giá trị bất thường",
        """
anom = df["DAYS_EMPLOYED"] == 365243
print(f"DAYS_EMPLOYED = 365243 (≈1000 năm, mã cho người không đi làm/hưu trí): {anom.sum():,} dòng ({anom.mean():.1%}), "
      f"bad rate {df.loc[anom, 'target'].mean():.1%} vs {df.loc[~anom, 'target'].mean():.1%}")
print("AMT_INCOME_TOTAL max:", f"{df.AMT_INCOME_TOTAL.max():,.0f}", "— median:", f"{df.AMT_INCOME_TOTAL.median():,.0f}")
print("CODE_GENDER:", df.CODE_GENDER.value_counts().to_dict())
print("ORGANIZATION_TYPE = 'XNA':", (df.ORGANIZATION_TYPE == "XNA").sum())
m = df.isna().mean()
print(f"Cột thiếu > 50%: {(m > 0.5).sum()} / {df.shape[1]} (chủ yếu nhóm đặc trưng căn hộ *_AVG/_MODE/_MEDI)")
flags = [c for c in df if c.startswith("FLAG_DOCUMENT")]
print(f"FLAG_DOCUMENT_* gần như hằng (<1% = 1): {sum(df[c].mean() < 0.01 for c in flags)} / {len(flags)}")
age = -df.DAYS_BIRTH / 365.25
display(df.groupby(pd.cut(age, [20, 30, 40, 50, 60, 70]), observed=True)["target"].agg(["size", "mean"]).rename_axis("tuổi"))
""",
    ],
    notes="""
- **Nhãn: mất cân bằng nặng** — bad rate ≈ **8,1%** (≈ 11:1), tương tự GMSC. Nhãn định nghĩa là “trễ hạn hơn X ngày trong Y kỳ đầu tiên” → nhãn *sớm* (early default), không phải vỡ nợ cuối cùng.
- **Dữ liệu rộng và thưa:** 121 đặc trưng, ~24% ô bị thiếu, **41 cột thiếu > 50%** (thông tin về nhà ở). Thiếu dữ liệu mang thông tin yếu (chênh bad rate chỉ vài điểm %).
- **Tín hiệu yếu và phân tán:** không biến đơn nào có AUC > 0,68; ba điểm tín dụng bên ngoài `EXT_SOURCE_1/2/3` là mạnh nhất, tiếp theo là tuổi (người trẻ bad rate ~11,5% so với ~5% ở nhóm 60+). Cần kết hợp nhiều đặc trưng và khai thác các bảng phụ.
- **Giá trị bất thường:** `DAYS_EMPLOYED = 365243` ở ~18% hồ sơ (mã cho người không đi làm; bad rate thấp hơn, ~5,4%); thu nhập max 117 triệu (skew ≈ 390); `CODE_GENDER = XNA`.
- 22/73 biến số có |skew| > 5; nhiều cặp biến trùng lặp gần như hoàn toàn (nhóm *_AVG/_MODE/_MEDI), 16/20 cờ tài liệu gần như hằng.
- **Cấu trúc quan hệ:** dữ liệu gồm 7 bảng (tới 27 triệu dòng) → cần tổng hợp (aggregation) theo khách hàng trước khi mô hình hoá — khác hẳn các bộ dữ liệu một bảng còn lại.
""",
    extra='{"days_employed_anomaly_pct": round(100 * anom.mean(), 2), "cols_missing_gt50": int((m > 0.5).sum())}',
)

SPECS["eda_08_lending_club"] = dict(
    title="""
# EDA 08 — Lending Club (2007 – 2018Q4)

Nguồn: Kaggle (wordsforthewise) — toàn bộ khoản vay P2P được **chấp nhận** (2,26 triệu, 151 cột) và hồ sơ **bị từ chối**
(27,6 triệu, 9 cột). File: `data/interim/lending_club/{accepted,rejected}.parquet`.

Nhãn được suy ra từ `loan_status`:
`Charged Off`, `Default`, `Does not meet the credit policy. Status:Charged Off` → 1;
`Fully Paid`, `Does not meet the credit policy. Status:Fully Paid` → 0;
các khoản **chưa kết thúc** (`Current`, `Late`, `In Grace Period`) → `target = NaN` (chưa biết kết cục).

Phần lớn phân tích bên dưới chỉ dùng các khoản đã kết thúc (*resolved*) và các biến có tại thời điểm cho vay.
""",
    load="""
NAME = "lending_club"
LC = INTERIM / "lending_club"
APP_COLS = ["loan_amnt", "term", "int_rate", "installment", "grade", "emp_length", "home_ownership",
            "annual_inc", "verification_status", "purpose", "dti", "delinq_2yrs", "fico_range_low",
            "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
            "mort_acc", "pub_rec_bankruptcies", "application_type"]
POST_COLS = ["recoveries", "total_rec_prncp", "total_pymnt", "last_fico_range_high", "debt_settlement_flag"]
full = pd.read_parquet(LC / "accepted.parquet", columns=APP_COLS + POST_COLS + ["issue_d", "loan_status", "target"])
full["issue_year"] = pd.to_datetime(full["issue_d"], format="%b-%Y").dt.year
print("Tất cả khoản vay:", full.shape)
print(full["loan_status"].value_counts(dropna=False))
df = full[full["target"].notna()].drop(columns=POST_COLS + ["issue_d", "loan_status"]).copy()
df["target"] = df["target"].astype(int)
df.shape
""",
    dist="""
top = ["int_rate", "fico_range_low", "dti", "annual_inc", "loan_amnt", "revol_util", "inq_last_6mths", "mort_acc"]
eda.plot_numeric_by_target(df, top, NAME, log=["annual_inc", "loan_amnt"])
""",
    cat="""
cats = ["term", "grade", "home_ownership", "verification_status", "purpose", "emp_length"]
display(eda.categorical_bad_rate(df, cats).query("n >= 500"))
eda.plot_categorical_bad_rate(df, cats, NAME)
""",
    checks=[
        "### Nhãn phụ thuộc thời gian (vintage) và khoản chưa kết thúc",
        """
by_year = full.groupby("issue_year").agg(
    n=("loan_status", "size"),
    resolved_pct=("target", lambda s: 100 * s.notna().mean()),
    bad_rate_resolved=("target", lambda s: 100 * s.mean()),
).round(1)
display(by_year)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(by_year.index, by_year["resolved_pct"], color="#bbb", label="% đã kết thúc")
ax2 = ax.twinx()
ax2.plot(by_year.index, by_year["bad_rate_resolved"], "o-", color="#DD5144", label="bad rate (đã kết thúc)")
ax.set_ylabel("% resolved"); ax2.set_ylabel("bad rate %")
ax.set_title("Lending Club: tỉ lệ đã kết thúc và bad rate theo năm giải ngân")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), fontsize=8)
eda.savefig(fig, f"{NAME}_vintage")
plt.show()
""",
        "### Rò rỉ thông tin từ các biến sau khi giải ngân",
        """
from sklearn.metrics import roc_auc_score
res = full[full["target"].notna()]
rows = []
for c in ["recoveries", "total_rec_prncp", "last_fico_range_high", "int_rate", "fico_range_low"]:
    ok = res[c].notna()
    a = roc_auc_score(res.loc[ok, "target"], res.loc[ok, c])
    rows.append({"feature": c, "kind": "post-loan (leak)" if c in POST_COLS else "application", "auc": round(max(a, 1 - a), 3)})
display(pd.DataFrame(rows))
display(res.groupby("debt_settlement_flag")["target"].agg(["size", "mean"]))
""",
        "### Giá trị bất thường và mức độ thiếu của bảng đầy đủ",
        """
import pyarrow.parquet as pq
t = pq.read_table(LC / "accepted.parquet")
null_pct = pd.Series({c: t[c].null_count / t.num_rows for c in t.column_names})
print(f"Cột thiếu > 50%: {(null_pct > 0.5).sum()} / {len(null_pct)};  > 90%: {(null_pct > 0.9).sum()}")
del t
print("annual_inc: max =", f"{df.annual_inc.max():,.0f}", "| = 0:", (df.annual_inc == 0).sum())
print("dti: min =", df.dti.min(), "| max =", df.dti.max(), "| > 100:", (df.dti > 100).sum())
""",
        "### Hồ sơ bị từ chối",
        """
rej = pd.read_parquet(LC / "rejected.parquet", columns=["Application Date", "Risk_Score", "Amount Requested"])
rej_year = pd.to_datetime(rej["Application Date"]).dt.year
cmp = pd.DataFrame({"accepted": full.groupby("issue_year").size(), "rejected": rej_year.value_counts().sort_index()})
cmp["acceptance_pct"] = (100 * cmp.accepted / (cmp.accepted + cmp.rejected)).round(1)
display(cmp)
print(f"Risk_Score thiếu ở {rej.Risk_Score.isna().mean():.0%} hồ sơ bị từ chối; median = {rej.Risk_Score.median():.0f} "
      f"(accepted fico_range_low median = {full.fico_range_low.median():.0f})")
""",
    ],
    notes="""
- **Nhãn không có sẵn mà phải suy ra**, và **~40% khoản vay chưa có kết cục** (`Current`/`Late`) → phải loại bỏ. Trên 1,35 triệu khoản đã kết thúc, bad rate ≈ **20%** (≈ 4:1).
- **Nhãn phụ thuộc thời gian — bất thường quan trọng nhất:** bad rate theo năm giải ngân dao động 14% → 23%; các năm 2017–2018 chỉ có 11–38% khoản đã kết thúc, nên mẫu “đã kết thúc” bị **thiên lệch** (khoản vỡ nợ sớm hoặc trả sớm). Kỳ hạn 60 tháng có bad rate gấp đôi 36 tháng (≈ 32% vs 16%). → cần chia train/test theo thời gian, không chia ngẫu nhiên.
- **Rò rỉ thông tin nghiêm trọng:** nhiều cột chỉ có sau khi giải ngân (`recoveries`, `total_rec_prncp`, `last_fico_range_*`, `debt_settlement_flag`…) cho AUC 0,84–0,94 và `debt_settlement_flag = Y` ⇒ bad ≈ 100%. Phải loại khỏi đặc trưng.
- **Nhãn gắn chặt với xếp hạng nội bộ:** bad rate tăng đều từ grade A (~6%) tới G (~50%); `int_rate` (AUC ≈ 0,68) là biến mạnh nhất vì nó *được định giá từ* rủi ro.
- **Thiên lệch chọn mẫu (selection bias):** chỉ có nhãn cho khoản được duyệt; tỉ lệ chấp nhận chỉ ~5–15% số hồ sơ (giảm còn ~5% năm 2018), điểm rủi ro của hồ sơ bị từ chối thấp hơn rõ (median 637 so với 690). Mô hình học trên đó không đại diện cho toàn bộ người nộp đơn (vấn đề *reject inference*).
- **Phân phối:** `annual_inc` cực lệch (max ≈ 11 triệu USD trong mẫu đã kết thúc, 110 triệu trong toàn bộ; 361 người thu nhập = 0), `dti` có giá trị -1 và 999 (mã lỗi, 533 dòng > 100); 44/152 cột thiếu > 50% (biến mới chỉ có từ ~2015, biến cho khoản vay chung).
""",
    extra='{"resolved_rows": int(len(df)), "all_rows": int(len(full)), "unresolved_pct": round(100 * full.target.isna().mean(), 2)}',
)

SPECS["eda_00_comparison"] = None  # built separately below

COMPARISON = [
    md("""
# EDA 00 — So sánh các bộ dữ liệu

Notebook này đọc các file tóm tắt `reports/eda/*.json` do từng notebook EDA sinh ra và so sánh:
kích thước, tỉ lệ nợ xấu, mức thiếu dữ liệu, độ lệch phân phối và sức phân biệt của biến mạnh nhất.
Chạy **sau** các notebook `eda_01` … `eda_08`.
"""),
    code(SETUP + "import json\n"),
    code("""
ORDER = ["hmeq", "fico_heloc", "taiwan_default_credit", "give_me_some_credit",
         "german_credit", "south_german_credit", "home_credit", "lending_club"]
S = {n: json.load(open(ROOT / "reports" / "eda" / f"{n}.json")) for n in ORDER}
tab = pd.DataFrame({
    n: {
        "rows (có nhãn)": s["target"]["n"],
        "features": s["overview"]["features"],
        "bad_rate_%": s["target"]["bad_rate_pct"],
        "good:bad": s["target"]["imbalance_ratio"],
        "ô thiếu %": s["overview"]["cells_missing_pct"],
        "cột có thiếu": s["overview"]["features_with_missing"],
        "dòng trùng": s["overview"]["duplicate_rows"],
        "biến số |skew|>2": f'{s["n_numeric_abs_skew_gt2"]}/{s["n_numeric"]}',
        "max |skew|": round(s["max_abs_skew"], 1),
        "AUC đơn biến tốt nhất": round(max(s["top_auc"].values()), 3),
    } for n, s in S.items()
}).T
tab
"""),
    code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
t = tab.sort_values("bad_rate_%")
axes[0].barh(t.index, t["bad_rate_%"].astype(float), color="#DD5144")
axes[0].set_title("Tỉ lệ nợ xấu (%)")
for y, v in enumerate(t["bad_rate_%"].astype(float)):
    axes[0].text(v + 0.5, y, f"{v:.1f}", va="center", fontsize=8)
axes[1].barh(t.index, np.log10(t["rows (có nhãn)"].astype(float)), color="#4C72B0")
axes[1].set_title("log10(số dòng có nhãn)")
axes[2].barh(t.index, t["AUC đơn biến tốt nhất"].astype(float), color="#55A868")
axes[2].set_xlim(0.5, 1); axes[2].set_title("AUC của biến đơn mạnh nhất")
for ax in axes[1:]:
    ax.set_yticklabels([])
fig.tight_layout()
eda.savefig(fig, "comparison_overview")
plt.show()
"""),
    code("""
fig, ax = plt.subplots(figsize=(7, 4))
x = tab["ô thiếu %"].astype(float); y = tab["max |skew|"].astype(float)
ax.scatter(x, np.log10(y), s=np.log10(tab["rows (có nhãn)"].astype(float)) * 40, c=tab["bad_rate_%"].astype(float),
           cmap="Reds", edgecolor="k")
for n in tab.index:
    ax.annotate(n, (x[n], np.log10(y[n])), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("% ô bị thiếu"); ax.set_ylabel("log10(max |skew|)")
ax.set_title("Chất lượng dữ liệu: thiếu vs độ lệch (kích thước điểm ~ số dòng, màu ~ bad rate)")
eda.savefig(fig, "comparison_quality")
plt.show()
"""),
    code("""
tab.to_csv(ROOT / "reports" / "tables" / "eda_comparison.csv")
print("Đã lưu reports/tables/eda_comparison.csv")
"""),
]


HMEQ_MISSING = [
    md("""
# EDA 09 — HMEQ: giá trị thiếu và rò rỉ thông tin

Notebook này phân tích riêng **mẫu giá trị thiếu** của HMEQ. Kiểm tra rò rỉ (`reports/models/validity/`) cho thấy
một mô hình chỉ biết *cột nào bị thiếu* (không thấy giá trị nào) đạt AUC 0,836 — cao hơn Logistic Regression dùng
toàn bộ dữ liệu (0,803). Câu hỏi: đây là thông tin hợp lệ về người vay, hay dấu vết của quy trình ghi nhận hồ sơ sau
khi biết kết cục (rò rỉ)? Và cách ngăn rò rỉ ảnh hưởng thế nào đến kết quả.

Theo mô tả gốc, dữ liệu gồm *"recent applicants granted credit"*: các khoản vay đã được cấp, nhãn `BAD` được ghi
nhận sau đó (vỡ nợ hoặc quá hạn nghiêm trọng).
"""),
    code(SETUP + """
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier
import seaborn as sns
NAME = "hmeq_missing"
df = pd.read_csv(INTERIM / "hmeq.csv")
X = df.drop(columns="target")
M = X.isna()
print(df.shape, f"bad rate {df.target.mean():.2%}")
"""),
    md("## 1. Mức thiếu và tỷ lệ xấu theo từng cột"),
    code("""
t = pd.DataFrame({
    "missing_pct": 100 * M.mean(),
    "bad_if_missing_pct": [100 * df.target[M[c]].mean() if M[c].any() else np.nan for c in X],
    "bad_if_present_pct": [100 * df.target[~M[c]].mean() for c in X],
}).round(1).sort_values("missing_pct", ascending=False)
display(t)
fig, ax = plt.subplots(figsize=(9, 3.6))
tt = t[t.missing_pct > 0]
x = np.arange(len(tt))
ax.bar(x - 0.2, tt.bad_if_missing_pct, 0.4, label="khi thiếu", color="#DD5144")
ax.bar(x + 0.2, tt.bad_if_present_pct, 0.4, label="khi có giá trị", color="#4C72B0")
ax.axhline(100 * df.target.mean(), color="k", ls="--", lw=1, label="tỷ lệ xấu chung")
ax.set_xticks(x, tt.index, rotation=30)
ax.set_ylabel("Tỷ lệ xấu (%)")
ax.set_title("HMEQ: tỷ lệ xấu khi cột bị thiếu và khi có giá trị")
ax.legend(fontsize=8)
eda.savefig(fig, "hmeq_missing_bad_rate")
plt.show()
"""),
    md("## 2. Tỷ lệ xấu theo số cột bị thiếu và theo mẫu thiếu"),
    code("""
n_miss = M.sum(axis=1)
display(df.groupby(n_miss.clip(upper=6))["target"].agg(n="size", bad_rate="mean").round(3)
        .rename_axis("số cột thiếu (≥6 gộp)").T)
pattern = M.apply(lambda r: " + ".join(c for c, v in r.items() if v) or "(đủ dữ liệu)", axis=1)
pat = df.groupby(pattern)["target"].agg(n="size", bad_rate="mean").sort_values("n", ascending=False)
display(pat.head(15).round(3))
big = pat[pat.n >= 30]
print(f"Các mẫu thiếu có ≥ 30 dòng: {len(big)}; tỷ lệ xấu từ {big.bad_rate.min():.1%} đến {big.bad_rate.max():.1%}")
"""),
    md("""
**Đọc kết quả.** Việc thiếu dữ liệu tách nhãn theo **cả hai chiều**:

- thiếu `DEBTINC` (tỷ lệ nợ/thu nhập) → khoảng 61% xấu; thiếu cả `VALUE` và `DEBTINC` → 100% xấu;
- thiếu các biến lịch sử tín dụng (`DEROG`, `DELINQ`, `NINQ`) hoặc `JOB` → gần như 0–5% xấu;
- hồ sơ đầy đủ → khoảng 9% xấu.

Một quy luật "sạch" như vậy (100% so với 0%) khó giải thích bằng hành vi của người vay tại thời điểm nộp đơn; nó
phù hợp hơn với giả thuyết các trường được **điền hoặc để trống khác nhau tùy theo kết cục** (ví dụ hồ sơ khoản vay
xấu được xử lý, cập nhật khác hồ sơ khoản vay tốt).
"""),
    md("## 3. Mẫu thiếu có cấu trúc: tương quan giữa các cờ thiếu"),
    code("""
mc = M.loc[:, M.any()].astype(int)
fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(mc.corr(), cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax,
            annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
ax.set_title("Tương quan giữa các cờ 'bị thiếu'")
eda.savefig(fig, "hmeq_missing_corr")
plt.show()
"""),
    md("## 4. Chỉ riêng mẫu thiếu dự đoán được nhãn đến đâu?"),
    code("""
cv = StratifiedKFold(5, shuffle=True, random_state=42)
ind = mc.astype(int)
auc_missing = cross_val_score(LGBMClassifier(n_estimators=100, verbose=-1), ind, df.target, cv=cv, scoring="roc_auc")
print(f"AUC 5-fold khi CHỈ dùng cờ thiếu (không dùng giá trị nào): {auc_missing.mean():.3f} ± {auc_missing.std():.3f}")
comp = df[~M.any(axis=1)]
print(f"Hồ sơ đầy đủ: {len(comp):,} dòng ({len(comp)/len(df):.1%}), tỷ lệ xấu {comp.target.mean():.1%}; "
      f"hồ sơ thiếu ít nhất một cột: {(M.any(axis=1)).sum():,} dòng, tỷ lệ xấu {df.target[M.any(axis=1)].mean():.1%}")
"""),
    md("""
## 5. Ngăn rò rỉ: hai phiên bản HMEQ

| Phiên bản | Cách làm | Ưu điểm | Nhược điểm |
|---|---|---|---|
| `hmeq` (gốc) | Tiền xử lý Baesens: median + nhóm `__missing__` | Đúng giao thức | Mô hình đọc được mẫu thiếu: nhóm `__missing__`, và cây nhận ra giá trị median được điền |
| `hmeq_complete` | Tầng A chỉ giữ hồ sơ đầy đủ (3.364 dòng) | Không còn giá trị thiếu nào | Đổi tổng thể: tỷ lệ xấu còn 8,9%; bỏ 44% dữ liệu |
| `hmeq_blind` | Giữ mọi dòng và đúng các fold; tầng B.1 điền mỗi giá trị thiếu bằng **một giá trị rút ngẫu nhiên** từ các giá trị quan sát được của cột đó trên train | Giữ tổng thể và đúng các fold; xóa nhóm `__missing__` và điểm median | Không xóa hết dấu vết: giá trị rút ngẫu nhiên phá vỡ quan hệ giữa các cột (ví dụ dư nợ và giá trị nhà), nên dòng từng thiếu vẫn nhận ra được một phần |
"""),
    code("""
from src.models.compare_models import NAMES, RUNS
rows = []
for r in RUNS:
    s = pd.read_csv(ROOT / "reports" / "models" / r / "summary.csv").set_index("dataset")
    rows.append({"model": NAMES[r], **{v: s.loc[v, "auc"] for v in ["hmeq", "hmeq_blind", "hmeq_complete"]}})
res = pd.DataFrame(rows).set_index("model")
res["blind − gốc"] = res.hmeq_blind - res.hmeq
res["complete − gốc"] = res.hmeq_complete - res.hmeq
display(res.round(4))
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(len(res)); w = 0.27
for i, (c, lab, col) in enumerate([("hmeq", "gốc", "#999999"), ("hmeq_blind", "che giá trị thiếu", "#4C72B0"),
                                    ("hmeq_complete", "chỉ hồ sơ đầy đủ", "#55A868")]):
    ax.bar(x + (i - 1) * w, res[c], w, label=lab, color=col)
ax.set_xticks(x, res.index); ax.set_ylim(0.6, 1.0); ax.set_ylabel("AUC test")
ax.set_title("HMEQ: AUC trước và sau khi ngăn rò rỉ qua giá trị thiếu"); ax.legend(fontsize=8)
eda.savefig(fig, "hmeq_leakage_control_auc")
plt.show()
"""),
    code("""
v = ROOT / "reports" / "models" / "validity" / "leakage.csv"
if v.exists():
    lk = pd.read_csv(v).set_index("dataset").loc[["hmeq", "hmeq_blind", "hmeq_complete"],
                                                ["missingness_only_auc", "missing_rows_detectable_auc"]]
    display(lk.round(3))
    print("missing_rows_detectable_auc: một mô hình có nhận ra được 'dòng này từng bị thiếu' từ ma trận sau tiền xử lý không "
          "(0,5 = không nhận ra).")
"""),
    md("""
## 6. Kết luận

- Mẫu giá trị thiếu của HMEQ dự đoán nhãn mạnh hơn cả Logistic Regression dùng toàn bộ dữ liệu, và tách nhãn theo
  cả hai chiều (tới 100% xấu và 0% xấu). Đây nhiều khả năng là **rò rỉ do quy trình ghi nhận hồ sơ**, không phải thông
  tin có sẵn khi xét duyệt.
- Với tiền xử lý Baesens, mô hình nhận ra gần như chắc chắn dòng nào từng bị thiếu (AUC 0,995) qua nhóm `__missing__`
  và giá trị median được điền. `hmeq_blind` giảm khả năng này xuống 0,907 — chưa xóa hết, vì giá trị rút ngẫu nhiên
  không khớp với các cột còn lại. `hmeq_complete` loại bỏ hoàn toàn (không còn giá trị thiếu) nhưng đổi tổng thể.
- Dấu vết còn lại của `hmeq_blind` gần như không mang thông tin về nhãn: AUC của `hmeq_blind` và `hmeq_complete`
  (sạch tuyệt đối) gần như trùng nhau ở mọi mô hình cây và LR (chênh ≤ 0,005).
- Sau khi ngăn rò rỉ, AUC của mọi mô hình giảm (mô hình cây giảm khoảng 0,03–0,04), nhưng **thứ tự các mô hình gần như
  giữ nguyên** và mô hình cây vẫn hơn LR rất xa: lợi thế của mô hình cây trên HMEQ là thật, chỉ có mức AUC tuyệt đối
  bị thổi phồng bởi rò rỉ.
"""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", nargs="*",
                    help="build (and execute) only these notebook stems; others are left untouched")
    args = ap.parse_args()
    NB_DIR.mkdir(exist_ok=True)

    def new_nb(cells):
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells
        nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        return nb

    notebooks = {name: build(spec) for name, spec in SPECS.items() if spec is not None}
    notebooks["eda_00_comparison"] = new_nb(COMPARISON)  # after eda_01..08: reads their summaries
    notebooks["eda_09_hmeq_missingness"] = new_nb(HMEQ_MISSING)

    paths = []
    for name, nb in notebooks.items():
        if args.only and name not in args.only:
            continue  # never overwrite (and wipe the outputs of) notebooks that were not asked for
        p = NB_DIR / f"{name}.ipynb"
        nbf.write(nb, p)
        paths.append(p)
    print("\n".join(str(x.relative_to(ROOT)) for x in paths))

    if args.execute:
        from nbclient import NotebookClient
        for p in paths:
            print("executing", p.name, flush=True)
            nb = nbf.read(p, as_version=4)
            NotebookClient(nb, timeout=1800, kernel_name="python3",
                           resources={"metadata": {"path": str(NB_DIR)}}).execute()
            nbf.write(nb, p)


if __name__ == "__main__":
    main()
