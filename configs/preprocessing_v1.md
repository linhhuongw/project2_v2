# Tiền xử lý và chia dữ liệu thống nhất theo giao thức Baesens et al. (2026)

## Viết tắt

| Viết tắt | Đầy đủ |
|---|---|
| CV | Cross-Validation (kiểm định chéo) |
| LR | Logistic Regression (hồi quy logistic) |
| RF | Random Forest (rừng ngẫu nhiên) |
| KNN | K-Nearest Neighbors (k láng giềng gần nhất) |
| NB | Naive Bayes |
| SVM | Support Vector Machine (máy vector hỗ trợ) |
| GBDT | Gradient Boosted Decision Trees (XGBoost, LightGBM, CatBoost) |
| FE | Feature Engineering (tạo và biến đổi đặc trưng) |
| AUC | Diện tích dưới đường ROC |
| GMSC | Bộ dữ liệu Give Me Some Credit |
| HELOC | Bộ dữ liệu FICO Home Equity Line of Credit |
| NaN | Giá trị thiếu (Not a Number) |

## Nguồn

| Vai trò | Paper |
|---|---|
| **Giao thức chính** (tiền xử lý, chia dữ liệu, tinh chỉnh) | Baesens, B., Goethals, A., Lessmann, S., De Vos, S., Bravo, C., Martens, D., Medina-Olivares, V., Mues, C., Óskarsdóttir, M., vanden Broucke, S., Verdonck, T., & Verbeke, W. (2026). *Foundation Models for Credit Risk Prediction: A Game Changer?* arXiv:2605.18147. Mục 4.2 và 4.3 |
| Tham chiếu HELOC | Dervovic, D., Lécué, F., Marchesotti, N., & Magazzeni, D. (2024). *Are Logistic Models Really Interpretable?* arXiv:2406.13427 |

**Cách diễn đạt trong báo cáo:** "áp dụng giao thức của Baesens et al. (2026)", không phải "tái lập kết quả của Baesens et al. (2026)". Paper chỉ công bố AUC trung bình trên 14 bộ, không có kết quả theo từng bộ, nên không đối chiếu được con số từng bộ.

---

## Tổng quan

```
Dữ liệu gốc
   │
   ▼
[Tầng A] Khai báo dữ liệu          ← BỔ SUNG; không học từ dữ liệu; chạy một lần trên toàn bộ dữ liệu
   │
   ▼
Chia dữ liệu                        ← 5-fold CV có nhóm theo dòng trùng lặp; Lending Club chia theo thời gian
   │
   ▼
[Tầng B.0] Sàng lọc cột             ← BỔ SUNG; fit CHỈ trên train của từng fold
   │
   ▼
[Tầng B] Pipeline Baesens           ← GIỮ ĐÚNG PAPER; fit CHỈ trên train của từng fold
   │
   ▼
Tinh chỉnh (Optuna) → đánh giá
```

**Nguyên tắc chống rò rỉ:** mọi bước **học từ dữ liệu** (median, trung bình, tần suất, tỷ lệ thiếu, số giá trị khác nhau…) chỉ được fit trên train của từng fold, rồi áp dụng sang validation và test của fold đó.

**Phép thử cho một bước bất kỳ:** nếu chỉ đưa vào **một dòng duy nhất** mà bước đó vẫn cho đúng kết quả như khi đưa cả bộ dữ liệu, bước đó không học từ dữ liệu và được phép chạy trước khi chia. Ngược lại, bước đó phải nằm ở tầng B.0 hoặc B.

---

## Tầng A — Khai báo dữ liệu (bổ sung)

### A.1. Quy tắc chung

| # | Thao tác | Bắt buộc trước khi chia? | Lý do |
|---|---|---|---|
| A1 | Chọn đúng bảng/tệp có nhãn | **Có** | Xác định dữ liệu được chia |
| A2 | Lọc dòng: bỏ dòng thiếu nhãn, khoản vay chưa kết thúc, hồ sơ ngoài phạm vi, ngoài cửa sổ thời gian | **Có** | Thay đổi số dòng; làm sau sẽ lệch tỷ lệ các fold |
| A3 | Mã hóa nhãn: **1 = vỡ nợ / xấu**, 0 = tốt | **Có** | Cần nhãn đúng để chia phân tầng |
| A4 | Chuyển cột thời gian dùng để chia sang dạng ngày | **Có** (Lending Club) | Cần cho chia theo thời gian |
| A5 | Bỏ cột ID, văn bản tự do, URL | Không (tiện làm trước) | Không mang thông tin dự báo |
| A6 | Bỏ cột rò rỉ (chỉ biết sau thời điểm quyết định cho vay) | Không (tiện làm trước) | Tránh hiệu năng ảo |
| A7 | Bỏ cột **chắc chắn thành hằng số do chính các bộ lọc ở A2** | Không (tiện làm trước) | Để log của B.0 chỉ còn các trường hợp bất ngờ |
| A8 | Đổi mã giá trị đặc biệt (theo tài liệu dữ liệu) thành NaN | Không (tiện làm trước) | Để tầng B xử lý như giá trị thiếu |
| A9 | Đổi định dạng: chuỗi → số (`"13.5%"` → 13.5), ngày → năm thập phân | Không (tiện làm trước) | Chỉ đổi định dạng, không tạo thông tin mới |
| A10 | Khai báo kiểu cột: số hay phân loại | Không (tiện làm trước) | Để tầng B chọn đúng nhánh xử lý |

**Không làm ở tầng A** (vì học từ dữ liệu hoặc là FE): điền giá trị thiếu, chuẩn hóa, one-hot, cắt ngoại lai, bỏ cột hằng số theo thống kê, bỏ cột thiếu nhiều, gom nhóm hiếm, tạo feature mới, chọn feature theo hiệu năng.

### A.2. Khai báo theo từng bộ dữ liệu

| Dataset | Bảng / tệp | Lọc dòng | Nhãn → 1 = xấu | Bỏ cột | Mã đặc biệt → NaN | Biến phân loại | Ghi chú |
|---|---|---|---|---|---|---|---|
| **GMSC** | `cs-training.csv` | Không | `SeriousDlqin2yrs` (đã là 1 = xấu) | Cột chỉ số dòng không tên | Không | Không có | Giá trị 96/98 bất thường ở các cột đếm trễ hạn: giữ nguyên, ghi chú trong báo cáo |
| **Taiwan** | Bảng chính | Không | `default payment next month` (đã là 1 = xấu) | `ID` | Không | `SEX`, `EDUCATION`, `MARRIAGE` | `PAY_0`…`PAY_6` giữ là biến số |
| **Home Credit** | `application_train.csv` (chỉ bảng chính) | Không | `TARGET` (đã là 1 = xấu) | Không bỏ `SK_ID_CURR`: **giữ làm khóa** (cột không phải feature) để nối bảng phụ ở bước FE; **phải bỏ khỏi X khi huấn luyện mô hình** | `DAYS_EMPLOYED = 365243` | Các cột kiểu chuỗi (`NAME_*`, `CODE_GENDER`, `OCCUPATION_TYPE`…) | Bảng phụ để dành cho bước FE; `application_test` không có nhãn. `SK_ID_CURR` nằm trong `non_features` của `meta.json`, nên `xy()` trong `src/data/splits.py` đã tự loại khỏi X |
| **HMEQ** | `hmeq.csv` | Không | `BAD` (đã là 1 = xấu) | Không | Không | `REASON`, `JOB` | |
| **South German / German** | Bảng chính | Không | Kiểm tra mã hóa của phiên bản đang dùng; đổi sao cho xấu = 1 (Statlog German gốc: 1 = tốt, 2 = xấu) | Không | Không | Các biến mã hóa dạng nhóm | Chỉ dùng **một** trong hai bộ |
| **HELOC** | Bản đã làm sạch (**n = 9.871**) | Bản gốc 10.459 dòng; bản sử dụng đã loại 588 dòng có **mọi** feature = −9 | `RiskPerformance`: `"Bad"` → 1, `"Good"` → 0 | Không | −7, −8, −9 ở mọi cột | `MaxDelq2PublicRecLast12M`, `MaxDelqEver` | 588 dòng bị loại không mang thông tin |
| **Lending Club** | Theo phiên bản sử dụng | Xem A.3 | `Charged Off`, `Default` → 1; `Fully Paid` → 0 | Xem A.3 | Không | `grade`, `sub_grade`, `home_ownership`, `verification_status`, `purpose`, `addr_state`, `emp_length`, `initial_list_status`, `disbursement_method` | Xem A.3 |

### A.3. Lending Club — khai báo chi tiết

**Lọc dòng, theo thứ tự:**

| Thứ tự | Bộ lọc | Lý do |
|---|---|---|
| 1 | Chỉ giữ khoản vay đã kết thúc: `Fully Paid`, `Charged Off`, `Default` (gộp các trạng thái dạng `"Does not meet the credit policy. Status:…"` vào trạng thái tương ứng) | Chỉ khoản vay đã kết thúc mới có nhãn |
| 2 | Chỉ giữ `application_type == "Individual"` | Hồ sơ đồng vay chỉ có từ khoảng 2015, tạo dịch chuyển tổng thể theo thời gian; loại tận gốc 13 cột `sec_app_*` và các trường joint |
| 3 | Chỉ giữ `term == 36` | Khoản vay 60 tháng đáo hạn chậm, làm nặng thêm vấn đề khoản vay chưa kết thúc |
| 4 | **Chỉ giữ `issue_d` từ năm X trở đi** (cửa sổ có đủ dữ liệu bureau) | 22 trường bureau (`tot_cur_bal`, `avg_cur_bal`, `num_*`, `mo_sin_*`…) không có ở khoản vay trước khoảng 2012–2013. Nếu giữ, khoảng 31% train thiếu các trường này trong khi validation và test đầy đủ; điền median khi đó thực chất là tín hiệu "khoản vay cũ" chỉ có ở train |

**Cách xác định năm X** (chỉ nhìn giai đoạn train, chỉ nhìn tỷ lệ thiếu của feature, không dùng nhãn):

```python
bureau = [...]                                   # 22 cột được cảnh báo ở B.0
yr = lc["issue_d"].dt.year
miss_by_year = lc.loc[yr < val_year, bureau].isna().mean(axis=1).groupby(yr[yr < val_year]).mean()
start_year = miss_by_year[miss_by_year <= 0.02].index.min()   # thiếu ≤ 2%
```

Sau khi xác định, **ghi cứng** năm X vào cấu hình tầng A; khi đó bộ lọc là một quy tắc cố định theo từng dòng. Ghi lại số dòng bị loại và tỷ lệ vỡ nợ của train trước và sau khi lọc.

**Kết quả đã chạy — năm X = 2013** (ghi cứng: `LC_START_YEAR = 2013` trong `src/data/datasets.py`).

Tỷ lệ thiếu trung bình của 22 trường bureau theo năm phát hành, trên giai đoạn train (sau bộ lọc 1–3, trước năm validation 2014):

| Năm | 2007–2011 | 2012 | 2013 |
|---|---:|---:|---:|
| Tỷ lệ thiếu | 100% | 52,0% | **0,2%** |

→ Năm đầu tiên có tỷ lệ thiếu ≤ 2% là **2013**, khớp với giá trị đã ghi cứng.

| Train (các năm < 2014) | Số dòng | Tỷ lệ vỡ nợ |
|---|---:|---:|
| Trước bộ lọc năm X (2007–2013) | 175.426 | 12,63% |
| Sau bộ lọc năm X (2013) | 100.422 | 12,33% |
| **Bị loại** | **75.004** | |

Tỷ lệ vỡ nợ của train gần như không đổi (−0,3 điểm %), nên bộ lọc không làm lệch nhãn.

**Cài đặt:** các bộ lọc chạy theo thứ tự "hồ sơ cá nhân → 36 tháng và `issue_d ≥ 2013` → khoản vay đã kết thúc". Thứ tự khác bảng trên nhưng kết quả giống hệt, vì mỗi bộ lọc là một điều kiện độc lập theo từng dòng.

**Bỏ cột:**

| Nhóm | Cột |
|---|---|
| ID / văn bản | `id`, `member_id`, `url`, `desc`, `title`, `emp_title` |
| Rò rỉ | `loan_status`, `total_pymnt*`, `total_rec_*`, `recoveries`, `collection_recovery_fee`, `out_prncp*`, `last_pymnt_*`, `next_pymnt_d`, `last_credit_pull_d`, `last_fico_range_*`, `debt_settlement_*`, `settlement_*`, `hardship_*`, `pymnt_plan`, `deferral_term`, `payment_plan_start_date`, `orig_projected_additional_accrued_interest` |
| Hằng số do bộ lọc | `application_type`, `verification_status_joint` (do lọc hồ sơ cá nhân); `term` (do lọc 36 tháng) |
| Khác | `zip_code`: gần như trùng thông tin với `addr_state`; mã bưu chính 3 số là biến đại diện được biết đến cho các đặc điểm được pháp luật bảo vệ trong cho vay; one-hot sinh 884 cột |

**Đổi định dạng:** `int_rate`, `revol_util` bỏ `%`; `earliest_cr_line` → năm thập phân (độ dài lịch sử tín dụng là feature mới nên để dành cho bước FE); `issue_d` **chỉ dùng để chia**, không đưa vào feature.

**Để B.0 xử lý:** `policy_code` (luôn bằng 1) và `disbursement_method` (chỉ một giá trị ở giai đoạn train, do được thêm vào muộn) sẽ bị B.0 bỏ và ghi log. Đây là hành vi mong đợi.

**Kết quả đã chạy:** `disbursement_method` thực tế chỉ có giá trị `Cash` ở cả train và test (khoản vay 36 tháng, 2013–2015), nên bị B.0 bỏ như dự kiến.

---

## Chia dữ liệu — sau tầng A, trước tầng B.0

### Quy tắc chung (Baesens et al. 2026, mục 4.3)

- Đánh giá trên phần test giữ lại trong sơ đồ **5-fold CV**.
- Tinh chỉnh siêu tham số trên phần validation bằng **20% tập train của mỗi fold**.

| Tầng | Tỷ lệ trên toàn bộ dữ liệu | Dùng cho |
|---|---|---|
| Test (1 fold) | 20% | Đánh giá |
| Validation (20% của train fold) | 16% | Tinh chỉnh siêu tham số |
| Train thực | 64% | Huấn luyện, fit tầng B.0 và B |

### Bổ sung: nhóm các dòng trùng lặp

Giữ **mọi dòng** nhưng dùng **`StratifiedGroupKFold`**, với nhóm là mã băm của toàn bộ giá trị feature. Các dòng có feature giống hệt nhau luôn nằm cùng một fold. Áp dụng cùng cách nhóm khi tách validation.

| Dataset | Dòng thuộc nhóm trùng lặp |
|---|---|
| GMSC | 1.000 dòng |
| HELOC | 1 cặp |
| Các bộ khác | Kiểm tra và ghi lại |

Không xóa dòng trùng vì các dòng giống hệt nhau có thể là những người vay khác nhau có hồ sơ đơn giản giống nhau.

### Theo từng bộ dữ liệu

| Dataset | Có trong Baesens | Số dòng dùng để chia | Cách chia | Test / fold | Validation / fold | Train / fold |
|---|---|---|---|---|---|---|
| GMSC | Có | 150.000 | Stratified group 5-fold | ~30.000 | ~24.000 | ~96.000 |
| Taiwan | Có | 30.000 | Stratified group 5-fold | ~6.000 | ~4.800 | ~19.200 |
| Home Credit | Có | 307.511 | Stratified group 5-fold | ~61.500 | ~49.200 | ~196.800 |
| HMEQ | Có | 5.960 | Stratified group 5-fold | ~1.192 | ~954 | ~3.814 |
| German / South German | Có (German) | 1.000 | Stratified group 5-fold | 200 | 160 | 640 |
| HELOC | Không | 9.871 | Stratified group 5-fold (ngoại lệ 2) | ~1.974 | ~1.579 | ~6.318 |
| Lending Club | Có (bản 9.578 dòng khác, chia ngẫu nhiên) | Sau các bộ lọc ở A.3 | **Chia theo thời gian** (ngoại lệ 1) | Năm test | Năm validation | Từ năm X đến trước năm validation |

Số dòng mỗi fold là xấp xỉ vì chia theo nhóm có thể làm lệch nhẹ kích thước các fold.

### Ngoại lệ 1 — Lending Club: chia theo cửa sổ đã đáo hạn

Không dùng 5-fold CV ngẫu nhiên vì dữ liệu có yếu tố thời gian.

**Vì sao chọn cửa sổ đã đáo hạn thay vì các năm mới nhất:** chỉ khoảng 24% khoản vay phát hành 2017–2018 đã kết thúc tại thời điểm chốt dữ liệu. Khi chỉ giữ khoản vay đã kết thúc, một tập test lấy từ các năm này sẽ thiên về các khoản vỡ nợ sớm hoặc trả hết sớm, không đại diện cho khoản vay của các năm đó. Chọn năm test mà gần như mọi khoản vay đã kết thúc tránh được sai lệch này.

| Tầng | Khoảng `issue_d` |
|---|---|
| Train thực | Năm X → năm trước năm validation |
| Validation | Năm liền trước năm test |
| Test | Năm phát hành gần nhất có **≥ 95%** khoản vay 36 tháng đã kết thúc |

**Cách xác định năm test** (tính **trước** bộ lọc "chỉ giữ khoản vay đã kết thúc", sau bộ lọc hồ sơ cá nhân và 36 tháng):

```python
FINISHED = ["Fully Paid", "Charged Off", "Default"]
raw36 = raw[(raw["term"] == 36) & (raw["application_type"] == "Individual")]
share = raw36["loan_status"].isin(FINISHED).groupby(raw36["issue_d"].dt.year).mean()
test_year = share[share >= 0.95].index.max()
val_year  = test_year - 1
```

**Quy tắc:**

1. Sắp xếp theo `issue_d` trước khi chia; **không xáo trộn**.
2. Chỉ có một lần chia, nên độ bất định được đo bằng **bootstrap trên tập test** (ví dụ 1.000 lần lặp).
3. Tầng B.0 và B fit trên train thực. Sau khi tinh chỉnh, fit lại mô hình cuối trên train + validation rồi đánh giá trên test.
4. Với mô hình chậm (RF, KNN, SVM), được phép lấy mẫu con khi tinh chỉnh, có ghi chú.
5. Sau khi áp bộ lọc năm X, chạy lại B.0 để xác nhận không còn cột nào có tỷ lệ thiếu lệch hơn 30 điểm phần trăm.

```python
y = lc["issue_d"].dt.year
train = lc[(y >= start_year) & (y < val_year)]
val   = lc[y == val_year]
test  = lc[y == test_year]
# issue_d không đưa vào X
```

**Kết quả đã chạy** (thư mục `data/splits/lending_club_B/`):

Tỷ lệ khoản vay 36 tháng (hồ sơ cá nhân) đã kết thúc theo năm phát hành:

| Năm | 2007–2014 | 2015 | 2016 | 2017 | 2018 |
|---|---:|---:|---:|---:|---:|
| Đã kết thúc | 100% | **99,95%** | 71,9% | 40,7% | 12,1% |

→ Năm test = **2015**, năm validation = **2014**, train = **2013** (năm X).

| Tầng | Khoảng `issue_d` | Số dòng | Tỷ lệ vỡ nợ |
|---|---|---:|---:|
| Train thực | 2013-01 → 2013-12 | 100.422 | 12,33% |
| Validation | 2014-01 → 2014-12 | 162.570 | 13,73% |
| Test | 2015-01 → 2015-12 | 282.787 | 14,88% |

Nhận xét:

- Train nhỏ hơn validation (1,6 lần) và test (2,8 lần) do quy mô Lending Club tăng nhanh theo năm → quy tắc 3 (fit lại trên train + validation, tức 2013–2014, 262.992 dòng) càng quan trọng.
- Tỷ lệ vỡ nợ tăng dần 12,3% → 13,7% → 14,9%: là trôi dạt thật theo thời gian, cần báo cáo độ lệch calibration trên test.
- Phiên bản Koklev (train 2007–2014, validation 2015, test 2017–2018) **không dùng** vì tập test chỉ gồm khoản vay kết thúc sớm (giới hạn 1).
- `data.parquet` còn giữ 386.155 khoản vay đã kết thúc của 2016–2018 với vai trò `unused` trong `folds.parquet`; các khoản này không thuộc train, validation hay test. (Theo A2 có thể lọc hẳn ở tầng A; không ảnh hưởng kết quả chia.)

### Ngoại lệ 2 — HELOC: không có trong Baesens

- Không có cột thời gian (các biến "months since…" là đặc trưng, không phải mốc thời gian của khoản vay), nên không thể chia theo thời gian.
- Tiền lệ: Dervovic et al. (2024) dùng stratified 10-fold CV trên HELOC.
- Đồ án dùng stratified group 5-fold để thống nhất với các bộ còn lại; ghi rõ khác biệt về số fold.

### Không được nêu trong paper (tự chọn, ghi rõ trong báo cáo)

| Mục | Giá trị |
|---|---|
| Phân tầng theo nhãn | Có |
| Nhóm dòng trùng lặp | `StratifiedGroupKFold`, nhóm = mã băm feature |
| Xáo trộn | `shuffle=True` (trừ Lending Club) |
| Seed | Cố định, ví dụ `random_state=42` |

### Khung code — 5-fold CV có nhóm

```python
from sklearn.model_selection import StratifiedGroupKFold

groups = pd.util.hash_pandas_object(X, index=False)

outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y, groups)):
    X_tr_full, y_tr_full, g_tr = X.iloc[tr_idx], y.iloc[tr_idx], groups.iloc[tr_idx]
    X_test,    y_test          = X.iloc[te_idx], y.iloc[te_idx]

    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fit_idx, val_idx = next(inner.split(X_tr_full, y_tr_full, g_tr))
    X_train, y_train = X_tr_full.iloc[fit_idx], y_tr_full.iloc[fit_idx]
    X_val,   y_val   = X_tr_full.iloc[val_idx], y_tr_full.iloc[val_idx]

    # Tầng B.0 và B: fit CHỈ trên (X_train, y_train)
```

### Dùng chung các fold cho mọi bước

Lưu chỉ số các fold (và mốc năm của Lending Club) ra file. Mọi bước dùng **đúng cùng một cách chia**:

```
Cùng một bộ chỉ số fold
   ├── Bước 1: Dummy, LR, RF, KNN, NB      (classic ML, không FE)
   └── Bước 2: GBDT không ràng buộc        (có FE)
```

Các bước sau này cũng dùng lại đúng bộ chỉ số fold này.

---

## Tầng B.0 — Sàng lọc cột (bổ sung, fit trên train của từng fold)

Baesens không mô tả bước này, nhưng nó cần thiết: `SimpleImputer` của scikit-learn **âm thầm bỏ** cột trống hoàn toàn (khi `keep_empty_features=False`, mặc định). Bước này làm việc bỏ cột trở nên tường minh và có ghi lại.

| Quy tắc (fit chỉ trên train) | Hành động |
|---|---|
| Cột thiếu 100% ở train | **Bỏ**; ghi log |
| Cột số chỉ có một giá trị ở train (không tính NaN) | **Bỏ**; ghi log |
| Cột phân loại chỉ có một giá trị ở train (tính NaN là một nhóm) | **Bỏ**; ghi log |
| Tỷ lệ thiếu chênh lệch giữa train và validation/test **> 30 điểm phần trăm** | **Giữ**, nhưng **đưa vào bảng cảnh báo** trong báo cáo |

```python
def screen_columns(X_train, num_cols, cat_cols):
    miss = X_train.isna().mean()
    drop = set(miss[miss == 1.0].index)
    drop |= {c for c in num_cols if X_train[c].nunique(dropna=True)  <= 1}
    drop |= {c for c in cat_cols if X_train[c].nunique(dropna=False) <= 1}
    return sorted(drop)

def missing_shift(X_train, X_other, thr=0.30):
    diff = (X_other.isna().mean() - X_train.isna().mean()).abs()
    return diff[diff > thr].sort_values(ascending=False)
```

Nếu bảng cảnh báo có cột, cần quay lại tầng A để tìm nguyên nhân (như trường hợp 22 trường bureau của Lending Club dẫn đến bộ lọc năm X), thay vì chỉ ghi chú.

**Kết quả B.0 cho Lending Club** (`data/splits/lending_club_B/fold1/screening.json`):

| Kết quả | Số cột | Cột |
|---|---:|---|
| Bỏ — thiếu 100% ở train | 29 | 12 cột `sec_app_*` và 3 cột đồng vay (`annual_inc_joint`, `dti_joint`, `revol_bal_joint`) — trống do chỉ giữ hồ sơ cá nhân; 14 trường bureau Lending Club bổ sung từ khoảng 2015–2016 (`open_acc_6m`, `open_act_il`, `open_il_12m`, `open_il_24m`, `mths_since_rcnt_il`, `total_bal_il`, `il_util`, `open_rv_12m`, `open_rv_24m`, `max_bal_bc`, `all_util`, `inq_fi`, `total_cu_tl`, `inq_last_12m`) |
| Bỏ — hằng số ở train | 2 | `policy_code`, `disbursement_method` |
| Cảnh báo tỷ lệ thiếu lệch > 30 điểm | **0** | Chênh lệch lớn nhất còn ~10 điểm (`mths_since_last_major_derog`) |
| **Giữ lại** | **70** | 62 biến số + 8 biến phân loại |

Trước khi áp bộ lọc năm X (train 2007–2013), B.0 báo 22 cột lệch khoảng 31 điểm; sau bộ lọc còn 0 → đạt quy tắc 5 của ngoại lệ 1.

---

## Tầng B — Pipeline Baesens (giữ đúng paper)

### B.1. Các bước

| Bước | Biến số | Biến phân loại |
|---|---|---|
| **1. Giá trị thiếu** | Điền bằng **median** (tính trên train fold) | Tạo **một nhóm mới** cho giá trị thiếu |
| **2. Mã hóa** | — | **Mã số nguyên** cho mô hình nhận biến phân loại trực tiếp; **one-hot** cho mô hình cần đầu vào số thực |
| **3. Chuẩn hóa** | **Standardize** về trung bình 0, phương sai 1 (tính trên train fold) | — |

### B.2. Áp dụng theo nhóm mô hình

| Nhóm mô hình | Giá trị thiếu | Mã hóa phân loại | Chuẩn hóa biến số |
|---|---|---|---|
| Classic ML (LR, RF, KNN, NB, SVM) | Có | One-hot | Có |
| GBDT (XGBoost, LightGBM, CatBoost) | Có | Mã số nguyên | Có |

### B.3. Khung code (scikit-learn)

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

def build_preprocessor(num_cols, cat_cols, family):
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_encoder = (OneHotEncoder(handle_unknown="ignore")
                   if family == "classic"
                   else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("encode", cat_encoder),
    ])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])

# Trong MỖI fold:
#   dropped = screen_columns(X_train, num_cols, cat_cols)        # tầng B.0
#   num_k   = [c for c in num_cols if c not in dropped]
#   cat_k   = [c for c in cat_cols if c not in dropped]
#   pre     = build_preprocessor(num_k, cat_k, family="classic")  # tầng B
#   model   = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
#   model.fit(X_train[num_k + cat_k], y_train)
```

Biến phân loại cần được ép về kiểu chuỗi trước khi vào `SimpleImputer(strategy="constant")`.

---

## Tinh chỉnh siêu tham số (Baesens et al. 2026, mục 4.3)

| Thành phần | Giá trị |
|---|---|
| Công cụ | Optuna |
| Số lần thử | Tối đa **20** mỗi fold |
| Mục tiêu | Tối đa hóa **AUC** trên validation (20% train fold) |
| Phạm vi | Tinh chỉnh độc lập trong từng fold |
| Độ đo báo cáo | AUC, Gini, KS, Brier, LogLoss, Average Precision; các độ đo theo ngưỡng dùng ngưỡng tối đa hóa F1 trên validation |
| Kiểm định thống kê | Friedman (hiệu chỉnh Iman–Davenport), rồi Wilcoxon từng cặp với hiệu chỉnh Holm |

Paper không công bố không gian tìm kiếm cụ thể; lấy theo cấu hình mặc định của TALENT hoặc repo code của paper và ghi rõ trong báo cáo.

---

## Không làm (để giữ đúng Baesens)

| Không làm | Ghi chú |
|---|---|
| Cắt / xử lý ngoại lai | Nếu muốn thử, báo cáo riêng như phân tích phụ |
| Cân bằng lớp (oversampling, undersampling, class weight) | Giữ nguyên phân phối nhãn |
| Xóa dòng trùng lặp | Giữ lại, xử lý bằng chia theo nhóm |
| Thêm cột đánh dấu giá trị thiếu | Không có trong pipeline Baesens |
| FE (biến tỷ lệ, tương tác, tổng hợp bảng phụ, độ dài lịch sử tín dụng) | Dành cho bước 2 |
| Binning, WoE | Dành cho bước 2 hoặc scorecard |
| Fit bất kỳ bước nào của tầng B.0 / B trên toàn bộ dữ liệu | Luôn fit trong từng fold |

---

## Giới hạn cần nêu trong báo cáo

| # | Giới hạn | Cách xử lý trong đồ án |
|---|---|---|
| 1 | Lending Club: các năm phát hành gần nhất có phần lớn khoản vay chưa kết thúc (2017–2018: chỉ khoảng 24%) | Chia theo cửa sổ đã đáo hạn, chỉ khoản vay 36 tháng |
| 2 | Lending Club: 22 trường bureau không có ở khoản vay trước năm X | Bắt đầu cửa sổ dữ liệu từ năm X = 2013; loại 75.004 dòng train (2007–2012), tỷ lệ vỡ nợ train 12,63% → 12,33% |
| 3 | Lending Club: một số cột chỉ có từ khoảng 2015–2017 | B.0 bỏ cột trống hoặc hằng số ở train; bảng cảnh báo tỷ lệ thiếu |
| 4 | Lending Club: chỉ hồ sơ cá nhân, chỉ khoản vay 36 tháng, bỏ `zip_code` | Ghi rõ lý do; kết luận Lending Club áp dụng cho tổng thể này |
| 5 | Lending Club: một lần chia duy nhất (train 2013, validation 2014, test 2015) | Bootstrap trên tập test |
| 6 | HELOC dùng bản đã làm sạch (9.871 dòng thay vì 10.459) | 588 dòng bị loại không mang thông tin |
| 7 | Dòng trùng lặp (GMSC 1.000 dòng, HELOC 1 cặp) | Giữ lại, chia theo nhóm |
| 8 | GMSC: giá trị 96/98 bất thường được giữ nguyên | Ghi chú; có thể thử xử lý ở phân tích phụ |
| 9 | Chỉ áp dụng giao thức Baesens; không đối chiếu được con số từng bộ | Ghi rõ trong báo cáo |

---

## Checklist trước khi chạy

**Tầng A**
- [ ] Nhãn của cả 7 bộ đã về dạng 1 = xấu
- [ ] Đã bỏ ID và cột rò rỉ (Lending Club: gồm cả `loan_status` và 4 cột bổ sung)
- [ ] Đã đổi mã đặc biệt thành NaN (HELOC, Home Credit)
- [x] Lending Club: lọc khoản vay đã kết thúc → hồ sơ cá nhân → 36 tháng → `issue_d ≥ năm X`
- [x] Lending Club: đã xác định và ghi cứng năm X (2013); ghi số dòng bị loại và tỷ lệ vỡ nợ trước/sau
- [x] Lending Club: đã bỏ `application_type`, `verification_status_joint`, `term`, `zip_code`; `issue_d` tách khỏi feature
- [ ] HELOC: dùng bản 9.871 dòng
- [ ] Chỉ dùng một trong hai bộ German / South German
- [ ] Đã khai báo danh sách biến số và biến phân loại cho từng bộ
- [ ] Home Credit: `SK_ID_CURR` giữ làm khóa, **bỏ khỏi X khi huấn luyện**

**Chia dữ liệu**
- [ ] Các bộ (trừ Lending Club): `StratifiedGroupKFold` 5 fold, nhóm theo mã băm feature; validation tách theo nhóm
- [x] Lending Club: đã tính năm test (≥ 95% đã kết thúc) và năm validation; không xáo trộn
- [ ] Đã lưu chỉ số các fold và mốc năm ra file

**Tầng B.0 và B**
- [ ] B.0 fit trên train của từng fold; đã lưu log cột bị bỏ và bảng cảnh báo tỷ lệ thiếu
- [x] Bảng cảnh báo của Lending Club không còn cột lệch hơn 30 điểm phần trăm sau khi lọc năm X
- [ ] B fit trên train của từng fold
- [ ] Mỗi nhóm mô hình dùng đúng nhánh mã hóa và chuẩn hóa theo bảng B.2

**Tinh chỉnh**
- [ ] Optuna, 20 lần thử, tối đa hóa AUC trên validation của từng fold
- [ ] Mọi điểm khác với paper đã được ghi vào phần giới hạn của báo cáo