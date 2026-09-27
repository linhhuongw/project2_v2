# Tiền xử lý dữ liệu thống nhất theo Baesens et al. (2026)

## Viết tắt

| Viết tắt | Đầy đủ |
|---|---|
| CV | Cross-Validation (kiểm định chéo) |
| LR | Logistic Regression (hồi quy logistic) |
| RF | Random Forest (rừng ngẫu nhiên) |
| KNN | K-Nearest Neighbors (k láng giềng gần nhất) |
| NB | Naive Bayes |
| GBDT | Gradient Boosted Decision Trees (XGBoost, LightGBM, CatBoost) |
| TFM | Tabular Foundation Model (ví dụ TabPFN) |
| FE | Feature Engineering (tạo và biến đổi đặc trưng) |
| NaN | Giá trị thiếu (Not a Number) |

## Nguồn

**Baesens, B., Goethals, A., Lessmann, S., De Vos, S., Bravo, C., Martens, D., Medina-Olivares, V., Mues, C., Óskarsdóttir, M., vanden Broucke, S., Verdonck, T., & Verbeke, W. (2026).** *Foundation Models for Credit Risk Prediction: A Game Changer?* arXiv:2605.18147.

- Mục 4.2 *Data preprocessing*: pipeline tiền xử lý (tầng B trong tài liệu này).
- Mục 4.3 *Benchmarking protocol*: chia dữ liệu và tinh chỉnh.
- Pipeline được cài đặt trong thư viện benchmark TALENT (Liu et al., 2024).

---

## Tổng quan

```
Dữ liệu gốc
   │
   ▼
[Tầng A] Khai báo dữ liệu  ← BỔ SUNG, không có trong paper; chạy một lần trên toàn bộ dữ liệu
   │                          (chỉ sửa cách hiểu dữ liệu, không học tham số từ dữ liệu)
   ▼
Chia 5-fold CV (+ 20% train fold làm validation)
   │
   ▼
[Tầng B] Pipeline Baesens   ← TÁI LẬP NGUYÊN VĂN; fit CHỈ trên train của từng fold
   │
   ▼
Huấn luyện / tinh chỉnh / đánh giá
```

**Nguyên tắc chống rò rỉ (theo paper):** mọi tham số tiền xử lý chỉ được học trên dữ liệu train của từng fold, rồi áp dụng sang validation và test của fold đó.

**Nguyên tắc của tầng A:** một thao tác được phép ở tầng A khi nó **không học bất kỳ giá trị nào từ dữ liệu** (không tính median, trung bình, tần suất…) và **không tạo ra thông tin mới**. Nhờ vậy tầng A chạy trước khi chia dữ liệu mà không gây rò rỉ.

---

## Tầng A — Khai báo dữ liệu (bổ sung)

### A.1. Quy tắc chung

| # | Thao tác | Mục đích |
|---|---|---|
| A1 | Chọn đúng bảng/tệp có nhãn | Thống nhất nguồn dữ liệu |
| A2 | Bỏ dòng thiếu nhãn hoặc nhãn không xác định | Nhãn nhị phân sạch |
| A3 | Mã hóa nhãn: **1 = vỡ nợ / xấu**, 0 = tốt | Thống nhất chiều nhãn giữa các bộ |
| A4 | Bỏ cột ID, cột văn bản tự do, cột URL | Không mang thông tin dự báo |
| A5 | Bỏ cột rò rỉ (thông tin chỉ biết sau thời điểm quyết định cho vay) | Tránh hiệu năng ảo |
| A6 | Đổi mã giá trị đặc biệt (đã được tài liệu dữ liệu định nghĩa) thành NaN | Để tầng B xử lý như giá trị thiếu |
| A7 | Đổi định dạng chuỗi sang số (`"13.5%"` → 13.5, `"36 months"` → 36) | Chỉ đổi định dạng, không tạo thông tin mới |
| A8 | Khai báo kiểu cột: số hay phân loại | Để tầng B chọn đúng nhánh xử lý |

**Không làm ở tầng A:** điền giá trị thiếu, chuẩn hóa, mã hóa one-hot, cắt ngoại lai, tạo feature mới, chọn feature theo hiệu năng. Các việc học từ dữ liệu thuộc tầng B; các việc còn lại là FE hoặc không có trong paper.

### A.2. Khai báo theo từng bộ dữ liệu

| Dataset | Bảng / tệp | Nhãn → 1 = xấu | Bỏ (ID, văn bản, rò rỉ) | Mã đặc biệt → NaN | Biến phân loại | Ghi chú |
|---|---|---|---|---|---|---|
| **Give Me Some Credit** | `cs-training.csv` | `SeriousDlqin2yrs` (đã là 1 = xấu) | Cột chỉ số dòng không tên | Không bắt buộc | Không có | Các cột đếm số lần trễ hạn có giá trị 96/98 bất thường; giữ nguyên để đúng Baesens, ghi chú trong báo cáo |
| **Taiwan** | Bảng chính | `default payment next month` (đã là 1 = xấu) | `ID` | Không | `SEX`, `EDUCATION`, `MARRIAGE` | `PAY_0`…`PAY_6` giữ là biến số |
| **Home Credit** | `application_train.csv` (chỉ bảng chính) | `TARGET` (đã là 1 = xấu) | `SK_ID_CURR` | `DAYS_EMPLOYED = 365243` | Các cột kiểu chuỗi (`NAME_*`, `CODE_GENDER`, `OCCUPATION_TYPE`…) | Bảng phụ (bureau, previous_application…) để dành cho bước FE; `application_test` không có nhãn nên không dùng |
| **HMEQ** | `hmeq.csv` | `BAD` (đã là 1 = xấu) | Không | Không | `REASON`, `JOB` | |
| **South German / German** | Bảng chính | Kiểm tra mã hóa của phiên bản đang dùng; đổi sao cho xấu = 1 (Statlog German gốc: 1 = tốt, 2 = xấu) | Không | Không | Các biến mã hóa dạng nhóm (tình trạng tài khoản, lịch sử tín dụng, mục đích vay…) | Chỉ dùng **một** trong hai bộ |
| **FICO HELOC** | `heloc_dataset_v1.csv` | `RiskPerformance`: `"Bad"` → 1, `"Good"` → 0 | Không | −7, −8, −9 ở mọi cột | `MaxDelq2PublicRecLast12M`, `MaxDelqEver` | Dòng có toàn bộ feature = −9 trở thành dòng toàn NaN; giữ nguyên để tầng B xử lý |
| **Lending Club** | Theo phiên bản sử dụng | `loan_status`: `Charged Off`, `Default` → 1; `Fully Paid` → 0; bỏ các trạng thái chưa kết thúc (`Current`, `In Grace Period`, `Late…`) | `id`, `member_id`, `url`, `desc`, `title`, `emp_title`; **rò rỉ:** `total_pymnt*`, `total_rec_*`, `recoveries`, `collection_recovery_fee`, `out_prncp*`, `last_pymnt_*`, `next_pymnt_d`, `last_credit_pull_d`, `last_fico_range_*`, `debt_settlement_*`, `settlement_*`, `hardship_*` | Không | `grade`, `sub_grade`, `home_ownership`, `verification_status`, `purpose`, `addr_state`, `emp_length`, `initial_list_status`, `application_type` | Đổi `int_rate`, `revol_util` (bỏ `%`) và `term` (bỏ `months`) sang số; `issue_d` **giữ riêng làm cột chia theo thời gian**, không đưa vào feature |

---

## Chia dữ liệu — chạy sau tầng A, trước tầng B

Theo Baesens et al. (2026), mục 4.3: đánh giá trên phần test giữ lại trong sơ đồ **5-fold CV**; tinh chỉnh siêu tham số trên phần validation bằng **20% tập train của mỗi fold**.

### Tỷ lệ chung

| Tầng | Tỷ lệ trên toàn bộ dữ liệu | Dùng cho |
|---|---|---|
| Test (1 fold) | 20% | Đánh giá |
| Validation (20% của train fold) | 16% | Tinh chỉnh siêu tham số |
| Train thực | 64% | Huấn luyện, học tham số tiền xử lý (tầng B) |

### Theo từng bộ dữ liệu

| Dataset | Có trong Baesens | Số dòng dùng để split | Cách split | Test / fold | Validation / fold | Train / fold |
|---|---|---|---|---|---|---|
| Give Me Some Credit | Có | 150.000 | 5-fold CV | ~30.000 | ~24.000 | ~96.000 |
| Taiwan | Có | 30.000 | 5-fold CV | ~6.000 | ~4.800 | ~19.200 |
| Home Credit | Có | 307.511 (`application_train`) | 5-fold CV | ~61.500 | ~49.200 | ~196.800 |
| HMEQ | Có | 5.960 | 5-fold CV | ~1.192 | ~954 | ~3.814 |
| German / South German | Có (German) | 1.000 | 5-fold CV | 200 | 160 | 640 |
| FICO HELOC | Không | 10.459 | **Stratified 5-fold CV** (ngoại lệ về nguồn, xem bên dưới) | ~2.092 | ~1.673 | ~6.694 |
| Lending Club | Có (bản 9.578 dòng, chia ngẫu nhiên) | Theo phiên bản sử dụng | **Chia theo thời gian `issue_d`** (ngoại lệ, xem bên dưới) | 2017–2018 | 2015 | 2007–2014 |

### Ngoại lệ 1 — Lending Club: chia theo thời gian

Không dùng 5-fold CV ngẫu nhiên của Baesens. Theo **Koklev (2025)** (*What's the Price of Monotonicity?*, arXiv:2512.17945): train trên các khoản vay giải ngân **2007–2015**, test trên **2017–2018**.

| Tầng | Khoảng `issue_d` | Dùng cho | Nguồn |
|---|---|---|---|
| Train thực | 2007-01 → 2014-12 | Huấn luyện, fit tầng B | Tự chọn (tách từ cửa sổ train của Koklev) |
| Validation | 2015-01 → 2015-12 | Tinh chỉnh siêu tham số | Tự chọn: năm cuối của cửa sổ train |
| *(Không dùng)* | 2016-01 → 2016-12 | — | Theo khoảng trống giữa train và test của Koklev |
| Test | 2017-01 → 2018-12 | Đánh giá | Koklev (2025) |

Quy tắc:

1. Sắp xếp theo `issue_d` **trước** khi chia; **không xáo trộn**.
2. Chỉ dùng khoản vay đã kết thúc (`Fully Paid`, `Charged Off`, `Default`), theo tầng A.
3. Chỉ có **một** lần chia, nên độ bất định được đo bằng **bootstrap trên tập test** (như Koklev), không phải bằng các fold.
4. Tầng B fit trên train thực (2007–2014); khi chốt mô hình cuối có thể fit lại trên 2007–2015 rồi đánh giá trên test.
5. Cần đối chiếu mốc thời gian với repo của Koklev nếu muốn tái lập số liệu Lending Club của bài đó.

Phương án thay thế (tham khảo): Dervovic et al. (2024) chia Lending Club thành các đoạn liên tiếp theo thời gian, mỗi đoạn 100.000 dòng, và coi mỗi đoạn là một bộ dữ liệu riêng.

### Ngoại lệ 2 — FICO HELOC: không có trong Baesens

- Bộ dữ liệu **không có cột thời gian** (các biến "months since…" là đặc trưng, không phải mốc thời gian của khoản vay), nên **không thể chia theo thời gian**.
- Tiền lệ trên HELOC: Dervovic et al. (2024) dùng **stratified 10-fold CV**.
- Đồ án giữ **stratified 5-fold CV** để thống nhất với các bộ còn lại; ghi rõ khác biệt số fold so với Dervovic.

### Không được nêu trong paper (tự chọn, ghi rõ trong báo cáo)

| Mục | Giá trị đề xuất |
|---|---|
| Phân tầng theo nhãn | `StratifiedKFold`; `stratify=y` khi tách validation |
| Xáo trộn | `shuffle=True` |
| Seed | Cố định, ví dụ `random_state=42` |

### Khung code

```python
from sklearn.model_selection import StratifiedKFold, train_test_split

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
    X_tr_full, y_tr_full = X.iloc[tr_idx], y.iloc[tr_idx]
    X_test,    y_test    = X.iloc[te_idx], y.iloc[te_idx]

    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.20,
        stratify=y_tr_full, random_state=42)

    # Tầng B: fit tiền xử lý CHỈ trên (X_train, y_train)
    # Tinh chỉnh trên (X_val, y_val); đánh giá trên (X_test, y_test)
```

Lưu ý: nên lưu chỉ số các fold (`tr_idx`, `te_idx`) ra file để mọi bước sau (GBDT, GBDT đơn điệu, TabPFN) dùng **đúng cùng một cách chia**.

Khung code riêng cho Lending Club:

```python
d = pd.to_datetime(df["issue_d"], format="%b-%Y")
train = df[(d >= "2007-01-01") & (d <= "2014-12-31")]
val   = df[(d >= "2015-01-01") & (d <= "2015-12-31")]
test  = df[(d >= "2017-01-01") & (d <= "2018-12-31")]
# issue_d không đưa vào X; tầng B fit trên train
```

---

## Tầng B — Pipeline Baesens (tái lập nguyên văn)

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
| TFM (TabPFN…) | Có | Mã số nguyên | **Không** — dùng giá trị thô |

Theo paper, việc chọn one-hot hay mã số nguyên được quy định và áp dụng theo từng phương pháp trong TALENT. Chuẩn hóa áp dụng cho mọi phương pháp trừ các TFM.

### B.3. Khung code (scikit-learn)

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

def build_preprocessor(num_cols, cat_cols, family):
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        *([] if family == "tfm" else [("scale", StandardScaler())]),
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
#   pre = build_preprocessor(num_cols, cat_cols, family="classic")
#   model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
#   model.fit(X_train_fold, y_train_fold)   # tiền xử lý chỉ học trên train fold
```

Lưu ý: biến phân loại cần được ép về kiểu chuỗi trước khi vào `SimpleImputer(strategy="constant")`.

---

## Không làm (để giữ đúng Baesens)

| Không làm | Ghi chú |
|---|---|
| Cắt / xử lý ngoại lai | Nếu muốn thử, báo cáo riêng như phân tích phụ |
| Cân bằng lớp (oversampling, undersampling, class weight) | Giữ nguyên phân phối nhãn |
| FE (biến tỷ lệ, tương tác, tổng hợp bảng phụ) | Dành cho bước 2 |
| Binning, WoE | Dành cho bước 2 hoặc scorecard |
| Chọn feature theo hiệu năng | Không có trong pipeline |
| Fit bất kỳ bước nào của tầng B trên toàn bộ dữ liệu | Luôn fit trong từng fold |

---

## Checklist trước khi chạy

- [ ] Tầng A: nhãn của cả 7 bộ đã về dạng 1 = xấu
- [ ] Tầng A: đã bỏ ID và cột rò rỉ (đặc biệt Lending Club)
- [ ] Tầng A: đã đổi mã đặc biệt thành NaN (HELOC, Home Credit)
- [ ] Tầng A: đã khai báo danh sách biến số và biến phân loại cho từng bộ
- [ ] Chỉ dùng một trong hai bộ German / South German
- [ ] Chia 5-fold CV **sau** tầng A, **trước** tầng B (trừ Lending Club)
- [ ] Lending Club: chia theo `issue_d` (train 2007–2014, validation 2015, bỏ 2016, test 2017–2018), không xáo trộn
- [ ] HELOC: stratified 5-fold CV, ghi chú khác Dervovic (10-fold)
- [ ] Tầng B fit trong từng fold, không fit trên toàn bộ dữ liệu
- [ ] Mỗi nhóm mô hình dùng đúng nhánh mã hóa và chuẩn hóa theo bảng B.2
- [ ] Ghi lại mọi điểm khác với paper trong báo cáo