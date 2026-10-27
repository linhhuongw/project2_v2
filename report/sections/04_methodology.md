# 4. Phương pháp

Đồ án áp dụng giao thức tiền xử lý và đánh giá của Baesens et al. (2026). Đặc tả đầy đủ nằm trong [`configs/preprocessing_v1.md`](../../configs/preprocessing_v1.md); chương này trình bày các lựa chọn chính và lý do.

## 4.1. Tổng quan quy trình

```
Dữ liệu gốc
   │
   ▼
[Tầng A]   Khai báo dữ liệu      ← không học từ dữ liệu; chạy một lần trên toàn bộ dữ liệu (chương 3)
   │
   ▼
Chia dữ liệu                      ← 5-fold CV có nhóm theo dòng trùng lặp; Lending Club chia theo thời gian
   │
   ▼
[Tầng B.0] Sàng lọc cột           ← fit CHỈ trên train của từng fold
   │
   ▼
[Tầng B.1] Pipeline Baesens       ← fit CHỈ trên train của từng fold
   │
   ▼
Huấn luyện → đánh giá
```

**Nguyên tắc chống rò rỉ.** Mọi bước học từ dữ liệu (median, trung bình, độ lệch chuẩn, danh sách nhóm, tỷ lệ thiếu, số giá trị khác nhau…) chỉ được fit trên phần train của từng fold, rồi áp dụng nguyên sang validation và test của fold đó.

## 4.2. Chia dữ liệu

### Sáu bộ dữ liệu không có yếu tố thời gian

Theo Baesens et al. (2026), mục 4.3: đánh giá trên phần test giữ lại trong sơ đồ **5-fold CV**; phần validation để tinh chỉnh là **20% tập train của mỗi fold**.

| Phần | Tỷ lệ trên toàn bộ dữ liệu | Dùng cho |
|---|---|---|
| Test (1 fold) | 20% | Đánh giá |
| Validation (20% của phần train) | 16% | Chọn ngưỡng, tinh chỉnh |
| Train | 64% | Huấn luyện, fit tầng B.0 và B.1 |

Các lựa chọn không được nêu trong paper:

| Mục | Giá trị | Lý do |
|---|---|---|
| Phân tầng theo nhãn | Có | Giữ tỷ lệ xấu như nhau giữa các fold |
| Nhóm dòng trùng lặp | `StratifiedGroupKFold`, nhóm = mã băm của toàn bộ giá trị đặc trưng; validation cũng tách theo nhóm | Các dòng giống hệt nhau luôn nằm cùng một phần, nên không có bản sao nào ở cả train và test |
| Xáo trộn, seed | `shuffle=True`, `random_state=42` | Tái lập được |

Mọi bước sau (các mô hình khác nhau) dùng **đúng cùng một bộ chỉ số fold**, được lưu trong `data/splits/<bộ dữ liệu>/folds.parquet`.

### Lending Club — chia theo thời gian

Lending Club không dùng 5-fold ngẫu nhiên vì dữ liệu có yếu tố thời gian: xáo trộn sẽ cho mô hình "nhìn thấy tương lai". Dữ liệu được sắp theo `issue_d`, không xáo trộn, và chỉ có **một lần chia**: train 2013, validation 2014, test 2015 (cách chọn mốc ở mục 3.5).

- Vì chỉ có một lần chia, độ bất định được đo bằng **bootstrap trên tập test** (1.000 lần lặp).
- Sau khi chọn ngưỡng trên validation, mô hình cuối được **fit lại trên train + validation** (2013–2014) rồi đánh giá trên test 2015.

## 4.3. Tầng B.0 — Sàng lọc cột

Bước này không có trong mô tả của Baesens nhưng cần thiết: `SimpleImputer` của scikit-learn âm thầm bỏ những cột trống hoàn toàn, và Lending Club có các cột trống ở giai đoạn train nhưng đầy đủ ở giai đoạn test. B.0 làm việc bỏ cột trở nên tường minh và có ghi lại.

| Quy tắc (fit chỉ trên train của fold) | Hành động |
|---|---|
| Cột thiếu 100% | Bỏ, ghi log |
| Cột số chỉ có một giá trị (không tính NaN) | Bỏ, ghi log |
| Cột phân loại chỉ có một giá trị (tính NaN là một nhóm) | Bỏ, ghi log |
| Tỷ lệ thiếu chênh lệch giữa train và validation/test > 30 điểm phần trăm | Giữ, đưa vào bảng cảnh báo |

Kết quả:

| Bộ dữ liệu | Cột bị bỏ | Cảnh báo |
|---|---|---|
| GMSC, Taiwan, HMEQ, South German, HELOC | Không | Không |
| Home Credit | `FLAG_MOBIL` ở fold 2, 4, 5 (chỉ có một dòng khác giá trị) | Không |
| Lending Club | 31 cột: 29 cột trống ở train (12 cột người đồng vay `sec_app_*`, 3 cột hồ sơ đồng vay, 14 trường bureau có từ khoảng 2015–2016) và 2 cột hằng số (`policy_code`, `disbursement_method`) | **Không** (trước khi lọc năm 2013 có 22 cột lệch ~31 điểm) |

## 4.4. Tầng B.1 — Tiền xử lý Baesens

| Bước | Biến số | Biến phân loại |
|---|---|---|
| 1. Giá trị thiếu | Điền bằng **median** của train | Đổi sang chuỗi, tạo **một nhóm mới** `__missing__` cho giá trị thiếu |
| 2. Mã hóa | — | **One-hot**; nhóm chưa từng xuất hiện ở train → toàn số 0 |
| 3. Chuẩn hóa | **Standardize** về trung bình 0, độ lệch chuẩn 1 theo thống kê của train | — |

Theo bảng B.2 của đặc tả, bước 2 có hai nhánh:

| Nhóm mô hình | Giá trị thiếu | Mã hóa biến phân loại | Chuẩn hóa biến số | Thư mục |
|---|---|---|---|---|
| Classic ML (LR, KNN, RF, NB) | Có | **One-hot** | Có | `data/processed/classic/` |
| GBDT (XGBoost, LightGBM, CatBoost) | Có | **Mã số nguyên** (`OrdinalEncoder`; nhóm chưa thấy ở train → −1) | Có | `data/processed/gbdt/` |

Hai nhánh dùng cùng fold, cùng cột sau B.0 và cùng median, trung bình, độ lệch chuẩn của train; chỉ khác cách mã hóa biến phân loại. Đồ án **không** thêm các bước ngoài pipeline này:

| Không làm | Lý do |
|---|---|
| Cắt hoặc xử lý ngoại lai | Giữ đúng pipeline Baesens; nếu thử thì báo cáo riêng như phân tích phụ |
| Cân bằng lớp (oversampling, undersampling, `class_weight`) | Giữ nguyên phân phối nhãn |
| Cột đánh dấu giá trị thiếu | Không có trong pipeline Baesens (thông tin thiếu của biến phân loại vẫn được giữ qua nhóm `__missing__`) |
| Tạo đặc trưng (tỷ lệ, tương tác, tổng hợp bảng phụ) | Dành cho bước sau |

**Kiểm tra tự động** cho mọi fold: không còn NaN hay giá trị vô cực; các biến số ở train có trung bình ≈ 0 và độ lệch chuẩn ≈ 1. Validation và test **không** có đúng trung bình 0 và độ lệch chuẩn 1 vì được biến đổi bằng thống kê của train — đây là dấu hiệu không có rò rỉ.

Số đặc trưng sau B.1:

| Bộ dữ liệu | Đầu vào | Nhánh classic (one-hot) | Nhánh GBDT (mã số nguyên) |
|---|---|---:|---:|
| GMSC | 10 số | 10 | 10 |
| Taiwan | 20 số + 3 phân loại | 33 | 23 |
| Home Credit | 104 số + 16 phân loại | 250 (249 ở các fold bỏ `FLAG_MOBIL`) | 120 (119) |
| HMEQ | 10 số + 2 phân loại | 20 | 12 |
| South German | 3 số + 17 phân loại | 71 | 20 |
| FICO HELOC | 21 số + 2 phân loại | 37 | 23 |
| Lending Club | 62 số + 8 phân loại | 184 (188 khi fit lại trên train + validation) | 70 |

## 4.5. Mô hình

Bốn mô hình thuộc nhóm classic ML của Baesens được huấn luyện với **tham số mặc định** của scikit-learn 1.7, chưa tinh chỉnh. Cả bốn dùng **cùng các fold** và **cùng ma trận đặc trưng** sau B.1 (one-hot, chuẩn hóa), nên khác biệt về kết quả chỉ đến từ mô hình.

| Mô hình | Cấu hình mặc định | Đặc điểm |
|---|---|---|
| **Logistic Regression (LR)** (Cox, 1958) | `C = 1,0`, phạt L2, solver `lbfgs`; `max_iter = 2000` theo khung code của đặc tả | Tuyến tính, dễ diễn giải; là chuẩn của ngành chấm điểm tín dụng |
| **K-Nearest Neighbors (KNN)** (Cover & Hart, 1967) | `k = 5`, trọng số đều, khoảng cách Euclid | Phi tham số, dựa trên khoảng cách; xác suất chỉ nhận 6 giá trị (0; 0,2; …; 1) |
| **Random Forest (RF)** (Breiman, 2001) | 100 cây, cây phát triển hết độ sâu, mỗi lần tách xét √p đặc trưng | Tổ hợp cây quyết định; nắm được quan hệ phi tuyến và tương tác; không nhạy với ngoại lai |
| **Naive Bayes (NB)** (Maron, 1961) | `GaussianNB` | Giả định các đặc trưng độc lập có điều kiện và có phân phối chuẩn |

Với LR, mọi lần fit hội tụ sau tối đa 89 vòng (dưới giới hạn mặc định 100 của scikit-learn), nên kết quả trùng với `LogisticRegression()` mặc định hoàn toàn. KNN **không được chấm điểm trên chính tập train**, vì mỗi điểm train là láng giềng gần nhất của chính nó (AUC train vô nghĩa) và phép tính khoảng cách train × train rất tốn kém.

Ba mô hình **gradient boosting (GBDT)** — các cách cài đặt khác nhau của gradient boosting trên cây quyết định (Friedman, 2001) — dùng nhánh mã số nguyên, với **tham số mặc định của từng thư viện**; chỉ đặt seed (42), số luồng CPU và tắt log:

| Mô hình | Thư viện | Cấu hình mặc định | Đặc điểm |
|---|---|---|---|
| **XGBoost (XGB)** (Chen & Guestrin, 2016) | xgboost 3.3 | 100 cây, độ sâu 6, learning rate 0,3, thuật toán `hist` | Boosting theo độ sâu (level-wise), có điều chuẩn L2 |
| **LightGBM (LGBM)** (Ke et al., 2017) | lightgbm 4.7 | 100 cây, 31 lá, learning rate 0,1 | Boosting theo lá (leaf-wise), chia theo histogram |
| **CatBoost** (Prokhorenkova et al., 2018; Dorogush et al., 2018) | catboost 1.2 | 1.000 cây, độ sâu 6, learning rate tự chọn theo dữ liệu | Cây đối xứng (oblivious), ordered boosting chống rò rỉ mục tiêu |

Mã số nguyên của biến phân loại được đưa vào cả ba mô hình **như biến số**, đúng theo đặc tả (bảng B.2). Cách này không tận dụng cơ chế xử lý biến phân loại riêng của CatBoost (target statistics) hay LightGBM; đây là lựa chọn để ba thư viện được so sánh trên cùng đầu vào. Không dùng dừng sớm (early stopping) vì không phải tham số mặc định.

SVM (mô hình classic thứ năm của Baesens) chưa được đưa vào: SVM hạt nhân có chi phí huấn luyện tăng xấp xỉ theo bình phương số dòng, không khả thi trên Home Credit và Lending Club (100–263 nghìn dòng train) nếu không lấy mẫu con, mà lấy mẫu con thì so sánh không còn cùng điều kiện.

## 4.6. Độ đo đánh giá

| Nhóm | Độ đo | Ý nghĩa |
|---|---|---|
| Khả năng xếp hạng | **AUC**, Gini = 2·AUC − 1, KS | Mô hình có xếp người xấu có rủi ro cao hơn người tốt không; không phụ thuộc ngưỡng |
| Chất lượng xác suất | Brier, LogLoss | Xác suất dự đoán có gần tỷ lệ thực không; phụ thuộc tỷ lệ xấu nên không so sánh được giữa các bộ |
| Lớp thiểu số | Average Precision (AP) | Tóm tắt đường precision–recall; mức cơ sở bằng tỷ lệ xấu |
| Theo ngưỡng | Precision, Recall, F1, Accuracy | Tại ngưỡng tối đa hóa F1 **trên validation**, áp nguyên sang test |

Với 6 bộ dữ liệu CV, kết quả được báo cáo là trung bình ± độ lệch chuẩn trên 5 fold test. Với Lending Club, kết quả là mô hình fit lại trên train + validation, kèm khoảng tin cậy 95% của AUC bằng bootstrap.

## 4.7. So sánh mô hình

Theo đặc tả, các mô hình được so sánh trên 7 bộ dữ liệu bằng:

1. **Hạng trung bình**: trên mỗi bộ, xếp hạng các mô hình theo độ đo (1 = tốt nhất), rồi lấy trung bình qua các bộ.
2. **Kiểm định Friedman** kèm hiệu chỉnh **Iman–Davenport**: kiểm tra giả thuyết "mọi mô hình có hạng như nhau".
3. **Wilcoxon signed-rank từng cặp** với hiệu chỉnh **Holm** cho nhiều phép so sánh.
4. Bổ sung ở mức từng bộ dữ liệu: số fold mô hình thắng LR (6 bộ CV); với Lending Club, **bootstrap ghép cặp** (1.000 lần) của chênh lệch AUC trên cùng 282.787 dòng test.

Ngoài AUC, các mô hình được so sánh trên **precision, recall và F1** tại ngưỡng tối đa F1 chọn trên validation (mục 4.6), cùng **đường precision–recall** trên test.

Với chỉ 7 bộ dữ liệu, Wilcoxon có năng lực thống kê thấp: p nhỏ nhất có thể đạt được (khi một mô hình thắng ở cả 7 bộ) là 0,0156 trước hiệu chỉnh. Với 4 mô hình (6 cặp) hay 7 mô hình (21 cặp), sau hiệu chỉnh Holm không cặp nào có thể xuống dưới 0,05.

Vì vậy, khi so sánh 7 mô hình, đồ án bổ sung hai kiểm định hậu nghiệm trên **hạng trung bình** theo Demšar (2006), là cách làm chuẩn khi có nhiều mô hình nhưng ít bộ dữ liệu:

- **So sánh với mô hình có hạng tốt nhất (control)**: kiểm định z trên chênh lệch hạng, z = (Rᵢ − R_best) / √(k(k+1)/(6N)), hiệu chỉnh Holm cho k − 1 phép so sánh. Ít phép so sánh hơn nên mạnh hơn so sánh mọi cặp.
- **Nemenyi critical difference (CD)**: CD = q₀.₀₅ · √(k(k+1)/(6N)); hai mô hình có hạng trung bình cách nhau ít hơn CD được coi là chưa khác biệt có ý nghĩa. Với k = 7, N = 7: CD = 3,41.

## 4.8. Learning curves

Learning curve cho biết hiệu năng thay đổi thế nào khi tăng lượng dữ liệu huấn luyện, từ đó phân biệt mô hình bị giới hạn bởi **dữ liệu** (đường validation còn đi lên) hay bởi **chính mô hình** (đường validation đã nằm ngang), và mức overfit (khoảng cách giữa AUC train và AUC validation).

| Mục | Cách làm |
|---|---|
| Tỷ lệ dữ liệu train | 2%, 5%, 10%, 20%, 40%, 70%, 100% phần train của fold; bỏ các mức dưới 50 dòng |
| Lấy mẫu con | Phân tầng theo nhãn và **lồng nhau** (các dòng ở mức nhỏ luôn nằm trong mức lớn hơn), seed cố định |
| Tiền xử lý | Tầng B.1 được **fit lại trên chính mẫu con** (cột theo kết quả B.0 của fold), để mẫu con nhỏ không được hưởng thống kê của các dòng mà mô hình không thấy |
| Đánh giá | AUC trên mẫu con (đường train; không tính cho KNN) và AUC trên **validation** của fold (đường validation). Tập test **không** được dùng |
| Fold | 5 fold cho các bộ CV; riêng Home Credit chỉ fold 1 (giới hạn chi phí của KNN); Lending Club: train 2013, validation 2014 |

## 4.9. Kiểm soát rò rỉ qua giá trị thiếu (HMEQ)

Để đo mức ảnh hưởng của rò rỉ ở HMEQ (mục 3.6), đồ án tạo thêm hai phiên bản, chạy đủ 7 mô hình và so sánh với bản gốc. Bản gốc `hmeq` vẫn được giữ trong so sánh chính để đúng giao thức.

| Phiên bản | Cách làm | Tầng | Ưu điểm | Nhược điểm |
|---|---|---|---|---|
| `hmeq` (gốc) | Tiền xử lý Baesens: median và nhóm `__missing__` | — | Đúng giao thức | Mô hình đọc được mẫu thiếu |
| `hmeq_complete` | Chỉ giữ hồ sơ đủ dữ liệu (3.364 dòng) | A (lọc dòng theo quy tắc cố định) | Không còn giá trị thiếu nào: sạch tuyệt đối | Đổi tổng thể: bỏ 43,6% dữ liệu, tỷ lệ xấu còn 8,9%; chia fold lại |
| `hmeq_blind` | Giữ mọi dòng và **đúng các fold** của bản gốc; mỗi giá trị thiếu (số và phân loại) được điền bằng **một giá trị rút ngẫu nhiên** từ các giá trị quan sát được của cột đó trên train | B.1 (học trên train của fold) | Giữ tổng thể; so sánh ghép cặp với bản gốc; xóa nhóm `__missing__` và điểm median | Không xóa hết dấu vết (mục 6.11) |

Cách điền ngẫu nhiên được cài đặt trong `src/features/preprocess.py` (`RandomDrawImputer`, seed cố định) và chỉ áp dụng cho `hmeq_blind`; mọi bộ dữ liệu khác vẫn dùng đúng tiền xử lý Baesens.

## 4.10. Kiểm tra rò rỉ và overfit

Script `src/models/validity_checks.py` chạy các kiểm tra sau trên fold 1 của mỗi bộ dữ liệu (chỉ phần test), kết quả trong `reports/models/validity/`:

| Kiểm tra | Câu hỏi | Dấu hiệu bất thường |
|---|---|---|
| Khoảng cách láng giềng gần nhất | Dòng test có phải bản sao gần đúng của dòng train không? | Khoảng cách test → train nhỏ hơn rõ so với train → train |
| Đặc trưng đơn mạnh nhất | Có cột nào gần như tiết lộ nhãn không? | AUC của một cột đơn lẻ (mô hình cây, giữ NaN) > 0,9 |
| Mô hình chỉ dùng mẫu thiếu | Việc thiếu dữ liệu có mang thông tin về nhãn không? | AUC cao, nhất là khi tiệm cận AUC của mô hình đầy đủ |
| Nhận ra dòng thiếu sau B.1 | Sau tiền xử lý, mô hình còn nhận ra được dòng nào từng bị thiếu không? | AUC gần 1 (đi cùng mẫu thiếu có thông tin thì là kênh rò rỉ) |
| Khoảng cách AUC train − test | Mô hình có học thuộc tập train không? | Khoảng cách lớn |
| Khoảng cách AUC validation − test | Lựa chọn trên validation (ngưỡng, sau này là tham số) có tổng quát hóa sang test không? | Validation cao hơn test một cách hệ thống |

Ngoài ra, các kiểm soát rò rỉ có sẵn trong thiết kế: tiền xử lý chỉ học trên train (mục 4.1), chia theo nhóm dòng trùng lặp, Lending Club chia theo thời gian và bỏ cột chỉ có sau khi giải ngân, ngưỡng chọn trên validation.
