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

## 4.4. Tầng B.1 — Tiền xử lý Baesens (nhánh classic ML)

| Bước | Biến số | Biến phân loại |
|---|---|---|
| 1. Giá trị thiếu | Điền bằng **median** của train | Đổi sang chuỗi, tạo **một nhóm mới** `__missing__` cho giá trị thiếu |
| 2. Mã hóa | — | **One-hot**; nhóm chưa từng xuất hiện ở train → toàn số 0 |
| 3. Chuẩn hóa | **Standardize** về trung bình 0, độ lệch chuẩn 1 theo thống kê của train | — |

Theo Baesens, các mô hình classic ML (LR, RF, KNN, NB, SVM) dùng one-hot và chuẩn hóa; các mô hình GBDT dùng mã số nguyên. Đồ án **không** thêm các bước ngoài pipeline này:

| Không làm | Lý do |
|---|---|
| Cắt hoặc xử lý ngoại lai | Giữ đúng pipeline Baesens; nếu thử thì báo cáo riêng như phân tích phụ |
| Cân bằng lớp (oversampling, undersampling, `class_weight`) | Giữ nguyên phân phối nhãn |
| Cột đánh dấu giá trị thiếu | Không có trong pipeline Baesens (thông tin thiếu của biến phân loại vẫn được giữ qua nhóm `__missing__`) |
| Tạo đặc trưng (tỷ lệ, tương tác, tổng hợp bảng phụ) | Dành cho bước sau |

**Kiểm tra tự động** cho mọi fold: không còn NaN hay giá trị vô cực; các biến số ở train có trung bình ≈ 0 và độ lệch chuẩn ≈ 1. Validation và test **không** có đúng trung bình 0 và độ lệch chuẩn 1 vì được biến đổi bằng thống kê của train — đây là dấu hiệu không có rò rỉ.

Số đặc trưng sau B.1:

| Bộ dữ liệu | Đầu vào | Số đặc trưng |
|---|---|---:|
| GMSC | 10 số | 10 |
| Taiwan | 20 số + 3 phân loại | 33 |
| Home Credit | 104 số + 16 phân loại | 250 (249 ở các fold bỏ `FLAG_MOBIL`) |
| HMEQ | 10 số + 2 phân loại | 20 |
| South German | 3 số + 17 phân loại | 71 |
| FICO HELOC | 21 số + 2 phân loại | 37 |
| Lending Club | 62 số + 8 phân loại | 184 (188 khi fit lại trên train + validation) |

## 4.5. Mô hình classic ML

Bốn mô hình thuộc nhóm classic ML của Baesens được huấn luyện với **tham số mặc định** của scikit-learn 1.7, chưa tinh chỉnh. Cả bốn dùng **cùng các fold** và **cùng ma trận đặc trưng** sau B.1 (one-hot, chuẩn hóa), nên khác biệt về kết quả chỉ đến từ mô hình.

| Mô hình | Cấu hình mặc định | Đặc điểm |
|---|---|---|
| **Logistic Regression (LR)** | `C = 1,0`, phạt L2, solver `lbfgs`; `max_iter = 2000` theo khung code của đặc tả | Tuyến tính, dễ diễn giải; là chuẩn của ngành chấm điểm tín dụng |
| **K-Nearest Neighbors (KNN)** | `k = 5`, trọng số đều, khoảng cách Euclid | Phi tham số, dựa trên khoảng cách; xác suất chỉ nhận 6 giá trị (0; 0,2; …; 1) |
| **Random Forest (RF)** | 100 cây, cây phát triển hết độ sâu, mỗi lần tách xét √p đặc trưng | Tổ hợp cây quyết định; nắm được quan hệ phi tuyến và tương tác; không nhạy với ngoại lai |
| **Naive Bayes (NB)** | `GaussianNB` | Giả định các đặc trưng độc lập có điều kiện và có phân phối chuẩn |

Với LR, mọi lần fit hội tụ sau tối đa 89 vòng (dưới giới hạn mặc định 100 của scikit-learn), nên kết quả trùng với `LogisticRegression()` mặc định hoàn toàn. KNN **không được chấm điểm trên chính tập train**, vì mỗi điểm train là láng giềng gần nhất của chính nó (AUC train vô nghĩa) và phép tính khoảng cách train × train rất tốn kém.

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

Với chỉ 7 bộ dữ liệu, Wilcoxon có năng lực thống kê thấp: p nhỏ nhất có thể đạt được (khi một mô hình thắng ở cả 7 bộ) là 0,0156 trước hiệu chỉnh; sau hiệu chỉnh Holm cho 6 cặp, không cặp nào có thể xuống dưới 0,05 nếu thứ tự hai mô hình không nhất quán tuyệt đối.

## 4.8. Learning curves

Learning curve cho biết hiệu năng thay đổi thế nào khi tăng lượng dữ liệu huấn luyện, từ đó phân biệt mô hình bị giới hạn bởi **dữ liệu** (đường validation còn đi lên) hay bởi **chính mô hình** (đường validation đã nằm ngang), và mức overfit (khoảng cách giữa AUC train và AUC validation).

| Mục | Cách làm |
|---|---|
| Tỷ lệ dữ liệu train | 2%, 5%, 10%, 20%, 40%, 70%, 100% phần train của fold; bỏ các mức dưới 50 dòng |
| Lấy mẫu con | Phân tầng theo nhãn và **lồng nhau** (các dòng ở mức nhỏ luôn nằm trong mức lớn hơn), seed cố định |
| Tiền xử lý | Tầng B.1 được **fit lại trên chính mẫu con** (cột theo kết quả B.0 của fold), để mẫu con nhỏ không được hưởng thống kê của các dòng mà mô hình không thấy |
| Đánh giá | AUC trên mẫu con (đường train; không tính cho KNN) và AUC trên **validation** của fold (đường validation). Tập test **không** được dùng |
| Fold | 5 fold cho các bộ CV; riêng Home Credit chỉ fold 1 (giới hạn chi phí của KNN); Lending Club: train 2013, validation 2014 |
