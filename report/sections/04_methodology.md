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

## 4.4. Tầng B.1 — Tiền xử lý Baesens (nhánh cho Logistic Regression)

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

## 4.5. Mô hình: Logistic Regression

Mô hình cơ sở là Logistic Regression của scikit-learn với **tham số mặc định**: `C = 1,0`, phạt L2, solver `lbfgs`, không `class_weight`. Giới hạn số vòng lặp đặt `max_iter = 2000` theo khung code của đặc tả; trên thực tế mọi lần fit hội tụ sau tối đa 89 vòng (dưới giới hạn mặc định 100 của scikit-learn), nên kết quả trùng với `LogisticRegression()` mặc định hoàn toàn. Chưa tinh chỉnh siêu tham số ở bước này.

## 4.6. Độ đo đánh giá

| Nhóm | Độ đo | Ý nghĩa |
|---|---|---|
| Khả năng xếp hạng | **AUC**, Gini = 2·AUC − 1, KS | Mô hình có xếp người xấu có rủi ro cao hơn người tốt không; không phụ thuộc ngưỡng |
| Chất lượng xác suất | Brier, LogLoss | Xác suất dự đoán có gần tỷ lệ thực không; phụ thuộc tỷ lệ xấu nên không so sánh được giữa các bộ |
| Lớp thiểu số | Average Precision (AP) | Tóm tắt đường precision–recall; mức cơ sở bằng tỷ lệ xấu |
| Theo ngưỡng | Precision, Recall, F1, Accuracy | Tại ngưỡng tối đa hóa F1 **trên validation**, áp nguyên sang test |

Với 6 bộ dữ liệu CV, kết quả được báo cáo là trung bình ± độ lệch chuẩn trên 5 fold test. Với Lending Club, kết quả là mô hình fit lại trên train + validation, kèm khoảng tin cậy 95% của AUC bằng bootstrap.
