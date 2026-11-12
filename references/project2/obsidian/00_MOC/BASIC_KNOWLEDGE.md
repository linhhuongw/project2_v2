# BASIC KNOWLEDGE — Kiến thức nền theo từng bước xây dựng mô hình

Trang này đi theo một workflow hoàn chỉnh của bài toán chấm điểm tín dụng trong đồ án: từ định nghĩa bài toán tới so sánh mô hình. Ở mỗi bước, trang liệt kê các khái niệm nền tảng trong `L0_Foundations` cần nắm, kèm lý do vì sao cần ở bước đó.

- Muốn đi sâu một mô hình cụ thể: xem [[03 Model Paths]].
- Muốn xem toàn bộ cây phân loại: xem [[00 Knowledge Tree]].
- Mỗi khái niệm có câu hỏi `question` và liên kết tới glossary của bài báo gốc trong chính ghi chú của nó.

```mermaid
flowchart LR
  S1[1. Định nghĩa bài toán] --> S2[2. Khám phá dữ liệu]
  S2 --> S3[3. Chia dữ liệu]
  S3 --> S4[4. Tiền xử lý]
  S4 --> S5[5. Chọn mô hình]
  S5 --> S6[6. Huấn luyện]
  S6 --> S7[7. Dự đoán và đánh giá]
  S7 --> S8[8. So sánh mô hình]
```

---

## 1. Định nghĩa bài toán

Mỗi khoản vay có vector đặc trưng x và nhãn y ∈ {0, 1} (1 = vỡ nợ). Mô hình cần ước lượng p(x) = P(y = 1 | x), sau đó so với một ngưỡng để ra quyết định.

| Khái niệm | Vì sao cần ở bước này |
|---|---|
| [[Probability]] | Đầu ra của mô hình là một xác suất, không phải một nhãn |
| [[Bernoulli distribution]] | Nhãn y chỉ có hai giá trị; đây là phân phối của y khi biết x |
| [[Conditional probability]] | Đại lượng cần ước lượng là P(y = 1 \| x) |
| [[Prior probability]] | Tỷ lệ vỡ nợ chung P(y = 1) — mức mất cân bằng nhãn của từng bộ dữ liệu |
| [[Odds]] | Cách nói khác của rủi ro: p / (1 − p); nền cho Logistic Regression |

## 2. Khám phá dữ liệu

Tìm hiểu phân phối từng biến, quan hệ của biến với nhãn và các vấn đề chất lượng dữ liệu.

| Khái niệm | Vì sao cần ở bước này |
|---|---|
| [[Statistics]] | Thống kê mô tả: trung bình, phân vị, tỷ lệ thiếu |
| [[Probability distributions]] · [[Gaussian distribution]] | Biến liên tục có lệch, có ngoại lai không; Gaussian NB giả định phân phối chuẩn trong từng lớp |
| [[Contingency table]] | Bảng chéo biến phân loại × nhãn: tỷ lệ vỡ nợ theo từng hạng mục |
| [[Odds ratio]] | So sánh rủi ro giữa hai hạng mục, ví dụ "đã từng trễ hạn" và "chưa từng" |
| [[Empty cells and small counts]] | Hạng mục hiếm làm ước lượng không ổn định → tầng B.0 loại cột thưa |
| [[Sample size]] | Bộ nhỏ (South German, 1.000 dòng) và bộ lớn (Lending Club) cần cách đánh giá khác nhau |

## 3. Chia dữ liệu

Mô hình phải được đánh giá trên dữ liệu chưa thấy. Đồ án dùng 5-fold CV cho 6 bộ dữ liệu, lấy 20% tập train của mỗi fold làm validation; Lending Club được chia theo thời gian.

| Khái niệm | Vì sao cần ở bước này |
|---|---|
| [[Sampling]] | Tập train, validation, test là các mẫu phải đại diện cho tổng thể; với dữ liệu có thời gian thì test phải là giai đoạn sau |
| [[Sample size]] | Validation quá nhỏ cho kết quả dao động mạnh, như South German |
| [[Bootstrap]] | Lấy mẫu có hoàn lại; dùng lại ở bước 5 (Random Forest) và bước 8 (bootstrap ghép cặp) |

## 4. Tiền xử lý

Tầng B.1 gồm điền giá trị thiếu, mã hóa biến phân loại và chuẩn hóa cho nhóm classic ML. Mọi tham số tiền xử lý chỉ được học trên tập train.

| Khái niệm | Vì sao cần ở bước này |
|---|---|
| [[Geometry]] · [[Euclidean distance]] | KNN đo khoảng cách nên biến có thang lớn sẽ lấn át → phải chuẩn hóa; one-hot làm số chiều tăng |
| [[Statistics]] | Trung bình, độ lệch chuẩn, trung vị dùng để chuẩn hóa và điền giá trị thiếu; ngoại lai kéo lệch các con số này (GMSC) |
| [[Empty cells and small counts]] | Hạng mục hiếm sau one-hot thành cột gần như toàn 0 |

## 5. Chọn mô hình — hiểu giả định của từng mô hình

Mỗi mô hình đặt một giả định khác nhau về P(y = 1 | x). Hiểu giả định giúp giải thích vì sao mô hình thắng thay đổi giữa các bộ dữ liệu.

| Mô hình | Giả định | Khái niệm L0 |
|---|---|---|
| Logistic Regression | Log-odds tuyến tính theo x | [[Odds]] → [[Logit]] → [[Logistic function]]; diễn giải hệ số bằng [[Odds ratio]] |
| Naive Bayes | Các đặc trưng độc lập khi biết lớp | [[Bayes theorem]], [[Prior probability]], [[Posterior probability]], [[Independence assumption]], [[Gaussian distribution]] |
| KNN | Khách hàng gần nhau có rủi ro giống nhau | [[Euclidean distance]], [[Bayes error]] (KNN có sai số tối đa gấp đôi sai số Bayes khi n → ∞) |
| Random Forest | Chia vùng bằng cây, lấy trung bình nhiều cây | [[Shannon entropy]] → [[Information gain]] (tiêu chí tách), [[Bootstrap]] (bagging) |
| XGBoost / LightGBM / CatBoost | Cộng dồn các cây học trên phần dư | [[Information theory]] (tiêu chí tách), và các khái niệm tối ưu ở bước 6 |

## 6. Huấn luyện — ước lượng tham số

Huấn luyện là tìm tham số làm dữ liệu train "hợp lý" nhất, đồng thời không để mô hình học thuộc.

| Khái niệm | Vì sao cần ở bước này |
|---|---|
| [[Parameter estimation]] | Bài toán chung: từ dữ liệu suy ra tham số |
| [[Likelihood]] → [[Maximum likelihood]] | LR được học bằng cực đại hợp lý; log-loss chính là −log-likelihood của phân phối Bernoulli |
| [[Sufficient statistic]] · [[Nuisance parameter]] | Lý thuyết gốc của Cox (1958): dữ liệu được tóm gọn thế nào và cách loại bỏ hệ số chặn khi suy luận |
| [[Optimization]] | Huấn luyện là cực tiểu một hàm mục tiêu |
| [[Gradient descent]] | Gradient boosting là gradient descent trong không gian hàm: mỗi cây đi theo −gradient của loss |
| [[Newton method]] | Solver `lbfgs` của LR là quasi-Newton; XGBoost, LightGBM và CatBoost dùng cả gradient và Hessian (bước Newton) để tính giá trị lá và độ lợi khi tách |
| [[Regularization]] | Phạt L2 trong LR (`C = 1`); λ trong XGBoost; `l2_leaf_reg` trong CatBoost |

## 7. Dự đoán và đánh giá

Mô hình cho xác suất trên test. Đồ án đánh giá khả năng xếp hạng (AUC), chất lượng xác suất (Brier, LogLoss) và nhãn tại một ngưỡng (precision, recall, F1).

| Khái niệm | Vì sao cần ở bước này |
|---|---|
| [[Posterior probability]] | Đầu ra p(x) chính là xác suất hậu nghiệm của lớp xấu |
| [[Bayes error]] | Giới hạn dưới của sai số: không mô hình nào vượt qua được |
| [[Likelihood]] | LogLoss đo mức hợp lý của xác suất dự đoán trên test |
| [[Prior probability]] | Nhãn mất cân bằng nên accuracy gây hiểu lầm; ngưỡng tối đa F1 được chọn trên validation |

## 8. So sánh mô hình

Chênh lệch giữa các mô hình phải được kiểm định, vì 7 bộ dữ liệu là ít.

| Khái niệm | Vì sao cần ở bước này |
|---|---|
| [[Hypothesis testing]] · [[Statistical significance]] | Khung chung: giả thuyết "các mô hình như nhau", p-value |
| [[Friedman test]] | So sánh nhiều mô hình trên nhiều bộ dữ liệu bằng hạng trung bình (có hiệu chỉnh Iman–Davenport) |
| [[Nemenyi test]] | Kiểm định hậu nghiệm: critical difference giữa các hạng trung bình |
| [[Wilcoxon signed-rank test]] | So sánh từng cặp mô hình trên cùng các bộ dữ liệu |
| [[Holm correction]] | Hiệu chỉnh khi thực hiện nhiều phép so sánh |
| [[Bootstrap]] | Bootstrap ghép cặp (1.000 lần) chênh lệch AUC trên test của Lending Club |
| [[Sample size]] | Với 7 bộ, Wilcoxon có năng lực thấp: sau hiệu chỉnh Holm không cặp nào xuống dưới 0,05 |
| [[Wilcoxon rank-sum test]] | Phân biệt với bản hạng có dấu: bản tổng hạng so sánh hai mẫu độc lập (Cox, 1958), không dùng để so sánh mô hình |

---

## Bảng tra ngược: khái niệm L0 → bước sử dụng

| Khái niệm | Bước |
|---|---|
| [[Bayes error]] | 5, 7 |
| [[Bayes theorem]] | 5 |
| [[Bernoulli distribution]] | 1 |
| [[Bootstrap]] | 3, 5, 8 |
| [[Conditional probability]] | 1 |
| [[Contingency table]] | 2 |
| [[Empty cells and small counts]] | 2, 4 |
| [[Euclidean distance]] | 4, 5 |
| [[Friedman test]] | 8 |
| [[Gaussian distribution]] | 2, 5 |
| [[Geometry]] | 4 |
| [[Gradient descent]] | 6 |
| [[Holm correction]] | 8 |
| [[Hypothesis testing]] | 8 |
| [[Independence assumption]] | 5 |
| [[Information gain]] | 5 |
| [[Information theory]] | 5 |
| [[Likelihood]] | 6, 7 |
| [[Logistic function]] | 5 |
| [[Logit]] | 5 |
| [[Maximum likelihood]] | 6 |
| [[Nemenyi test]] | 8 |
| [[Newton method]] | 6 |
| [[Nuisance parameter]] | 6 |
| [[Odds]] | 1, 5 |
| [[Odds ratio]] | 2, 5 |
| [[Optimization]] | 6 |
| [[Parameter estimation]] | 6 |
| [[Posterior probability]] | 5, 7 |
| [[Prior probability]] | 1, 5, 7 |
| [[Probability]] | 1 |
| [[Probability distributions]] | 2 |
| [[Regularization]] | 6 |
| [[Sample size]] | 2, 3, 8 |
| [[Sampling]] | 3 |
| [[Shannon entropy]] | 5 |
| [[Statistical significance]] | 8 |
| [[Statistics]] | 2, 4 |
| [[Sufficient statistic]] | 6 |
| [[Wilcoxon rank-sum test]] | 8 |
| [[Wilcoxon signed-rank test]] | 8 |
