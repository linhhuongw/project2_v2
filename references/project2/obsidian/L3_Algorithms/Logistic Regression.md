---
type: algorithm
level: L3
up: '[[Linear probabilistic models]]'
base:
- '[[Linear probabilistic models]]'
inherited:
- '[[Maximum likelihood]]'
- '[[Log-loss]]'
new:
- '[[Logistic function]]'
- '[[Odds ratio]]'
config: '[[LR config]]'
question: Hồi quy logistic dự đoán xác suất vỡ nợ thế nào?
glossary:
- '[[LR_cox_1958_glossary#2. The logistic model]]'
- '[[LR_cox_1958_glossary#7. Estimation methods for β]]'
papers:
- '[[Cox 1958]]'
aliases:
- LR
- Hồi quy logistic
---

# Logistic Regression (Hồi quy logistic)

Mô hình phân biệt (discriminative): mô hình hóa trực tiếp [[Predicted probability|xác suất]] vỡ nợ qua [[Logistic function|hàm logistic]] của tổ hợp tuyến tính các [[Feature|đặc trưng]]; tuyến tính trên thang [[Logit]].

- Nguồn gốc: [[Cox 1958]]
- Huấn luyện: [[Maximum likelihood]] (scikit-learn `LogisticRegression`, mặc định phạt L2)
- Diễn giải hệ số: [[Odds ratio]]
- So sánh: [[Naive Bayes]] mô hình hóa $P(x\mid y)$ rồi dùng [[Bayes theorem]]; LR mô hình hóa $P(y\mid x)$ trực tiếp, không cần [[Independence assumption|giả định độc lập]] giữa các đặc trưng.

Thuộc [[Project2 - Credit scoring]].

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Logistic function]] · [[Maximum likelihood]] · [[Odds ratio]] · [[Log-loss]] · [[Linear probabilistic models]]
**➡ Mở ra:** [[LR config]]
**🎯 Dùng trong mô hình:** [[Logistic Regression]]
**↳ Nhánh con (cây phân loại):** [[LR config]]

### Lộ trình ngược (tự sinh)

1. [[Bernoulli distribution]] `L0` ▫️ kiến thức trước — Một biến chỉ có hai kết quả 0/1 được mô tả bằng phân phối nào? [[LR_cox_1958_glossary#1. Basic set-up and data|📖 LR_cox_1958 §1]]
2. [[Gradient descent]] `L0` ▫️ kiến thức trước — Đi theo hướng nào để giảm hàm mục tiêu nhanh nhất? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
3. [[Likelihood]] `L0` ▫️ kiến thức trước — Mô hình giải thích dữ liệu quan sát được tốt đến mức nào? [[LR_cox_1958_glossary#3. Likelihood, sufficiency and conditioning|📖 LR_cox_1958 §3]]
4. [[Newton method]] `L0` ▫️ kiến thức trước — Dùng thêm đạo hàm bậc hai để chọn bước đi tốt hơn thế nào? [[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting|📖 XGB_chen_guestrin_2016 §2]]
5. [[Odds]] `L0` ▫️ kiến thức trước — Tỷ lệ giữa khả năng xảy ra và không xảy ra là gì? [[LR_cox_1958_glossary#2. The logistic model|📖 LR_cox_1958 §2]]
6. [[Logit]] `L0` ▫️ kiến thức trước — Vì sao lấy log của odds để có một thang tuyến tính? [[LR_cox_1958_glossary#2. The logistic model|📖 LR_cox_1958 §2]]
7. [[Supervised learning]] `L1` ▫️ kiến thức trước — Học một hàm từ đặc trưng *x* đến nhãn *y* trên dữ liệu đã có nhãn là gì? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
8. [[Feature]] `L1` ▫️ kiến thức trước — Thông tin nào của khách hàng được đưa vào mô hình? [[LR_cox_1958_glossary#1. Basic set-up and data|📖 LR_cox_1958 §1]] [[NB_maron_1961_glossary#2. Basic notions: words, evidence, hypotheses, predictions|📖 NB_maron_1961 §2]]
9. [[Feature scaling]] `L1` ▫️ kiến thức trước — Vì sao cần đưa các đặc trưng về cùng thang đo? [[RF_breiman_2001_glossary#4. Random features: Forest-RI and Forest-RC|📖 RF_breiman_2001 §4]]
10. [[Generalization]] `L1` ▫️ kiến thức trước — Mô hình có dự đoán tốt trên dữ liệu chưa từng thấy không? [[RF_breiman_2001_glossary#2. Margin, convergence, strength and correlation|📖 RF_breiman_2001 §2]]
11. [[Loss function]] `L1` ▫️ kiến thức trước — Đo mức sai của dự đoán bằng hàm nào? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
12. [[Overfitting]] `L1` ▫️ kiến thức trước — Vì sao mô hình tốt trên tập huấn luyện lại kém trên dữ liệu mới? [[GB_friedman_2001_glossary#5. Regularization|📖 GB_friedman_2001 §5]] [[NB_maron_1961_glossary#3. Corpus and data split|📖 NB_maron_1961 §3]]
13. [[Regularization]] `L0` ▫️ kiến thức trước — Phạt độ phức tạp của mô hình để nó không học thuộc dữ liệu thế nào? [[XGB_chen_guestrin_2016_glossary#1. Tree ensemble model and regularized objective|📖 XGB_chen_guestrin_2016 §1]] [[GB_friedman_2001_glossary#5. Regularization|📖 GB_friedman_2001 §5]]
14. [[Maximum likelihood]] `L0` 🟨 kế thừa — Chọn tham số sao cho dữ liệu quan sát được có khả năng xảy ra cao nhất thế nào? [[LR_cox_1958_glossary#7. Estimation methods for β|📖 LR_cox_1958 §7]]
15. [[Logistic function]] `L0` 🟥 mới — Đổi một điểm số bất kỳ thành xác suất trong [0, 1] bằng cách nào? [[LR_cox_1958_glossary#2. The logistic model|📖 LR_cox_1958 §2]]
16. [[Linear probabilistic models]] `L2` 🟦 nền — Mô hình xác suất dựa trên tổ hợp tuyến tính của đặc trưng là gì? [[LR_cox_1958_glossary#2. The logistic model|📖 LR_cox_1958 §2]]
17. [[Log-loss]] `L1` 🟨 kế thừa — Hàm mất mát nào dùng cho bài toán xác suất 0/1, và vì sao nó chính là âm log hợp lý? [[GB_friedman_2001_glossary#3. Loss functions and specific algorithms|📖 GB_friedman_2001 §3]]
18. [[Odds ratio]] `L0` 🟥 mới — Một hệ số hồi quy logistic được diễn giải ra sao? [[LR_cox_1958_glossary#6. Contingency tables and classical tests as special cases|📖 LR_cox_1958 §6]]
19. [[Logistic Regression]] `L3` 🎯 mô hình — Hồi quy logistic dự đoán xác suất vỡ nợ thế nào? [[LR_cox_1958_glossary#2. The logistic model|📖 LR_cox_1958 §2]] [[LR_cox_1958_glossary#7. Estimation methods for β|📖 LR_cox_1958 §7]]
20. [[LR config]] `L4` ⚙️ đồ án — Đồ án cấu hình Logistic Regression thế nào? [[LR_cox_1958_glossary#14. Relevance to this project (credit scoring with LR)|📖 LR_cox_1958 §14]]
<!-- tree:auto end -->
