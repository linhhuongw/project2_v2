---
type: algorithm
level: L3
up: '[[Generative models]]'
base:
- '[[Generative models]]'
inherited:
- '[[Bayes theorem]]'
- '[[Independence assumption]]'
- '[[Posterior probability]]'
new:
- '[[Naive Bayes formula]]'
- '[[Smoothing]]'
config: '[[NB config]]'
question: Naive Bayes kết hợp xác suất của từng đặc trưng thế nào?
glossary:
- '[[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)]]'
- '[[NB_maron_1961_glossary#7. Parameter estimation and smoothing]]'
papers:
- '[[Maron 1961]]'
aliases:
- NB
- Bayes ngây thơ
---

# Naive Bayes (Bayes ngây thơ)

Mô hình sinh (generative): ước lượng [[Prior probability|xác suất tiên nghiệm]] của lớp và [[Likelihood|xác suất có điều kiện]] của từng [[Feature|đặc trưng]] theo lớp, rồi kết hợp bằng [[Bayes theorem|định lý Bayes]] dưới [[Independence assumption|giả định độc lập có điều kiện]].

- Nguồn gốc: [[Maron 1961]] — công thức: [[Naive Bayes formula]]
- Trong đồ án: `GaussianNB` (mỗi đặc trưng theo lớp là phân phối chuẩn), khác với cách đếm từ của Maron
- [[Smoothing]]: `var_smoothing` là hậu duệ của việc thay 0,001 cho ô rỗng
- So sánh: [[Logistic Regression]]

Thuộc [[Project2 - Credit scoring]].

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Bayes theorem]] · [[Independence assumption]] · [[Posterior probability]] · [[Smoothing]] · [[Generative models]] · [[Naive Bayes formula]]
**➡ Mở ra:** [[NB config]]
**🎯 Dùng trong mô hình:** [[Naive Bayes]]
**↳ Nhánh con (cây phân loại):** [[Naive Bayes formula]] · [[NB config]]

### Lộ trình ngược (tự sinh)

1. [[Conditional probability]] `L0` ▫️ kiến thức trước — Biết một điều thì xác suất của điều khác thay đổi thế nào? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]]
2. [[Contingency table]] `L0` ▫️ kiến thức trước — Bảng đếm theo hai biến phân loại cho biết gì? [[LR_cox_1958_glossary#6. Contingency tables and classical tests as special cases|📖 LR_cox_1958 §6]]
3. [[Empty cells and small counts]] `L0` ▫️ kiến thức trước — Ước lượng hỏng ra sao khi một ô có rất ít hoặc không có mẫu? [[NB_maron_1961_glossary#7. Parameter estimation and smoothing|📖 NB_maron_1961 §7]]
4. [[Gaussian distribution]] `L0` ▫️ kiến thức trước — Phân phối chuẩn mô tả một biến liên tục thế nào? [[NB_maron_1961_glossary#14. Relevance to this project (credit scoring with NB)|📖 NB_maron_1961 §14]]
5. [[Prior probability]] `L0` ▫️ kiến thức trước — Trước khi nhìn dữ liệu của một khách hàng, khả năng vỡ nợ là bao nhiêu? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]]
6. [[Supervised learning]] `L1` ▫️ kiến thức trước — Học một hàm từ đặc trưng *x* đến nhãn *y* trên dữ liệu đã có nhãn là gì? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
7. [[Feature]] `L1` ▫️ kiến thức trước — Thông tin nào của khách hàng được đưa vào mô hình? [[LR_cox_1958_glossary#1. Basic set-up and data|📖 LR_cox_1958 §1]] [[NB_maron_1961_glossary#2. Basic notions: words, evidence, hypotheses, predictions|📖 NB_maron_1961 §2]]
8. [[Bayes theorem]] `L0` 🟨 kế thừa — Từ P(đặc trưng | lớp) suy ngược ra P(lớp | đặc trưng) thế nào? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]]
9. [[Generative models]] `L2` 🟦 nền — Mô hình hóa P(x | y) rồi suy ra P(y | x) là gì? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]]
10. [[Independence assumption]] `L0` 🟨 kế thừa — Khi nào được nhân các xác suất của từng đặc trưng với nhau? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]]
11. [[Posterior probability]] `L0` 🟨 kế thừa — Sau khi thấy dữ liệu, xác suất mỗi lớp là bao nhiêu? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]] [[KNN_cover_hart_1967_glossary#3. Bayes procedure (the benchmark)|📖 KNN_cover_hart_1967 §3]]
12. [[Smoothing]] `L1` 🟥 mới — Tránh xác suất bằng 0 khi một tổ hợp chưa từng xuất hiện thế nào? [[NB_maron_1961_glossary#7. Parameter estimation and smoothing|📖 NB_maron_1961 §7]]
13. [[Naive Bayes formula]] `L3` 🟥 mới — Xác suất hậu nghiệm = tiên nghiệm × tích các xác suất có điều kiện được suy ra thế nào? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]]
14. [[Naive Bayes]] `L3` 🎯 mô hình — Naive Bayes kết hợp xác suất của từng đặc trưng thế nào? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]] [[NB_maron_1961_glossary#7. Parameter estimation and smoothing|📖 NB_maron_1961 §7]]
15. [[NB config]] `L4` ⚙️ đồ án — Đồ án cấu hình Naive Bayes thế nào? [[NB_maron_1961_glossary#14. Relevance to this project (credit scoring with NB)|📖 NB_maron_1961 §14]]
<!-- tree:auto end -->
