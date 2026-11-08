---
type: algorithm
level: L3
up: '[[Distance-based models]]'
base:
- '[[Distance-based models]]'
inherited:
- '[[Posterior probability]]'
new:
- '[[Nearest neighbor error bound]]'
config: '[[KNN config]]'
question: KNN phân loại theo các láng giềng gần nhất thế nào?
glossary:
- '[[KNN_cover_hart_1967_glossary#2. The nearest neighbor rule]]'
- '[[KNN_cover_hart_1967_glossary#5. Main results: error bounds]]'
papers:
- '[[Cover Hart 1967]]'
aliases:
- k-nearest neighbors
- k láng giềng gần nhất
---

# KNN (k láng giềng gần nhất)

**Định nghĩa:** Dự đoán theo tỷ lệ lớp trong *k* quan sát huấn luyện gần nhất; *k* = 1 là luật nearest neighbor của Cover & Hart.

**Suy ra từ:** [[Distance-based models]]

**Nguồn:** [[Cover Hart 1967]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Posterior probability]] · [[Distance-based models]] · [[Nearest neighbor error bound]]
**➡ Mở ra:** [[KNN config]]
**🎯 Dùng trong mô hình:** [[KNN]]
**↳ Nhánh con (cây phân loại):** [[Nearest neighbor error bound]] · [[KNN config]]

### Lộ trình ngược (tự sinh)

1. [[Conditional probability]] `L0` ▫️ kiến thức trước — Biết một điều thì xác suất của điều khác thay đổi thế nào? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]]
2. [[Euclidean distance]] `L0` ▫️ kiến thức trước — Đo khoảng cách giữa hai khách hàng trong không gian đặc trưng thế nào? [[KNN_cover_hart_1967_glossary#1. The classification problem|📖 KNN_cover_hart_1967 §1]]
3. [[Prior probability]] `L0` ▫️ kiến thức trước — Trước khi nhìn dữ liệu của một khách hàng, khả năng vỡ nợ là bao nhiêu? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]]
4. [[Bayes theorem]] `L0` ▫️ kiến thức trước — Từ P(đặc trưng | lớp) suy ngược ra P(lớp | đặc trưng) thế nào? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]]
5. [[Supervised learning]] `L1` ▫️ kiến thức trước — Học một hàm từ đặc trưng *x* đến nhãn *y* trên dữ liệu đã có nhãn là gì? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
6. [[Feature]] `L1` ▫️ kiến thức trước — Thông tin nào của khách hàng được đưa vào mô hình? [[LR_cox_1958_glossary#1. Basic set-up and data|📖 LR_cox_1958 §1]] [[NB_maron_1961_glossary#2. Basic notions: words, evidence, hypotheses, predictions|📖 NB_maron_1961 §2]]
7. [[Feature scaling]] `L1` ▫️ kiến thức trước — Vì sao cần đưa các đặc trưng về cùng thang đo? [[RF_breiman_2001_glossary#4. Random features: Forest-RI and Forest-RC|📖 RF_breiman_2001 §4]]
8. [[Distance-based models]] `L2` 🟦 nền — Dự đoán dựa trên các khách hàng giống nhất là gì? [[KNN_cover_hart_1967_glossary#2. The nearest neighbor rule|📖 KNN_cover_hart_1967 §2]]
9. [[Posterior probability]] `L0` 🟨 kế thừa — Sau khi thấy dữ liệu, xác suất mỗi lớp là bao nhiêu? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]] [[KNN_cover_hart_1967_glossary#3. Bayes procedure (the benchmark)|📖 KNN_cover_hart_1967 §3]]
10. [[Bayes error]] `L0` ▫️ kiến thức trước — Sai số nhỏ nhất mà bất kỳ mô hình nào có thể đạt là bao nhiêu? [[KNN_cover_hart_1967_glossary#3. Bayes procedure (the benchmark)|📖 KNN_cover_hart_1967 §3]]
11. [[Nearest neighbor error bound]] `L3` 🟥 mới — Vì sao sai số của luật láng giềng gần nhất không quá hai lần sai số Bayes? [[KNN_cover_hart_1967_glossary#5. Main results: error bounds|📖 KNN_cover_hart_1967 §5]]
12. [[KNN]] `L3` 🎯 mô hình — KNN phân loại theo các láng giềng gần nhất thế nào? [[KNN_cover_hart_1967_glossary#2. The nearest neighbor rule|📖 KNN_cover_hart_1967 §2]] [[KNN_cover_hart_1967_glossary#5. Main results: error bounds|📖 KNN_cover_hart_1967 §5]]
13. [[KNN config]] `L4` ⚙️ đồ án — Đồ án cấu hình KNN thế nào? [[KNN_cover_hart_1967_glossary#10. Relevance to this project (credit scoring with KNN)|📖 KNN_cover_hart_1967 §10]]
<!-- tree:auto end -->
