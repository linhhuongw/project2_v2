---
type: algorithm
level: L3
up: '[[Gradient Boosting]]'
base:
- '[[Gradient Boosting]]'
inherited:
- '[[Categorical encoding]]'
- '[[Data leakage]]'
new:
- '[[Prediction shift]]'
- '[[Ordered boosting]]'
- '[[Ordered target statistics]]'
- '[[Oblivious tree]]'
config: '[[CatBoost config]]'
question: CatBoost chống dịch chuyển dự đoán và xử lý biến phân loại thế nào?
glossary:
- '[[CB_prokhorenkova_2018_glossary#3. Prediction shift and ordered boosting]]'
- '[[CB_dorogush_2018_glossary#4. Fast scorer and oblivious trees]]'
papers:
- '[[Prokhorenkova 2018]]'
- '[[Dorogush 2018]]'
---

# CatBoost

**Định nghĩa:** GBDT khắc phục prediction shift bằng ordered boosting và ordered target statistics, dùng cây đối xứng, hỗ trợ trực tiếp biến phân loại.

**Suy ra từ:** [[Gradient Boosting]]

**Nguồn:** [[Prokhorenkova 2018]] · [[Dorogush 2018]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Categorical encoding]] · [[Data leakage]] · [[Gradient Boosting]] · [[Oblivious tree]] · [[Ordered boosting]] · [[Ordered target statistics]] · [[Prediction shift]]
**➡ Mở ra:** [[CatBoost config]]
**🎯 Dùng trong mô hình:** [[CatBoost]]
**↳ Nhánh con (cây phân loại):** [[Ordered target statistics]] · [[Prediction shift]] · [[CatBoost config]]

### Lộ trình ngược (tự sinh)

1. [[Bernoulli distribution]] `L0` ▫️ kiến thức trước — Một biến chỉ có hai kết quả 0/1 được mô tả bằng phân phối nào? [[LR_cox_1958_glossary#1. Basic set-up and data|📖 LR_cox_1958 §1]]
2. [[Gradient descent]] `L0` ▫️ kiến thức trước — Đi theo hướng nào để giảm hàm mục tiêu nhanh nhất? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
3. [[Likelihood]] `L0` ▫️ kiến thức trước — Mô hình giải thích dữ liệu quan sát được tốt đến mức nào? [[LR_cox_1958_glossary#3. Likelihood, sufficiency and conditioning|📖 LR_cox_1958 §3]]
4. [[Newton method]] `L0` ▫️ kiến thức trước — Dùng thêm đạo hàm bậc hai để chọn bước đi tốt hơn thế nào? [[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting|📖 XGB_chen_guestrin_2016 §2]]
5. [[Odds]] `L0` ▫️ kiến thức trước — Tỷ lệ giữa khả năng xảy ra và không xảy ra là gì? [[LR_cox_1958_glossary#2. The logistic model|📖 LR_cox_1958 §2]]
6. [[Logit]] `L0` ▫️ kiến thức trước — Vì sao lấy log của odds để có một thang tuyến tính? [[LR_cox_1958_glossary#2. The logistic model|📖 LR_cox_1958 §2]]
7. [[Logistic function]] `L0` ▫️ kiến thức trước — Đổi một điểm số bất kỳ thành xác suất trong [0, 1] bằng cách nào? [[LR_cox_1958_glossary#2. The logistic model|📖 LR_cox_1958 §2]]
8. [[Prior probability]] `L0` ▫️ kiến thức trước — Trước khi nhìn dữ liệu của một khách hàng, khả năng vỡ nợ là bao nhiêu? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]]
9. [[Supervised learning]] `L1` ▫️ kiến thức trước — Học một hàm từ đặc trưng *x* đến nhãn *y* trên dữ liệu đã có nhãn là gì? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
10. [[Feature]] `L1` ▫️ kiến thức trước — Thông tin nào của khách hàng được đưa vào mô hình? [[LR_cox_1958_glossary#1. Basic set-up and data|📖 LR_cox_1958 §1]] [[NB_maron_1961_glossary#2. Basic notions: words, evidence, hypotheses, predictions|📖 NB_maron_1961 §2]]
11. [[Generalization]] `L1` ▫️ kiến thức trước — Mô hình có dự đoán tốt trên dữ liệu chưa từng thấy không? [[RF_breiman_2001_glossary#2. Margin, convergence, strength and correlation|📖 RF_breiman_2001 §2]]
12. [[Loss function]] `L1` ▫️ kiến thức trước — Đo mức sai của dự đoán bằng hàm nào? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
13. [[Log-loss]] `L1` ▫️ kiến thức trước — Hàm mất mát nào dùng cho bài toán xác suất 0/1, và vì sao nó chính là âm log hợp lý? [[GB_friedman_2001_glossary#3. Loss functions and specific algorithms|📖 GB_friedman_2001 §3]]
14. [[Overfitting]] `L1` ▫️ kiến thức trước — Vì sao mô hình tốt trên tập huấn luyện lại kém trên dữ liệu mới? [[GB_friedman_2001_glossary#5. Regularization|📖 GB_friedman_2001 §5]] [[NB_maron_1961_glossary#3. Corpus and data split|📖 NB_maron_1961 §3]]
15. [[Train-test split]] `L1` ▫️ kiến thức trước — Tách dữ liệu thế nào để đánh giá trung thực? [[NB_maron_1961_glossary#3. Corpus and data split|📖 NB_maron_1961 §3]]
16. [[Decision tree]] `L2` ▫️ kiến thức trước — Chia không gian đặc trưng bằng các ngưỡng liên tiếp thế nào? [[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)|📖 GB_friedman_2001 §4]]
17. [[Tree ensembles]] `L2` ▫️ kiến thức trước — Vì sao kết hợp nhiều cây tốt hơn một cây? [[RF_breiman_2001_glossary#1. Ensembles and random vectors|📖 RF_breiman_2001 §1]]
18. [[Boosting]] `L2` ▫️ kiến thức trước — Huấn luyện các cây tuần tự, cây sau sửa lỗi cây trước thế nào? [[GB_friedman_2001_glossary#2. Optimization in function space|📖 GB_friedman_2001 §2]]
19. [[Tree growth strategy]] `L2` ▫️ kiến thức trước — Mở rộng cây theo tầng, theo lá hay đối xứng? [[LGBM_ke_2017_glossary#5. Tree growth and experiments|📖 LGBM_ke_2017 §5]]
20. [[Pseudo-residual]] `L3` ▫️ kiến thức trước — Phần dư giả (gradient âm) là gì và vì sao mỗi cây học nó? [[GB_friedman_2001_glossary#2. Optimization in function space|📖 GB_friedman_2001 §2]]
21. [[Shrinkage]] `L3` ▫️ kiến thức trước — Vì sao nhân mỗi cây với learning rate nhỏ lại tổng quát tốt hơn? [[GB_friedman_2001_glossary#5. Regularization|📖 GB_friedman_2001 §5]]
22. [[Gradient Boosting]] `L3` 🟦 nền — Gradient boosting cộng dần các cây khớp với gradient âm thế nào? [[GB_friedman_2001_glossary#2. Optimization in function space|📖 GB_friedman_2001 §2]] [[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)|📖 GB_friedman_2001 §4]]
23. [[Categorical encoding]] `L1` 🟨 kế thừa — Đổi biến phân loại thành số thế nào? [[CB_prokhorenkova_2018_glossary#2. Categorical features and target statistics|📖 CB_prokhorenkova_2018 §2]]
24. [[Data leakage]] `L1` 🟨 kế thừa — Thông tin của nhãn lọt vào lúc huấn luyện bằng những đường nào? [[CB_prokhorenkova_2018_glossary#2. Categorical features and target statistics|📖 CB_prokhorenkova_2018 §2]]
25. [[Oblivious tree]] `L3` 🟥 mới — Cây dùng chung một điều kiện chia cho cả tầng có lợi gì? [[CB_dorogush_2018_glossary#4. Fast scorer and oblivious trees|📖 CB_dorogush_2018 §4]] [[CB_prokhorenkova_2018_glossary#4. CatBoost implementation|📖 CB_prokhorenkova_2018 §4]]
26. [[Ordered target statistics]] `L3` 🟥 mới — Mã hóa biến phân loại bằng trung bình nhãn mà không rò rỉ thế nào? [[CB_prokhorenkova_2018_glossary#2. Categorical features and target statistics|📖 CB_prokhorenkova_2018 §2]]
27. [[Prediction shift]] `L3` 🟥 mới — Vì sao tính gradient trên chính dữ liệu đã học lại gây chệch? [[CB_prokhorenkova_2018_glossary#3. Prediction shift and ordered boosting|📖 CB_prokhorenkova_2018 §3]] [[CB_dorogush_2018_glossary#3. Fighting gradient bias|📖 CB_dorogush_2018 §3]]
28. [[Ordered boosting]] `L3` 🟥 mới — Tính phần dư bằng mô hình chưa thấy mẫu đó thế nào? [[CB_prokhorenkova_2018_glossary#3. Prediction shift and ordered boosting|📖 CB_prokhorenkova_2018 §3]] [[CB_dorogush_2018_glossary#3. Fighting gradient bias|📖 CB_dorogush_2018 §3]]
29. [[CatBoost]] `L3` 🎯 mô hình — CatBoost chống dịch chuyển dự đoán và xử lý biến phân loại thế nào? [[CB_prokhorenkova_2018_glossary#3. Prediction shift and ordered boosting|📖 CB_prokhorenkova_2018 §3]] [[CB_dorogush_2018_glossary#4. Fast scorer and oblivious trees|📖 CB_dorogush_2018 §4]]
30. [[CatBoost config]] `L4` ⚙️ đồ án — Đồ án cấu hình CatBoost thế nào? [[CB_prokhorenkova_2018_glossary#9. Relevance to this project (credit scoring with CatBoost)|📖 CB_prokhorenkova_2018 §9]]
<!-- tree:auto end -->
