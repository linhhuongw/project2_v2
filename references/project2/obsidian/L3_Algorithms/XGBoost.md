---
type: algorithm
level: L3
up: '[[Gradient Boosting]]'
base:
- '[[Gradient Boosting]]'
inherited:
- '[[Regularization]]'
- '[[Random feature subspace]]'
new:
- '[[Regularized objective]]'
- '[[Split gain]]'
- '[[Weighted quantile sketch]]'
- '[[Sparsity-aware split finding]]'
config: '[[XGBoost config]]'
question: XGBoost thêm điều chuẩn và xấp xỉ bậc hai vào gradient boosting thế nào?
glossary:
- '[[XGB_chen_guestrin_2016_glossary#1. Tree ensemble model and regularized objective]]'
- '[[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting]]'
papers:
- '[[Chen Guestrin 2016]]'
---

# XGBoost

**Định nghĩa:** GBDT với mục tiêu có điều chuẩn, xấp xỉ bậc hai, tìm điểm chia nhận biết giá trị thiếu và hệ thống tính toán mở rộng được.

**Suy ra từ:** [[Gradient Boosting]]

**Nguồn:** [[Chen Guestrin 2016]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Regularization]] · [[Gradient Boosting]] · [[Random feature subspace]] · [[Regularized objective]] · [[Sparsity-aware split finding]] · [[Split gain]] · [[Weighted quantile sketch]]
**➡ Mở ra:** [[XGBoost config]]
**🎯 Dùng trong mô hình:** [[XGBoost]]
**↳ Nhánh con (cây phân loại):** [[Regularized objective]] · [[Sparsity-aware split finding]] · [[Split gain]] · [[Weighted quantile sketch]] · [[XGBoost config]]

### Lộ trình ngược (tự sinh)

1. [[Bernoulli distribution]] `L0` ▫️ kiến thức trước — Một biến chỉ có hai kết quả 0/1 được mô tả bằng phân phối nào? [[LR_cox_1958_glossary#1. Basic set-up and data|📖 LR_cox_1958 §1]]
2. [[Conditional probability]] `L0` ▫️ kiến thức trước — Biết một điều thì xác suất của điều khác thay đổi thế nào? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]]
3. [[Gradient descent]] `L0` ▫️ kiến thức trước — Đi theo hướng nào để giảm hàm mục tiêu nhanh nhất? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
4. [[Likelihood]] `L0` ▫️ kiến thức trước — Mô hình giải thích dữ liệu quan sát được tốt đến mức nào? [[LR_cox_1958_glossary#3. Likelihood, sufficiency and conditioning|📖 LR_cox_1958 §3]]
5. [[Newton method]] `L0` ▫️ kiến thức trước — Dùng thêm đạo hàm bậc hai để chọn bước đi tốt hơn thế nào? [[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting|📖 XGB_chen_guestrin_2016 §2]]
6. [[Odds]] `L0` ▫️ kiến thức trước — Tỷ lệ giữa khả năng xảy ra và không xảy ra là gì? [[LR_cox_1958_glossary#2. The logistic model|📖 LR_cox_1958 §2]]
7. [[Logit]] `L0` ▫️ kiến thức trước — Vì sao lấy log của odds để có một thang tuyến tính? [[LR_cox_1958_glossary#2. The logistic model|📖 LR_cox_1958 §2]]
8. [[Logistic function]] `L0` ▫️ kiến thức trước — Đổi một điểm số bất kỳ thành xác suất trong [0, 1] bằng cách nào? [[LR_cox_1958_glossary#2. The logistic model|📖 LR_cox_1958 §2]]
9. [[Shannon entropy]] `L0` ▫️ kiến thức trước — Đo độ bất định của một phân phối lớp bằng con số nào? [[NB_maron_1961_glossary#5. Clue word selection (feature selection)|📖 NB_maron_1961 §5]]
10. [[Information gain]] `L0` ▫️ kiến thức trước — Biết thêm một thông tin thì độ bất định giảm bao nhiêu? [[NB_maron_1961_glossary#5. Clue word selection (feature selection)|📖 NB_maron_1961 §5]]
11. [[Supervised learning]] `L1` ▫️ kiến thức trước — Học một hàm từ đặc trưng *x* đến nhãn *y* trên dữ liệu đã có nhãn là gì? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
12. [[Feature]] `L1` ▫️ kiến thức trước — Thông tin nào của khách hàng được đưa vào mô hình? [[LR_cox_1958_glossary#1. Basic set-up and data|📖 LR_cox_1958 §1]] [[NB_maron_1961_glossary#2. Basic notions: words, evidence, hypotheses, predictions|📖 NB_maron_1961 §2]]
13. [[Generalization]] `L1` ▫️ kiến thức trước — Mô hình có dự đoán tốt trên dữ liệu chưa từng thấy không? [[RF_breiman_2001_glossary#2. Margin, convergence, strength and correlation|📖 RF_breiman_2001 §2]]
14. [[Loss function]] `L1` ▫️ kiến thức trước — Đo mức sai của dự đoán bằng hàm nào? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
15. [[Log-loss]] `L1` ▫️ kiến thức trước — Hàm mất mát nào dùng cho bài toán xác suất 0/1, và vì sao nó chính là âm log hợp lý? [[GB_friedman_2001_glossary#3. Loss functions and specific algorithms|📖 GB_friedman_2001 §3]]
16. [[Log-loss gradient and Hessian]] `L1` ▫️ kiến thức trước — Đạo hàm bậc một g = p − y và bậc hai h = p(1 − p) của log-loss có ý nghĩa gì? [[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting|📖 XGB_chen_guestrin_2016 §2]] [[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)|📖 GB_friedman_2001 §4]]
17. [[Overfitting]] `L1` ▫️ kiến thức trước — Vì sao mô hình tốt trên tập huấn luyện lại kém trên dữ liệu mới? [[GB_friedman_2001_glossary#5. Regularization|📖 GB_friedman_2001 §5]] [[NB_maron_1961_glossary#3. Corpus and data split|📖 NB_maron_1961 §3]]
18. [[Decision tree]] `L2` ▫️ kiến thức trước — Chia không gian đặc trưng bằng các ngưỡng liên tiếp thế nào? [[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)|📖 GB_friedman_2001 §4]]
19. [[Split criterion]] `L2` ▫️ kiến thức trước — Chọn biến và ngưỡng chia tốt nhất theo tiêu chí nào? [[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)|📖 GB_friedman_2001 §4]] [[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting|📖 XGB_chen_guestrin_2016 §2]]
20. [[Tree ensembles]] `L2` ▫️ kiến thức trước — Vì sao kết hợp nhiều cây tốt hơn một cây? [[RF_breiman_2001_glossary#1. Ensembles and random vectors|📖 RF_breiman_2001 §1]]
21. [[Boosting]] `L2` ▫️ kiến thức trước — Huấn luyện các cây tuần tự, cây sau sửa lỗi cây trước thế nào? [[GB_friedman_2001_glossary#2. Optimization in function space|📖 GB_friedman_2001 §2]]
22. [[Tree growth strategy]] `L2` ▫️ kiến thức trước — Mở rộng cây theo tầng, theo lá hay đối xứng? [[LGBM_ke_2017_glossary#5. Tree growth and experiments|📖 LGBM_ke_2017 §5]]
23. [[Histogram-based split]] `L3` ▫️ kiến thức trước — Gom giá trị vào các bin để tìm điểm chia nhanh hơn thế nào? [[LGBM_ke_2017_glossary#1. GBDT and its cost|📖 LGBM_ke_2017 §1]] [[XGB_chen_guestrin_2016_glossary#4. Split finding algorithms|📖 XGB_chen_guestrin_2016 §4]]
24. [[Level-wise growth]] `L3` ▫️ kiến thức trước — Mở rộng mọi nút cùng độ sâu trước khi xuống tầng dưới là gì? [[LGBM_ke_2017_glossary#5. Tree growth and experiments|📖 LGBM_ke_2017 §5]]
25. [[Pseudo-residual]] `L3` ▫️ kiến thức trước — Phần dư giả (gradient âm) là gì và vì sao mỗi cây học nó? [[GB_friedman_2001_glossary#2. Optimization in function space|📖 GB_friedman_2001 §2]]
26. [[Shrinkage]] `L3` ▫️ kiến thức trước — Vì sao nhân mỗi cây với learning rate nhỏ lại tổng quát tốt hơn? [[GB_friedman_2001_glossary#5. Regularization|📖 GB_friedman_2001 §5]]
27. [[Gradient Boosting]] `L3` 🟦 nền — Gradient boosting cộng dần các cây khớp với gradient âm thế nào? [[GB_friedman_2001_glossary#2. Optimization in function space|📖 GB_friedman_2001 §2]] [[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)|📖 GB_friedman_2001 §4]]
28. [[Regularization]] `L0` 🟨 kế thừa — Phạt độ phức tạp của mô hình để nó không học thuộc dữ liệu thế nào? [[XGB_chen_guestrin_2016_glossary#1. Tree ensemble model and regularized objective|📖 XGB_chen_guestrin_2016 §1]] [[GB_friedman_2001_glossary#5. Regularization|📖 GB_friedman_2001 §5]]
29. [[Random feature subspace]] `L3` 🟨 kế thừa — Chỉ xét ngẫu nhiên một số đặc trưng ở mỗi nút giúp gì? [[RF_breiman_2001_glossary#4. Random features: Forest-RI and Forest-RC|📖 RF_breiman_2001 §4]]
30. [[Regularized objective]] `L3` 🟥 mới — Hàm mục tiêu của XGBoost phạt số lá và trọng số lá thế nào? [[XGB_chen_guestrin_2016_glossary#1. Tree ensemble model and regularized objective|📖 XGB_chen_guestrin_2016 §1]]
31. [[Split gain]] `L3` 🟥 mới — Trọng số lá w* = −G/(H+λ) và Gain của một phép chia được tính thế nào? [[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting|📖 XGB_chen_guestrin_2016 §2]]
32. [[Sparsity-aware split finding]] `L3` 🟥 mới — Mẫu thiếu giá trị đi về nhánh nào? [[XGB_chen_guestrin_2016_glossary#5. Sparsity-aware split finding|📖 XGB_chen_guestrin_2016 §5]]
33. [[Weighted quantile sketch]] `L3` 🟥 mới — Chọn các điểm chia ứng viên có trọng số h thế nào? [[XGB_chen_guestrin_2016_glossary#4. Split finding algorithms|📖 XGB_chen_guestrin_2016 §4]]
34. [[XGBoost]] `L3` 🎯 mô hình — XGBoost thêm điều chuẩn và xấp xỉ bậc hai vào gradient boosting thế nào? [[XGB_chen_guestrin_2016_glossary#1. Tree ensemble model and regularized objective|📖 XGB_chen_guestrin_2016 §1]] [[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting|📖 XGB_chen_guestrin_2016 §2]]
35. [[XGBoost config]] `L4` ⚙️ đồ án — Đồ án cấu hình XGBoost thế nào? [[XGB_chen_guestrin_2016_glossary#11. Relevance to this project (credit scoring with XGBoost)|📖 XGB_chen_guestrin_2016 §11]]
<!-- tree:auto end -->
