---
type: algorithm
level: L3
up: '[[Gradient Boosting]]'
base:
- '[[Gradient Boosting]]'
inherited:
- '[[Split gain]]'
- '[[Histogram-based split]]'
- '[[Leaf-wise growth]]'
new:
- '[[GOSS]]'
- '[[EFB]]'
config: '[[LightGBM config]]'
question: LightGBM làm gradient boosting nhanh hơn thế nào?
glossary:
- '[[LGBM_ke_2017_glossary#1. GBDT and its cost]]'
- '[[LGBM_ke_2017_glossary#3. Gradient-based One-Side Sampling (GOSS)]]'
- '[[LGBM_ke_2017_glossary#4. Exclusive Feature Bundling (EFB)]]'
papers:
- '[[Ke 2017]]'
---

# LightGBM

**Định nghĩa:** GBDT tối ưu tốc độ: tìm điểm chia bằng histogram, phát triển cây theo lá, GOSS và EFB.

**Suy ra từ:** [[Gradient Boosting]]

**Nguồn:** [[Ke 2017]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[EFB]] · [[GOSS]] · [[Gradient Boosting]] · [[Histogram-based split]] · [[Leaf-wise growth]] · [[Split gain]]
**➡ Mở ra:** [[LightGBM config]]
**🎯 Dùng trong mô hình:** [[LightGBM]]
**↳ Nhánh con (cây phân loại):** [[EFB]] · [[GOSS]] · [[Histogram-based split]] · [[LightGBM config]]

### Lộ trình ngược (tự sinh)

1. [[Bernoulli distribution]] `L0` ▫️ kiến thức trước — Một biến chỉ có hai kết quả 0/1 được mô tả bằng phân phối nào? [[LR_cox_1958_glossary#1. Basic set-up and data|📖 LR_cox_1958 §1]]
2. [[Conditional probability]] `L0` ▫️ kiến thức trước — Biết một điều thì xác suất của điều khác thay đổi thế nào? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]]
3. [[Gradient descent]] `L0` ▫️ kiến thức trước — Đi theo hướng nào để giảm hàm mục tiêu nhanh nhất? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
4. [[Likelihood]] `L0` ▫️ kiến thức trước — Mô hình giải thích dữ liệu quan sát được tốt đến mức nào? [[LR_cox_1958_glossary#3. Likelihood, sufficiency and conditioning|📖 LR_cox_1958 §3]]
5. [[Newton method]] `L0` ▫️ kiến thức trước — Dùng thêm đạo hàm bậc hai để chọn bước đi tốt hơn thế nào? [[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting|📖 XGB_chen_guestrin_2016 §2]]
6. [[Odds]] `L0` ▫️ kiến thức trước — Tỷ lệ giữa khả năng xảy ra và không xảy ra là gì? [[LR_cox_1958_glossary#2. The logistic model|📖 LR_cox_1958 §2]]
7. [[Logit]] `L0` ▫️ kiến thức trước — Vì sao lấy log của odds để có một thang tuyến tính? [[LR_cox_1958_glossary#2. The logistic model|📖 LR_cox_1958 §2]]
8. [[Logistic function]] `L0` ▫️ kiến thức trước — Đổi một điểm số bất kỳ thành xác suất trong [0, 1] bằng cách nào? [[LR_cox_1958_glossary#2. The logistic model|📖 LR_cox_1958 §2]]
9. [[Sampling]] `L0` ▫️ kiến thức trước — Lấy một tập con dữ liệu thế nào mà vẫn đại diện được cho toàn bộ? [[LGBM_ke_2017_glossary#2. Related approaches|📖 LGBM_ke_2017 §2]]
10. [[Shannon entropy]] `L0` ▫️ kiến thức trước — Đo độ bất định của một phân phối lớp bằng con số nào? [[NB_maron_1961_glossary#5. Clue word selection (feature selection)|📖 NB_maron_1961 §5]]
11. [[Information gain]] `L0` ▫️ kiến thức trước — Biết thêm một thông tin thì độ bất định giảm bao nhiêu? [[NB_maron_1961_glossary#5. Clue word selection (feature selection)|📖 NB_maron_1961 §5]]
12. [[Supervised learning]] `L1` ▫️ kiến thức trước — Học một hàm từ đặc trưng *x* đến nhãn *y* trên dữ liệu đã có nhãn là gì? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
13. [[Feature]] `L1` ▫️ kiến thức trước — Thông tin nào của khách hàng được đưa vào mô hình? [[LR_cox_1958_glossary#1. Basic set-up and data|📖 LR_cox_1958 §1]] [[NB_maron_1961_glossary#2. Basic notions: words, evidence, hypotheses, predictions|📖 NB_maron_1961 §2]]
14. [[Categorical encoding]] `L1` ▫️ kiến thức trước — Đổi biến phân loại thành số thế nào? [[CB_prokhorenkova_2018_glossary#2. Categorical features and target statistics|📖 CB_prokhorenkova_2018 §2]]
15. [[Generalization]] `L1` ▫️ kiến thức trước — Mô hình có dự đoán tốt trên dữ liệu chưa từng thấy không? [[RF_breiman_2001_glossary#2. Margin, convergence, strength and correlation|📖 RF_breiman_2001 §2]]
16. [[Loss function]] `L1` ▫️ kiến thức trước — Đo mức sai của dự đoán bằng hàm nào? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
17. [[Log-loss]] `L1` ▫️ kiến thức trước — Hàm mất mát nào dùng cho bài toán xác suất 0/1, và vì sao nó chính là âm log hợp lý? [[GB_friedman_2001_glossary#3. Loss functions and specific algorithms|📖 GB_friedman_2001 §3]]
18. [[Log-loss gradient and Hessian]] `L1` ▫️ kiến thức trước — Đạo hàm bậc một g = p − y và bậc hai h = p(1 − p) của log-loss có ý nghĩa gì? [[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting|📖 XGB_chen_guestrin_2016 §2]] [[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)|📖 GB_friedman_2001 §4]]
19. [[Overfitting]] `L1` ▫️ kiến thức trước — Vì sao mô hình tốt trên tập huấn luyện lại kém trên dữ liệu mới? [[GB_friedman_2001_glossary#5. Regularization|📖 GB_friedman_2001 §5]] [[NB_maron_1961_glossary#3. Corpus and data split|📖 NB_maron_1961 §3]]
20. [[Regularization]] `L0` ▫️ kiến thức trước — Phạt độ phức tạp của mô hình để nó không học thuộc dữ liệu thế nào? [[XGB_chen_guestrin_2016_glossary#1. Tree ensemble model and regularized objective|📖 XGB_chen_guestrin_2016 §1]] [[GB_friedman_2001_glossary#5. Regularization|📖 GB_friedman_2001 §5]]
21. [[Decision tree]] `L2` ▫️ kiến thức trước — Chia không gian đặc trưng bằng các ngưỡng liên tiếp thế nào? [[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)|📖 GB_friedman_2001 §4]]
22. [[Split criterion]] `L2` ▫️ kiến thức trước — Chọn biến và ngưỡng chia tốt nhất theo tiêu chí nào? [[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)|📖 GB_friedman_2001 §4]] [[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting|📖 XGB_chen_guestrin_2016 §2]]
23. [[Tree ensembles]] `L2` ▫️ kiến thức trước — Vì sao kết hợp nhiều cây tốt hơn một cây? [[RF_breiman_2001_glossary#1. Ensembles and random vectors|📖 RF_breiman_2001 §1]]
24. [[Boosting]] `L2` ▫️ kiến thức trước — Huấn luyện các cây tuần tự, cây sau sửa lỗi cây trước thế nào? [[GB_friedman_2001_glossary#2. Optimization in function space|📖 GB_friedman_2001 §2]]
25. [[Tree growth strategy]] `L2` ▫️ kiến thức trước — Mở rộng cây theo tầng, theo lá hay đối xứng? [[LGBM_ke_2017_glossary#5. Tree growth and experiments|📖 LGBM_ke_2017 §5]]
26. [[Pseudo-residual]] `L3` ▫️ kiến thức trước — Phần dư giả (gradient âm) là gì và vì sao mỗi cây học nó? [[GB_friedman_2001_glossary#2. Optimization in function space|📖 GB_friedman_2001 §2]]
27. [[Regularized objective]] `L3` ▫️ kiến thức trước — Hàm mục tiêu của XGBoost phạt số lá và trọng số lá thế nào? [[XGB_chen_guestrin_2016_glossary#1. Tree ensemble model and regularized objective|📖 XGB_chen_guestrin_2016 §1]]
28. [[Shrinkage]] `L3` ▫️ kiến thức trước — Vì sao nhân mỗi cây với learning rate nhỏ lại tổng quát tốt hơn? [[GB_friedman_2001_glossary#5. Regularization|📖 GB_friedman_2001 §5]]
29. [[Gradient Boosting]] `L3` 🟦 nền — Gradient boosting cộng dần các cây khớp với gradient âm thế nào? [[GB_friedman_2001_glossary#2. Optimization in function space|📖 GB_friedman_2001 §2]] [[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)|📖 GB_friedman_2001 §4]]
30. [[Histogram-based split]] `L3` 🟨 kế thừa — Gom giá trị vào các bin để tìm điểm chia nhanh hơn thế nào? [[LGBM_ke_2017_glossary#1. GBDT and its cost|📖 LGBM_ke_2017 §1]] [[XGB_chen_guestrin_2016_glossary#4. Split finding algorithms|📖 XGB_chen_guestrin_2016 §4]]
31. [[Split gain]] `L3` 🟨 kế thừa — Trọng số lá w* = −G/(H+λ) và Gain của một phép chia được tính thế nào? [[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting|📖 XGB_chen_guestrin_2016 §2]]
32. [[Leaf-wise growth]] `L3` 🟨 kế thừa — Luôn chia lá có Gain lớn nhất thì cây khác gì? [[LGBM_ke_2017_glossary#5. Tree growth and experiments|📖 LGBM_ke_2017 §5]]
33. [[EFB]] `L3` 🟥 mới — Gộp các cột thưa hiếm khi cùng khác 0 thành một cột thế nào? [[LGBM_ke_2017_glossary#4. Exclusive Feature Bundling (EFB)|📖 LGBM_ke_2017 §4]]
34. [[GOSS]] `L3` 🟥 mới — Bớt số dòng mà Gain vẫn gần đúng bằng cách nào? [[LGBM_ke_2017_glossary#3. Gradient-based One-Side Sampling (GOSS)|📖 LGBM_ke_2017 §3]]
35. [[LightGBM]] `L3` 🎯 mô hình — LightGBM làm gradient boosting nhanh hơn thế nào? [[LGBM_ke_2017_glossary#1. GBDT and its cost|📖 LGBM_ke_2017 §1]] [[LGBM_ke_2017_glossary#3. Gradient-based One-Side Sampling (GOSS)|📖 LGBM_ke_2017 §3]] [[LGBM_ke_2017_glossary#4. Exclusive Feature Bundling (EFB)|📖 LGBM_ke_2017 §4]]
36. [[LightGBM config]] `L4` ⚙️ đồ án — Đồ án cấu hình LightGBM thế nào? [[LGBM_ke_2017_glossary#9. Relevance to this project (credit scoring with LightGBM)|📖 LGBM_ke_2017 §9]]
<!-- tree:auto end -->
