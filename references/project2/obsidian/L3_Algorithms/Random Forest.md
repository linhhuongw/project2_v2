---
type: algorithm
level: L3
up: '[[Bagging]]'
base:
- '[[Bagging]]'
inherited:
- '[[Split criterion]]'
- '[[Bias-variance tradeoff]]'
new:
- '[[Random feature subspace]]'
- '[[Out-of-bag error]]'
- '[[Strength and correlation]]'
config: '[[RF config]]'
question: Random Forest kết hợp các cây ngẫu nhiên thế nào?
glossary:
- '[[RF_breiman_2001_glossary#2. Margin, convergence, strength and correlation]]'
- '[[RF_breiman_2001_glossary#4. Random features: Forest-RI and Forest-RC]]'
papers:
- '[[Breiman 2001]]'
aliases:
- RF
- Rừng ngẫu nhiên
---

# Random Forest (Rừng ngẫu nhiên)

**Định nghĩa:** Tập cây, mỗi cây học trên một mẫu bootstrap và mỗi nút chỉ xét một tập con đặc trưng ngẫu nhiên; dự đoán bằng trung bình.

**Suy ra từ:** [[Bagging]]

**Nguồn:** [[Breiman 2001]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Bias-variance tradeoff]] · [[Bagging]] · [[Split criterion]] · [[Out-of-bag error]] · [[Random feature subspace]] · [[Strength and correlation]]
**➡ Mở ra:** [[RF config]]
**🎯 Dùng trong mô hình:** [[Random Forest]]
**↳ Nhánh con (cây phân loại):** [[Out-of-bag error]] · [[Random feature subspace]] · [[Strength and correlation]] · [[RF config]]

### Lộ trình ngược (tự sinh)

1. [[Conditional probability]] `L0` ▫️ kiến thức trước — Biết một điều thì xác suất của điều khác thay đổi thế nào? [[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)|📖 NB_maron_1961 §6]]
2. [[Sampling]] `L0` ▫️ kiến thức trước — Lấy một tập con dữ liệu thế nào mà vẫn đại diện được cho toàn bộ? [[LGBM_ke_2017_glossary#2. Related approaches|📖 LGBM_ke_2017 §2]]
3. [[Bootstrap]] `L0` ▫️ kiến thức trước — Lấy mẫu có hoàn lại tạo ra nhiều tập huấn luyện khác nhau thế nào? [[RF_breiman_2001_glossary#3. Out-of-bag estimation|📖 RF_breiman_2001 §3]]
4. [[Shannon entropy]] `L0` ▫️ kiến thức trước — Đo độ bất định của một phân phối lớp bằng con số nào? [[NB_maron_1961_glossary#5. Clue word selection (feature selection)|📖 NB_maron_1961 §5]]
5. [[Information gain]] `L0` ▫️ kiến thức trước — Biết thêm một thông tin thì độ bất định giảm bao nhiêu? [[NB_maron_1961_glossary#5. Clue word selection (feature selection)|📖 NB_maron_1961 §5]]
6. [[Supervised learning]] `L1` ▫️ kiến thức trước — Học một hàm từ đặc trưng *x* đến nhãn *y* trên dữ liệu đã có nhãn là gì? [[GB_friedman_2001_glossary#1. Function estimation as optimization|📖 GB_friedman_2001 §1]]
7. [[Feature]] `L1` ▫️ kiến thức trước — Thông tin nào của khách hàng được đưa vào mô hình? [[LR_cox_1958_glossary#1. Basic set-up and data|📖 LR_cox_1958 §1]] [[NB_maron_1961_glossary#2. Basic notions: words, evidence, hypotheses, predictions|📖 NB_maron_1961 §2]]
8. [[Generalization]] `L1` ▫️ kiến thức trước — Mô hình có dự đoán tốt trên dữ liệu chưa từng thấy không? [[RF_breiman_2001_glossary#2. Margin, convergence, strength and correlation|📖 RF_breiman_2001 §2]]
9. [[Overfitting]] `L1` ▫️ kiến thức trước — Vì sao mô hình tốt trên tập huấn luyện lại kém trên dữ liệu mới? [[GB_friedman_2001_glossary#5. Regularization|📖 GB_friedman_2001 §5]] [[NB_maron_1961_glossary#3. Corpus and data split|📖 NB_maron_1961 §3]]
10. [[Decision tree]] `L2` ▫️ kiến thức trước — Chia không gian đặc trưng bằng các ngưỡng liên tiếp thế nào? [[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)|📖 GB_friedman_2001 §4]]
11. [[Tree ensembles]] `L2` ▫️ kiến thức trước — Vì sao kết hợp nhiều cây tốt hơn một cây? [[RF_breiman_2001_glossary#1. Ensembles and random vectors|📖 RF_breiman_2001 §1]]
12. [[Bagging]] `L2` 🟦 nền — Huấn luyện các cây độc lập trên các mẫu bootstrap rồi lấy trung bình thế nào? [[RF_breiman_2001_glossary#1. Ensembles and random vectors|📖 RF_breiman_2001 §1]]
13. [[Bias-variance tradeoff]] `L1` 🟨 kế thừa — Sai số đến từ mô hình quá đơn giản hay quá nhạy với dữ liệu? [[RF_breiman_2001_glossary#8. Conclusions|📖 RF_breiman_2001 §8]]
14. [[Split criterion]] `L2` 🟨 kế thừa — Chọn biến và ngưỡng chia tốt nhất theo tiêu chí nào? [[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)|📖 GB_friedman_2001 §4]] [[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting|📖 XGB_chen_guestrin_2016 §2]]
15. [[Gini impurity]] `L2` ▫️ kiến thức trước — Độ vẩn đục Gini đo mức trộn lẫn các lớp trong một nút thế nào? [[RF_breiman_2001_glossary#12. Relevance to this project (credit scoring with RF)|📖 RF_breiman_2001 §12]]
16. [[Out-of-bag error]] `L3` 🟥 mới — Ước lượng sai số mà không cần tập kiểm tra riêng thế nào? [[RF_breiman_2001_glossary#3. Out-of-bag estimation|📖 RF_breiman_2001 §3]]
17. [[Random feature subspace]] `L3` 🟥 mới — Chỉ xét ngẫu nhiên một số đặc trưng ở mỗi nút giúp gì? [[RF_breiman_2001_glossary#4. Random features: Forest-RI and Forest-RC|📖 RF_breiman_2001 §4]]
18. [[Strength and correlation]] `L3` 🟥 mới — Sai số của rừng phụ thuộc vào độ mạnh của từng cây và tương quan giữa các cây thế nào? [[RF_breiman_2001_glossary#2. Margin, convergence, strength and correlation|📖 RF_breiman_2001 §2]]
19. [[Random Forest]] `L3` 🎯 mô hình — Random Forest kết hợp các cây ngẫu nhiên thế nào? [[RF_breiman_2001_glossary#2. Margin, convergence, strength and correlation|📖 RF_breiman_2001 §2]] [[RF_breiman_2001_glossary#4. Random features: Forest-RI and Forest-RC|📖 RF_breiman_2001 §4]]
20. [[RF config]] `L4` ⚙️ đồ án — Đồ án cấu hình Random Forest thế nào? [[RF_breiman_2001_glossary#12. Relevance to this project (credit scoring with RF)|📖 RF_breiman_2001 §12]]
<!-- tree:auto end -->
