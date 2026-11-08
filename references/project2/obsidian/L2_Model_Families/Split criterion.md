---
type: concept
level: L2
up: '[[Decision tree]]'
requires:
- '[[Decision tree]]'
- '[[Information gain]]'
question: Chọn biến và ngưỡng chia tốt nhất theo tiêu chí nào?
glossary:
- '[[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)]]'
- '[[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting]]'
papers:
- '[[Breiman 2001]]'
- '[[Chen Guestrin 2016]]'
aliases:
- Tiêu chí chia nút
---

# Split criterion (Tiêu chí chia nút)

**Định nghĩa:** Tiêu chí chọn đặc trưng và ngưỡng chia: Gini hoặc entropy (RF), Gain tính từ g, h (XGBoost).

**Suy ra từ:** [[Decision tree]]

**Nguồn:** [[Breiman 2001]] · [[Chen Guestrin 2016]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Information gain]] · [[Decision tree]]
**➡ Mở ra:** [[Gini impurity]] · [[Histogram-based split]] · [[Random Forest]] · [[Split gain]]
**🎯 Dùng trong mô hình:** [[Random Forest]] · [[XGBoost]] · [[LightGBM]]
**↳ Nhánh con (cây phân loại):** [[Gini impurity]]
<!-- tree:auto end -->
