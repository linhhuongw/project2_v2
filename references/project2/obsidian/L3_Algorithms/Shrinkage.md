---
type: algorithm
level: L3
up: '[[Gradient Boosting]]'
requires:
- '[[Overfitting]]'
question: Vì sao nhân mỗi cây với learning rate nhỏ lại tổng quát tốt hơn?
glossary:
- '[[GB_friedman_2001_glossary#5. Regularization]]'
papers:
- '[[Friedman 2001]]'
aliases:
- Learning rate
- η
- Tốc độ học
---

# Shrinkage (η)

**Định nghĩa:** Nhân mỗi cây với $\eta\in(0,1]$; η nhỏ cần nhiều cây hơn nhưng thường tổng quát tốt hơn.

**Suy ra từ:** [[Gradient Boosting]]

**Nguồn:** [[Friedman 2001]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Overfitting]]
**➡ Mở ra:** [[Gradient Boosting]]
**🎯 Dùng trong mô hình:** [[XGBoost]] · [[LightGBM]] · [[CatBoost]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
