---
type: algorithm
level: L3
up: '[[Boosting]]'
base:
- '[[Boosting]]'
inherited:
- '[[Gradient descent]]'
- '[[Loss function]]'
- '[[Log-loss]]'
- '[[Newton method]]'
new:
- '[[Pseudo-residual]]'
- '[[Shrinkage]]'
question: Gradient boosting cộng dần các cây khớp với gradient âm thế nào?
glossary:
- '[[GB_friedman_2001_glossary#2. Optimization in function space]]'
- '[[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)]]'
papers:
- '[[Friedman 2001]]'
aliases:
- GBDT
- Gradient boosting machine
---

# Gradient Boosting (GBDT)

**Định nghĩa:** Mô hình cộng tính $F_M(x)=F_0+\sum_m \eta\,h_m(x)$, mỗi cây $h_m$ khớp với gradient âm (phần dư giả) của hàm mất mát.

**Suy ra từ:** [[Boosting]]

**Nguồn:** [[Friedman 2001]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Gradient descent]] · [[Newton method]] · [[Log-loss]] · [[Loss function]] · [[Boosting]] · [[Pseudo-residual]] · [[Shrinkage]]
**➡ Mở ra:** [[CatBoost]] · [[LightGBM]] · [[XGBoost]]
**🎯 Dùng trong mô hình:** [[XGBoost]] · [[LightGBM]] · [[CatBoost]]
**↳ Nhánh con (cây phân loại):** [[CatBoost]] · [[LightGBM]] · [[Pseudo-residual]] · [[Shrinkage]] · [[XGBoost]]
<!-- tree:auto end -->
