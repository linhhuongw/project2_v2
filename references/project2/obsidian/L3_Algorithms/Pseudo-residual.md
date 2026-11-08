---
type: algorithm
level: L3
up: '[[Gradient Boosting]]'
requires:
- '[[Gradient descent]]'
- '[[Loss function]]'
question: Phần dư giả (gradient âm) là gì và vì sao mỗi cây học nó?
glossary:
- '[[GB_friedman_2001_glossary#2. Optimization in function space]]'
papers:
- '[[Friedman 2001]]'
aliases:
- Phần dư giả
---

# Pseudo-residual (Phần dư giả)

**Định nghĩa:** $r_i=-\partial L/\partial F(x_i)$; với log-loss, $r_i=y_i-p_i$.

**Suy ra từ:** [[Gradient Boosting]]

**Nguồn:** [[Friedman 2001]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Gradient descent]] · [[Loss function]]
**➡ Mở ra:** [[Gradient Boosting]] · [[Prediction shift]]
**🎯 Dùng trong mô hình:** [[XGBoost]] · [[LightGBM]] · [[CatBoost]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
