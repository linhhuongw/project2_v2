---
type: algorithm
level: L3
up: '[[XGBoost]]'
requires:
- '[[Regularization]]'
- '[[Loss function]]'
question: Hàm mục tiêu của XGBoost phạt số lá và trọng số lá thế nào?
glossary:
- '[[XGB_chen_guestrin_2016_glossary#1. Tree ensemble model and regularized objective]]'
papers:
- '[[Chen Guestrin 2016]]'
aliases:
- Mục tiêu có điều chuẩn
---

# Regularized objective (Mục tiêu có điều chuẩn)

**Định nghĩa:** $\mathrm{Obj}=\sum_i l(y_i,\hat y_i)+\sum_k\big(\gamma T_k+\tfrac12\lambda\lVert w_k\rVert^2\big)$.

**Suy ra từ:** [[XGBoost]]

**Nguồn:** [[Chen Guestrin 2016]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Regularization]] · [[Loss function]]
**➡ Mở ra:** [[Split gain]] · [[XGBoost]]
**🎯 Dùng trong mô hình:** [[XGBoost]] · [[LightGBM]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
