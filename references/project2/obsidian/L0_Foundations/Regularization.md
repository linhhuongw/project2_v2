---
type: concept
level: L0
up: '[[Optimization]]'
requires:
- '[[Overfitting]]'
question: Phạt độ phức tạp của mô hình để nó không học thuộc dữ liệu thế nào?
glossary:
- '[[XGB_chen_guestrin_2016_glossary#1. Tree ensemble model and regularized objective]]'
- '[[GB_friedman_2001_glossary#5. Regularization]]'
papers:
- '[[Chen Guestrin 2016]]'
aliases:
- Điều chuẩn
---

# Regularization (Điều chuẩn)

**Định nghĩa:** Thêm vào hàm mục tiêu một khoản phạt độ phức tạp của mô hình để giảm overfitting.

**Suy ra từ:** [[Optimization]]

**Trong đồ án:** L2 với C = 1 (LR); λ, γ (XGBoost); `l2_leaf_reg` (CatBoost).

**Nguồn:** [[Chen Guestrin 2016]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Overfitting]]
**➡ Mở ra:** [[Regularized objective]] · [[XGBoost]] · [[LR config]]
**🎯 Dùng trong mô hình:** [[Logistic Regression]] · [[XGBoost]] · [[LightGBM]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
