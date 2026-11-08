---
type: algorithm
level: L3
up: '[[XGBoost]]'
requires:
- '[[Regularized objective]]'
- '[[Log-loss gradient and Hessian]]'
- '[[Split criterion]]'
question: Trọng số lá w* = −G/(H+λ) và Gain của một phép chia được tính thế nào?
glossary:
- '[[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting]]'
papers:
- '[[Chen Guestrin 2016]]'
aliases:
- Gain
- Optimal leaf weight
---

# Split gain (Gain)

**Định nghĩa:** Trọng số lá tối ưu $w^*=-G/(H+\lambda)$; $\mathrm{Gain}=\tfrac12\big[\tfrac{G_L^2}{H_L+\lambda}+\tfrac{G_R^2}{H_R+\lambda}-\tfrac{(G_L+G_R)^2}{H_L+H_R+\lambda}\big]-\gamma$.

**Suy ra từ:** [[XGBoost]]

**Nguồn:** [[Chen Guestrin 2016]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Log-loss gradient and Hessian]] · [[Split criterion]] · [[Regularized objective]]
**➡ Mở ra:** [[GOSS]] · [[Leaf-wise growth]] · [[LightGBM]] · [[Sparsity-aware split finding]] · [[Weighted quantile sketch]] · [[XGBoost]]
**🎯 Dùng trong mô hình:** [[XGBoost]] · [[LightGBM]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
