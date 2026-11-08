---
type: concept
level: L1
up: '[[Log-loss]]'
requires:
- '[[Log-loss]]'
- '[[Newton method]]'
question: Đạo hàm bậc một g = p − y và bậc hai h = p(1 − p) của log-loss có ý nghĩa gì?
glossary:
- '[[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting]]'
- '[[GB_friedman_2001_glossary#4. Regression trees as base learners (TreeBoost)]]'
papers:
- '[[Friedman 2001]]'
- '[[Chen Guestrin 2016]]'
aliases:
- Gradient và Hessian của log-loss
- g = p − y
- h = p(1 − p)
---

# Log-loss gradient and Hessian (Gradient và Hessian của log-loss)

**Định nghĩa:** Theo điểm $F$ (log-odds): $g=\partial L/\partial F=p-y$, $h=\partial^2L/\partial F^2=p(1-p)$.

**Suy ra từ:** [[Log-loss]]

**Nguồn:** [[Friedman 2001]] · [[Chen Guestrin 2016]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Newton method]] · [[Log-loss]]
**➡ Mở ra:** [[Split gain]]
**🎯 Dùng trong mô hình:** [[XGBoost]] · [[LightGBM]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
