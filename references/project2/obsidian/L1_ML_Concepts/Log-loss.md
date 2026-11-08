---
type: concept
level: L1
up: '[[Likelihood]]'
requires:
- '[[Likelihood]]'
- '[[Loss function]]'
- '[[Logistic function]]'
question: Hàm mất mát nào dùng cho bài toán xác suất 0/1, và vì sao nó chính là âm log hợp lý?
glossary:
- '[[GB_friedman_2001_glossary#3. Loss functions and specific algorithms]]'
papers:
- '[[Cox 1958]]'
- '[[Friedman 2001]]'
- '[[Chen Guestrin 2016]]'
aliases:
- Cross-entropy
- Hàm mất mát logistic
---

# Log-loss (Hàm mất mát logistic)

**Định nghĩa:** $-[y\log p+(1-y)\log(1-p)]$: bằng âm log hợp lý Bernoulli, nên cực tiểu log-loss = cực đại hợp lý.

**Suy ra từ:** [[Likelihood]]

**Trong đồ án:** Hàm mục tiêu của LR và cả ba thư viện GBDT.

**Nguồn:** [[Cox 1958]] · [[Friedman 2001]] · [[Chen Guestrin 2016]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Likelihood]] · [[Logistic function]] · [[Loss function]]
**➡ Mở ra:** [[Log-loss gradient and Hessian]] · [[Gradient Boosting]] · [[Logistic Regression]]
**🎯 Dùng trong mô hình:** [[Logistic Regression]] · [[XGBoost]] · [[LightGBM]] · [[CatBoost]]
**↳ Nhánh con (cây phân loại):** [[Log-loss gradient and Hessian]]
<!-- tree:auto end -->
