---
type: concept
level: L0
up: '[[Odds]]'
requires:
- '[[Odds]]'
question: Vì sao lấy log của odds để có một thang tuyến tính?
glossary:
- '[[LR_cox_1958_glossary#2. The logistic model]]'
papers:
- '[[Cox 1958]]'
aliases:
- Logit (log-odds)
- Log-odds
tags:
- cox
---

# Logit (Logit (log-odds))

$\operatorname{logit}\theta=\log\{\theta/(1-\theta)\}=\alpha+\beta x$, eq. (1): là logarit tự nhiên của odds - mô hình tuyến tính trên thang logit.
Logit nằm trong (−∞, +∞). Nó đối xứng quanh 0: p = 0.5 cho logit = 0, còn p = 0.8 và p = 0.2 cho logit lần lượt là +1.386 và −1.386.

Nguồn: [[Cox 1958]]

## Liên quan
[[Logistic function]] · [[Odds ratio]] · [[Feature]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Odds]]
**➡ Mở ra:** [[Logistic function]]
**🎯 Dùng trong mô hình:** [[Logistic Regression]] · [[XGBoost]] · [[LightGBM]] · [[CatBoost]]
**↳ Nhánh con (cây phân loại):** [[Logistic function]]
<!-- tree:auto end -->
