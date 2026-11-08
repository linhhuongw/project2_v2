---
type: concept
level: L0
up: '[[Statistics]]'
requires:
- '[[Bernoulli distribution]]'
question: Mô hình giải thích dữ liệu quan sát được tốt đến mức nào?
glossary:
- '[[LR_cox_1958_glossary#3. Likelihood, sufficiency and conditioning]]'
papers:
- '[[Cox 1958]]'
- '[[Maron 1961]]'
- '[[Friedman 2001]]'
aliases:
- Hàm hợp lý
- Xác suất có điều kiện (hợp lý)
tags:
- shared
---

# Likelihood (Hàm hợp lý)

Xác suất của dữ liệu quan sát được khi biết tham số (hoặc khi biết lớp).

## Trong [[Cox 1958]]
Hàm hợp lý của toàn bộ chuỗi 0/1 theo (α, β), eq. (23): $e^{\alpha Y+\beta X}/\prod(1+e^{\alpha+\beta x_i})$. Được cực đại hóa để ước lượng tham số (§4.1).

## Trong [[Maron 1961]]
$P(W_i\mid C_j)$: xác suất tài liệu thuộc danh mục *j* chứa từ $W_i$ (§6.1). Là một thừa số trong công thức Bayes.

## Trong đồ án
LR cực tiểu log-loss = cực đại hợp lý; NB dùng $P(x_j\mid y)$ dạng Gauss.

## Liên quan
[[Maximum likelihood]] · [[Conditional probability]] · [[Bayes theorem]] · [[Parameter estimation]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Bernoulli distribution]]
**➡ Mở ra:** [[Maximum likelihood]] · [[Log-loss]]
**🎯 Dùng trong mô hình:** [[Logistic Regression]] · [[XGBoost]] · [[LightGBM]] · [[CatBoost]]
**↳ Nhánh con (cây phân loại):** [[Maximum likelihood]] · [[Sufficient statistic]] · [[Log-loss]]
<!-- tree:auto end -->
