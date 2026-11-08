---
type: concept
level: L0
up: '[[Logit]]'
requires:
- '[[Logit]]'
question: Đổi một điểm số bất kỳ thành xác suất trong [0, 1] bằng cách nào?
glossary:
- '[[LR_cox_1958_glossary#2. The logistic model]]'
papers:
- '[[Cox 1958]]'
aliases:
- Hàm logistic
- Luật logistic
- Sigmoid
tags:
- cox
---

# Logistic function (Hàm logistic)
Là hàm ngược của logit (log-odds) - hai hàm nghịch đảo

$\theta = e^{\alpha+\beta x}/(1+e^{\alpha+\beta x})$, eq. (2). Luôn nằm trong [0, 1], khác với quan hệ tuyến tính (linear)
Hàm logistic (sigmoid) này ánh xạ ℝ → (0, 1). Đặt z = logit(p) rồi giải ngược ra p:
$$z = \ln\frac{p}{1-p} \;\Rightarrow\; e^{z} = \frac{p}{1-p} \;\Rightarrow\; p = \frac{e^{z}}{1+e^{z}} = \frac{1}{1+e^{-z}} = \sigma(z)$$
**Mô hình hoá xác suất qua sigmoid:** p = σ(β₀ + β₁x₁ + … + βₖxₖ) tương đương với **Mô hình hoá log-odds tuyến tính:** logit(p) = β₀ + β₁x₁ + … + βₖxₖ
--> Hồi quy logistic thực chất là **hồi quy tuyến tính trên thang log-odds**.
**Tóm lại:** logit đưa xác suất lên thang tuyến tính không bị chặn, nơi mô hình tuyến tính hoạt động được. Hàm logistic đưa kết quả ngược về thang xác suất.

Nguồn: [[Cox 1958]]

## Liên quan
[[Logit]] · [[Predicted probability]] · [[Logistic Regression]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Logit]]
**➡ Mở ra:** [[Log-loss]] · [[Linear probabilistic models]] · [[Logistic Regression]]
**🎯 Dùng trong mô hình:** [[Logistic Regression]] · [[XGBoost]] · [[LightGBM]] · [[CatBoost]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
