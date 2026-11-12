---
type: concept
level: L1
up: '[[Binary outcome]]'
requires:
- '[[Binary outcome]]'
question: Vì sao mô hình nên trả ra xác suất chứ không chỉ một nhãn?
glossary:
- '[[LR_cox_1958_glossary#2. The logistic model]]'
- '[[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)]]'
papers:
- '[[Cox 1958]]'
- '[[Maron 1961]]'
aliases:
- Xác suất dự đoán
- Attribute number
- Số thuộc tính
- θ
tags:
- shared
---

# Predicted probability (Xác suất dự đoán)

Đầu ra của mô hình là một xác suất cho mỗi lớp, không chỉ một nhãn.

## Trong [[Cox 1958]]
$\theta_i = \Pr(Y_i=1)$ cho bởi [[Logistic function|hàm logistic]] (§2).

## Trong [[Maron 1961]]
**Attribute number** $P(C_j\mid W_k\cdots W_r)$, chuẩn hóa tổng bằng 1, eq. (8); Maron chỉ đánh giá thứ hạng, không đánh giá độ chính xác của xác suất.

## Trong đồ án
Dùng cho AUC (thứ hạng) và Brier score / hiệu chỉnh xác suất (độ chính xác của xác suất).

## Liên quan
[[Posterior probability]] · [[Logistic function]] · [[Binary outcome]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Binary outcome]]
**➡ Mở ra:** [[PR curve]] · [[ROC curve]] · [[Threshold selection]]
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** [[Probability calibration]] · [[Threshold selection]]
<!-- tree:auto end -->
