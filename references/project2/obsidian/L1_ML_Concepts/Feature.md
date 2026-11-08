---
type: concept
level: L1
up: '[[Supervised learning]]'
requires:
- '[[Supervised learning]]'
question: Thông tin nào của khách hàng được đưa vào mô hình?
glossary:
- '[[LR_cox_1958_glossary#1. Basic set-up and data]]'
- '[[NB_maron_1961_glossary#2. Basic notions: words, evidence, hypotheses, predictions]]'
papers:
- '[[Cox 1958]]'
- '[[Maron 1961]]'
- '[[Breiman 2001]]'
aliases:
- Đặc trưng
- Biến độc lập
- Biến giải thích
- Từ gợi ý
- Clue word
- Independent variable
tags:
- shared
---

# Feature (Đặc trưng)

Thông tin đo được trên mỗi quan sát, dùng để dự đoán kết quả.

## Trong [[Cox 1958]]
Gọi là **independent variable** $x_i$ (cũng có $u_i$), có thể xác định trước hoặc tính từ lịch sử chuỗi (§2).

## Trong [[Maron 1961]]
Gọi là **clue word** / **property**: các từ gợi ý xuất hiện trong tài liệu; phép đo là danh sách thuộc tính (§2.2).

## Trong đồ án
Các cột của từng tập dữ liệu sau tầng B.0/B.1.

## Liên quan
[[Feature selection]] · [[Independence assumption]] · [[Logit]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Supervised learning]]
**➡ Mở ra:** [[Categorical encoding]] · [[Feature scaling]] · [[Decision tree]] · [[Distance-based models]] · [[Generative models]] · [[Linear probabilistic models]] · [[Preprocessing B0-B1]]
**🎯 Dùng trong mô hình:** [[Logistic Regression]] · [[Naive Bayes]] · [[KNN]] · [[Random Forest]] · [[XGBoost]] · [[LightGBM]] · [[CatBoost]]
**↳ Nhánh con (cây phân loại):** [[Categorical encoding]] · [[Feature scaling]] · [[Feature selection]] · [[Preprocessing B0-B1]]
<!-- tree:auto end -->
