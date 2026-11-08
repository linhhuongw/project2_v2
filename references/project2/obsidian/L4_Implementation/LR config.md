---
type: config
level: L4
up: '[[Logistic Regression]]'
requires:
- '[[Regularization]]'
- '[[Feature scaling]]'
question: Đồ án cấu hình Logistic Regression thế nào?
glossary:
- '[[LR_cox_1958_glossary#14. Relevance to this project (credit scoring with LR)]]'
aliases:
- Cấu hình LR
---

# LR config (Cấu hình LR)

**Định nghĩa:** `LogisticRegression(max_iter=2000)`: phạt L2, C = 1.0, solver lbfgs; dữ liệu one-hot + chuẩn hóa.

**Suy ra từ:** [[Logistic Regression]]

**Trong đồ án:** `src/models/train_classic.py`

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Regularization]] · [[Feature scaling]] · [[Logistic Regression]]
**➡ Mở ra:** —
**🎯 Dùng trong mô hình:** [[Logistic Regression]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
