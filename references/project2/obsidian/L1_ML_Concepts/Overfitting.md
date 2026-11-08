---
type: concept
level: L1
up: '[[Generalization]]'
requires:
- '[[Generalization]]'
question: Vì sao mô hình tốt trên tập huấn luyện lại kém trên dữ liệu mới?
glossary:
- '[[GB_friedman_2001_glossary#5. Regularization]]'
- '[[NB_maron_1961_glossary#3. Corpus and data split]]'
papers:
- '[[Maron 1961]]'
- '[[Breiman 2001]]'
- '[[Ke 2017]]'
- '[[Prokhorenkova 2018]]'
aliases:
- Quá khớp
---

# Overfitting (Quá khớp)

**Định nghĩa:** Mô hình khớp quá sát dữ liệu huấn luyện (kể cả nhiễu) nên kém trên dữ liệu mới. Maron: 91% trên tập huấn luyện so với 51,8% trên tập kiểm tra.

**Suy ra từ:** [[Generalization]]

**Nguồn:** [[Maron 1961]] · [[Breiman 2001]] · [[Ke 2017]] · [[Prokhorenkova 2018]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Generalization]]
**➡ Mở ra:** [[Regularization]] · [[Bias-variance tradeoff]] · [[Shrinkage]]
**🎯 Dùng trong mô hình:** [[Logistic Regression]] · [[Random Forest]] · [[XGBoost]] · [[LightGBM]] · [[CatBoost]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
