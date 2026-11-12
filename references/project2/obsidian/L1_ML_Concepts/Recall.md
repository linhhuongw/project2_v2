---
type: concept
level: L1
up: '[[Confusion matrix]]'
requires:
- '[[Confusion matrix]]'
- '[[Conditional probability]]'
question: Trong số người xấu thật, mô hình bắt được bao nhiêu?
aliases:
- Độ nhạy
- TPR
- Sensitivity
- True positive rate
---

# Recall (Độ nhạy)

**Định nghĩa:** TP / (TP + FN) = P(ŷ = 1 | y = 1). Recall thấp nghĩa là nhiều khoản vay xấu lọt qua. Là trục tung của [[ROC curve]].

**Suy ra từ:** [[Confusion matrix]]

**Trong đồ án:** Tính tại ngưỡng tối đa F1 trên validation (`metrics.py`).

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Conditional probability]] · [[Confusion matrix]]
**➡ Mở ra:** [[F1 score]] · [[PR curve]] · [[ROC curve]]
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
