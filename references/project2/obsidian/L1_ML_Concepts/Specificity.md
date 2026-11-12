---
type: concept
level: L1
up: '[[Confusion matrix]]'
requires:
- '[[Confusion matrix]]'
- '[[Conditional probability]]'
question: Trong số người tốt, mô hình giữ lại được bao nhiêu?
aliases:
- Độ đặc hiệu
- TNR
- FPR
- False positive rate
---

# Specificity (Độ đặc hiệu)

**Định nghĩa:** TN / (TN + FP) = P(ŷ = 0 | y = 0). **FPR** = 1 − Specificity là tỷ lệ người tốt bị từ chối nhầm, trục hoành của [[ROC curve]].

**Suy ra từ:** [[Confusion matrix]]

**Trong đồ án:** Không báo cáo riêng; FPR chỉ dùng bên trong ROC và KS.

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Conditional probability]] · [[Confusion matrix]]
**➡ Mở ra:** [[ROC curve]]
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
