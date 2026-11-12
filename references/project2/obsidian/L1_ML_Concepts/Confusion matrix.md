---
type: concept
level: L1
up: '[[Model evaluation]]'
requires:
- '[[Binary outcome]]'
- '[[Contingency table]]'
question: Tại một ngưỡng, mô hình đúng và sai theo những kiểu nào?
aliases:
- Ma trận nhầm lẫn
- TP FP FN TN
---

# Confusion matrix (Ma trận nhầm lẫn)

**Định nghĩa:** [[Contingency table|Bảng tiếp liên]] 2×2 giữa nhãn thật y và nhãn dự đoán ŷ = 1{p(x) ≥ t}. Lớp dương là người vay xấu: **TP** bắt đúng người xấu, **FN** cho người xấu vay (mất vốn), **FP** từ chối nhầm người tốt (mất khách), **TN** cho người tốt vay.

**Suy ra từ:** [[Model evaluation]]

**Trong đồ án:** Không xuất ra trực tiếp; precision, recall, accuracy được tính từ nó (`src/models/metrics.py`).

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Contingency table]] · [[Binary outcome]]
**➡ Mở ra:** [[Accuracy]] · [[Precision]] · [[Recall]] · [[Specificity]]
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** [[Accuracy]] · [[F1 score]] · [[Precision]] · [[Recall]] · [[Specificity]]
<!-- tree:auto end -->
