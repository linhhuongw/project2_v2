---
type: concept
level: L1
up: '[[Confusion matrix]]'
requires:
- '[[Confusion matrix]]'
- '[[Conditional probability]]'
question: Trong số người bị gắn nhãn xấu, bao nhiêu người thật sự xấu?
aliases:
- Độ chính xác dương
- PPV
---

# Precision (Độ chính xác dương)

**Định nghĩa:** TP / (TP + FP) = P(y = 1 | ŷ = 1). Precision thấp nghĩa là từ chối nhầm nhiều khách hàng tốt.

**Suy ra từ:** [[Confusion matrix]]

**Trong đồ án:** Tính tại ngưỡng tối đa F1 trên validation (`metrics.py`); LR có precision tốt nhất ở 5/7 bộ.

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Conditional probability]] · [[Confusion matrix]]
**➡ Mở ra:** [[F1 score]] · [[PR curve]]
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
