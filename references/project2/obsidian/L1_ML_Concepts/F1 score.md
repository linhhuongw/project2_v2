---
type: concept
level: L1
up: '[[Confusion matrix]]'
requires:
- '[[Precision]]'
- '[[Recall]]'
question: Gộp precision và recall thành một con số thế nào?
aliases:
- Điểm F1
- F1
---

# F1 score (Điểm F1)

**Định nghĩa:** 2 · Precision · Recall / (Precision + Recall): trung bình điều hòa, chỉ cao khi cả hai cùng cao.

**Suy ra từ:** [[Confusion matrix]]

**Trong đồ án:** Ngưỡng quyết định được chọn để **tối đa F1 trên validation** rồi áp sang test (`f1_threshold` trong `metrics.py`; [[Threshold selection]]).

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Precision]] · [[Recall]]
**➡ Mở ra:** —
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
