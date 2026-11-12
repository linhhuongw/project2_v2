---
type: concept
level: L1
up: '[[Model evaluation]]'
requires:
- '[[Precision]]'
- '[[Recall]]'
- '[[Predicted probability]]'
question: Khi lớp xấu rất hiếm, mô hình đánh đổi precision và recall thế nào?
aliases:
- Đường precision–recall
- Precision-recall curve
---

# PR curve (Đường precision–recall)

**Định nghĩa:** Đường (Recall, Precision) khi quét ngưỡng. Tập trung vào lớp thiểu số nên nhạy hơn ROC khi tỷ lệ xấu thấp.

**Suy ra từ:** [[Model evaluation]]

**Trong đồ án:** Vẽ trong `compare_models.py`, kèm điểm hoạt động tại ngưỡng F1.

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Precision]] · [[Predicted probability]] · [[Recall]]
**➡ Mở ra:** [[Average Precision]]
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** [[Average Precision]]
<!-- tree:auto end -->
