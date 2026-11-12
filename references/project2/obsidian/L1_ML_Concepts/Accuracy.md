---
type: concept
level: L1
up: '[[Confusion matrix]]'
requires:
- '[[Confusion matrix]]'
- '[[Prior probability]]'
question: Vì sao tỷ lệ dự đoán đúng gây hiểu lầm khi nhãn mất cân bằng?
aliases:
- Độ chính xác tổng thể
---

# Accuracy (Độ chính xác tổng thể)

**Định nghĩa:** (TP + TN) / N. Khi tỷ lệ xấu thấp (ví dụ 7%), đoán "tốt" cho mọi người đã đạt 93%, nên accuracy gần như chỉ phản ánh [[Prior probability|tỷ lệ tiên nghiệm]].

**Suy ra từ:** [[Confusion matrix]]

**Trong đồ án:** Có trong bảng kết quả nhưng **không** dùng để xếp hạng hay kiểm định (`compare_models.py`, `HIGHER_IS_BETTER`).

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Prior probability]] · [[Confusion matrix]]
**➡ Mở ra:** —
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
