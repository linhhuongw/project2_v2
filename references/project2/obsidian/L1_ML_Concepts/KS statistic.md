---
type: concept
level: L1
up: '[[ROC curve]]'
requires:
- '[[ROC curve]]'
question: Phân phối điểm của người xấu và người tốt tách nhau xa nhất bao nhiêu?
aliases:
- Thống kê KS
- Kolmogorov–Smirnov
- KS
---

# KS statistic (Thống kê Kolmogorov–Smirnov)

**Định nghĩa:** KS = max_t (TPR(t) − FPR(t)): khoảng cách lớn nhất giữa hai hàm phân phối tích lũy của điểm số; điểm trên ROC xa đường chéo nhất.

**Suy ra từ:** [[ROC curve]]

**Trong đồ án:** `ks_stat` trong `metrics.py`; có bảng hạng trung bình riêng.

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[ROC curve]]
**➡ Mở ra:** —
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
