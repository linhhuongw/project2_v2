---
type: concept
level: L1
up: '[[Generalization]]'
requires:
- '[[Bias-variance tradeoff]]'
- '[[Train-test split]]'
question: Thêm dữ liệu huấn luyện có giúp mô hình tốt hơn không?
aliases:
- Đường học
- Learning curves
---

# Learning curve (Đường học)

**Định nghĩa:** Hiệu năng theo tỷ lệ dữ liệu train. Đường validation còn đi lên: mô hình bị giới hạn bởi dữ liệu; nằm ngang sớm: giới hạn bởi chính mô hình (bias). Khoảng cách train − validation lớn: overfit.

**Suy ra từ:** [[Generalization]]

**Trong đồ án:** `src/models/learning_curves.py` (mục 4.6, Hình 4.8).

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Bias-variance tradeoff]] · [[Train-test split]]
**➡ Mở ra:** —
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
