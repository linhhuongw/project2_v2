---
type: concept
level: L1
up: '[[PR curve]]'
requires:
- '[[PR curve]]'
- '[[Prior probability]]'
question: Tóm tắt đường precision–recall bằng một con số thế nào?
aliases:
- AP
- Độ chính xác trung bình
- PR-AUC
---

# Average Precision (Độ chính xác trung bình)

**Định nghĩa:** Tổng precision có trọng số theo mức tăng recall. **Mức cơ sở bằng tỷ lệ xấu** (không phải 0,5), nên không so sánh AP giữa các bộ dữ liệu có tỷ lệ xấu khác nhau.

**Suy ra từ:** [[PR curve]]

**Trong đồ án:** `average_precision_score` trong `metrics.py`; có bảng hạng trung bình riêng.

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Prior probability]] · [[PR curve]]
**➡ Mở ra:** —
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
