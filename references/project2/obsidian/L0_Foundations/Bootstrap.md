---
type: concept
level: L0
up: '[[Sampling]]'
requires:
- '[[Sampling]]'
question: Lấy mẫu có hoàn lại tạo ra nhiều tập huấn luyện khác nhau thế nào?
glossary:
- '[[RF_breiman_2001_glossary#3. Out-of-bag estimation]]'
papers:
- '[[Breiman 2001]]'
aliases:
- Lấy mẫu bootstrap
---

# Bootstrap (Lấy mẫu bootstrap)

**Định nghĩa:** Lấy mẫu **có hoàn lại** *n* quan sát từ *n*; khoảng 36,8% quan sát không được chọn (out-of-bag).

**Suy ra từ:** [[Sampling]]

**Trong đồ án:** Mỗi cây RF học trên một mẫu bootstrap; khoảng tin cậy AUC của Lending Club dùng 1.000 lần bootstrap.

**Nguồn:** [[Breiman 2001]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Sampling]]
**➡ Mở ra:** [[Bagging]] · [[Out-of-bag error]]
**🎯 Dùng trong mô hình:** [[Random Forest]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
