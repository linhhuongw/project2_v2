---
type: concept
level: L2
up: '[[Split criterion]]'
requires:
- '[[Split criterion]]'
question: Độ vẩn đục Gini đo mức trộn lẫn các lớp trong một nút thế nào?
glossary:
- '[[RF_breiman_2001_glossary#12. Relevance to this project (credit scoring with RF)]]'
papers:
- '[[Breiman 2001]]'
aliases:
- Độ vẩn đục Gini
---

# Gini impurity (Độ vẩn đục Gini)

**Định nghĩa:** $1-\sum_k p_k^2$: xác suất phân loại sai nếu gán nhãn ngẫu nhiên theo tỷ lệ lớp trong nút.

**Suy ra từ:** [[Split criterion]]

**Trong đồ án:** Tiêu chí mặc định của `RandomForestClassifier`.

**Nguồn:** [[Breiman 2001]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Split criterion]]
**➡ Mở ra:** [[RF config]]
**🎯 Dùng trong mô hình:** [[Random Forest]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
