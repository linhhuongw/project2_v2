---
type: config
level: L4
up: '[[Random Forest]]'
requires:
- '[[Gini impurity]]'
question: Đồ án cấu hình Random Forest thế nào?
glossary:
- '[[RF_breiman_2001_glossary#12. Relevance to this project (credit scoring with RF)]]'
aliases:
- Cấu hình RF
---

# RF config (Cấu hình RF)

**Định nghĩa:** `RandomForestClassifier()`: 100 cây phát triển đầy đủ, `max_features = sqrt`, tiêu chí Gini.

**Suy ra từ:** [[Random Forest]]

**Trong đồ án:** `src/models/train_classic.py`

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Gini impurity]] · [[Random Forest]]
**➡ Mở ra:** —
**🎯 Dùng trong mô hình:** [[Random Forest]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
