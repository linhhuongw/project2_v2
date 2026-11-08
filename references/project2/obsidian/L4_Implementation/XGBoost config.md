---
type: config
level: L4
up: '[[XGBoost]]'
requires:
- '[[Histogram-based split]]'
- '[[Level-wise growth]]'
question: Đồ án cấu hình XGBoost thế nào?
glossary:
- '[[XGB_chen_guestrin_2016_glossary#11. Relevance to this project (credit scoring with XGBoost)]]'
aliases:
- Cấu hình XGBoost
---

# XGBoost config (Cấu hình XGBoost)

**Định nghĩa:** 100 cây, độ sâu 6, η = 0.3, λ = 1, γ = 0, `hist`, phát triển theo tầng.

**Suy ra từ:** [[XGBoost]]

**Trong đồ án:** `src/models/train_classic.py`

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Histogram-based split]] · [[Level-wise growth]] · [[XGBoost]]
**➡ Mở ra:** —
**🎯 Dùng trong mô hình:** [[XGBoost]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
