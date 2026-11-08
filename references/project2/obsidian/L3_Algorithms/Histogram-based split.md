---
type: algorithm
level: L3
up: '[[LightGBM]]'
requires:
- '[[Split criterion]]'
question: Gom giá trị vào các bin để tìm điểm chia nhanh hơn thế nào?
glossary:
- '[[LGBM_ke_2017_glossary#1. GBDT and its cost]]'
- '[[XGB_chen_guestrin_2016_glossary#4. Split finding algorithms]]'
papers:
- '[[Ke 2017]]'
- '[[Chen Guestrin 2016]]'
aliases:
- Histogram
- hist
---

# Histogram-based split (Histogram)

**Định nghĩa:** Gom giá trị liên tục vào các bin rồi tìm ngưỡng trên histogram, nhanh hơn duyệt mọi giá trị.

**Suy ra từ:** [[LightGBM]]

**Trong đồ án:** LightGBM và XGBoost (`tree_method = hist`).

**Nguồn:** [[Ke 2017]] · [[Chen Guestrin 2016]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Split criterion]]
**➡ Mở ra:** [[EFB]] · [[LightGBM]] · [[XGBoost config]]
**🎯 Dùng trong mô hình:** [[XGBoost]] · [[LightGBM]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
