---
type: algorithm
level: L3
up: '[[Tree growth strategy]]'
requires:
- '[[Tree growth strategy]]'
- '[[Split gain]]'
question: Luôn chia lá có Gain lớn nhất thì cây khác gì?
glossary:
- '[[LGBM_ke_2017_glossary#5. Tree growth and experiments]]'
papers:
- '[[Ke 2017]]'
aliases:
- Best-first
- Phát triển theo lá
---

# Leaf-wise growth (Phát triển theo lá)

**Định nghĩa:** Luôn chia lá có Gain lớn nhất; cây sâu không đều, dễ overfit khi dữ liệu nhỏ.

**Suy ra từ:** [[Tree growth strategy]]

**Trong đồ án:** Mặc định của LightGBM (giới hạn bằng `num_leaves`).

**Nguồn:** [[Ke 2017]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Tree growth strategy]] · [[Split gain]]
**➡ Mở ra:** [[LightGBM]]
**🎯 Dùng trong mô hình:** [[LightGBM]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
