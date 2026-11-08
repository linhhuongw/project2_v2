---
type: algorithm
level: L3
up: '[[LightGBM]]'
requires:
- '[[Sampling]]'
- '[[Split gain]]'
question: Bớt số dòng mà Gain vẫn gần đúng bằng cách nào?
glossary:
- '[[LGBM_ke_2017_glossary#3. Gradient-based One-Side Sampling (GOSS)]]'
papers:
- '[[Ke 2017]]'
aliases:
- Gradient-based One-Side Sampling
---

# GOSS (Gradient-based One-Side Sampling)

**Định nghĩa:** Giữ mọi mẫu có |gradient| lớn, lấy mẫu ngẫu nhiên một phần mẫu gradient nhỏ và nhân trọng số bù.

**Suy ra từ:** [[LightGBM]]

**Trong đồ án:** Không dùng: `boosting_type = gbdt`.

**Nguồn:** [[Ke 2017]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Sampling]] · [[Split gain]]
**➡ Mở ra:** [[LightGBM]]
**🎯 Dùng trong mô hình:** [[LightGBM]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
