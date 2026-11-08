---
type: algorithm
level: L3
up: '[[Tree growth strategy]]'
requires:
- '[[Tree growth strategy]]'
question: Cây dùng chung một điều kiện chia cho cả tầng có lợi gì?
glossary:
- '[[CB_dorogush_2018_glossary#4. Fast scorer and oblivious trees]]'
- '[[CB_prokhorenkova_2018_glossary#4. CatBoost implementation]]'
papers:
- '[[Prokhorenkova 2018]]'
- '[[Dorogush 2018]]'
aliases:
- Symmetric tree
- Cây đối xứng
---

# Oblivious tree (Cây đối xứng)

**Định nghĩa:** Mọi nút ở cùng độ sâu dùng chung một điều kiện chia; cây cân bằng, dự đoán nhanh, ít overfit.

**Suy ra từ:** [[Tree growth strategy]]

**Trong đồ án:** Mặc định của CatBoost (`SymmetricTree`).

**Nguồn:** [[Prokhorenkova 2018]] · [[Dorogush 2018]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Tree growth strategy]]
**➡ Mở ra:** [[CatBoost]]
**🎯 Dùng trong mô hình:** [[CatBoost]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
