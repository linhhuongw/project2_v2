---
type: algorithm
level: L3
up: '[[CatBoost]]'
requires:
- '[[Categorical encoding]]'
- '[[Data leakage]]'
- '[[Prior probability]]'
question: Mã hóa biến phân loại bằng trung bình nhãn mà không rò rỉ thế nào?
glossary:
- '[[CB_prokhorenkova_2018_glossary#2. Categorical features and target statistics]]'
papers:
- '[[Prokhorenkova 2018]]'
- '[[Dorogush 2018]]'
aliases:
- Ordered TS
---

# Ordered target statistics (Ordered TS)

**Định nghĩa:** Mã hóa biến phân loại bằng trung bình nhãn của các mẫu cùng hạng mục đứng trước trong hoán vị, cộng một prior.

**Suy ra từ:** [[CatBoost]]

**Trong đồ án:** Không dùng: biến phân loại được truyền dưới dạng mã số nguyên.

**Nguồn:** [[Prokhorenkova 2018]] · [[Dorogush 2018]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Prior probability]] · [[Categorical encoding]] · [[Data leakage]]
**➡ Mở ra:** [[CatBoost]]
**🎯 Dùng trong mô hình:** [[CatBoost]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
