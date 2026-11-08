---
type: algorithm
level: L3
up: '[[Prediction shift]]'
requires:
- '[[Prediction shift]]'
question: Tính phần dư bằng mô hình chưa thấy mẫu đó thế nào?
glossary:
- '[[CB_prokhorenkova_2018_glossary#3. Prediction shift and ordered boosting]]'
- '[[CB_dorogush_2018_glossary#3. Fighting gradient bias]]'
papers:
- '[[Prokhorenkova 2018]]'
---

# Ordered boosting

**Định nghĩa:** Theo một hoán vị ngẫu nhiên, phần dư của mẫu *i* được tính bằng mô hình chỉ học trên các mẫu đứng trước.

**Suy ra từ:** [[Prediction shift]]

**Trong đồ án:** Không dùng: `boosting_type = Plain`.

**Nguồn:** [[Prokhorenkova 2018]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Prediction shift]]
**➡ Mở ra:** [[CatBoost]]
**🎯 Dùng trong mô hình:** [[CatBoost]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
