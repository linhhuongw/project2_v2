---
type: algorithm
level: L3
up: '[[CatBoost]]'
requires:
- '[[Pseudo-residual]]'
- '[[Data leakage]]'
question: Vì sao tính gradient trên chính dữ liệu đã học lại gây chệch?
glossary:
- '[[CB_prokhorenkova_2018_glossary#3. Prediction shift and ordered boosting]]'
- '[[CB_dorogush_2018_glossary#3. Fighting gradient bias]]'
papers:
- '[[Prokhorenkova 2018]]'
aliases:
- Dịch chuyển dự đoán
---

# Prediction shift (Dịch chuyển dự đoán)

**Định nghĩa:** Gradient của mỗi mẫu được tính bằng mô hình đã học trên chính mẫu đó, nên phân phối $F(x_i)$ trên tập huấn luyện lệch khỏi trên dữ liệu mới — một dạng rò rỉ nhãn.

**Suy ra từ:** [[CatBoost]]

**Nguồn:** [[Prokhorenkova 2018]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Data leakage]] · [[Pseudo-residual]]
**➡ Mở ra:** [[CatBoost]] · [[Ordered boosting]]
**🎯 Dùng trong mô hình:** [[CatBoost]]
**↳ Nhánh con (cây phân loại):** [[Ordered boosting]]
<!-- tree:auto end -->
