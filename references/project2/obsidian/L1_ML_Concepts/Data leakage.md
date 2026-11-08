---
type: concept
level: L1
up: '[[Train-test split]]'
requires:
- '[[Train-test split]]'
question: Thông tin của nhãn lọt vào lúc huấn luyện bằng những đường nào?
glossary:
- '[[CB_prokhorenkova_2018_glossary#2. Categorical features and target statistics]]'
papers:
- '[[Prokhorenkova 2018]]'
aliases:
- Rò rỉ dữ liệu
- Target leakage
---

# Data leakage (Rò rỉ dữ liệu)

**Định nghĩa:** Thông tin từ tập kiểm tra, từ tương lai hoặc từ chính nhãn lọt vào quá trình huấn luyện, làm kết quả đánh giá lạc quan giả tạo.

**Suy ra từ:** [[Train-test split]]

**Trong đồ án:** Tiền xử lý chỉ fit trên train; kiểm soát rò rỉ qua giá trị thiếu ở HMEQ.

**Nguồn:** [[Prokhorenkova 2018]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Train-test split]]
**➡ Mở ra:** [[CatBoost]] · [[Ordered target statistics]] · [[Prediction shift]]
**🎯 Dùng trong mô hình:** [[CatBoost]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
