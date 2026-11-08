---
type: concept
level: L0
up: '[[Shannon entropy]]'
requires:
- '[[Shannon entropy]]'
- '[[Conditional probability]]'
question: Biết thêm một thông tin thì độ bất định giảm bao nhiêu?
glossary:
- '[[NB_maron_1961_glossary#5. Clue word selection (feature selection)]]'
papers:
- '[[Maron 1961]]'
- '[[Breiman 2001]]'
aliases:
- Độ lợi thông tin
---

# Information gain (Độ lợi thông tin)

**Định nghĩa:** Lượng entropy giảm đi sau khi biết một thông tin (một từ, một phép chia nút).

Information Gain (IG) = (entropy trước khi tách) − (entropy sau khi tách, lấy trung bình theo số mẫu mỗi nhánh).
Công thức entropy sau khi tách:  $\sum$ (Tỷ lệ mẫu nhãn * H(nhãn)) với p lấy theo tỷ lệ thiểu số **trong nhánh đó** 
Nhánh được chia theo feature:
- **Đặc trưng dạng số**. Ví dụ: chọn một ngưỡng, hỏi “giá trị ≥ ngưỡng không?”, được 2 nhánh.
- **Đặc trưng dạng phân loại:** chia theo giá trị. Ví dụ “loại thuê bao” có 3 giá trị trả trước / trả sau / doanh nghiệp thì có thể tách thành 3 nhánh, hoặc gộp thành 2 nhóm.

Có rất nhiều cách tách (nhiều đặc trưng, mỗi đặc trưng nhiều ngưỡng). Cây quyết định thử lần lượt:
1. Với mỗi đặc trưng, thử nhiều ngưỡng khác nhau (≥ 1, ≥ 2, ≥ 3…).
2. Mỗi cách tách tạo ra các nhánh, tính H từng nhánh, lấy trung bình có trọng số, ra IG.
3. Chọn cách tách có **IG lớn nhất**.

**Suy ra từ:** [[Shannon entropy]]

**Nguồn:** [[Maron 1961]] · [[Breiman 2001]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Conditional probability]] · [[Shannon entropy]]
**➡ Mở ra:** [[Split criterion]]
**🎯 Dùng trong mô hình:** [[Random Forest]] · [[XGBoost]] · [[LightGBM]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
