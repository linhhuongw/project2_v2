---
type: concept
level: L0
up: '[[Sampling]]'
requires:
- '[[Sampling]]'
question: Lấy mẫu có hoàn lại tạo ra nhiều tập huấn luyện khác nhau thế nào?
glossary:
- '[[RF_breiman_2001_glossary#3. Out-of-bag estimation]]'
papers:
- '[[Breiman 2001]]'
aliases:
- Lấy mẫu bootstrap
---

# Bootstrap (Lấy mẫu bootstrap)

**Định nghĩa:** Lấy mẫu **có hoàn lại** *n* quan sát từ *n*; khoảng 36,8% quan sát không được chọn (out-of-bag).

Vì có hoàn lại nên một phần tử có thể xuất hiện nhiều lần hoặc không xuất hiện lần nào. Xác suất một quan sát **không** được chọn là
							$(1 − 1/n)ⁿ$≈ $e⁻¹$≈ **36,8%**. 
							với n là số observation

Nghĩa là mỗi mẫu bootstrap chỉ chứa khoảng **63,2% số quan sát khác nhau** của tập gốc. Khoảng 36,8% còn lại gọi là các quan sát **ngoài túi** (out-of-bag). Có thể dùng chúng làm tập kiểm tra miễn phí.

**Ý tưởng cốt lõi:** ta không thể lấy thêm dữ liệu từ tổng thể thật. Vì vậy ta coi tập mẫu là “tổng thể thu nhỏ” và lấy mẫu lại từ nó nhiều lần (thường 1.000 đến 10.000 lần). Cách này cho thấy một thống kê (trung bình, hệ số, chỉ số đánh giá mô hình…) **dao động bao nhiêu**, mà không cần giả định phân phối.

**Suy ra từ:** [[Sampling]]

**Trong đồ án:** Mỗi cây RF học trên một mẫu bootstrap; khoảng tin cậy AUC của Lending Club dùng 1.000 lần bootstrap.

**Nguồn:** [[Breiman 2001]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Sampling]]
**➡ Mở ra:** [[Bagging]] · [[Out-of-bag error]]
**🎯 Dùng trong mô hình:** [[Random Forest]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
