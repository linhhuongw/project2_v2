---
type: concept
level: L0
up: '[[Conditional probability]]'
requires:
- '[[Conditional probability]]'
question: Khi nào được nhân các xác suất của từng đặc trưng với nhau?
glossary:
- '[[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)]]'
papers:
- '[[Cox 1958]]'
- '[[Maron 1961]]'
aliases:
- Giả định độc lập
- Độc lập lẫn nhau
- Giả định độc lập có điều kiện
- Conditional independence
tags:
- shared
---
	
# Independence assumption (Giả định độc lập)

Giả định rằng một số biến ngẫu nhiên không ảnh hưởng đến nhau, để hàm xác suất chung tách thành tích.

## Trong [[Cox 1958]]
Các **phép thử** $Y_i$ độc lập lẫn nhau (§2–5); bỏ giả định này khi xét phụ thuộc vào kết quả trước (§6–7). Nhờ đó hàm hợp lý là tích theo *i*.

## Trong [[Maron 1961]]
Các **từ gợi ý** độc lập khi biết danh mục, eq. (5); Maron thừa nhận giả định này sai, eq. (6), nhưng dùng để đơn giản tính toán.

## Trong đồ án
Cả LR và NB giả định các khoản vay độc lập; chỉ NB giả định thêm các đặc trưng độc lập theo lớp — không đúng với dữ liệu tín dụng có đặc trưng tương quan mạnh.

## Liên quan
[[Simplifying assumption]] · [[Naive Bayes formula]] · [[Likelihood]] · [[Feature]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Conditional probability]]
**➡ Mở ra:** [[Naive Bayes]] · [[Naive Bayes formula]]
**🎯 Dùng trong mô hình:** [[Naive Bayes]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
