---
type: concept
level: L0
up: '[[Statistics]]'
papers:
- '[[Cox 1958]]'
- '[[Maron 1961]]'
aliases:
- Ước lượng tham số
- Estimation
- Ước lượng
tags:
- shared
---

# Parameter estimation (Ước lượng tham số)

****
**Ước lượng tham số** là việc dùng dữ liệu mẫu (sample) để đoán giá trị của tham số đó. Có hai khái niệm cần phân biệt:
- **Hàm ước lượng (estimator)** là công thức hay quy tắc tính toán, ví dụ “lấy số khách hàng xấu chia cho tổng số khách hàng”.
- **Giá trị ước lượng (estimate)** là con số cụ thể thu được khi áp công thức đó lên một bộ dữ liệu, ví dụ 0,07.

Có hai dạng kết quả:
- **Ước lượng điểm (point estimate)** cho ra một con số duy nhất.
- **Ước lượng khoảng (interval estimate / confidence interval)** cho ra một khoảng tin cậy, ví dụ từ 6,5% đến 7,5% với độ tin cậy 95%.

## Trong [[Cox 1958]]
Bốn cách ước lượng β: phân phối có điều kiện chính xác, khai triển cumulant, [[Maximum likelihood|hợp lý cực đại]] (lặp), logit χ² cực tiểu (không lặp) (§4.3).

## Trong [[Maron 1961]]
Ước lượng bằng **tần suất tương đối** (đếm): $P(C_j)$ = số mục chỉ mục của *j* / tổng; $P(W_i\mid C_j)$ = số lần từ *i* trong *j* / tổng số lượt từ gợi ý trong *j* (§6.3). Maron lưu ý đây chỉ là ước lượng.

## Trong đồ án
LR: tối ưu lặp (lbfgs); GaussianNB: công thức đóng (trung bìn  o lớp).

## Liên quan
[[Maximum likelihood]] · [[Likelihood]] · [[Sample size]] · [[Smoothing]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** —
**➡ Mở ra:** —
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** [[Nuisance parameter]]
<!-- tree:auto end -->
