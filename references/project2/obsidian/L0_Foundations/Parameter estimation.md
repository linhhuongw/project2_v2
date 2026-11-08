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

Suy ra giá trị tham số mô hình từ dữ liệu.

## Trong [[Cox 1958]]
Bốn cách ước lượng β: phân phối có điều kiện chính xác, khai triển cumulant, [[Maximum likelihood|hợp lý cực đại]] (lặp), logit χ² cực tiểu (không lặp) (§4.3).

## Trong [[Maron 1961]]
Ước lượng bằng **tần suất tương đối** (đếm): $P(C_j)$ = số mục chỉ mục của *j* / tổng; $P(W_i\mid C_j)$ = số lần từ *i* trong *j* / tổng số lượt từ gợi ý trong *j* (§6.3). Maron lưu ý đây chỉ là ước lượng.

## Trong đồ án
LR: tối ưu lặp (lbfgs); GaussianNB: công thức đóng (trung bình, phương sai theo lớp).

## Liên quan
[[Maximum likelihood]] · [[Likelihood]] · [[Sample size]] · [[Smoothing]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** —
**➡ Mở ra:** —
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** [[Nuisance parameter]]
<!-- tree:auto end -->
