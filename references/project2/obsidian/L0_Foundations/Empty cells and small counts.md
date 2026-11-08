---
type: concept
level: L0
up: '[[Contingency table]]'
requires:
- '[[Contingency table]]'
question: Ước lượng hỏng ra sao khi một ô có rất ít hoặc không có mẫu?
glossary:
- '[[NB_maron_1961_glossary#7. Parameter estimation and smoothing]]'
papers:
- '[[Cox 1958]]'
- '[[Maron 1961]]'
aliases:
- Ô rỗng và tần số nhỏ
- Zero-probability problem
- Vấn đề xác suất bằng 0
- Ô rỗng
tags:
- shared
---

# Empty cells and small counts (Ô rỗng và tần số nhỏ)

Khi một tổ hợp (lớp, giá trị) có rất ít hoặc không có quan sát, ước lượng trở nên không ổn định hoặc suy biến.

## Trong [[Cox 1958]]
Xấp xỉ chuẩn cho giá trị vô lý khi một ô của bảng 2×2 nhỏ (Ví dụ 1, §4.2); logit χ² cực tiểu cần gộp nhóm để ít ô toàn 0 hoặc toàn 1 (§7).

## Trong [[Maron 1961]]
Ô rỗng trong ma trận từ–danh mục làm tích xác suất bằng 0, eq. (9); khắc phục bằng cách thay 0,001 (§7.3).

## Trong đồ án
Hạng mục hiếm của biến phân loại; tầng B.0 loại cột thưa.

## Liên quan
[[Smoothing]] · [[Contingency table]] · [[Sample size]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Contingency table]]
**➡ Mở ra:** [[Smoothing]]
**🎯 Dùng trong mô hình:** [[Naive Bayes]]
**↳ Nhánh con (cây phân loại):** [[Smoothing]]
<!-- tree:auto end -->
