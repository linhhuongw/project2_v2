---
type: concept
level: L1
up: '[[Empty cells and small counts]]'
requires:
- '[[Empty cells and small counts]]'
question: Tránh xác suất bằng 0 khi một tổ hợp chưa từng xuất hiện thế nào?
glossary:
- '[[NB_maron_1961_glossary#7. Parameter estimation and smoothing]]'
papers:
- '[[Maron 1961]]'
aliases:
- Làm trơn
- Laplace smoothing
- var_smoothing
tags:
- maron
---

# Smoothing (Làm trơn)

Thay 0,001 cho mỗi ô rỗng để tích xác suất không bằng 0 mà không đổi thứ tự (§7.3, đề xuất của Paul Baran). Tiền thân của làm trơn Laplace; trong `GaussianNB` là `var_smoothing`.

Nguồn: [[Maron 1961]]

## Liên quan
[[Empty cells and small counts]] · [[Naive Bayes]] · [[Parameter estimation]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Empty cells and small counts]]
**➡ Mở ra:** [[Naive Bayes]]
**🎯 Dùng trong mô hình:** [[Naive Bayes]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
