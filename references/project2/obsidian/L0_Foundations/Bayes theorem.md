---
type: concept
level: L0
up: '[[Conditional probability]]'
requires:
- '[[Conditional probability]]'
- '[[Prior probability]]'
question: Từ P(đặc trưng | lớp) suy ngược ra P(lớp | đặc trưng) thế nào?
glossary:
- '[[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)]]'
papers:
- '[[Maron 1961]]'
- '[[Cover Hart 1967]]'
aliases:
- Định lý Bayes
- Bayes' rule
tags:
- maron
---

# Bayes theorem (Định lý Bayes)

$P(A\mid B)=P(B) * P(B\mid A)/P(A)$, eq. (1): suy luận ngược từ bằng chứng đến danh mục.

Trong đó:
$P(A\mid B)$ là xác suât hậu nghiệm: niềm tin sau khi thấy bằng chứng
$P(B)$ là xác suất tiền nghiệm: trước khi có gì xảy ra, xác suất này luôn tồn tại (VD: trong 100 trẻ em thì có 1 trẻ bị suy dinh dưỡng --> xác suất là 1/100)
$P(B\mid A)$ là hàm hợp lý (likelihood) – bằng chứng “khớp” với từng giả thuyết đến đâu.
$P(A)$ là bằng chứng: xác suất thấy kết quả đó nói chung

Nguồn: [[Maron 1961]]

## Liên quan
[[Prior probability]] · [[Posterior probability]] · [[Likelihood]] · [[Naive Bayes formula]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Conditional probability]] · [[Prior probability]]
**➡ Mở ra:** [[Posterior probability]] · [[Generative models]] · [[Naive Bayes]] · [[Naive Bayes formula]]
**🎯 Dùng trong mô hình:** [[Naive Bayes]] · [[KNN]]
**↳ Nhánh con (cây phân loại):** [[Bayes error]] · [[Posterior probability]] · [[Prior probability]]
<!-- tree:auto end -->
