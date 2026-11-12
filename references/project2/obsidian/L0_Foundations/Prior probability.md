---
type: concept
level: L0
up: '[[Bayes theorem]]'
question: Trước khi nhìn dữ liệu của một khách hàng, khả năng vỡ nợ là bao nhiêu?
glossary:
- '[[NB_maron_1961_glossary#6. The prediction method (Naive Bayes)]]'
papers:
- '[[Maron 1961]]'
aliases:
- Xác suất tiên nghiệm
- A priori probability
- P(C_j)
tags:
- maron
---

# Prior probability (Xác suất tiên nghiệm)

$P(C_j)$: xác suất tài liệu thuộc danh mục *j* trước khi xét từ nào; ước lượng bằng tần suất tương đối (§6.3).

Ví dụ: **Bối cảnh.** Một bệnh hiếm, cứ 100 người có 1 người mắc. Có một loại xét nghiệm: người bệnh đi xét nghiệm thì 90% ra dương tính, người khỏe thì vẫn có 9% bị báo dương tính nhầm. An đi xét nghiệm và nhận kết quả **dương tính**. An có thật sự bị bệnh không?

**1. Tiền nghiệm – niềm tin trước khi có bằng chứng.** Trước khi xét nghiệm, điều duy nhất ta biết về An là An thuộc một dân số có tỷ lệ mắc 1%. Vậy **xác suất tiền nghiệm** P(bệnh) = 1%, và P(khỏe) = 99%.

Nguồn: [[Maron 1961]]

## Liên quan
[[Bayes theorem]] · [[Shannon entropy]] · [[Parameter estimation]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** —
**➡ Mở ra:** [[Bayes theorem]] · [[Accuracy]] · [[Average Precision]] · [[Ordered target statistics]]
**🎯 Dùng trong mô hình:** [[Naive Bayes]] · [[KNN]] · [[CatBoost]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
