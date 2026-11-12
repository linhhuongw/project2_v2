---
type: concept
level: L0
up: '[[Bayes theorem]]'
requires:
- '[[Posterior probability]]'
question: Sai số nhỏ nhất mà bất kỳ mô hình nào có thể đạt là bao nhiêu?
glossary:
- '[[KNN_cover_hart_1967_glossary#3. Bayes procedure (the benchmark)]]'
papers:
- '[[Cover Hart 1967]]'
aliases:
- Sai số Bayes
- Bayes risk
---

# Bayes error (Sai số Bayes)

**Định nghĩa:** Sai số nhỏ nhất mà bất kỳ bộ phân loại nào có thể đạt được khi biết phân phối thật; đạt được bằng cách chọn lớp có xác suất hậu nghiệm lớn nhất.
**Bayes error** = trung bình sai số này trên mọi x, có trọng số là P(x):

$$\text{Bayes error} = \sum_x P(x)\,\big(1 - \max_y P(y \mid x)\big)$$

**Suy ra từ:** [[Bayes theorem]]

**Nguồn:** [[Cover Hart 1967]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Posterior probability]]
**➡ Mở ra:** [[Nearest neighbor error bound]]
**🎯 Dùng trong mô hình:** [[KNN]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
