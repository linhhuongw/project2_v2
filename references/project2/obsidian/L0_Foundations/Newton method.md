---
type: concept
level: L0
up: '[[Optimization]]'
requires:
- '[[Gradient descent]]'
question: Dùng thêm đạo hàm bậc hai để chọn bước đi tốt hơn thế nào?
glossary:
- '[[XGB_chen_guestrin_2016_glossary#2. Second-order gradient boosting]]'
papers:
- '[[Chen Guestrin 2016]]'
aliases:
- Phương pháp Newton
- Second-order
---

# Newton method (Phương pháp Newton)

**Định nghĩa:** Dùng cả đạo hàm bậc một (gradient) và bậc hai (Hessian) để chọn bước cập nhật.

**Suy ra từ:** [[Optimization]]

**Trong đồ án:** LR dùng lbfgs (tựa Newton); XGBoost xấp xỉ bậc hai hàm mất mát.

**Nguồn:** [[Chen Guestrin 2016]]

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Gradient descent]]
**➡ Mở ra:** [[Maximum likelihood]] · [[Log-loss gradient and Hessian]] · [[Gradient Boosting]]
**🎯 Dùng trong mô hình:** [[Logistic Regression]] · [[XGBoost]] · [[LightGBM]] · [[CatBoost]]
**↳ Nhánh con (cây phân loại):** —
<!-- tree:auto end -->
