---
type: concept
level: L1
up: '[[ROC curve]]'
requires:
- '[[ROC curve]]'
- '[[Wilcoxon rank-sum test]]'
question: Mô hình xếp người xấu trên người tốt đúng đến mức nào, không phụ thuộc ngưỡng?
aliases:
- ROC AUC
- Diện tích dưới đường ROC
---

# AUC (Diện tích dưới đường ROC)

**Định nghĩa:** Xác suất mô hình chấm một khách hàng xấu ngẫu nhiên cao hơn một khách hàng tốt ngẫu nhiên; đo khả năng xếp hạng, không phụ thuộc ngưỡng.

Bằng diện tích dưới [[ROC curve]], và bằng thống kê Mann–Whitney ([[Wilcoxon rank-sum test]]) chia cho n_xấu · n_tốt.

**Suy ra từ:** [[ROC curve]]

**Trong đồ án:** độ đo chính — xếp hạng mô hình, kiểm định, learning curves, kiểm tra overfit và rò rỉ.

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Wilcoxon rank-sum test]] · [[ROC curve]]
**➡ Mở ra:** [[Gini coefficient]]
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** [[Gini coefficient]]
<!-- tree:auto end -->
