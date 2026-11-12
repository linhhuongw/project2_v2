---
type: concept
level: L1
up: '[[Model evaluation]]'
requires:
- '[[Recall]]'
- '[[Specificity]]'
- '[[Predicted probability]]'
question: Khi quét mọi ngưỡng, mô hình đánh đổi giữa bắt người xấu và từ chối nhầm người tốt thế nào?
aliases:
- Đường ROC
- Receiver operating characteristic
---

# ROC curve (Đường ROC)

**Định nghĩa:** Đường (FPR, TPR) khi ngưỡng t chạy từ 1 về 0. Đường chéo là đoán ngẫu nhiên; càng sát góc trên trái càng tốt.

**Suy ra từ:** [[Model evaluation]]

**Trong đồ án:** Vẽ trong `report_classic.py` (Hình 4.4); là cơ sở của [[AUC]] và [[KS statistic]].

<!-- tree:auto start -->
**⬅ Cần biết trước:** [[Predicted probability]] · [[Recall]] · [[Specificity]]
**➡ Mở ra:** [[AUC]] · [[KS statistic]]
**🎯 Dùng trong mô hình:** —
**↳ Nhánh con (cây phân loại):** [[AUC]] · [[KS statistic]]
<!-- tree:auto end -->
