# 6. Kết quả

Mục 6.1–6.6 phân tích mô hình cơ sở Logistic Regression; mục 6.7 so sánh bốn mô hình classic ML (LR, KNN, RF, NB), tất cả với tham số mặc định. Số liệu gốc nằm trong `reports/models/lr_default/` (`summary.csv`, `fold_metrics.csv`, `run.json`).

## 6.1. Kết quả chính

![AUC trên test theo bộ dữ liệu](../../reports/models/lr_default/figures/auc_by_dataset.png)

| Bộ dữ liệu | Đánh giá | AUC | Gini | KS | Brier | LogLoss | AP |
|---|---|---|---|---|---|---|---|
| HMEQ | 5-fold CV | **0,8030 ± 0,0161** | 0,6061 ± 0,0322 | 0,4655 ± 0,0394 | 0,1196 ± 0,0071 | 0,3884 ± 0,0184 | 0,5919 ± 0,0182 |
| FICO HELOC | 5-fold CV | **0,8019 ± 0,0098** | 0,6038 ± 0,0196 | 0,4736 ± 0,0194 | 0,1813 ± 0,0044 | 0,5442 ± 0,0101 | 0,7947 ± 0,0043 |
| South German | 5-fold CV | 0,7663 ± 0,0281 | 0,5327 ± 0,0562 | 0,4340 ± 0,0305 | 0,1716 ± 0,0131 | 0,5196 ± 0,0416 | 0,5983 ± 0,0445 |
| Home Credit | 5-fold CV | 0,7461 ± 0,0022 | 0,4922 ± 0,0044 | 0,3678 ± 0,0043 | 0,0687 ± 0,0010 | 0,2500 ± 0,0027 | 0,2225 ± 0,0044 |
| Taiwan | 5-fold CV | 0,7237 ± 0,0082 | 0,4473 ± 0,0165 | 0,3821 ± 0,0173 | 0,1449 ± 0,0038 | 0,4659 ± 0,0102 | 0,5043 ± 0,0094 |
| Lending Club | Fit lại train + val | 0,7018 (KTC 95%: 0,6992–0,7045) | 0,4037 | 0,2955 | 0,1185 | 0,3885 | 0,2800 |
| GMSC | 5-fold CV | **0,6981 ± 0,0110** | 0,3962 ± 0,0220 | 0,3196 ± 0,0221 | 0,0576 ± 0,0019 | 0,2253 ± 0,0065 | 0,2344 ± 0,0088 |

Brier, LogLoss và AP phụ thuộc tỷ lệ xấu nên không so sánh được giữa các bộ: GMSC có Brier thấp nhất nhưng AUC thấp nhất; HELOC có AP cao nhất (0,79) một phần vì tỷ lệ xấu 52%.

**Độ đo theo ngưỡng** (ngưỡng tối đa F1 trên validation):

| Bộ dữ liệu | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| HMEQ | 0,538 ± 0,023 | 0,543 ± 0,045 | 0,539 ± 0,018 | 0,815 ± 0,006 |
| FICO HELOC | 0,694 ± 0,021 | 0,839 ± 0,016 | 0,759 ± 0,006 | 0,723 ± 0,017 |
| South German | 0,551 ± 0,041 | 0,588 ± 0,103 | 0,563 ± 0,034 | 0,729 ± 0,027 |
| Home Credit | 0,223 ± 0,014 | 0,426 ± 0,043 | 0,291 ± 0,002 | 0,833 ± 0,016 |
| Taiwan | 0,547 ± 0,043 | 0,477 ± 0,030 | 0,508 ± 0,008 | 0,795 ± 0,019 |
| Lending Club | 0,239 | 0,662 | 0,352 | 0,637 |
| GMSC | 0,306 ± 0,014 | 0,319 ± 0,013 | 0,312 ± 0,009 | 0,906 ± 0,004 |

Accuracy cao của GMSC (0,906) và Home Credit (0,833) chỉ phản ánh mất cân bằng nhãn: đoán "tốt" cho mọi người đã đạt 93% và 92%.

## 6.2. Overfit: gần như không có, trừ South German

| Bộ dữ liệu | AUC train | AUC validation | AUC test | Train − test |
|---|---|---|---|---|
| GMSC | 0,6997 | 0,6945 | 0,6981 | +0,002 |
| Taiwan | 0,7271 | 0,7233 | 0,7237 | +0,003 |
| Home Credit | 0,7495 | 0,7465 | 0,7461 | +0,003 |
| HMEQ | 0,8081 | 0,8054 | 0,8030 | +0,005 |
| **South German** | **0,8365** | 0,8228 | **0,7663** | **+0,070** |
| FICO HELOC | 0,8059 | 0,8006 | 0,8019 | +0,004 |
| Lending Club | 0,6856 | 0,6827 | 0,7018 | −0,016 |

- Với 6 bộ, AUC train gần bằng AUC test: giới hạn hiệu năng của LR đến từ **dạng quan hệ tuyến tính**, không phải từ overfit.
- **South German** là ngoại lệ: 71 đặc trưng (68 cột one-hot) cho 640 dòng train dẫn đến overfit rõ (+0,07) và AUC dao động lớn nhất giữa các fold (0,734 ở fold 1 đến 0,790 ở fold 3). Đây là bộ hưởng lợi nhiều nhất khi tinh chỉnh tham số phạt `C`.
- Lending Club có AUC test **cao hơn** AUC train không phải do lỗi: mô hình cuối được fit trên 2013–2014 (262.992 dòng) thay vì chỉ 2013. Mô hình chỉ fit trên 2013 cho AUC test 0,6936; fit lại trên train + validation tăng AUC thêm **+0,008**.

## 6.3. GMSC: AUC thấp do ngoại lai cực đoan

AUC 0,698 của GMSC thấp hơn cả AUC đơn biến của biến mạnh nhất, tỷ lệ sử dụng hạn mức tín dụng `RevolvingUtilizationOfUnsecuredLines` (0,776 ở giai đoạn EDA). Nguyên nhân:

- Biến này lẽ ra nằm trong [0, 1] nhưng có giá trị tới ~50.000. Sau chuẩn hóa, độ lệch chuẩn bị ngoại lai kéo lên ~205, nên gần như mọi người vay dồn về cùng một giá trị (~ −0,02). Hệ số LR của biến này chỉ ≈ −0,015: mô hình tuyến tính **không khai thác được** biến mạnh nhất.
- Mã 96/98 trong các cột đếm trễ hạn làm ba cột này gần như cộng tuyến trên các dòng đó, dẫn đến hệ số trái dấu. Tuy nhiên, bỏ 49 dòng có mã 96/98 khỏi test không cải thiện AUC (0,683 → 0,678), nên đây không phải nguyên nhân chính.

**Phân tích phụ** (fold 1; *không* thuộc pipeline chính vì đặc tả không cắt ngoại lai): cắt mỗi biến số về khoảng phân vị 1%–99% **học trên train**, sau đó chạy lại đúng B.1 và LR mặc định.

| Bộ dữ liệu | AUC test fold 1 (pipeline chuẩn) | AUC sau khi cắt 1%–99% | Chênh lệch |
|---|---|---|---|
| **GMSC** | 0,6825 | **0,8517** | **+0,169** |
| Home Credit | 0,7497 | 0,7499 | +0,000 |
| Lending Club | 0,6937 | 0,6943 | +0,001 |

Với GMSC, gần như toàn bộ khoảng cách là do ngoại lai. Kết quả LR của GMSC theo pipeline chuẩn vì vậy **đánh giá thấp** mô hình tuyến tính; khi so sánh với các mô hình cây (vốn không nhạy với ngoại lai), cần nêu rõ điểm này để không quy toàn bộ chênh lệch cho tính phi tuyến. Home Credit và Lending Club, dù cũng có giá trị chuẩn hóa tới hàng trăm độ lệch chuẩn, không bị ảnh hưởng vì ngoại lai không nằm ở các biến quan trọng.

## 6.4. Calibration

![ROC và calibration trên test](../../reports/models/lr_default/figures/roc_calibration.png)

| Bộ dữ liệu | Tỷ lệ xấu thực tế (test) | Xác suất dự đoán trung bình |
|---|---:|---:|
| GMSC | 6,68% | 6,70% |
| Taiwan | 22,12% | 22,16% |
| Home Credit | 8,07% | 8,10% |
| HMEQ | 19,95% | 19,90% |
| South German | 30,00% | 30,60% |
| FICO HELOC | 52,03% | 52,10% |
| Lending Club | **14,88%** | **13,51%** |

- Ở 6 bộ CV, xác suất dự đoán trung bình gần như bằng tỷ lệ xấu thực (chênh ≤ 0,6 điểm %), đúng kỳ vọng với LR không cân bằng lớp khi train và test cùng phân phối.
- **Lending Club** dự đoán thấp hơn thực tế 1,4 điểm %: mô hình học trên 2013–2014 (tỷ lệ xấu 13,2%) nhưng năm 2015 có tỷ lệ xấu 14,9%. Đây là trôi dạt theo thời gian — mô hình vẫn xếp hạng tốt nhưng mức rủi ro cần được hiệu chỉnh lại khi áp dụng cho năm sau.
- **South German** quá tự tin ở nhóm rủi ro cao (dự đoán ~0,8 nhưng thực tế ~0,68), là hệ quả của overfit ở mục 6.2.
- **Taiwan** lệch ở vùng giữa (dự đoán 0,2–0,25 nhưng thực tế ~0,12): quan hệ phi tuyến của các biến trạng thái thanh toán `PAY_*` (giữ là biến số) mà LR không nắm được.

## 6.5. Độ ổn định

- Độ lệch chuẩn AUC giữa các fold tỷ lệ nghịch với kích thước bộ dữ liệu: Home Credit ± 0,002 (307.511 dòng) → HMEQ ± 0,016 → South German ± 0,028 (1.000 dòng).
- Lending Club có khoảng tin cậy bootstrap rất hẹp (± 0,0027) vì tập test lớn (282.787 dòng). Khoảng này chỉ phản ánh biến động do lấy mẫu test, không phản ánh biến động do việc chọn năm.

| Bộ dữ liệu | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|
| GMSC | 0,6825 | 0,7057 | 0,6910 | 0,7087 | 0,7026 |
| Taiwan | 0,7315 | 0,7270 | 0,7296 | 0,7183 | 0,7120 |
| Home Credit | 0,7497 | 0,7437 | 0,7458 | 0,7458 | 0,7454 |
| HMEQ | 0,8268 | 0,7956 | 0,8090 | 0,7838 | 0,8000 |
| South German | 0,7336 | 0,7377 | 0,7895 | 0,7841 | 0,7868 |
| FICO HELOC | 0,8009 | 0,7990 | 0,8171 | 0,7899 | 0,8025 |

## 6.6. Tóm tắt

| Mức độ khó (theo AUC của LR) | Bộ dữ liệu | Nhận xét |
|---|---|---|
| Dễ (AUC ≈ 0,80) | HMEQ, FICO HELOC | HMEQ hưởng lợi từ việc thiếu dữ liệu mang thông tin (qua nhóm `__missing__`); HELOC có sẵn điểm rủi ro tổng hợp |
| Trung bình (0,72–0,77) | South German, Home Credit, Taiwan | South German bị giới hạn bởi kích thước; Taiwan bởi quan hệ phi tuyến của `PAY_*` |
| Khó (≈ 0,70) | Lending Club, GMSC | Lending Club: tín hiệu yếu thật và có trôi dạt theo thời gian. GMSC: do ngoại lai, không phải do tín hiệu yếu (mục 6.3) |

**Giới hạn của kết quả này:**

1. Tham số mặc định, chưa tinh chỉnh — đặc biệt tham số phạt `C` với South German.
2. Theo đặc tả, không cắt ngoại lai, nên LR của GMSC bị đánh giá thấp nghiêm trọng (mục 6.3).
3. Ngưỡng F1 của Lending Club lấy từ mô hình fit trên 2013 (validation 2014) và áp cho mô hình fit lại trên 2013–2014, vì mô hình fit lại không còn phần validation.
4. Lending Club chỉ có một lần chia; khoảng tin cậy bootstrap không phản ánh biến động do chọn năm.

## 6.7. So sánh các mô hình classic ML

Bốn mô hình dùng cùng các fold, cùng ma trận đặc trưng và cùng cách đánh giá (chương 4), nên chênh lệch chỉ đến từ mô hình. Số liệu gốc: `reports/models/comparison/` (`tables.md`, `auc.csv`, `ranks.csv`, `stats.json`).

![AUC test của bốn mô hình trên 7 bộ dữ liệu](../../reports/models/comparison/figures/auc_by_model.png)

### AUC trên test

| Bộ dữ liệu | LR | KNN | RF | NB | Tốt nhất |
|---|---|---|---|---|---|
| GMSC | 0,6981 ± 0,0110 | 0,6859 ± 0,0038 | **0,8399 ± 0,0047** | 0,7042 ± 0,0119 | RF |
| Taiwan | 0,7237 ± 0,0082 | 0,6985 ± 0,0126 | **0,7633 ± 0,0081** | 0,7323 ± 0,0142 | RF |
| Home Credit | **0,7461 ± 0,0022** | 0,5819 ± 0,0009 | 0,7084 ± 0,0030 | 0,5309 ± 0,0068 | LR |
| HMEQ | 0,8030 ± 0,0161 | 0,8993 ± 0,0195 | **0,9703 ± 0,0060** | 0,7704 ± 0,0179 | RF |
| South German | 0,7663 ± 0,0281 | 0,7084 ± 0,0431 | **0,7710 ± 0,0284** | 0,7157 ± 0,0414 | RF |
| FICO HELOC | **0,8019 ± 0,0098** | 0,7427 ± 0,0075 | 0,7946 ± 0,0066 | 0,7587 ± 0,0069 | LR |
| Lending Club | **0,7018** | 0,5782 | 0,6777 | 0,6744 | LR |
| **Hạng trung bình** | 2,00 | 3,57 | **1,43** | 3,00 | |

### Hạng trung bình theo từng độ đo (1 = tốt nhất)

| Độ đo | LR | KNN | RF | NB |
|---|---|---|---|---|
| AUC / Gini | 2,00 | 3,57 | **1,43** | 3,00 |
| KS | 2,00 | 3,29 | **1,57** | 3,14 |
| AP | 1,71 | 3,29 | **1,43** | 3,57 |
| F1 | 1,86 | 3,43 | **1,43** | 3,29 |
| Brier | 1,71 | 2,86 | **1,43** | 4,00 |
| LogLoss | **1,43** | 3,29 | 1,57 | 3,71 |

Thứ tự **RF > LR > NB, KNN** giữ nguyên ở hầu hết các độ đo. Riêng LogLoss, LR xếp nhất vì xác suất của LR được hiệu chỉnh tốt nhất (mục 6.4), trong khi RF với cây phát triển hết độ sâu cho xác suất sát 0 hoặc 1 nhiều hơn.

### Kiểm định thống kê

| Kiểm định | Kết quả |
|---|---|
| Friedman (N = 7 bộ, k = 4 mô hình) | χ²_F = 11,74; p = 0,0083 |
| Iman–Davenport | F = 7,61; **p = 0,0017** |

→ Bác bỏ giả thuyết "bốn mô hình có hạng như nhau": có khác biệt có ý nghĩa thống kê giữa các mô hình.

| Cặp (A vs B) | Số bộ A thắng / B thắng | Chênh AUC trung bình (A − B) | p | p (Holm) |
|---|---|---|---|---|
| RF vs KNN | 7 / 0 | +0,0900 | 0,0156 | 0,0938 |
| RF vs NB | 7 / 0 | +0,0912 | 0,0156 | 0,0938 |
| LR vs NB | 5 / 2 | +0,0506 | 0,0781 | 0,3125 |
| LR vs KNN | 6 / 1 | +0,0494 | 0,1562 | 0,4688 |
| RF vs LR | 4 / 3 | +0,0406 | 0,4688 | 0,9375 |
| NB vs KNN | 5 / 2 | −0,0012 | 0,8125 | 0,9375 |

- RF thắng KNN và NB ở **cả 7 bộ**, nhưng sau hiệu chỉnh Holm p = 0,094, chưa đạt mức 0,05. Đây là giới hạn của số bộ dữ liệu: với N = 7, p nhỏ nhất có thể có của Wilcoxon là 0,0156, và nhân với 6 cặp theo Holm thì không cặp nào có thể xuống dưới 0,05. Kết luận "RF tốt hơn KNN và NB" là nhất quán nhưng chưa được khẳng định thống kê ở mức từng cặp.
- **RF và LR không khác biệt có ý nghĩa** (4 thắng / 3 thua, p = 0,47): mỗi mô hình mạnh trên một nhóm bộ dữ liệu khác nhau (xem dưới).

### Ở mức từng bộ dữ liệu

Số fold (trên 5) mô hình thắng LR về AUC test:

| Bộ dữ liệu | KNN | RF | NB |
|---|---|---|---|
| GMSC | 1/5 | **5/5** | 4/5 |
| Taiwan | 0/5 | **5/5** | **5/5** |
| Home Credit | 0/5 | 0/5 | 0/5 |
| HMEQ | **5/5** | **5/5** | 0/5 |
| South German | 0/5 | 3/5 | 0/5 |
| FICO HELOC | 0/5 | 1/5 | 0/5 |

Lending Club — chênh lệch AUC so với LR trên cùng 282.787 dòng test (bootstrap ghép cặp, 1.000 lần):

| Mô hình | Chênh AUC | KTC 95% |
|---|---|---|
| KNN − LR | −0,1237 | −0,1268 đến −0,1207 |
| RF − LR | −0,0241 | −0,0261 đến −0,0221 |
| NB − LR | −0,0270 | −0,0285 đến −0,0253 |

Trên Lending Club, LR tốt hơn cả ba mô hình còn lại một cách rõ ràng (mọi khoảng tin cậy đều nằm hoàn toàn dưới 0).

### Nhận xét theo mô hình

**Random Forest — tốt nhất nói chung, mạnh ở dữ liệu nhỏ và vừa có quan hệ phi tuyến.**

- Thắng rõ trên **GMSC** (0,840 so với 0,698 của LR): RF chia theo ngưỡng nên không bị ngoại lai ảnh hưởng, khai thác được biến tỷ lệ sử dụng hạn mức mà LR bỏ qua (mục 6.3). Kết quả này gần với mức LR đạt được khi cắt ngoại lai ở phân tích phụ (0,852), củng cố nhận định rằng khoảng cách của LR trên GMSC chủ yếu do ngoại lai.
- Thắng rất lớn trên **HMEQ** (0,970 so với 0,803). Mức tăng này không đến từ mẫu giá trị thiếu: trên các dòng test không thiếu giá trị nào, RF vẫn đạt 0,972 so với 0,798 của LR, và RF chỉ huấn luyện trên các dòng đầy đủ vẫn đạt 0,957. Biến quan trọng nhất là tỷ lệ nợ/thu nhập `DEBTINC` (22% độ quan trọng), có quan hệ phi tuyến với nhãn.
- Thắng trên **Taiwan** (0,763): nắm được quan hệ phi tuyến của các biến trạng thái thanh toán `PAY_*`.
- **Thua LR** trên Home Credit, HELOC và Lending Club — các bộ có nhiều đặc trưng (HELOC 37, Lending Club 184, Home Credit 250), tín hiệu phân tán, gần tuyến tính. Với tham số mặc định, mỗi cây phát triển hết độ sâu nên **AUC train = 1,000 ở mọi bộ** (học thuộc dữ liệu train); trên các bộ tín hiệu yếu và nhiễu, việc học thuộc này làm giảm khả năng tổng quát. Đây là mô hình hưởng lợi nhiều nhất khi tinh chỉnh (độ sâu, số dòng tối thiểu ở lá).

**Logistic Regression — ổn định và được hiệu chỉnh tốt nhất.**

- Tốt nhất trên 3 bộ lớn/rộng (Home Credit, HELOC, Lending Club) và thắng khẳng định trên Lending Club.
- Xác suất dự đoán trung bình khớp tỷ lệ xấu thực trong khoảng 0,6 điểm % ở mọi bộ CV; xếp nhất về LogLoss.
- Điểm yếu: không nắm được quan hệ phi tuyến (HMEQ, Taiwan) và nhạy với ngoại lai (GMSC).

**K-Nearest Neighbors — tệ nhất, sụp đổ khi số chiều lớn.**

- Trên Home Credit (250 đặc trưng) và Lending Club (184 đặc trưng), AUC chỉ còn 0,58: với số chiều lớn, khoảng cách Euclid giữa các điểm gần như bằng nhau ("lời nguyền số chiều"), lại bị chi phối bởi các biến có ngoại lai lớn sau chuẩn hóa.
- Ngược lại, KNN tốt thứ hai trên HMEQ (0,899, hơn LR 0,10): HMEQ có ít chiều (20) và cấu trúc cục bộ rõ, cùng hướng với kết quả của RF. Mức tăng lớn của cả hai mô hình dựa trên láng giềng/phân vùng trên HMEQ có thể một phần do các hồ sơ gần giống nhau trong dữ liệu; điều này chưa được kiểm tra (việc nhóm dòng trùng chỉ xử lý các dòng **giống hệt** nhau).
- Chậm nhất khi dự đoán (chiếm khoảng 80% tổng thời gian chạy, mục 5.3).

**Naive Bayes — xếp hạng kém và xác suất sai lệch nghiêm trọng.**

- Thất bại trên **Home Credit** (AUC 0,531, gần ngẫu nhiên). Nguyên nhân: giả định các đặc trưng độc lập bị vi phạm nặng — Home Credit có nhiều nhóm cột gần trùng nhau (các biến nhà ở dạng `*_AVG`/`*_MODE`/`*_MEDI`), nên tích các xác suất có điều kiện bị bão hòa: 98,5% dự đoán rơi sát 0 hoặc 1, thứ tự xếp hạng bị phá vỡ. Cắt ngoại lai không khắc phục được (0,547).
- Xác suất sai lệch rất lớn: trung bình xác suất dự đoán cao hơn tỷ lệ xấu thực **69 điểm %** trên Taiwan và **85 điểm %** trên Home Credit, nên NB xếp cuối về Brier ở mọi bộ; F1 trên Home Credit bằng 0.
- Vẫn khá trên các bộ ít biến và ít tương quan (Taiwan 0,732, hơn LR).

### Kết luận so sánh

| Nhóm bộ dữ liệu | Mô hình tốt nhất | Lý do |
|---|---|---|
| Nhỏ và vừa, ít đặc trưng, quan hệ phi tuyến hoặc có ngoại lai (GMSC, Taiwan, HMEQ, South German) | **RF** | Chia theo ngưỡng: không nhạy với ngoại lai, nắm được phi tuyến và tương tác |
| Lớn hoặc rộng, tín hiệu phân tán, có trôi dạt theo thời gian (Home Credit, HELOC, Lending Club) | **LR** | Mô hình đơn giản, ổn định, ít học thuộc nhiễu; xác suất được hiệu chỉnh tốt |
| — | KNN, NB | Không phù hợp làm mô hình chính: KNN hỏng khi số chiều lớn; NB vi phạm giả định độc lập và cho xác suất sai lệch |

Với tham số mặc định, **RF có hạng trung bình tốt nhất (1,43)** và khác biệt giữa các mô hình là có ý nghĩa thống kê theo Friedman (p = 0,0017); nhưng RF **không vượt trội LR một cách nhất quán** (4 thắng / 3 thua). Hai hướng tiếp theo có khả năng thay đổi thứ tự: tinh chỉnh tham số (đặc biệt độ sâu cây của RF và tham số phạt `C` của LR) và các mô hình GBDT.

**Giới hạn của so sánh:** tất cả là tham số mặc định, chưa tinh chỉnh; mới có 7 bộ dữ liệu nên năng lực của kiểm định từng cặp thấp; chưa có SVM (mục 4.5); ma trận đặc trưng one-hot và chuẩn hóa là lựa chọn của đặc tả cho nhóm classic, không nhất thiết tối ưu cho từng mô hình (ví dụ NB và RF không cần chuẩn hóa).
