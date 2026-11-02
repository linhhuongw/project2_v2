# Chương 5. Kết luận và hướng phát triển

## 5.1. Kết luận

### 5.1.1. Tóm tắt công việc

Đồ án xây dựng một quy trình chấm điểm tín dụng thống nhất trên 7 bộ dữ liệu (GMSC, Taiwan, Home Credit, HMEQ, South German, FICO HELOC, Lending Club), trong đó mọi mô hình được huấn luyện và đánh giá trong cùng điều kiện:

- **Dữ liệu**: khai báo dữ liệu không học từ dữ liệu (tầng A); 5-fold CV phân tầng có nhóm theo dòng trùng lặp; Lending Club chia theo thời gian (train 2013, validation 2014, test 2015) trên cửa sổ khoản vay đã đáo hạn.
- **Tiền xử lý**: sàng lọc cột (B.0) và tiền xử lý (B.1) đều fit chỉ trên train của từng fold, với hai nhánh: one-hot cho classic ML và mã số nguyên cho GBDT.
- **Mô hình**: 7 mô hình với tham số mặc định — Logistic Regression, KNN, Random Forest, Naive Bayes, XGBoost, LightGBM, CatBoost — trên cùng các fold.
- **Đánh giá**: AUC, Gini, KS, Brier, LogLoss, AP, precision, recall, F1; learning curves; kiểm định Friedman, Wilcoxon–Holm và kiểm định hậu nghiệm trên hạng; kiểm tra rò rỉ dữ liệu và overfit.

### 5.1.2. Kết quả chính

1. **CatBoost là mô hình tốt nhất tổng thể** với tham số mặc định: hạng trung bình 1,57 theo AUC, đứng đầu 5/7 bộ và đứng đầu ở mọi độ đo trừ recall. LightGBM đứng thứ hai (2,86), RF thứ ba (3,29).
2. **Bước nhảy lớn nhất là từ mô hình tuyến tính sang mô hình cây**, không phải từ RF sang boosting: so với LR, mô hình cây tăng tới +0,17 AUC (GMSC), còn so với RF, GBDT chỉ tăng thêm tối đa +0,03.
3. **Khác biệt giữa các mô hình là có thật** (Friedman/Iman–Davenport p < 0,0001), nhưng với 7 bộ dữ liệu, chỉ khẳng định được KNN và NB kém CatBoost có ý nghĩa thống kê; LR, RF, XGBoost, LightGBM, CatBoost chưa phân biệt được về mặt thống kê.
4. **LR vẫn cạnh tranh ở các bộ tín hiệu gần tuyến tính hoặc có trôi dạt theo thời gian** (HELOC, Lending Club), xác suất được hiệu chỉnh tốt và không overfit — vẫn là lựa chọn hợp lý khi cần mô hình dễ diễn giải.
5. **Mô hình tốt nhất phụ thuộc vào dữ liệu**: mô hình cây vượt trội khi có quan hệ phi tuyến hoặc ngoại lai (GMSC, Taiwan, HMEQ); KNN hỏng khi số chiều lớn; NB hỏng khi các đặc trưng tương quan mạnh (Home Credit).

### 5.1.3. Độ tin cậy của kết quả: rò rỉ dữ liệu và overfit

**Rò rỉ dữ liệu**

- **Không phát hiện rò rỉ do pipeline hay do cách chia** ở bộ nào: dòng test không phải bản sao gần đúng của dòng train, không cột đơn lẻ nào gần như tiết lộ nhãn, tiền xử lý chỉ học trên train, rò rỉ sau giải ngân của Lending Club đã được loại ở tầng A.
- **HMEQ có rò rỉ nằm trong chính dữ liệu**: mẫu giá trị thiếu dự đoán nhãn (AUC 0,836) tốt hơn LR dùng toàn bộ dữ liệu, với các mẫu thiếu có tỷ lệ xấu từ 0% đến 100%. Nhiều khả năng các trường được ghi nhận khác nhau tùy theo kết cục khoản vay. Tiền xử lý B.1 truyền tín hiệu này cho mô hình (dòng thiếu nhận ra được với AUC 0,995).
- **Ngăn rò rỉ ở HMEQ** bằng hai cách — chỉ giữ hồ sơ đủ dữ liệu, hoặc điền giá trị thiếu bằng giá trị rút ngẫu nhiên — cho kết quả nhất quán: AUC của HMEQ giảm 0,02 (LR) đến 0,04 (GBDT), nhưng **thứ tự các mô hình và mọi kết luận so sánh giữ nguyên**, và mô hình cây vẫn hơn LR khoảng 0,13–0,16 AUC. Nghĩa là rò rỉ làm phóng đại mức AUC tuyệt đối của HMEQ chứ không tạo ra lợi thế của mô hình cây.

**Overfit**

- **RF với tham số mặc định học thuộc mọi bộ** (AUC train = 1); việc này gây hại ở Home Credit, HELOC và Lending Club, nơi RF thua LR. XGBoost là GBDT overfit nhiều nhất (learning rate 0,3). LR và NB không overfit.
- **South German là điểm yếu nhất**: mọi mô hình overfit, validation lạc quan hơn test 0,02–0,06, và learning curve cho thấy bộ này thiếu dữ liệu. Kết quả trên South German có độ bất định cao nhất (độ lệch chuẩn AUC giữa các fold tới 0,03–0,05).
- Ở các bộ còn lại, AUC validation và test chênh nhau không quá 0,007 (riêng NB trên HMEQ 0,015): các lựa chọn dựa trên validation tổng quát hóa tốt.

### 5.1.4. Giới hạn

1. Mọi mô hình dùng tham số mặc định; chưa tinh chỉnh.
2. Chỉ có 7 bộ dữ liệu, nên năng lực của các kiểm định từng cặp thấp.
3. Pipeline không cắt ngoại lai: LR bị đánh giá thấp trên GMSC (phân tích phụ cho thấy cắt ngoại lai đưa AUC từ 0,683 lên 0,852).
4. Biến phân loại được đưa vào GBDT dưới dạng mã số nguyên, chưa dùng cơ chế xử lý biến phân loại riêng của CatBoost và LightGBM.
5. HMEQ: kết quả trong so sánh chính dùng bản gốc (cùng cách tiền xử lý với các bộ khác) nên mức AUC tuyệt đối bị phóng đại bởi rò rỉ qua giá trị thiếu; kết quả đã ngăn rò rỉ được báo cáo ở mục 4.8.
6. Lending Club chỉ có một lần chia và có trôi dạt theo thời gian; khoảng tin cậy bootstrap không phản ánh biến động do chọn năm.
7. Chưa có SVM và các mô hình học sâu hay tabular foundation model.

## 5.2. Hướng phát triển trong tương lai

1. **Tinh chỉnh tham số** bằng Optuna (20 lần thử mỗi fold, tối đa AUC trên validation), ưu tiên các mô hình hưởng lợi nhiều nhất: RF (độ sâu, số dòng tối thiểu ở lá), XGBoost (learning rate), LR (tham số phạt `C`, đặc biệt trên South German). Với South German, nên dùng CV lặp lại khi tinh chỉnh vì validation quá nhỏ.
2. **Tạo đặc trưng (FE)** cho nhánh GBDT, đặc biệt tổng hợp các bảng phụ của Home Credit.
3. **Hiệu chỉnh xác suất theo thời gian** cho Lending Club (mọi mô hình dự đoán thấp hơn tỷ lệ xấu năm sau).
4. Mở rộng so sánh sang SVM, mô hình học sâu và tabular foundation model, và thêm bộ dữ liệu để tăng năng lực kiểm định.
