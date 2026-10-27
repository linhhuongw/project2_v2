# 1. Giới thiệu

## 1.1. Bối cảnh

Chấm điểm tín dụng là bài toán ước lượng khả năng một người vay không trả được nợ, dựa trên thông tin có tại thời điểm xét duyệt. Đầu ra của mô hình được dùng để quyết định cho vay hay không, định giá khoản vay và ước tính tổn thất, nên một mô hình xếp hạng rủi ro tốt hơn mang lại giá trị trực tiếp cho tổ chức cho vay.

Trong nhiều thập kỷ, Logistic Regression (Cox, 1958) là mô hình chuẩn của ngành vì dễ diễn giải và cho xác suất được hiệu chỉnh tốt. Các mô hình học máy như KNN (Cover & Hart, 1967), Naive Bayes (Maron, 1961), Random Forest (Breiman, 2001) và đặc biệt là gradient boosting trên cây quyết định (Friedman, 2001) — với các thư viện XGBoost (Chen & Guestrin, 2016), LightGBM (Ke et al., 2017) và CatBoost (Prokhorenkova et al., 2018) — có khả năng nắm bắt quan hệ phi tuyến và tương tác giữa các biến, nhưng mức cải thiện thực tế so với Logistic Regression thay đổi nhiều giữa các nghiên cứu.

Một lý do là các nghiên cứu thường khác nhau về bộ dữ liệu, cách chia, cách tiền xử lý và cách đánh giá, nên kết quả khó so sánh với nhau. Một số lỗi phổ biến còn làm kết quả lạc quan quá mức: tiền xử lý học trên toàn bộ dữ liệu trước khi chia, dùng biến chỉ biết sau khi giải ngân, hoặc kết luận "mô hình A tốt hơn B" chỉ từ một bộ dữ liệu mà không có kiểm định thống kê. Baesens et al. (2026) đề xuất một giao thức đánh giá thống nhất cho chấm điểm tín dụng nhằm giải quyết vấn đề này; đồ án áp dụng giao thức đó.

## 1.2. Mục tiêu và câu hỏi nghiên cứu

Mục tiêu của đồ án là **so sánh công bằng các mô hình chấm điểm tín dụng** trên nhiều bộ dữ liệu, dưới cùng một quy trình tiền xử lý và đánh giá, và kiểm tra rằng kết quả không bị sai lệch bởi rò rỉ dữ liệu hay overfit. Cụ thể, đồ án trả lời bốn câu hỏi:

1. **Các mô hình học máy có vượt trội Logistic Regression không**, và mức vượt trội là bao nhiêu trên từng bộ dữ liệu?
2. **Mô hình nào tốt nhất tổng thể**, và khác biệt giữa các mô hình có ý nghĩa thống kê không khi chỉ có 7 bộ dữ liệu?
3. **Đặc điểm nào của dữ liệu** (quy mô, số chiều, ngoại lai, giá trị thiếu, trôi dạt theo thời gian) quyết định mô hình nào thắng?
4. **Kết quả có đáng tin không**: có rò rỉ dữ liệu, overfit, hay khác biệt giữa validation và test làm sai lệch kết luận không?

## 1.3. Phạm vi

- **Dữ liệu**: 7 bộ dữ liệu công khai — Give Me Some Credit, Taiwan (Yeh & Lien, 2009), Home Credit, HMEQ, South German (Grömping, 2019), FICO HELOC và Lending Club — từ 1.000 đến hơn 500 nghìn khoản vay, gồm cả thẻ tín dụng, vay tiêu dùng, vay thế chấp và vay P2P.
- **Mô hình**: 7 mô hình chia làm hai nhóm — classic ML (Logistic Regression, KNN, Random Forest, Naive Bayes) và gradient boosting (XGBoost, LightGBM, CatBoost).
- **Thiết lập**: mọi mô hình dùng **tham số mặc định** của thư viện, trên **cùng các fold** và cùng tiền xử lý. Đây là mốc cơ sở; tinh chỉnh tham số và tạo đặc trưng là hướng phát triển (mục 8.5).
- **Ngoài phạm vi**: SVM, mô hình học sâu và tabular foundation model; dữ liệu nhiều bảng (chỉ dùng bảng chính của Home Credit).

## 1.4. Đóng góp

1. **Một quy trình tái lập được** từ dữ liệu gốc đến bảng kết quả (`run_pipeline.py`), cài đặt giao thức Baesens et al. (2026) với nguyên tắc mọi bước học từ dữ liệu chỉ fit trên phần train của từng fold.
2. **So sánh 7 mô hình trên 7 bộ dữ liệu** theo nhiều độ đo (AUC, Gini, KS, Brier, LogLoss, AP, precision, recall, F1), kèm kiểm định Friedman, Wilcoxon–Holm và kiểm định hậu nghiệm trên hạng theo Demšar (2006).
3. **Kiểm tra độ tin cậy của kết quả**: phát hiện rò rỉ qua mẫu giá trị thiếu trong HMEQ, đề xuất hai cách ngăn rò rỉ và cho thấy kết luận so sánh không đổi; đo mức overfit và độ lệch giữa validation và test của từng mô hình.
4. **Phân tích theo đặc điểm dữ liệu**, giải thích vì sao mô hình tốt nhất thay đổi giữa các bộ, kèm learning curves để phân biệt giới hạn do dữ liệu với giới hạn do mô hình.

## 1.5. Cấu trúc báo cáo

- **Chương 2** tổng quan các nghiên cứu liên quan.
- **Chương 3** mô tả 7 bộ dữ liệu, chất lượng dữ liệu và bước khai báo dữ liệu.
- **Chương 4** trình bày phương pháp: chia dữ liệu, tiền xử lý, mô hình, độ đo và kiểm định thống kê.
- **Chương 5** mô tả thiết lập thực nghiệm.
- **Chương 6** trình bày kết quả, so sánh mô hình và kiểm tra rò rỉ dữ liệu, overfit.
- **Chương 7** thảo luận kết quả.
- **Chương 8** kết luận, nêu giới hạn và hướng phát triển.
