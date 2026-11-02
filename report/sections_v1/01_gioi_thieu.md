# Chương 1. Giới thiệu

## 1.1. Đặt vấn đề

Chấm điểm tín dụng là bài toán ước lượng khả năng một người vay không trả được nợ, dựa trên thông tin có tại thời điểm xét duyệt. Đầu ra của mô hình được dùng để quyết định cho vay hay không, định giá khoản vay và ước tính tổn thất, nên một mô hình xếp hạng rủi ro tốt hơn mang lại giá trị trực tiếp cho tổ chức cho vay.

Về mặt kỹ thuật, đây là bài toán phân loại nhị phân trên dữ liệu dạng bảng, với nhãn 1 là người vay xấu và 0 là người vay tốt. Dữ liệu tín dụng thường mất cân bằng nhãn, có giá trị thiếu, ngoại lai và có thể trôi dạt theo thời gian. Các đặc điểm này khác nhau giữa các sản phẩm vay và các thị trường, nên một mô hình tốt trên một bộ dữ liệu chưa chắc tốt trên bộ khác.

## 1.2. Các giải pháp hiện tại và hạn chế

Trong nhiều thập kỷ, Logistic Regression (Cox, 1958) là mô hình chuẩn của ngành vì dễ diễn giải và cho xác suất được hiệu chỉnh tốt. Các mô hình học máy như KNN (Cover & Hart, 1967), Naive Bayes (Maron, 1961), Random Forest (Breiman, 2001) và đặc biệt là Gradient Boosting trên cây quyết định (Friedman, 2001), với các thư viện XGBoost (Chen & Guestrin, 2016), LightGBM (Ke et al., 2017) và CatBoost (Prokhorenkova et al., 2018), có khả năng nắm bắt quan hệ phi tuyến và tương tác giữa các biến. Tuy vậy, mức cải thiện thực tế của chúng so với Logistic Regression thay đổi nhiều giữa các nghiên cứu.

Lessmann et al. (2015) tổng quan các nghiên cứu so sánh mô hình chấm điểm tín dụng và chỉ ra bốn hạn chế chính. Thứ nhất, phần lớn nghiên cứu chỉ dùng ít bộ dữ liệu, trung bình 1,9 bộ, thường có kích thước nhỏ, ít biến và chủ yếu là hai bộ German Credit và Australian Credit của UCI; các bộ này không đại diện cho dữ liệu tín dụng trong thực tế, nên kết luận khó tổng quát hóa. Thứ hai, nhiều nghiên cứu đề xuất một phương pháp mới rồi so với một vài mô hình tham chiếu, trong khi phương pháp mới thường được tác giả tinh chỉnh kỹ hơn, nên phép so sánh có thể thiên lệch. Thứ ba, các nghiên cứu thường chỉ dùng một độ đo hoặc các độ đo cùng loại, và không nghiên cứu nào được tổng quan đánh giá chất lượng xác suất dự đoán, ví dụ bằng Brier score, dù tổ chức cho vay cần xác suất vỡ nợ được hiệu chỉnh tốt. Thứ tư, nhiều nghiên cứu không dùng kiểm định thống kê, hoặc dùng các kiểm định không phù hợp như t-test, hay so sánh từng cặp mà không hiệu chỉnh cho nhiều phép so sánh; chỉ ba nghiên cứu dùng kiểm định Friedman kèm kiểm định hậu nghiệm như Demšar (2006) khuyến nghị.

Ngoài ra, bản thân nghiên cứu của Lessmann et al. (2015) được công bố trước LightGBM (Ke et al., 2017) và CatBoost (Prokhorenkova et al., 2018), nên chưa phản ánh các thư viện gradient boosting hiện được dùng phổ biến trên dữ liệu dạng bảng.

## 1.3. Mục tiêu và định hướng giải pháp

Mục tiêu của đồ án là so sánh công bằng các mô hình chấm điểm tín dụng trên nhiều bộ dữ liệu, dưới cùng một quy trình tiền xử lý và đánh giá, và kiểm tra rằng kết quả không bị sai lệch bởi rò rỉ dữ liệu hay overfit. Cụ thể, đồ án trả lời bốn câu hỏi: (i) các mô hình học máy có vượt trội Logistic Regression không, và mức vượt trội là bao nhiêu trên từng bộ dữ liệu; (ii) mô hình nào tốt nhất tổng thể, và khác biệt giữa các mô hình có ý nghĩa thống kê không khi chỉ có bảy bộ dữ liệu; (iii) đặc điểm nào của dữ liệu, như quy mô, số chiều, ngoại lai, giá trị thiếu hay trôi dạt theo thời gian, dẫn đến quyết định mô hình nào thắng; và (iv) kết quả có đáng tin không (tức là xét xem có rò rỉ dữ liệu, overfit, hay khác biệt giữa validation và test làm sai lệch kết luận không).

Định hướng giải pháp đi theo các hạn chế nêu ở mục 1.2. Với hạn chế về dữ liệu, đồ án dùng bảy bộ dữ liệu thuộc nhiều loại sản phẩm vay, trong đó có các bộ lớn tới hàng trăm nghìn khoản vay và hơn một trăm biến, và dùng bản South German đã sửa lỗi mã hóa thay cho bản German Credit gốc. Với hạn chế về thiên lệch, đồ án không đề xuất mô hình mới mà so sánh các mô hình trong cùng điều kiện: mọi mô hình dùng tham số mặc định, cùng các fold và cùng tiền xử lý. Với hạn chế về độ đo, đồ án đánh giá cả ba loại: khả năng xếp hạng (AUC, KS), chất lượng xác suất (Brier, LogLoss) và chất lượng phân loại tại ngưỡng (precision, recall, F1). Với hạn chế về kiểm định, các mô hình được so sánh bằng hạng trung bình, kiểm định Friedman và các kiểm định hậu nghiệm theo Demšar (2006). Cuối cùng, đồ án đưa vào 3 mô hình Gradient Boosting: XGBoost, LightGBM và CatBoost. Đồ án cũng kiểm tra rò rỉ dữ liệu và overfit cho từng bộ để bảo đảm các kết luận của chính mình đáng tin cậy.

Phạm vi của đồ án gồm bảy bộ dữ liệu công khai là Give Me Some Credit, Taiwan (Yeh & Lien, 2009), Home Credit, HMEQ, South German (Grömping, 2019), FICO HELOC và Lending Club, có từ 1.000 đến hơn 500 nghìn khoản vay thuộc thẻ tín dụng, vay tiêu dùng, vay thế chấp và vay P2P. Bảy mô hình được chia làm hai nhóm: classic ML gồm Logistic Regression, KNN, Random Forest, Naive Bayes, và Gradient Boosting gồm XGBoost, LightGBM, CatBoost. Mọi mô hình dùng tham số mặc định của thư viện, làm mốc cơ sở cho việc tinh chỉnh sau này. SVM, mô hình học sâu, tabular foundation model và dữ liệu nhiều bảng nằm ngoài phạm vi.

## 1.4. Đóng góp của đồ án

Đồ án có bốn đóng góp chính như sau:

1. Đồ án xây dựng một quy trình tái lập được từ dữ liệu gốc đến bảng kết quả, với nguyên tắc mọi bước học từ dữ liệu chỉ fit trên phần train của từng fold.
2. Đồ án so sánh bảy mô hình trên bảy bộ dữ liệu theo nhiều độ đo (AUC, Gini, KS, Brier, LogLoss, AP, precision, recall, F1), kèm kiểm định Friedman, Wilcoxon–Holm và kiểm định hậu nghiệm trên hạng theo Demšar (2006).
3. Đồ án kiểm tra độ tin cậy của kết quả: phát hiện rò rỉ qua mẫu giá trị thiếu trong HMEQ, đề xuất hai cách ngăn rò rỉ và cho thấy kết luận so sánh không đổi; đo mức overfit và độ lệch giữa validation và test của từng mô hình.
4. Đồ án phân tích kết quả theo đặc điểm dữ liệu để giải thích vì sao mô hình tốt nhất thay đổi giữa các bộ, kèm learning curves để phân biệt giới hạn do dữ liệu với giới hạn do mô hình.

## 1.5. Bố cục đồ án

Phần còn lại của báo cáo được tổ chức như sau.

Chương 2 trình bày cơ sở lý thuyết của đồ án. Chương này mô tả ngữ cảnh của bài toán chấm điểm tín dụng, tổng quan các nghiên cứu liên quan, rồi trình bày nguyên lý của bốn mô hình classic ML và của gradient boosting trên cây quyết định cùng ba cách cài đặt XGBoost, LightGBM và CatBoost.

Chương 3 trình bày giải pháp đề xuất, đó là một quy trình chấm điểm tín dụng thống nhất trên bảy bộ dữ liệu. Chương này mô tả các bộ dữ liệu và bước khai báo dữ liệu, cách chia dữ liệu theo nhóm và theo thời gian, hai tầng tiền xử lý chỉ học trên train, cấu hình của bảy mô hình, cùng cách kiểm soát và kiểm tra rò rỉ dữ liệu.

Chương 4 trình bày đánh giá thực nghiệm. Sau khi nêu các độ đo, phương pháp so sánh thống kê và thiết lập thí nghiệm, chương này phân tích mô hình cơ sở Logistic Regression, so sánh các mô hình classic ML với nhau và với gradient boosting theo AUC, precision, recall và F1, trình bày learning curves, và kiểm tra rò rỉ dữ liệu, overfit.

Chương 5 tổng kết các kết quả chính, đánh giá độ tin cậy của chúng, nêu các giới hạn của đồ án và đề xuất hướng phát triển.
