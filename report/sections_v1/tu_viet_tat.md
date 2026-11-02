# Danh mục từ viết tắt

| Từ viết tắt | Nghĩa |
|---|---|
| AP | Độ chính xác trung bình (Average Precision) |
| AUC | Diện tích dưới đường cong ROC (Area Under the ROC Curve) |
| CD | Khoảng cách tới hạn (Critical Difference) |
| CPU | Bộ xử lý trung tâm (Central Processing Unit) |
| CV | Kiểm định chéo (Cross-Validation) |
| EDA | Phân tích khám phá dữ liệu (Exploratory Data Analysis) |
| EFB | Gộp các đặc trưng loại trừ nhau (Exclusive Feature Bundling) |
| FE | Tạo đặc trưng (Feature Engineering) |
| GBDT | Gradient boosting trên cây quyết định (Gradient Boosting Decision Tree) |
| GMSC | Bộ dữ liệu Give Me Some Credit |
| GOSS | Lấy mẫu một phía dựa trên gradient (Gradient-based One-Side Sampling) |
| HELOC | Hạn mức tín dụng thế chấp bằng giá trị nhà (Home Equity Line of Credit) |
| HMEQ | Bộ dữ liệu khoản vay thế chấp bằng giá trị nhà (Home Equity) |
| KC | Khoảng cách |
| KNN | K láng giềng gần nhất (K-Nearest Neighbors) |
| KS | Thống kê Kolmogorov–Smirnov |
| KTC | Khoảng tin cậy |
| LGBM | LightGBM (Light Gradient Boosting Machine) |
| LR | Hồi quy logistic (Logistic Regression) |
| ML | Học máy (Machine Learning) |
| NaN | Giá trị không xác định (Not a Number) |
| NB | Naive Bayes |
| P2P | Cho vay ngang hàng (Peer-to-Peer) |
| RF | Rừng ngẫu nhiên (Random Forest) |
| ROC | Đường cong đặc trưng hoạt động của bộ thu nhận (Receiver Operating Characteristic) |
| SVM | Máy vector hỗ trợ (Support Vector Machine) |
| UCI | Kho dữ liệu học máy của Đại học California, Irvine (UCI Machine Learning Repository) |
| XGB | XGBoost (Extreme Gradient Boosting) |

Khai báo LaTeX (gói `glossaries`), theo mẫu trong `report/template.md`. Dùng trong bài bằng `\gls{<khóa>}`; lần dùng đầu tiên in nội dung trường `first`, các lần sau in `name`.

```
%\makeglossaries
\makenoidxglossaries

% Danh mục thuật ngữ và từ viết tắt
\newglossaryentry{AP}{
    type=\acronymtype,
    name={AP},
    description={Độ chính xác trung bình (Average Precision)},
    first={độ chính xác trung bình (Average Precision - AP)}
}
\newglossaryentry{AUC}{
    type=\acronymtype,
    name={AUC},
    description={Diện tích dưới đường cong ROC (Area Under the ROC Curve)},
    first={diện tích dưới đường cong ROC (Area Under the ROC Curve - AUC)}
}
\newglossaryentry{CD}{
    type=\acronymtype,
    name={CD},
    description={Khoảng cách tới hạn (Critical Difference)},
    first={khoảng cách tới hạn (Critical Difference - CD)}
}
\newglossaryentry{CPU}{
    type=\acronymtype,
    name={CPU},
    description={Bộ xử lý trung tâm (Central Processing Unit)},
    first={CPU}
}
\newglossaryentry{CV}{
    type=\acronymtype,
    name={CV},
    description={Kiểm định chéo (Cross-Validation)},
    first={kiểm định chéo (Cross-Validation - CV)}
}
\newglossaryentry{EDA}{
    type=\acronymtype,
    name={EDA},
    description={Phân tích khám phá dữ liệu (Exploratory Data Analysis)},
    first={phân tích khám phá dữ liệu (Exploratory Data Analysis - EDA)}
}
\newglossaryentry{EFB}{
    type=\acronymtype,
    name={EFB},
    description={Gộp các đặc trưng loại trừ nhau (Exclusive Feature Bundling)},
    first={Exclusive Feature Bundling (EFB)}
}
\newglossaryentry{FE}{
    type=\acronymtype,
    name={FE},
    description={Tạo đặc trưng (Feature Engineering)},
    first={tạo đặc trưng (Feature Engineering - FE)}
}
\newglossaryentry{GBDT}{
    type=\acronymtype,
    name={GBDT},
    description={Gradient boosting trên cây quyết định (Gradient Boosting Decision Tree)},
    first={gradient boosting trên cây quyết định (Gradient Boosting Decision Tree - GBDT)}
}
\newglossaryentry{GMSC}{
    type=\acronymtype,
    name={GMSC},
    description={Bộ dữ liệu Give Me Some Credit},
    first={Give Me Some Credit (GMSC)}
}
\newglossaryentry{GOSS}{
    type=\acronymtype,
    name={GOSS},
    description={Lấy mẫu một phía dựa trên gradient (Gradient-based One-Side Sampling)},
    first={Gradient-based One-Side Sampling (GOSS)}
}
\newglossaryentry{HELOC}{
    type=\acronymtype,
    name={HELOC},
    description={Hạn mức tín dụng thế chấp bằng giá trị nhà (Home Equity Line of Credit)},
    first={Home Equity Line of Credit (HELOC)}
}
\newglossaryentry{HMEQ}{
    type=\acronymtype,
    name={HMEQ},
    description={Bộ dữ liệu khoản vay thế chấp bằng giá trị nhà (Home Equity)},
    first={HMEQ (Home Equity)}
}
\newglossaryentry{KC}{
    type=\acronymtype,
    name={KC},
    description={Khoảng cách},
    first={khoảng cách (KC)}
}
\newglossaryentry{KNN}{
    type=\acronymtype,
    name={KNN},
    description={K láng giềng gần nhất (K-Nearest Neighbors)},
    first={K láng giềng gần nhất (K-Nearest Neighbors - KNN)}
}
\newglossaryentry{KS}{
    type=\acronymtype,
    name={KS},
    description={Thống kê Kolmogorov–Smirnov},
    first={thống kê Kolmogorov–Smirnov (KS)}
}
\newglossaryentry{KTC}{
    type=\acronymtype,
    name={KTC},
    description={Khoảng tin cậy},
    first={khoảng tin cậy (KTC)}
}
\newglossaryentry{LGBM}{
    type=\acronymtype,
    name={LGBM},
    description={LightGBM (Light Gradient Boosting Machine)},
    first={LightGBM (LGBM)}
}
\newglossaryentry{LR}{
    type=\acronymtype,
    name={LR},
    description={Hồi quy logistic (Logistic Regression)},
    first={Logistic Regression (LR)}
}
\newglossaryentry{ML}{
    type=\acronymtype,
    name={ML},
    description={Học máy (Machine Learning)},
    first={học máy (Machine Learning - ML)}
}
\newglossaryentry{NaN}{
    type=\acronymtype,
    name={NaN},
    description={Giá trị không xác định (Not a Number)},
    first={NaN (Not a Number)}
}
\newglossaryentry{NB}{
    type=\acronymtype,
    name={NB},
    description={Naive Bayes},
    first={Naive Bayes (NB)}
}
\newglossaryentry{P2P}{
    type=\acronymtype,
    name={P2P},
    description={Cho vay ngang hàng (Peer-to-Peer)},
    first={cho vay ngang hàng (Peer-to-Peer - P2P)}
}
\newglossaryentry{RF}{
    type=\acronymtype,
    name={RF},
    description={Rừng ngẫu nhiên (Random Forest)},
    first={Random Forest (RF)}
}
\newglossaryentry{ROC}{
    type=\acronymtype,
    name={ROC},
    description={Đường cong đặc trưng hoạt động của bộ thu nhận (Receiver Operating Characteristic)},
    first={Receiver Operating Characteristic (ROC)}
}
\newglossaryentry{SVM}{
    type=\acronymtype,
    name={SVM},
    description={Máy vector hỗ trợ (Support Vector Machine)},
    first={máy vector hỗ trợ (Support Vector Machine - SVM)}
}
\newglossaryentry{UCI}{
    type=\acronymtype,
    name={UCI},
    description={Kho dữ liệu học máy của Đại học California, Irvine (UCI Machine Learning Repository)},
    first={UCI Machine Learning Repository (UCI)}
}
\newglossaryentry{XGB}{
    type=\acronymtype,
    name={XGB},
    description={XGBoost (Extreme Gradient Boosting)},
    first={XGBoost (XGB)}
}
```
