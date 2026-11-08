# Glossary — Breiman (2001), *Random Forests*

**Source:** L. Breiman (2001). "Random Forests". *Machine Learning*, 45(1), 5–32. Statistics Department, University of California, Berkeley; editor R. E. Schapire; received November 1999, accepted April 2001. File: `references/project2/refs/random_forest_breiman_2001.pdf` (28 pages, pp. 5–32).

**What the paper does:** A random forest is a set of trees, each grown with its own independent random vector Θ, that vote for the most popular class. Breiman:
- proves that the forest's generalization error **converges** as trees are added, so adding trees does not overfit (Theorem 1.2);
- bounds that error by the **strength** of the individual trees and the **correlation** between them: $PE^*\le\bar\rho(1-s^2)/s^2$ (Theorem 2.3);
- builds forests with **bagging + random feature selection at each node** (Forest-RI) or random linear combinations of inputs (Forest-RC), which match or beat Adaboost and are more robust to label noise;
- introduces **out-of-bag (OOB)** estimates of error, strength, correlation and **variable importance**;
- extends the method to regression.

The paper is the standard reference for **Random Forest**. Its benchmark includes the **German credit** dataset, also used in this project.

**How to read this file:** Terms are grouped by topic, in roughly the order the paper uses them.
- Each entry gives the English term, a suggested Vietnamese term, a definition in Vietnamese (định nghĩa), the formula where there is one, and where it appears (section § / journal page).
- The Vietnamese terms are suggestions for the report (`report/sections_v1`); check them against `tu_viet_tat.md` and chapter 2.
- Equation numbers (1)–(14) and (A1)–(A2) are Breiman's own. He writes Θ for the random vector of a tree, $h(x,\Theta)$ for a tree and *M* for the number of inputs.

---

## 1. Ensembles and random vectors

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Ensemble of trees / voting** | Tập hợp cây / bỏ phiếu | Huấn luyện nhiều cây rồi cho chúng bỏ phiếu chọn lớp phổ biến nhất; cải thiện đáng kể độ chính xác so với một cây. | §1.1 p.5 |
| **Random vector $\Theta_k$** | Vectơ ngẫu nhiên $\Theta_k$ | Với cây thứ *k*, sinh một vectơ ngẫu nhiên $\Theta_k$ độc lập với các vectơ trước nhưng **cùng phân phối**; cây được dựng từ tập huấn luyện và $\Theta_k$, cho bộ phân loại $h(x,\Theta_k)$. Đây là điểm chung của mọi phương pháp tạo ngẫu nhiên. | §1.1 p.5 |
| **Bagging** | Bagging (bootstrap aggregating) | Mỗi cây học trên một mẫu lấy ngẫu nhiên từ tập huấn luyện (Breiman 1996). Θ là số lần mỗi mẫu được chọn khi "ném *N* phi tiêu vào *N* ô", tức lấy mẫu **có hoàn lại**. (Phần mở đầu ghi "without replacement", nhưng §3.1 và cách mô tả Θ đều là có hoàn lại.) | §1.1, §3.1 |
| **Random split selection** | Chọn điểm chia ngẫu nhiên | Dietterich (1998): ở mỗi nút chọn ngẫu nhiên một trong *K* điểm chia tốt nhất. | §1.1 |
| **Random subspace method** | Phương pháp không gian con ngẫu nhiên | Ho (1998): chọn ngẫu nhiên một tập con đặc trưng để dựng **mỗi cây**. Random forest thì chọn ngẫu nhiên **ở mỗi nút**. | §1.1 |
| **Randomizing outputs** | Ngẫu nhiên hóa đầu ra | Breiman (1998b): tạo tập huấn luyện mới bằng cách thêm nhiễu vào nhãn/giá trị đầu ra. | §1.1, §12 |
| **Amit & Geman (1997)** | | Nhận dạng chữ viết tay bằng cây chọn điểm chia trong một tập ngẫu nhiên các đặc trưng hình học; Breiman nói đây là nguồn ý tưởng chính. | §1.1, §3 |
| **Random forest (Definition 1.1)** | Rừng ngẫu nhiên | Bộ phân loại gồm các cây $\{h(x,\Theta_k)\}$, với $\Theta_k$ i.i.d., mỗi cây bỏ **một phiếu** cho lớp phổ biến nhất tại *x*. | §1.1 p.6 |
| **Unit vote (hard voting)** | Bỏ phiếu đơn vị | Mỗi cây bỏ đúng một phiếu cho một lớp. (scikit-learn thì lấy trung bình xác suất của các cây.) | §1.1 |

## 2. Margin, convergence, strength and correlation

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Margin function mg(X, Y)** | Hàm lề | $mg(X,Y)=av_k I(h_k(X)=Y)-\max_{j\ne Y}av_k I(h_k(X)=j)$: tỷ lệ phiếu cho lớp đúng trừ tỷ lệ phiếu lớn nhất cho một lớp sai. Lề càng lớn, phân loại càng chắc chắn. | §2.1 p.7 |
| **Indicator function I(·)** | Hàm chỉ thị | Bằng 1 nếu điều kiện đúng, 0 nếu sai. | §2.1 |
| **Generalization error PE\*** | Sai số tổng quát | $PE^*=P_{X,Y}(mg(X,Y)<0)$: xác suất lề âm, tức xác suất phân loại sai trên dữ liệu mới. | §2.1 |
| **Convergence (Theorem 1.2)** | Sự hội tụ | Khi số cây tăng, $PE^*$ hội tụ hầu chắc chắn về $P_{X,Y}\big(P_\Theta(h(X,\Theta)=Y)-\max_{j\ne Y}P_\Theta(h(X,\Theta)=j)<0\big)$ (1). | §2.1 eq. (1), App. I |
| **Strong Law of Large Numbers** | Luật số lớn mạnh | Công cụ chứng minh Theorem 1.2: trung bình phiếu của các cây hội tụ về xác suất theo Θ. | §2.1, App. I |
| **"Random forests do not overfit"** | Rừng ngẫu nhiên không overfit khi thêm cây | Hệ quả của Theorem 1.2: thêm cây không làm sai số tăng mà dẫn tới một giá trị giới hạn. Lưu ý: điều này nói về **số cây**, không có nghĩa từng cây hay toàn rừng không thể khớp quá sát dữ liệu. | §2.1, §13 |
| **Margin function for a forest mr(X, Y)** | Hàm lề của rừng | $mr(X,Y)=P_\Theta(h(X,\Theta)=Y)-\max_{j\ne Y}P_\Theta(h(X,\Theta)=j)$ (2). | §2.2 eq. (2) |
| **Strength s** | Độ mạnh *s* | $s=E_{X,Y}\,mr(X,Y)$ (3): lề kỳ vọng, đo độ chính xác của từng cây. | §2.2 eq. (3) |
| **Chebyshev's inequality bound** | Cận theo bất đẳng thức Chebyshev | Nếu $s\ge0$: $PE^*\le\mathrm{var}(mr)/s^2$ (4). | §2.2 eq. (4) |
| **Raw margin function rmg** | Hàm lề thô | $rmg(\Theta,X,Y)=I(h(X,\Theta)=Y)-I(h(X,\Theta)=\hat j(X,Y))$; *mr* là kỳ vọng của *rmg* theo Θ. | §2.2 Def. 2.2 |
| **Mean correlation $\bar\rho$** | Tương quan trung bình $\bar\rho$ | Tương quan trung bình giữa hàm lề thô của hai cây độc lập. Hai lớp với nhãn ±1: $\bar\rho$ là tương quan trung bình giữa dự đoán của hai cây. | §2.2 eq. (6)–(7) |
| **Upper bound (Theorem 2.3)** | Cận trên sai số | $PE^*\le\bar\rho(1-s^2)/s^2$. Cận có thể lỏng nhưng cho thấy hai thành phần quyết định: độ mạnh của từng cây và tương quan giữa các cây. | §2.2 p.9 |
| **c/s² ratio** | Tỷ số c/s² | $\bar\rho/s^2$: càng nhỏ càng tốt. | §2.2 Def. 2.4 |
| **VC-type bounds** | Cận kiểu VC | Cận lý thuyết cho các loại bộ phân loại khác; cận của Breiman đóng vai trò gợi ý tương tự. | §2.2 |
| **Weak learning condition** | Điều kiện học yếu | Hai lớp: yêu cầu độ mạnh dương tương tự $E_{X,Y}P_\Theta(h(X,\Theta)=Y)>0.5$, tức mỗi cây tốt hơn đoán ngẫu nhiên. | §2.2 p.9 |
| **Strength relative to class j** | Độ mạnh theo lớp *j* | Với nhiều lớp: $s_j=E_{X,Y}(P_\Theta(h=Y)-P_\Theta(h=j))$, không phụ thuộc vào rừng; cho cận (9). | §2.2 eq. (9) |
| **Design principle** | Nguyên tắc thiết kế | Tính ngẫu nhiên đưa vào phải **giảm tương quan $\bar\rho$ mà vẫn giữ độ mạnh**. | §3, §6 |

## 3. Out-of-bag estimation

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Bootstrap training set $T_k$** | Tập huấn luyện bootstrap | Mỗi cây học trên một tập lấy **có hoàn lại** từ tập gốc; cây **không được tỉa** (unpruned). | §3.1 p.11 |
| **Out-of-bag classifier** | Bộ phân loại ngoài túi | Với mỗi mẫu (y, x), chỉ tổng hợp phiếu của các cây mà $T_k$ **không chứa** mẫu đó. | §3.1 |
| **Out-of-bag (OOB) estimate** | Ước lượng ngoài túi | Tỷ lệ lỗi của bộ phân loại ngoài túi trên tập huấn luyện. Chính xác ngang dùng một tập kiểm tra cùng cỡ, nên **không cần tách tập kiểm tra riêng**. | §3.1 |
| **About one-third left out** | Khoảng một phần ba bị bỏ ra | Mỗi mẫu bootstrap bỏ ra khoảng 1/3 số quan sát (chính xác là $(1-1/N)^N\approx e^{-1}\approx36.8\%$). | §3.1 |
| **OOB bias** | Thiên lệch của OOB | Ước lượng OOB chỉ dùng khoảng 1/3 số cây nên có xu hướng **ước lượng cao** sai số hiện tại; cần chạy đủ nhiều cây. Khác với kiểm định chéo, Breiman cho rằng OOB không chệch khi đủ cây. | §3.1 |
| **OOB strength and correlation** | Độ mạnh và tương quan ngoài túi | $Q(x,j)$ = tỷ lệ phiếu ngoài túi cho lớp *j* tại *x*, ước lượng $P_\Theta(h(x,\Theta)=j)$; thay vào định nghĩa để ước lượng *s* và $\bar\rho$, (A1)–(A2). | §3.1, App. II |

## 4. Random features: Forest-RI and Forest-RC

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Forest-RI (random input selection)** | Rừng chọn ngẫu nhiên đầu vào | Ở mỗi nút chọn ngẫu nhiên *F* biến đầu vào và tìm điểm chia tốt nhất trong số đó; cây CART phát triển tối đa, không tỉa. Đây là dạng **Random Forest** thông dụng. | §4 p.11 |
| **Group size F** | Số biến chọn ở mỗi nút *F* | Hai giá trị thử nghiệm: *F* = 1 và *F* = int(log₂ *M* + 1). Tương ứng tham số `max_features` của scikit-learn (mặc định √*M*). | §4 |
| **CART methodology** | Phương pháp CART | Cách dựng cây phân loại/hồi quy của Breiman et al. (1984): chia nhị phân, chọn điểm chia theo độ vẩn đục. | §4 |
| **Insensitivity to F** | Không nhạy với *F* | Kết quả ít thay đổi theo *F*; chênh lệch trung bình giữa *F* = 1 và *F* lớn hơn dưới 1%. Thường 1–2 biến đã gần tối ưu. | §1.2, §4 |
| **Compute time ratio** | Tỷ lệ thời gian tính | So với cây dùng mọi biến: $F\log_2(N)/M$; ví dụ zip-code với *F* = 1 nhanh hơn khoảng 40 lần. | §4 p.14 |
| **Forest-RC (random linear combinations)** | Rừng tổ hợp tuyến tính ngẫu nhiên | Ở mỗi nút tạo *F* đặc trưng mới, mỗi đặc trưng là tổng của *L* biến với hệ số ngẫu nhiên đều trên [−1, 1]; dùng *L* = 3, *F* = 2 hoặc 8. | §5 |
| **Normalization of inputs** | Chuẩn hóa đầu vào | Khi các biến không cùng thang đo, trừ trung bình và chia độ lệch chuẩn **tính trên tập huấn luyện** trước khi tổ hợp. | §5 p.14 |
| **Categorical variables** | Biến phân loại | Mỗi lần chọn biến phân loại để chia, chọn ngẫu nhiên một tập con các hạng mục và tạo biến 0/1 thay thế. Biến có *I* hạng mục được chọn với xác suất gấp *I* − 1 lần biến số. Tránh được chi phí $O(2^{I-1})$ khi tìm điểm chia tốt nhất. | §5.1 |
| **Dummy 0–1 variables** | Biến giả 0–1 | Biến phân loại *I* giá trị có thể mã hóa thành *I* − 1 biến 0/1 (tương tự one-hot). | §5.1 |
| **Strength–correlation trade-off in F** | Đánh đổi độ mạnh – tương quan theo *F* | Thực nghiệm (sonar, breast, satellite): tăng *F* làm tương quan tăng đều, còn độ mạnh nhanh chóng chững lại; nên sai số thấp nhất ở *F* nhỏ. Với dữ liệu lớn hơn, độ mạnh tăng lâu hơn. | §6, Fig. 1–3 |

## 5. Comparisons and robustness

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Adaboost** | Adaboost | Thuật toán boosting tất định (Freund & Schapire 1996): tăng dần trọng số các mẫu vừa bị phân loại sai. Là chuẩn so sánh chính trong bài. | §1.2, §3, §7 |
| **Arcing (adaptive reweighting)** | Tái trọng số thích nghi | Tên Breiman gọi các thuật toán thay đổi trọng số tập huấn luyện theo lịch sử, như Adaboost. | §3, §13 |
| **Five desirable properties** | Năm ưu điểm | (i) chính xác như Adaboost, đôi khi tốt hơn; (ii) khá vững với ngoại lai và nhiễu; (iii) nhanh hơn bagging hoặc boosting; (iv) cho ước lượng nội bộ về sai số, độ mạnh, tương quan, độ quan trọng của biến; (v) đơn giản, dễ song song hóa. | §3 p.10 |
| **Conjecture: Adaboost is a random forest** | Giả thuyết: Adaboost là một rừng ngẫu nhiên | Ở giai đoạn sau, Adaboost có thể tương đương một rừng ngẫu nhiên với trọng số tập huấn luyện được rút từ một phân phối bất biến; điều này sẽ giải thích vì sao Adaboost không overfit khi thêm cây. | §7 |
| **Ergodic operator / invariant measure** | Toán tử ergodic / độ đo bất biến | Công cụ trong giả thuyết trên: dãy trọng số $w(k+1)=\phi(w(k))$ được giả định là ergodic. | §7 |
| **Output (label) noise** | Nhiễu nhãn | Đổi ngẫu nhiên 5% nhãn huấn luyện: Adaboost xấu đi rõ rệt (tới +43% ở Breast cancer, +49% ở Votes), rừng ngẫu nhiên thay đổi ít, vì không dồn trọng số vào các mẫu bị gán nhãn sai. | §8, Table 4 |
| **Many weak inputs** | Nhiều đầu vào yếu | Dữ liệu có hàng trăm/hàng nghìn biến, mỗi biến mang ít thông tin. Dữ liệu mô phỏng 1.000 biến nhị phân: sai số Bayes 1,0%, Naive Bayes 6,2%, Forest-RI 2,8–3,0% dù mỗi cây sai 60–80%. Rừng làm việc được với cây rất yếu miễn là tương quan thấp. | §9 |

## 6. Variable importance

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Black box** | Hộp đen | Rừng khó diễn giải trực tiếp; cần công cụ để hiểu biến nào quan trọng. | §1.2, §10 |
| **Permutation variable importance** | Độ quan trọng hoán vị | Sau mỗi cây, hoán vị ngẫu nhiên giá trị biến *m* trong các mẫu ngoài túi, cho lại qua cây; độ quan trọng là **% tăng tỷ lệ lỗi** so với OOB khi giữ nguyên biến. | §10 p.23 |
| **Noising up a variable** | Làm nhiễu một biến | Tên gọi của Breiman cho bước hoán vị ở trên. | §10 |
| **Correlated (dependent) variables** | Biến tương quan | Hai biến mang cùng thông tin có thể cùng có độ quan trọng cao, nhưng thêm biến thứ hai khi đã có biến thứ nhất không giảm lỗi (ví dụ biến 2 và 8 của Diabetes). | §10 |

## 7. Regression forests

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Mean-squared generalization error** | Sai số tổng quát bình phương trung bình | $E_{X,Y}(Y-h(X))^2$ (11). | §11 eq. (11) |
| **Convergence (Theorem 11.1)** | Hội tụ (hồi quy) | Khi số cây tăng, sai số của trung bình các cây hội tụ hầu chắc chắn về $E_{X,Y}(Y-E_\Theta h(X,\Theta))^2$ (12). | §11 |
| **PE\*(forest) ≤ ρ̄ PE\*(tree) (Theorem 11.2)** | Cận sai số rừng hồi quy | Nếu mỗi cây không chệch, sai số của rừng không quá $\bar\rho$ lần sai số trung bình của một cây, với $\bar\rho$ là tương quan có trọng số giữa phần dư của hai cây. | §11 eq. (13)–(14) |
| **Adaptive bagging** | Bagging thích nghi | Breiman (1999): biến thể bagging nhằm giảm bias trong hồi quy; dùng để so sánh. | §12, §13 |
| **Minimum node size** | Cỡ nút tối thiểu | Quy tắc "không chia nếu nút có dưới 5 mẫu" trong thí nghiệm hồi quy. | §12 |

## 8. Conclusions

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Bias and variance reduction** | Giảm bias và variance | Boosting/arcing giảm được cả bias lẫn variance; rừng ngẫu nhiên cạnh tranh được với chúng nên có vẻ cũng giảm bias, dù cơ chế chưa rõ. | §13 |
| **Random features with boosting** | Kết hợp đặc trưng ngẫu nhiên với boosting | Câu hỏi mở: thêm chọn đặc trưng ngẫu nhiên vào boosting có thể giảm lỗi trên dữ liệu lớn (zip-code 5,1%, letters 2,2%). Ý tưởng này sau được dùng trong GBDT hiện đại (lấy mẫu cột). | §13 |
| **Kernel view** | Góc nhìn kernel | Breiman (2000): với hai lớp, rừng ngẫu nhiên tương đương một kernel tác động lên lề thật. | §13 |

---

## 9. Notation summary

| Symbol | Ý nghĩa |
|---|---|
| $h(x,\Theta_k)$ | Cây thứ *k*, dựng với vectơ ngẫu nhiên $\Theta_k$ |
| *mg*, *mr*, *rmg* | Hàm lề của tập cây; hàm lề của rừng; hàm lề thô |
| $PE^*$ | Sai số tổng quát |
| *s* | Độ mạnh (lề kỳ vọng) |
| $\bar\rho$ | Tương quan trung bình giữa các cây |
| *F* | Số biến (hoặc tổ hợp) chọn ngẫu nhiên ở mỗi nút |
| *L* | Số biến trong một tổ hợp tuyến tính (Forest-RC) |
| *M*, *N* | Số biến đầu vào; số mẫu huấn luyện |
| $T_k$ | Tập huấn luyện bootstrap của cây *k* |
| $Q(x,j)$ | Tỷ lệ phiếu ngoài túi cho lớp *j* tại *x* |

## 10. Key numbers

| Item | Value |
|---|---|
| Benchmark | 20 datasets (13 UCI small, 3 large, 4 synthetic), incl. **German credit** (1,000 rows, 24 inputs, 2 classes) |
| German credit test error | Adaboost 23.5%, Forest-RI 24.4% (F=1: 26.2%), Forest-RC 22.8%, one tree 33.3% |
| Trees | 100 per forest (200 for zip-code); Adaboost 50 |
| Label noise (5%) | Adaboost error up to +48.9%; Forest-RI mostly < +8% |
| Many weak inputs | Bayes 1.0%, Naive Bayes 6.2%, Forest-RI 2.8% (F = 25) |

## 11. People and works cited

- **Breiman 1996a:** bagging predictors. **Breiman 1996b:** out-of-bag estimation.
- **Breiman 1998a/b, 1999, 2000:** arcing; randomizing outputs; adaptive bagging; infinity theory for ensembles.
- **Amit & Geman 1997:** randomized trees for shape recognition (main inspiration).
- **Ho 1998:** random subspace method.
- **Dietterich 1998:** random split selection; noise experiments.
- **Freund & Schapire 1996:** Adaboost. **Schapire et al. 1998:** margin explanation of boosting.
- **Bauer & Kohavi 1999; Grove & Schuurmans 1998; Kleinberg 2000; Tibshirani 1996; Wolpert & Macready 1997.**
- **Breiman et al. (1985 in text; CART):** classification and regression trees.

---

## 12. Relevance to this project (credit scoring with RF)

These points connect the paper to the report's chapter 2 (Random Forest section).

1. **Same algorithm, different defaults.** The project's `RandomForestClassifier` (100 fully grown trees, bootstrap, `max_features = sqrt`) is Breiman's Forest-RI. Differences: scikit-learn picks √*M* features per node (Breiman tried 1 and int(log₂ *M* + 1)), and it **averages tree probabilities** instead of counting unit votes.
2. **German credit appears in the paper.** Breiman reports about 24–26% test error for Forest-RI and 23.5% for Adaboost on the same 1,000-row German credit data used in the project. This is a useful external reference point for chapter 4, but note the different metric (error rate, not AUC) and protocol.
3. **"Does not overfit" needs care.** Theorem 1.2 says error converges as **trees** are added. Fully grown trees can still fit the training data almost perfectly, which can produce a large train–test gap (check the overfit heatmap in chapter 4). Cite Breiman for the convergence result, not as proof that RF cannot overfit.
4. **Why RF works: strength vs correlation.** $PE^*\le\bar\rho(1-s^2)/s^2$ explains the role of random feature selection: it lowers correlation between trees while keeping each tree reasonably strong.
5. **Robust to label noise.** Credit labels are noisy (default depends on events after the decision). Breiman's §8 shows RF is far less affected by noisy labels than Adaboost.
6. **Probabilities.** The paper studies votes and error rate only. With fully grown trees, RF probabilities tend to be pushed away from 0 and 1 and are often poorly calibrated; check the Brier score rather than citing the paper for this.
7. **Link to the other papers.** §9 compares with **Naive Bayes** (Maron 1961) and the **Bayes error** (Cover & Hart 1967). The closing question about combining random features with boosting points forward to XGBoost/LightGBM/CatBoost, which all support row and column subsampling.
8. **What the paper does not cover.** AUC, calibration, class imbalance, hyperparameter tuning beyond *F*, or impurity-based (Gini) importance. Do not cite it for those.

**Suggested citation sentence (VI):**
> *Rừng ngẫu nhiên (Breiman, 2001) kết hợp nhiều cây quyết định, mỗi cây học trên một mẫu bootstrap và chỉ xét một tập con đặc trưng ngẫu nhiên tại mỗi nút; sai số tổng quát của rừng phụ thuộc vào độ mạnh của từng cây và tương quan giữa các cây.*
