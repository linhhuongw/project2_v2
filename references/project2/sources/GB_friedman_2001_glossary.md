# Glossary — Friedman (2001), *Greedy Function Approximation: A Gradient Boosting Machine*

**Source:** J. H. Friedman (2001). "Greedy Function Approximation: A Gradient Boosting Machine". *The Annals of Statistics*, 29(5), 1189–1232. 1999 Reitz Lecture; Stanford University; received May 1999, revised April 2001. File: `references/project2/refs/gradien_boosting_friedman_2001.pdf` (44 pages, pp. 1189–1232).

**What the paper does:** Friedman views fitting a model as **numerical optimization in function space**: each new base learner is a step of **steepest descent** on the loss. This gives a single recipe, **Gradient Boost** (Algorithm 1), for any differentiable loss:
- fit a base learner to the **negative gradient** ("pseudo-responses") by least squares;
- choose the step size by a **line search** on the original loss;
- add the step to the model.

He then:
- derives versions for least squares, least absolute deviation, Huber loss and **logistic log-likelihood** (two-class and *K*-class);
- specialises to **regression trees** ("TreeBoost"), with a separate optimal value in each leaf, using a **Newton step** for the logistic case;
- introduces **shrinkage** (learning rate ν) as the main regularizer;
- studies tree size as control of the **interaction order**;
- proposes **relative variable importance** and **partial dependence plots** for interpretation.

The paper is the standard reference for **gradient boosting / GBDT**, the common ancestor of XGBoost, LightGBM and CatBoost.

**How to read this file:** Terms are grouped by topic, in roughly the order the paper uses them.
- Each entry gives the English term, a suggested Vietnamese term, a definition in Vietnamese (định nghĩa), the formula where there is one, and where it appears (section § / journal page).
- The Vietnamese terms are suggestions for the report (`report/sections_v1`); check them against `tu_viet_tat.md` and chapter 2.
- Equation numbers (1)–(58) are Friedman's own. **Convention warning:** for classification Friedman uses $y\in\{-1,1\}$ and $F=\tfrac12$ log-odds, so his formulas differ by constant factors from the $y\in\{0,1\}$, $F$ = log-odds convention used in the report (§2.4).

---

## 1. Function estimation as optimization

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Function estimation / predictive learning** | Ước lượng hàm / học dự đoán | Từ mẫu huấn luyện $\{y_i,x_i\}_1^N$, tìm hàm $\hat F(x)$ xấp xỉ $F^*=\arg\min_F E_{y,x}L(y,F(x))$ (1). | §1 p.1189 |
| **Output / response, input / explanatory variables** | Biến đầu ra / phản hồi, biến đầu vào / giải thích | *y* là biến cần dự đoán; $x=\{x_1,\dots,x_n\}$ là các biến đầu vào. | §1 |
| **Loss function L(y, F)** | Hàm mất mát | Đo thiệt hại khi dự đoán *F* cho giá trị thật *y*: bình phương $(y-F)^2$, trị tuyệt đối $|y-F|$, hoặc âm log hợp lý nhị thức $\log(1+e^{-2yF})$ với $y\in\{-1,1\}$. | §1 |
| **Additive expansion** | Khai triển cộng tính | $F(x;\{\beta_m,a_m\}_1^M)=\sum_{m=1}^M\beta_m h(x;a_m)$ (2): tổng của nhiều hàm cơ sở đơn giản *h*. Mạng nơ-ron, MARS, wavelet, SVM và cây đều có dạng này. | §1 eq. (2) |
| **Base learner h(x; a)** | Bộ học cơ sở | Hàm tham số đơn giản trong khai triển; ở đây chủ yếu là **cây hồi quy nhỏ** (CART). Với cây, *a* gồm biến chia, ngưỡng chia và giá trị trung bình ở lá. | §1, §3 |
| **Parameter optimization** | Tối ưu trong không gian tham số | Chọn họ hàm có tham số $F(x;P)$ rồi tìm $P^*=\arg\min_P\Phi(P)$ (3); thường giải bằng tổng các bước $P^*=\sum_m p_m$ (4). | §1.1 |
| **Steepest descent** | Hạ dốc nhanh nhất | Bước $p_m=-\rho_m g_m$ với $g_m$ là gradient hiện tại; $-g_m$ là hướng giảm nhanh nhất. | §1.2 |
| **Line search** | Tìm kiếm theo đường | Chọn độ dài bước $\rho_m=\arg\min_\rho\Phi(P_{m-1}-\rho g_m)$ (5). | §1.2 eq. (5) |
| **"Steps" or "boosts"** | Các bước tăng cường | Các số gia liên tiếp $p_m$ (hay $f_m(x)$), mỗi bước dựa trên các bước trước; nguồn gốc tên "boosting" trong cách nhìn này. | §1.1, §2 |

## 2. Optimization in function space

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Numerical optimization in function space** | Tối ưu số trong không gian hàm | Coi giá trị $F(x)$ tại mỗi điểm *x* là một "tham số" và tối ưu trực tiếp theo $F(x)$, thay vì theo tham số của một họ hàm. Đây là ý tưởng trung tâm của bài. | §2 p.1191 |
| **Nonparametric approach** | Cách tiếp cận phi tham số | Không giả định dạng hàm cho *F*; chỉ cộng dần các hàm số gia $f_m(x)$. | §2 |
| **Function-space gradient $g_m(x)$** | Gradient trong không gian hàm | $g_m(x)=E_y\big[\partial L(y,F(x))/\partial F(x)\mid x\big]_{F=F_{m-1}}$ (7). | §2 eq. (7) |
| **Function-space steepest descent** | Hạ dốc trong không gian hàm | $f_m(x)=-\rho_m g_m(x)$ (6), với $\rho_m$ tìm bằng line search (8). | §2 eq. (6), (8) |
| **Finite data problem** | Vấn đề dữ liệu hữu hạn | Với dữ liệu hữu hạn, gradient chỉ biết tại các điểm huấn luyện và không tổng quát cho *x* khác; cần "mượn sức" từ các điểm lân cận bằng cách áp đặt độ trơn (dùng hàm cơ sở có tham số). | §3 p.1192 |
| **Greedy stagewise approach** | Cách tiếp cận tham lam theo giai đoạn | Với $m=1..M$: $(\beta_m,a_m)=\arg\min\sum_i L(y_i,F_{m-1}(x_i)+\beta h(x_i;a))$ (9) rồi $F_m=F_{m-1}+\beta_m h(x;a_m)$ (10). **Không điều chỉnh lại** các số hạng đã thêm (khác với *stepwise*). | §3 eq. (9)–(10) |
| **Matching pursuit** | Matching pursuit | Tên gọi của chiến lược theo giai đoạn trong xử lý tín hiệu (Mallat & Zhang 1993), với mất mát bình phương. | §3 |
| **Boosting (as stagewise fitting)** | Boosting | Trong học máy, (9)–(10) với $y\in\{-1,1\}$ và mất mát mũ $e^{-yF}$ (AdaBoost) hoặc âm log hợp lý nhị thức (FHT00) được gọi là boosting. | §3 |
| **Weak learner** | Bộ học yếu | Tên khác của bộ học cơ sở trong boosting, thường là cây phân loại. | §3 |
| **Pseudo-responses / pseudo-residuals $\tilde y_i$** | Phản hồi giả / phần dư giả | $\tilde y_i=-\big[\partial L(y_i,F(x_i))/\partial F(x_i)\big]_{F=F_{m-1}}$: gradient âm tại từng điểm huấn luyện. | §3, Alg. 1 |
| **Constrained negative gradient** | Gradient âm bị ràng buộc | Chọn $h(x;a_m)$ "song song" nhất với $-g_m$ bằng bình phương tối thiểu: $a_m=\arg\min_{a,\beta}\sum_i[-g_m(x_i)-\beta h(x_i;a)]^2$ (11); sau đó line search trên mất mát gốc (12). Nhờ vậy mọi mất mát khả vi đều dùng được, miễn là có thuật toán bình phương tối thiểu cho *h*. | §3 eq. (11)–(12) |
| **Gradient Boost (Algorithm 1)** | Thuật toán Gradient Boost | 1. $F_0=\arg\min_\rho\sum L(y_i,\rho)$; với *m* = 1..*M*: 3. tính $\tilde y_i$; 4. khớp $h(x;a_m)$ với $\tilde y_i$ bằng bình phương tối thiểu; 5. line search $\rho_m$; 6. $F_m=F_{m-1}+\rho_m h(x;a_m)$. | §3 p.1193 |
| **Margin yF** | Lề *yF* | Với $y\in\{-1,1\}$ và mất mát chỉ phụ thuộc *yF*, boosting tương đương hạ dốc trong không gian lề. Cách nhìn theo hàm *F* tổng quát hơn. | §3 p.1194 |

## 3. Loss functions and specific algorithms

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **LS Boost (Algorithm 2)** | Boosting bình phương tối thiểu | $L=(y-F)^2/2$ ⇒ $\tilde y_i=y_i-F_{m-1}(x_i)$: khớp lặp đi lặp lại **phần dư hiện tại**. $F_0=\bar y$. | §4.1 |
| **Least absolute deviation (LAD)** | Độ lệch tuyệt đối nhỏ nhất | $L=|y-F|$ ⇒ $\tilde y_i=\mathrm{sign}(y_i-F_{m-1}(x_i))$ (13); line search là **trung vị có trọng số** (14). | §4.2 |
| **Huber loss / M-regression** | Mất mát Huber / hồi quy M | Bình phương khi $|y-F|\le\delta$, tuyệt đối khi lớn hơn (19); vững với phân phối đuôi dài và ngoại lai mà vẫn hiệu quả với sai số chuẩn. | §4.4 eq. (19) |
| **Transition point δ / breakdown point** | Điểm chuyển δ / điểm sụp đổ | δ chọn bằng phân vị α của phần dư hiện tại; điểm sụp đổ là tỷ lệ quan sát có thể bị thay đổi tùy ý mà không làm hỏng kết quả. | §4.4 |
| **Negative binomial log-likelihood** | Âm log hợp lý nhị thức | $L(y,F)=\log(1+\exp(-2yF))$, $y\in\{-1,1\}$, với $F(x)=\tfrac12\log\frac{\Pr(y=1\mid x)}{\Pr(y=-1\mid x)}$ (21). Đây là **log-loss** viết theo quy ước ±1. | §4.5 eq. (21) |
| **Logistic pseudo-response** | Phản hồi giả logistic | $\tilde y_i=2y_i/(1+\exp(2y_iF_{m-1}(x_i)))$ (22). Theo quy ước 0/1 và *F* = log-odds, đại lượng này tương ứng $y_i-p_i$. | §4.5 eq. (22) |
| **Probability estimates** | Ước lượng xác suất | $p_+(x)=1/(1+e^{-2F_M(x)})$, $p_-(x)=1/(1+e^{2F_M(x)})$: đổi điểm *F* thành xác suất. | §4.5 p.1199 |
| **Cost-sensitive classification** | Phân loại theo chi phí | Dùng xác suất để phân loại với chi phí $c(\hat y,y)$ cho từng loại sai; ngưỡng phụ thuộc chi phí, không nhất thiết 0,5. | §4.5, §4.6 |
| **Multiclass logistic (softmax)** | Logistic nhiều lớp | $p_k=\exp F_k/\sum_l\exp F_l$ (30); $\tilde y_{ik}=y_{ik}-p_k(x_i)$ (31): mỗi vòng học *K* cây, mỗi cây cho một lớp. | §4.6 |
| **Symmetric multiple logistic transform** | Biến đổi logistic đối xứng | $F_k=\log p_k-\tfrac1K\sum_l\log p_l$ (29). | §4.6 eq. (29) |

## 4. Regression trees as base learners (TreeBoost)

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **J-terminal node regression tree** | Cây hồi quy *J* lá | $h(x;\{b_j,R_j\}_1^J)=\sum_j b_j\mathbf 1(x\in R_j)$ (15): các vùng rời nhau $R_j$ phủ toàn không gian; nếu $x\in R_j$ thì $h(x)=b_j$. | §4.3 eq. (15) |
| **Terminal node / region $R_{jm}$** | Nút lá / vùng | Mỗi lá của cây ở vòng *m* ứng với một vùng $R_{jm}$ của không gian đầu vào. | §4.3 |
| **Separate update per leaf $\gamma_{jm}$** | Cập nhật riêng cho từng lá | Thay vì một hệ số chung $\rho_m$, chọn giá trị tối ưu cho từng lá theo **mất mát gốc**: $\gamma_{jm}=\arg\min_\gamma\sum_{x_i\in R_{jm}}L(y_i,F_{m-1}(x_i)+\gamma)$ (18). Cây chỉ dùng để chọn vùng; giá trị lá tính lại theo mất mát. | §4.3 eq. (17)–(18) |
| **TreeBoost** | TreeBoost | Tên chung cho gradient boosting với cây hồi quy và cập nhật theo từng lá (LS, LAD, M, LK TreeBoost). | Abstract, §4 |
| **LAD TreeBoost (Algorithm 3)** | | Cây khớp dấu của phần dư; giá trị lá là trung vị phần dư trong lá. Rất vững vì chỉ dùng thông tin thứ tự của đầu vào và phản hồi giả chỉ có hai giá trị. | §4.3 |
| **M TreeBoost (Algorithm 4)** | | Phiên bản Huber; giá trị lá xấp xỉ bằng một bước lặp Huber từ trung vị. | §4.4 |
| **LK TreeBoost (Algorithm 5, two-class)** | Gradient boosting logistic hai lớp | $F_0=\tfrac12\log\frac{1+\bar y}{1-\bar y}$; mỗi vòng: tính $\tilde y_i$ (22), dựng cây *J* lá, giá trị lá $\gamma_{jm}=\sum\tilde y_i/\sum|\tilde y_i|(2-|\tilde y_i|)$, cập nhật *F*. **Đây là dạng gần nhất với GBDT dùng cho chấm điểm tín dụng.** | §4.5 p.1199 |
| **Newton–Raphson step for leaf values** | Bước Newton–Raphson cho giá trị lá | Không có nghiệm đóng cho (23), nên xấp xỉ bằng một bước Newton: tổng gradient chia tổng đạo hàm bậc hai trong lá. Theo quy ước 0/1, đó là $\sum(y-p)/\sum p(1-p)$. XGBoost về sau tổng quát hóa ý này. | §4.5, §4.6 |
| **LK TreeBoost (Algorithm 6, K-class)** | Gradient boosting logistic *K* lớp | Mỗi vòng dựng *K* cây; giá trị lá $\gamma_{jkm}=\frac{K-1}{K}\frac{\sum\tilde y_{ik}}{\sum|\tilde y_{ik}|(1-|\tilde y_{ik}|)}$ (32) dùng xấp xỉ chéo của Hessian. Với *K* = 2 trùng với Algorithm 5. | §4.6 |
| **LogitBoost (FHT00)** | LogitBoost | Thuật toán của Friedman, Hastie & Tibshirani (2000) dựa trên Newton–Raphson; dùng phản hồi giả có trọng số $p(1-p)$ (33)–(34). Kém ổn định số khi trọng số gần 0; LK TreeBoost ổn định hơn. | §4.6 |
| **Split criterion (least-squares improvement)** | Tiêu chí chia (mức cải thiện bình phương) | $i^2(R_l,R_r)=\frac{w_lw_r}{w_l+w_r}(\bar y_l-\bar y_r)^2$ (35); chọn điểm chia làm tối đa đại lượng này. | §4.6 eq. (35) |
| **Influence trimming** | Cắt bỏ theo ảnh hưởng | Quan sát có $y_iF_{m-1}(x_i)$ rất lớn (đã được dự đoán đúng rất chắc) hầu như không ảnh hưởng tới vòng hiện tại; bỏ chúng (trọng số dưới ngưỡng α ∈ [0,05; 0,2]) giảm tính toán 10–20 lần mà không mất độ chính xác. Tiền thân ý tưởng của **GOSS** trong LightGBM. | §4.5.1 eq. (24)–(27) |
| **Influence / weight of an observation** | Ảnh hưởng / trọng số của quan sát | Đo bằng đạo hàm bậc hai của mất mát theo $F(x_i)$: $w_i=|\tilde y_i|(2-|\tilde y_i|)$ (26), tương đương $p(1-p)$ theo quy ước 0/1. | §4.5.1 |

## 5. Regularization

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Overfitting** | Quá khớp | Giảm mất mát trên tập huấn luyện quá một mức nào đó làm mất mát kỳ vọng trên tổng thể ngừng giảm rồi tăng lên. | §5 p.1203 |
| **Regularization** | Điều chuẩn | Ràng buộc quá trình khớp để tránh overfitting. | §5 |
| **Number of components M** | Số thành phần (số cây) *M* | Tham số điều chuẩn tự nhiên của khai triển cộng tính; chọn bằng tập kiểm tra độc lập hoặc kiểm định chéo. | §5 |
| **Shrinkage / learning rate ν** | Co rút / tốc độ học ν | Thay bước 6 bằng $F_m=F_{m-1}+\nu\cdot\rho_m h(x;a_m)$, $0<\nu\le1$ (36). Thường cho kết quả tốt hơn so với chỉ giới hạn *M* (theo Copas 1983). | §5 eq. (36) |
| **ν–M trade-off** | Đánh đổi ν–*M* | ν nhỏ cần *M* lớn hơn nhưng cho độ chính xác cao hơn, với lợi ích giảm dần khi ν < 0,125. Khuyến nghị: chọn *M* lớn nhất có thể về tính toán rồi chỉnh ν để mất mát tối thiểu gần *M*. | §5, Table 1 |
| **Incremental vs global shrinkage** | Co rút từng bước / co rút toàn cục | Co rút từng bước (36) cho mô hình rất khác và tốt hơn nhiều so với co rút toàn bộ mô hình cuối (38); lý do khi đó còn chưa rõ. | §5 eq. (38) |
| **Lack of fit (LOF)** | Mức độ không khớp | Thước đo sai số trên tập kiểm định: sai số tuyệt đối tương đối (37), deviance (−2 log-likelihood) hoặc tỷ lệ phân loại sai. | §5 |
| **Deviance vs error rate under overfitting** | Deviance và tỷ lệ lỗi khi overfit | Tỷ lệ phân loại sai vẫn giảm sau khi log-likelihood đã đạt tối ưu: overfitting làm hỏng **độ lớn** của *F* (tức xác suất) mà không làm hỏng **dấu** của nó. Tỷ lệ lỗi ít nhạy với overfitting hơn. | §5 p.1204–1205 |
| **Signal-to-noise ratio** | Tỷ số tín hiệu – nhiễu | Tỷ lệ giữa độ biến thiên của hàm thật và của nhiễu trong thí nghiệm mô phỏng (2/1 hoặc 1/1). | §5, §6.2 |

## 6. Simulation studies

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Monte Carlo simulation** | Mô phỏng Monte Carlo | Sinh dữ liệu theo nhiều kịch bản để đánh giá phương pháp trên nhiều hàm mục tiêu, thay vì vài bộ dữ liệu thật. | §6 |
| **Random function generator** | Bộ sinh hàm ngẫu nhiên | Hàm mục tiêu $F^*(x)=\sum_{l=1}^{20}a_lg_l(z_l)$ (39), mỗi $g_l$ là hàm Gauss trên một tập con ngẫu nhiên các biến (40); thiên về tương tác bậc thấp. 100 hàm, 10 biến đầu vào chuẩn. | §6.1 |
| **Slash distribution** | Phân phối slash | Phân phối sai số đuôi rất dày dùng để kiểm tra độ vững: LS TreeBoost thất bại, LAD và M TreeBoost tốt. | §6.2 |
| **Piecewise constant approximation** | Xấp xỉ hằng từng khúc | Mô hình cây cho hàm bậc thang; so với MARS (liên tục), không phải bất lợi nghiêm trọng và vững hơn với đuôi dài và ngoại lai ở đầu vào. | §6.3 |
| **AdaBoost.MH, K-class LogitBoost** | | Các thuật toán so sánh trên bài toán 5 lớp. Khi thêm co rút cho cả ba, hiệu năng gần như như nhau: mức độ "hung hăng" (learning rate) quan trọng hơn khác biệt cấu trúc. | §6.4 |

## 7. Tree size and interactions

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Metaparameters** | Siêu tham số | Ba siêu tham số chính của TreeBoost: số vòng *M*, learning rate ν và số lá *J* của mỗi cây. | §7 p.1214 |
| **Best-first induced tree** | Cây phát triển theo kiểu best-first | Cây được mở rộng bằng cách luôn chia nút tốt nhất cho tới khi đủ *J* lá (cùng ý với **leaf-wise** trong LightGBM). | §5, §7 |
| **ANOVA expansion** | Khai triển ANOVA | $F(x)=\sum_jf_j(x_j)+\sum_{j,k}f_{jk}(x_j,x_k)+\dots$ (42): hiệu ứng chính, tương tác hai biến, ba biến… | §7 eq. (42) |
| **Main effects / additive model** | Hiệu ứng chính / mô hình cộng tính | Tổng các hàm một biến; Naive Bayes cũng cho xấp xỉ thuần cộng tính (trên thang log). | §7 |
| **Interaction effects / order** | Hiệu ứng tương tác / bậc tương tác | Phần phụ thuộc đồng thời vào nhiều biến. Cây *J* lá tạo hàm có bậc tương tác tối đa $\min(J-1,n)$; vì boosting là cộng tính, bậc tương tác của cả mô hình không vượt quá bậc của từng cây. | §7 |
| **Choice of tree size J** | Chọn số lá *J* | *J* nên khớp với bậc tương tác thật của hàm mục tiêu: *J* = 2 chỉ có hiệu ứng chính; *J* = 6–21 ổn định hơn; cây lớn hiếm khi cần. Chọn bằng kiểm định chéo hoặc tập giữ lại. | §7, Fig. 6 |

## 8. Interpretation

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Relative influence / importance $I_j$** | Độ ảnh hưởng tương đối | Định nghĩa lý thuyết $I_j=\big(E_x[\partial\hat F/\partial x_j]^2\cdot\mathrm{var}_x[x_j]\big)^{1/2}$ (43); với cây dùng xấp xỉ: tổng mức cải thiện bình phương tại các nút chia theo biến *j* (44), trung bình trên mọi cây (45). Đây là gốc của **feature importance (gain)** trong các thư viện GBDT. | §8.1 eq. (43)–(45) |
| **Surrogate splits / masking** | Điểm chia thay thế / che khuất | Trong CART, điểm chia thay thế giúp phát hiện biến bị biến khác tương quan che khuất; với boosting ít cần hơn vì có rất nhiều cơ hội chia. | §8.1 |
| **Partial dependence function** | Hàm phụ thuộc riêng phần | $\bar F_l(z_l)=E_{z_{\setminus l}}\hat F(z_l,z_{\setminus l})$ (51), ước lượng bằng $\frac1N\sum_i\hat F(z_l,z_{i,\setminus l})$ (53): ảnh hưởng trung bình của tập biến $z_l$ sau khi lấy trung bình theo các biến còn lại. | §8.2 eq. (51), (53) |
| **Partial dependence plot (PDP)** | Biểu đồ phụ thuộc riêng phần | Vẽ $\bar F_l(z_l)$ theo một hoặc hai biến; áp dụng được cho mọi mô hình "hộp đen" (mạng nơ-ron, SVM, láng giềng gần nhất…). | §8.2–8.3 |
| **Marginal vs conditional averaging** | Trung bình theo biên / theo điều kiện | PDP lấy trung bình theo **phân phối biên** của các biến còn lại (51); lấy theo phân phối có điều kiện (56) lẫn vào cả phụ thuộc do tương quan giữa các biến. | §8.2 eq. (56) |
| **Additive / multiplicative dependence** | Phụ thuộc cộng / nhân | Nếu $\hat F$ phụ thuộc vào $z_l$ theo kiểu cộng (54) hoặc nhân (55), PDP mô tả đầy đủ ảnh hưởng của $z_l$. | §8.2 |
| **Weighted tree traversal** | Duyệt cây có trọng số | Cách tính PDP trực tiếp từ cây mà không cần dữ liệu: đi theo nhánh nếu biến chia thuộc $z_l$, nếu không thì đi cả hai nhánh với trọng số bằng tỷ lệ mẫu. | §8.2 |

## 9. Real data and data-mining properties

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Garnet / demographic data** | Dữ liệu garnet / nhân khẩu học | Hai ví dụ thật: garnet (13.317 mẫu, nhiều ngoại lai; tương tác mạnh giữa Ga và Zr) và khảo sát nhân khẩu học (9.409 mẫu, nhiều giá trị thiếu; mô hình cộng tính *J* = 2 là đủ). | §9 |
| **Invariance to monotone transformations** | Bất biến với biến đổi đơn điệu | Mọi thủ tục TreeBoost cho cùng kết quả nếu thay $x_j$ bằng $\log x_j$, $e^{x_j}$, $x_j^a$…; không cần biến đổi đầu vào và không nhạy với đuôi dài hay ngoại lai ở đầu vào. | §10 p.1228 |
| **Internal feature selection** | Chọn đặc trưng nội tại | Cây tự chọn biến khi chia nên khá vững khi thêm biến không liên quan. | §10 |
| **Missing values in trees** | Giá trị thiếu trong cây | Mô hình cây xử lý giá trị thiếu một cách thống nhất (CART), không cần điền giá trị từ bên ngoài. | §10 |
| **Instability of single trees** | Tính không ổn định của cây đơn | Cây đơn kém chính xác vì xấp xỉ thô, không ổn định (nhất là cây lớn) và chủ yếu có tương tác bậc cao; boosting giảm cả ba nhược điểm. | §10 |
| **Computational scaling** | Độ phức tạp tính toán | Sau khi sắp xếp các biến, thời gian tăng tuyến tính theo *N*, *n*, *M* và gần như theo log *J*; garnet (*N* = 13.317, *M* = 500) mất 20 giây trên Pentium III 933 MHz. | §10 |
| **Early stopping / restart** | Dừng sớm / tiếp tục huấn luyện | Lỗi giảm nhanh lúc đầu rồi chậm lại (gần tối ưu ở *M* ≈ 100); có thể dừng và tiếp tục từ chỗ cũ, hoặc huấn luyện tiếp trên dữ liệu mới, với tập kiểm tra độc lập để tránh overfit. | §10 |

---

## 10. Notation summary

| Symbol | Ý nghĩa |
|---|---|
| $F^*(x)$, $\hat F(x)$ | Hàm mục tiêu tối ưu; hàm ước lượng |
| $L(y,F)$ | Hàm mất mát |
| $h(x;a)$ | Bộ học cơ sở với tham số *a* |
| $F_m(x)$ | Mô hình sau *m* vòng |
| $g_m(x)$, $\tilde y_i$ | Gradient; phản hồi giả (gradient âm) |
| $\rho_m$ | Độ dài bước tìm bằng line search |
| $R_{jm}$, $\gamma_{jm}$ | Vùng lá *j* của cây vòng *m*; giá trị cập nhật của lá đó |
| *J* | Số lá của mỗi cây |
| *M* | Số vòng (số cây) |
| ν | Learning rate (shrinkage) |
| *K* | Số lớp |
| $I_j$ | Độ ảnh hưởng tương đối của biến *j* |
| $\bar F_l(z_l)$ | Hàm phụ thuộc riêng phần theo tập biến $z_l$ |

## 11. Algorithms in the paper

| # | Name | Loss | Pseudo-response | Leaf value |
|---|---|---|---|---|
| 1 | Gradient Boost | any differentiable | $-\partial L/\partial F$ | line search $\rho_m$ |
| 2 | LS Boost | $(y-F)^2/2$ | $y-F$ | (same as fit) |
| 3 | LAD TreeBoost | $|y-F|$ | $\mathrm{sign}(y-F)$ | median of residuals |
| 4 | M TreeBoost | Huber | clipped residual | one Huber step from median |
| 5 | LK TreeBoost (2-class) | $\log(1+e^{-2yF})$ | $2y/(1+e^{2yF})$ | Newton step |
| 6 | LK TreeBoost (*K*-class) | multinomial deviance | $y_k-p_k$ | Newton step, diagonal Hessian |

## 12. People and works cited

- **Breiman, Friedman, Olshen & Stone (1983, CART):** regression trees, variable importance, surrogate splits, missing values.
- **Friedman, Hastie & Tibshirani (2000, "FHT00"):** additive logistic regression, LogitBoost, statistical view of boosting.
- **Freund & Schapire (1996); Schapire & Singer (1998):** AdaBoost, AdaBoost.MH.
- **Breiman (1997, 1999):** pasting bites; arcing algorithms. **Ratsch et al. (1998); Duffy & Helmbold (1999):** boosting as gradient descent in margin space.
- **Huber (1964):** robust loss. **Copas (1983):** shrinkage. **Mallat & Zhang (1993):** matching pursuit.
- **Friedman (1991):** MARS. **Hastie & Tibshirani (1990):** generalized additive models.
- **Warner et al. (1961):** "naive" Bayes for medical diagnosis.
- Others: Rumelhart et al. (1986), Powell (1987), Donoho (1993), Vapnik (1995), Drucker (1997), Becker & Cleveland (1996), Ripley (1996).

---

## 13. Relevance to this project (credit scoring with GBDT)

These points connect the paper to the report's chapter 2 (§2.4 Gradient boosting).

1. **The common ancestor.** XGBoost, LightGBM and CatBoost all implement Algorithm 1 with regression trees and log-loss (LK TreeBoost). Section 2.4.2 of the report (initial log-odds, pseudo-residual *y − p*, tree fit, Newton leaf value $\sum(y-p)/\sum p(1-p)$, update with η) is Algorithm 5 rewritten in the 0/1 convention.
2. **Convention.** Friedman's $y\in\{-1,1\}$ and $F=\tfrac12\log$-odds produce the factors of 2 in (21)–(22). When citing his formulas next to the report's, say that they are equivalent up to this change of convention.
3. **Learning rate and number of trees.** The ν–*M* trade-off explains the project's defaults: XGBoost (η = 0.3, 100 trees) and LightGBM (η = 0.1, 100 trees) are more "aggressive" than CatBoost (1,000 trees, small automatic η). Friedman's results suggest smaller η with more trees usually generalises better.
4. **Tree size controls interactions.** XGBoost's depth 6, LightGBM's 31 leaves and CatBoost's depth 6 all allow high-order interactions; Friedman found *J* ≈ 6–21 leaves robust.
5. **Probability quality vs classification.** §5 shows that overfitting can damage the log-likelihood (probability quality) while the error rate keeps improving. This supports reporting **Brier score / log-loss** alongside AUC and threshold metrics in chapter 4.
6. **Robustness.** Invariance to monotone transformations explains why GBDT models do not need the standardisation of tier B.1 (it is applied for uniformity, not because trees need it).
7. **Ideas later reused.** Newton leaf values → XGBoost's second-order objective; influence trimming → LightGBM's GOSS; relative influence → feature importance (gain); partial dependence plots → standard interpretation tool.
8. **What the paper does not cover.** Row/column subsampling (stochastic gradient boosting is Friedman 2002), explicit L1/L2 penalties on leaf weights, histogram split finding, categorical encoding, or prediction shift. Cite the later papers for those.

**Suggested citation sentence (VI):**
> *Friedman (2001) xem việc xây dựng mô hình cộng tính là phép hạ gradient trong không gian hàm: ở mỗi vòng, một cây hồi quy được khớp với gradient âm (phần dư giả) của hàm mất mát và được cộng vào mô hình với một hệ số co rút (learning rate).*
