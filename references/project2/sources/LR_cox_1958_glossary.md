# Glossary — Cox (1958), *The Regression Analysis of Binary Sequences*

**Source:** D. R. Cox (1958). "The Regression Analysis of Binary Sequences". *Journal of the Royal Statistical Society, Series B (Methodological)*, 20(2), 215–242. Read before the Research Section of the RSS on 5 March 1958. File: `references/project2/refs/LR_cox_1958.pdf` (18 pages: the paper is pp. 215–231, followed by the start of the discussion, pp. 232–…).

**What the paper does:** You observe a sequence of 0/1 outcomes and suspect that the chance of a 1 depends on one or more explanatory variables. Cox proposes the **logistic law** to model that dependence. He then builds tests and estimates for the slope β, treating the intercept α as a nuisance parameter. Two cases are covered:
- The explanatory variable is fixed in advance (§4–5).
- The explanatory variable is a function of the earlier outcomes in the sequence (§6–7).

The paper is usually cited as the origin of **logistic regression** as a general regression method.

**How to read this file:** Terms are grouped by topic, in roughly the order the paper uses them.
- Each entry gives the English term, a suggested Vietnamese term, a definition in Vietnamese (định nghĩa), the formula where there is one, and where it appears (section § / journal page).
- The Vietnamese terms are suggestions for the report (`report/sections_v1`); check them against `tu_viet_tat.md` and chapter 2.
- Equation numbers (1)–(59) are Cox's own numbers.

---

## 1. Basic set-up and data

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Binary sequence** | Dãy nhị phân | Một chuỗi phép thử, mỗi kết quả được mã hóa 0 hoặc 1, ví dụ thành công/thất bại, lỗi/không lỗi, bình thường/bất thường. Đây là đối tượng nghiên cứu của bài báo. | Summary, §1 p.215 |
| **Trial** | Phép thử | Một quan sát trong chuỗi, chỉ có đúng hai kết quả có thể xảy ra. | §1 |
| **(0, 1) variate / binary response** | Biến (0,1), biến phản hồi nhị phân | Biến ngẫu nhiên $Y_i \in \{0,1\}$ ghi nhận ở phép thử *i*. Trong chấm điểm tín dụng, $Y_i = 1$ nghĩa là vỡ nợ (khách hàng "xấu"). | §2 p.216 |
| **Independent variable** (regression variable, regressor) | Biến độc lập / biến giải thích | Biến $x_i$ gắn với mỗi phép thử, có thể làm thay đổi xác suất ra kết quả 1. Cox dùng "independent variable" cho khái niệm mà ngày nay gọi là "đặc trưng" (feature) hay "biến đồng biến" (covariate). | §1, §2 |
| **Preassigned / predetermined independent variable** | Biến độc lập xác định trước | Biến $x_i$ có giá trị cố định trước khi quan sát dữ liệu (liều thuốc, thứ tự sinh, tuổi). Khi đó các $Y_i$ độc lập lẫn nhau. Được xét ở §4–5. | §2, §4, §5 |
| **Independent variable that is a function of the sequence** | Biến độc lập phụ thuộc chuỗi | Biến $x_i$ tính từ các kết quả trước đó, như $Y_{i-1}$ hoặc $Y_1+\dots+Y_{i-1}$. Khi đó các $Y_i$ **không còn độc lập**. Được xét ở §6–7. | §1, §2 |
| **Mutually (statistically) independent** | Độc lập lẫn nhau | Kết quả của các phép thử khác nhau không ảnh hưởng đến nhau. Giả định này dùng ở §2–5 và bị bỏ ở §6–7. | §2 |
| **Success / failure probability** $\theta_i$ | Xác suất thành công | $\theta_i = \Pr(Y_i = 1)$, xác suất phép thử *i* cho kết quả 1. | §2 eq. (2) |
| **Simple / multiple regression** | Hồi quy đơn / bội | Phương pháp chuẩn khi biến phản hồi *liên tục*. Mục tiêu của Cox là xây dựng phương pháp tương tự hồi quy cho biến phản hồi 0/1. | §1 |

## 2. The logistic model

| Term (EN)                              | Vietnamese                                      | Định nghĩa / công thức                                                                                                                                                                                                                                             | Where            |
| -------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- |
| **Logistic law** (logistic function)   | Luật logistic / hàm logistic                    | Quan hệ được đề xuất giữa $\theta_i$ và $x_i$: $\theta_i = \dfrac{e^{\alpha+\beta x_i}}{1+e^{\alpha+\beta x_i}}$ (2), $\quad 1-\theta_i = \dfrac{1}{1+e^{\alpha+\beta x_i}}$ (3). Giá trị luôn nằm trong [0, 1], khác với đường thẳng.                             | §2 eqs. (1)–(3)  |
| **Logit**                              | Logit (log-odds)                                | $\operatorname{logit}\theta_i \equiv \log\{\theta_i/(1-\theta_i)\} = \alpha + \beta x_i$ (1). Mô hình **tuyến tính trên thang logit**. Trong báo cáo, $F(x)$ chính là điểm log-odds (chương 2, Bảng 2.1).                                                          | §2 eq. (1)       |
| **Odds**                               | Tỷ lệ cược                                      | $\theta/(1-\theta)$. Cox không gọi tên riêng khái niệm này, nhưng logit chính là logarit của nó.                                                                                                                                                                   | eq. (1)          |
| **Why logistic and not linear**        | Lý do chọn logistic                             | Quan hệ tuyến tính không phù hợp "trừ khi trong một khoảng hẹp, vì $\theta_i$ phải nằm trong [0, 1]". Cox cần một dạng hàm (a) có ý nghĩa trực quan, (b) xấp xỉ hợp lý các quan hệ thực tế, và (c) dễ xử lý về mặt toán học.                                       | §2 p.216         |
| **α (intercept / position)**           | Hệ số chặn, tham số vị trí                      | Quyết định mức chung của xác suất. Cox coi đây là **tham số phiền nhiễu**.                                                                                                                                                                                         | §2               |
| **β (slope / regression coefficient)** | Hệ số hồi quy (độ dốc)                          | Đo mức thay đổi của logit khi $x$ tăng một đơn vị. Đây là tham số cần quan tâm.                                                                                                                                                                                    | §2               |
| **Interpretation of β**                | Ý nghĩa của β                                   | Nếu $\theta_i$ nhỏ, β là *mức tăng tương đối* của $\theta_i$ khi $x_i$ tăng một đơn vị. Nếu $1-\theta_i$ nhỏ, β là *mức giảm tương đối* của $1-\theta_i$. Nếu mọi $\theta_i$ đều gần $\theta^0$, đường cong gần tuyến tính với độ dốc $\beta\theta^0(1-\theta^0)$. | §2 p.216         |
| **Natural generalization of (1)**      | Mở rộng nhiều biến                              | Với nhiều biến, logit tuyến tính theo tất cả các biến, ví dụ $\theta_i = e^{\alpha+\beta x_i+\gamma u_i}/(1+e^{\alpha+\beta x_i+\gamma u_i})$ (39). Đây chính là hồi quy logistic nhiều biến ngày nay.                                                             | §2, §5 eq. (39)  |
| **Partial regression coefficients**    | Hệ số hồi quy riêng phần                        | β và γ trong (39): ảnh hưởng của một biến khi giữ cố định biến còn lại.                                                                                                                                                                                            | §5 p.226         |
| **"Position" and "slope"**             | Vị trí và độ dốc                                | Trong thử nghiệm liều–đáp ứng, đó là vị trí (α) và độ dốc (β) của đường cong. Cox chỉ tập trung vào độ dốc, nên ông nói bài báo ít liên quan trực tiếp đến thử nghiệm định lượng (assay).                                                                          | §1 p.216         |
| **Bioassay / quantal response**        | Thử nghiệm sinh học / phản ứng định tính        | Thí nghiệm trong đó mỗi cá thể hoặc có phản ứng hoặc không ở một liều nhất định. Đường cong logistic đã được dùng trong lĩnh vực này, đặc biệt bởi **Berkson**.                                                                                                    | §1, §2           |
| **Ogive / cumulative Gaussian form**   | Đường cong chữ S / dạng Gauss tích lũy (probit) | Đường cong liều–đáp ứng hình chữ S. Trong phần thảo luận, C. A. B. Smith nhận xét rằng thường có thể "nắn" nó về dạng logistic hoặc dạng chuẩn tích lũy (probit) bằng cách biến đổi liều.                                                                          | Discussion p.232 |
| **Linear regression of the logit**     | Hồi quy tuyến tính của logit                    | Cách Smith gọi mô hình logistic: logit là hàm tuyến tính của các biến.                                                                                                                                                                                             | Discussion p.232 |

## 3. Likelihood, sufficiency and conditioning

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Likelihood** | Hàm hợp lý | Xác suất của dữ liệu quan sát được, xem như hàm của các tham số. Với một biến: $L = \dfrac{\exp\{\sum(\alpha+\beta x_i)y_i\}}{\prod(1+e^{\alpha+\beta x_i})} = \dfrac{e^{\alpha Y+\beta X}}{\prod(1+e^{\alpha+\beta x_i})}$ (23). Đối của logarit hàm này chính là **log-loss** dùng để huấn luyện LR trong đồ án. | §4.1 eq. (23) |
| **Sufficient statistic / jointly sufficient** | Thống kê đủ / đủ đồng thời | Một đại lượng tóm tắt dữ liệu chứa toàn bộ thông tin về tham số. Ở đây $Y=\sum y_i$ (số kết quả 1) và $X=\sum y_i x_i$ là thống kê đủ đồng thời cho (α, β). Garwood (1941) đã chỉ ra điều này trước đó. | §4.1 |
| **Y** (total number of 1s) | Tổng số kết quả 1 | $Y=\sum Y_i$. Mang thông tin về α nhưng "không cho biết gì về β". | §4.1 |
| **X** (test statistic) | Thống kê X | $X=\sum Y_i x_i$, tổng các giá trị $x$ tại những phép thử cho kết quả 1. Đây là thống kê đủ cho β. | §4.1 |
| **Nuisance parameter** | Tham số phiền nhiễu | Tham số cần có trong mô hình nhưng không phải đối tượng quan tâm (ở đây là α). Cox loại bỏ nó bằng cách lấy điều kiện. | §1, §2, §4.1 |
| **Conditional inference** (arguing conditionally) | Suy diễn có điều kiện | Suy diễn về β từ phân phối của $X$ *với điều kiện* $Y = y$ quan sát được. Cách này loại bỏ α hoàn toàn: $\Pr(X=x\mid Y=y)=\dfrac{N_{x,y}e^{\beta x}}{\sum_x N_{x,y}e^{\beta x}}$ (26). | §4.1 eqs. (24)–(26) |
| **$N_{x,y}$** | Số tổ hợp | Số tập khác nhau gồm *y* giá trị lấy từ $x_1,\dots,x_n$ có tổng bằng $x$. | §4.1 |
| **Nuisance-free distribution** | Phân phối không chứa tham số phiền nhiễu | (26) chỉ phụ thuộc vào β. Đây là cơ sở của kiểm định chính xác. | §4.3 |
| **Null hypothesis of no regression** | Giả thuyết không (không có hồi quy) | $H_0: \beta = 0$: xác suất ra kết quả 1 không phụ thuộc vào $x$. | §2, §4 |
| **Non-parametric test (in Cox's sense)** | Kiểm định phi tham số | Luật logistic chỉ dùng để *chọn* thống kê kiểm định. Phân phối của thống kê dưới giả thuyết không không phụ thuộc vào giả định logistic. | §2 p.217, §4.3 |
| **Optimum / uniquely appropriate test** | Kiểm định tối ưu | Vì $X$ là thống kê đủ, kiểm định dựa trên $X$ là kiểm định tốt nhất trước **các đối thuyết logistic**, kể cả với mẫu nhỏ. | §4.3, §4.4 |
| **Locally optimum** | Tối ưu cục bộ | Tốt nhất với các đối thuyết gần giả thuyết không (β nhỏ). Dùng cho thống kê *R* ở §7. | §7 |
| **Logistic alternatives** | Đối thuyết logistic | Các đối thuyết trong đó xác suất tuân theo luật logistic với β ≠ 0. | §4.3, §4.4 |

## 4. Sampling without replacement and distribution theory

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Sampling without replacement from a finite population** | Lấy mẫu không hoàn lại từ tổng thể hữu hạn | Dưới $H_0$ và với điều kiện $Y=y$, $X$ có phân phối như tổng của *y* số rút ngẫu nhiên không hoàn lại từ $\{x_1,\dots,x_n\}$. Đây là lý do có §3. | §3 |
| **$X_{0,y}$** | | Tổng ngẫu nhiên đó khi β = 0. $X_{\beta,y}$ là cùng đại lượng khi β là giá trị thật. | §3, §4.1 |
| **Moments of $X_{0,y}$** | Các mô-men | $E = y m_1$ (4); $V = y(n-y)m_2/(n-1)$ (5); mô-men trung tâm bậc ba và bậc bốn (6)–(7). Các công thức lấy từ Irwin & Kendall (1944). | §3 |
| **Population moments $m_r$** | Mô-men tổng thể | $m_1=\sum x_i/n$ (8); $m_r=\sum(x_i-m_1)^r/n$ (9). | §3 |
| **Power sums $s_r$** | Tổng lũy thừa | $s_r=\sum x_i^r$. Khi tính toán số, $m_r$ được tính từ các tổng này, (10)–(13). | §3 |
| **Sampling fraction $f$** | Tỷ lệ lấy mẫu | $f=y/n$. Các khai triển (14)–(17) giữ $f$ cố định và cho *n* lớn. | §3 |
| **Cumulant ratios $\gamma_1, \gamma_2$** | Hệ số bất đối xứng / độ nhọn | $\gamma_1$ là cumulant bậc ba đã chuẩn hóa (độ lệch, skewness) và $\gamma_2$ là cumulant bậc bốn đã chuẩn hóa (độ nhọn vượt chuẩn, excess kurtosis). Chúng đo mức lệch khỏi phân phối chuẩn. | §3 eqs. (16)–(17) |
| **Cumulant $\kappa_r$** | Tích lũy (cumulant) | Các hệ số của hàm sinh tích lũy: trung bình, phương sai, cumulant bậc ba, v.v. | §3, §4.1 |
| **Moment generating function (m.g.f.) $M(s)$** | Hàm sinh mô-men | $E[e^{sX}]$. Kết quả then chốt: $M_{\beta,y}(s)=M_{0,y}(\beta+s)/M_{0,y}(\beta)$ (28). | §4.1 |
| **Cumulant generating function (c.g.f.) $K(s)$** | Hàm sinh tích lũy | $\log M(s)$. Kết quả then chốt: $K_{\beta,y}(s)=K_{0,y}(\beta+s)-K_{0,y}(\beta)$ (29). Do đó các cumulant với β bất kỳ là đạo hàm của hàm sinh tích lũy dưới giả thuyết không, tính tại β. Người thảo luận Smith nhấn mạnh đây là lợi ích toán học chính của dạng logistic. | §4.1 eq. (29), §5 eq. (40) |
| **Series expansion in powers of β** | Khai triển chuỗi theo β | $E(X_{\beta,y}) = m_{0,y}+\beta\sigma^2_{0,y}+\tfrac12\beta^2\sigma^3_{0,y}\gamma_{1(0,y)}+\dots$ (30); $V(X_{\beta,y})=\sigma^2_{0,y}+\beta\sigma^3_{0,y}\gamma_{1(0,y)}+\dots$ (31); tổng quát $\kappa^{(r)}_{\beta,y}=\sum_t \kappa^{(t+r)}_{0,y}\beta^t/t!$ (32). | §4.1 |
| **Probability generating function (p.g.f.)** | Hàm sinh xác suất | Dùng cho trường hợp siêu bội (18) và trường hợp $x_i=i$ (19, Haldane & Smith). | §3 |
| **Hypergeometric distribution** | Phân phối siêu bội | Phân phối của $X_{0,y}$ khi các $x_i$ chỉ gồm 0 và 1, tức là trường hợp bảng 2×2. | §3 eq. (18), §4.2 |
| **Bivariate finite population / joint cumulants $\kappa_{ij}$** | Tổng thể hữu hạn hai chiều / tích lũy hỗn hợp | Các cặp $(x_i, u_i)$ được lấy mẫu cùng nhau. $\kappa_{11}=y(n-y)m_{11}/(n-1)$ (20) và $\kappa_{21}$ (21). Cox cho biết các công thức này chưa từng được tính trước đó; chúng cần cho trường hợp hai biến ở §5. | §3 eqs. (20)–(22) |
| **Bernoulli numbers** | Số Bernoulli | Được Haldane & Smith dùng để biểu diễn các cumulant khi $x_i = i$. | §3 |

## 5. Asymptotics and approximations

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Asymptotic normality** | Tiệm cận chuẩn | Khi *n* → ∞, $X$ đã chuẩn hóa hội tụ về N(0, 1). Được chứng minh bởi Madow (1948), Hoeffding (1951) và Motoo (1957). | §3, §4.1 |
| **Central limit effect** | Hiệu ứng giới hạn trung tâm | Khi lấy mẫu không hoàn lại, $X$ tiến về phân phối chuẩn *nhanh hơn* so với mẫu có hoàn lại cỡ min(y, n−y). | §3 p.218 |
| **Convergence in distribution** | Hội tụ theo phân phối | Dạng hội tụ trong định lý giới hạn trung tâm. | §3 |
| **Combinatorial central limit theorem (Hoeffding)** | Định lý giới hạn trung tâm tổ hợp | Định lý giới hạn trung tâm cho các thống kê xây dựng từ hoán vị ngẫu nhiên. Dùng cho *R* ở §7. | §3, §7 |
| **Lyapunov's central limit theorem / condition (38)** | Định lý Lyapunov | Đảm bảo tính chuẩn cho tổng các số hạng độc lập nhưng không cùng phân phối. Điều kiện: $\lim \sum|x_i|^r/(\sum x_i^2)^{r/2}=0$ với một $r>2$ nào đó. | §4.6 eq. (38) |
| **Normal approximation** | Xấp xỉ chuẩn | Coi $X_{\beta,y}$ có phân phối chuẩn với trung bình $m_{0,y}+\beta\sigma^2_{0,y}$ và phương sai $\sigma^2_{0,y}$ (33). Hợp lệ khi $\beta\sqrt{m_2 y}=o(1)$, tức β nhỏ. | §4.1 eq. (33) |
| **o(1), O(·) notation** | Ký hiệu o nhỏ | o(1) là đại lượng tiến về 0. | §4.1, §4.6 |
| **Edgeworth expansion** | Khai triển Edgeworth | Hiệu chỉnh xấp xỉ chuẩn bằng các cumulant bậc cao. | §4.1, §4.2 |
| **Fisher–Cornish inversion** | Nghịch đảo Cornish–Fisher | Chuyển chuỗi Edgeworth thành các điểm phần trăm (ngưỡng tới hạn) đã hiệu chỉnh. | §4.3 |
| **Continuity correction** | Hiệu chỉnh liên tục | Điều chỉnh thống kê rời rạc thêm ½ trước khi dùng xấp xỉ chuẩn. | §4.2, §4.6, §7 |
| **With-replacement approximation** | Xấp xỉ có hoàn lại | Khi $y/n \ll 1$, coi $X_{0,y}$ như tổng của mẫu có hoàn lại (Haldane & Smith). | §4.1 |
| **Power function** | Hàm lực lượng | Xác suất bác bỏ $H_0$ như một hàm của tham số thật. Patnaik (1948) đưa ra xấp xỉ chuẩn cho kiểm định chính xác trong bảng 2×2. | §4.1 |
| **Standard error, asymptotic standard error** | Sai số chuẩn | Độ lệch chuẩn của một ước lượng. Với ước lượng hợp lý cực đại, nó được tính bằng công thức mẫu lớn thông thường. | §4.3 |
| **Significance level, two-sided test, tail area** | Mức ý nghĩa, kiểm định hai phía, xác suất đuôi | Các thuật ngữ kiểm định chuẩn, dùng trong Ví dụ 1–4. | §4.2, §4.6, §7 |
| **Exact test / enumeration** | Kiểm định chính xác / liệt kê | Tính chính xác phân phối dưới giả thuyết không bằng cách liệt kê mọi cách sắp xếp. Chỉ khả thi khi *n* nhỏ. | §4.3, §4.6 |

## 6. Contingency tables and classical tests as special cases

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **2×2 contingency table** | Bảng tiếp liên 2×2 | Trường hợp $x_i$ chỉ nhận hai giá trị (0/1). $X$ là ô dưới bên phải. Lấy điều kiện cho ra **kiểm định chính xác Fisher** (Fisher 1935), dựa trên phân phối siêu bội. | §1, §4.2 |
| **Odds ratio ψ** | Tỷ số odds | $\psi=e^{\beta}=\dfrac{\Pr(Y=1\mid x=1)\Pr(Y=0\mid x=0)}{\Pr(Y=0\mid x=1)\Pr(Y=1\mid x=0)}$ (34). **Vậy β là log của tỷ số odds**, cách diễn giải chuẩn cho hệ số LR. | §4.2 eq. (34) |
| **Test of association** | Kiểm định tính liên kết | Kiểm định chính xác cho $\beta=0$ trong bảng 2×2. | §4.2 |
| **Confidence limits / interval** | Giới hạn / khoảng tin cậy | Ví dụ 1 so sánh giới hạn 95% của Cornfield cho ψ, (0,030; 0,623), với xấp xỉ chuẩn của Cox, (0,045; 0,613). | §4.2 |
| **Retrospective study** (case–control) | Nghiên cứu hồi cứu (bệnh–chứng) | Dữ liệu của Cornfield (1956) về bác sĩ hút thuốc / ung thư phổi trong Bảng 1. | §4.2 |
| **2×k contingency table** | Bảng tiếp liên 2×k | $x$ nhận *k* giá trị, mỗi giá trị có nhiều phép thử, tức là bảng 2×k có điểm số trên các cột. | §1, §4.3 |
| **Test for trend; single degree of freedom for linear regression from χ²** | Kiểm định xu hướng (Cochran–Armitage) | Tách một bậc tự do từ $\chi^2_{k-1}$ cho xu hướng tuyến tính (Yates 1948; Armitage 1955; Cochran 1955). Cox chỉ ra cách này gần như trùng với kiểm định của ông, và phiên bản của ông là tối ưu trước các đối thuyết logistic. | §1, §4.3 |
| **Degree of freedom** | Bậc tự do | Khái niệm χ² thông thường. | §1, §4.3 |
| **Least squares / formal linear regression on (0, 1) data** | Bình phương tối thiểu | Hồi quy kết quả 0/1 theo $x$ bằng bình phương tối thiểu thông thường (OLS) cho ra kiểm định Armitage và ước lượng xấp xỉ bậc nhất của β. | §4.3, §5 |
| **Score (column score)** | Điểm số cột | Các giá trị số gán cho *k* cột. | §4.3 |
| **Discrete rectangular population** | Phân phối đều rời rạc | Ví dụ 2: $x = -3,\dots,3$ với trọng số bằng nhau. | §4.3 |

## 7. Estimation methods for β

Cox lists **four ways** to get an interval estimate of β (§4.3).

| Term (EN) | Vietnamese | Định nghĩa | Where |
|---|---|---|---|
| **1. Exact conditional distribution** | Phân phối có điều kiện chính xác | Dùng trực tiếp (26). Không chứa tham số phiền nhiễu nhưng không khả thi, trừ khi số giá trị cực trị có thể của $X$ là ít. | §4.3 |
| **2. Cumulant expansions** | Khai triển tích lũy | Dùng (29)–(32). Tốt khi β nhỏ và xác suất nằm trong khoảng ¼ đến ¾, "khoảng mà đường cong logistic gần như tuyến tính". | §4.3 |
| **3. Maximum likelihood (ML)** | Ước lượng hợp lý cực đại | Cực đại hóa (23) **bằng phép lặp**; Berkson (1957) cung cấp bảng tra. Đây là cách LR được huấn luyện ngày nay: `LogisticRegression` của scikit-learn cực tiểu hóa log-loss, mặc định có thêm phạt L2 (C = 1). | §4.3, §5, §7 |
| **Iterative multiple regression calculation** | Tính lặp dạng hồi quy bội | Mỗi bước ML giải một hệ bình phương tối thiểu có trọng số với (số biến + 1) phương trình tuyến tính. Đây là tiền thân của IRLS / Newton–Raphson. | §7 p.231 |
| **4. Minimum logit χ² (Berkson 1955)** | Phương pháp logit χ² cực tiểu | **Không lặp**: bình phương tối thiểu có trọng số trên các tỷ lệ đã biến đổi logit từ dữ liệu. Cần dữ liệu đã gộp nhóm, với ít nhóm toàn 0 hoặc toàn 1. Cox khuyên dùng nó làm giá trị khởi đầu cho ML. | §4.3, §5, §7 |
| **Empirical weights** | Trọng số thực nghiệm | Trọng số ước lượng từ dữ liệu trong phương pháp logit χ² cực tiểu. | §4.3 |
| **Asymptotically equivalent** | Tương đương tiệm cận | Phương pháp 3 và 4 cho cùng kết quả khi mẫu lớn. | §4.3 |
| **Mean square error (MSE) of a point estimate** | Sai số bình phương trung bình | Berkson chỉ ra rằng trong một số trường hợp logit χ² cực tiểu có MSE nhỏ hơn ML; Silverstone (1957) tranh luận về điểm này. | §4.3 |
| **Point estimate vs interval estimate** | Ước lượng điểm / khoảng | Thuật ngữ chuẩn: ước lượng bằng một giá trị so với ước lượng bằng một khoảng. | §4.3 |
| **Unbiased estimate (first approximation)** | Ước lượng không chệch | Ở §5, bỏ các số hạng bậc hai trong (41)–(42) cho các ước lượng hồi quy tuyến tính $B_1=\kappa^{20}X'+\kappa^{11}U'$ và $C_1=\kappa^{11}X'+\kappa^{02}U'$ (43). Hiệu chỉnh các số hạng bậc hai cho ra (44). | §5 eqs. (41)–(44) |
| **Inverse covariance matrix $\kappa^{ij}$** | Ma trận hiệp phương sai nghịch đảo | Nghịch đảo của $\begin{pmatrix}\kappa_{20}&\kappa_{11}\\\kappa_{11}&\kappa_{02}\end{pmatrix}$, với Δ là định thức của ma trận. | §5 |
| **Cox's recommendation** | Khuyến nghị | Nếu **kiểm định ý nghĩa** là quan trọng nhất, dùng phương pháp có điều kiện. Nếu **ước lượng** là quan trọng nhất và mẫu không quá nhỏ, dùng ML hoặc logit χ² cực tiểu. | §4.3 p.224 |

## 8. Special tests derived from the framework

| Term (EN) | Vietnamese | Định nghĩa | Where |
|---|---|---|---|
| **Serial order test** | Kiểm định thứ tự | Với $x_i = i$, $X$ là **tổng hạng** của các phép thử cho kết quả 1. Kiểm định này phát hiện xu hướng theo thời gian. Do Haldane & Smith (1948) đề xuất, kể cả khi thiếu một số số thứ tự. | §4.4 |
| **Rank sum / rank numbers** | Tổng hạng | Tổng các vị trí của những kết quả 1. | §4.4, §4.5 |
| **Test of randomness** | Kiểm định tính ngẫu nhiên | Kiểm tra xem các kết quả 1 có phân bố ngẫu nhiên trong chuỗi hay không. | §4.4 |
| **Wilcoxon's two-sample test** (Mann–Whitney) | Kiểm định Wilcoxon hai mẫu | Kiểm định thứ tự *chính là* kiểm định tổng hạng Wilcoxon áp dụng cho hạng của nhóm 0 so với nhóm 1. Do đó **kiểm định Wilcoxon là tối ưu trước các đối thuyết logistic** trong bối cảnh này. (Đồ án dùng kiểm định liên quan là Wilcoxon *hạng có dấu* kèm hiệu chỉnh Holm để so sánh mô hình.) | §4.4; discussion p.232 |
| **Comparison of locations of two samples** | So sánh vị trí hai mẫu | Ứng dụng thông thường của kiểm định Wilcoxon. Cox đặt câu hỏi liệu nó có tối ưu cho bài toán vị trí nào không. | §4.4 |
| **Point process in continuous time** | Quá trình điểm thời gian liên tục | Nếu xác suất ra 1 luôn nhỏ và *x* là thời gian, chuỗi tiến gần tới một quá trình điểm. Khi đó thống kê là tổng các thời điểm xảy ra sự kiện (Cox 1955). | §4.4 |
| **Combining several series** | Kết hợp nhiều chuỗi | Nhiều chuỗi độc lập có chung β nhưng mỗi chuỗi có $\alpha^{(j)}$ riêng (ví dụ 51 gia đình). Lấy điều kiện theo tổng riêng của từng chuỗi: hàm hợp lý là (35), và các thống kê đủ đồng thời là $Y^{(1)},\dots,Y^{(k)}$ và $X=\sum X^{(j)}$. | §4.5 eq. (35) |
| **Stratification by series (one α per series)** | Phân tầng | Cách lấy điều kiện theo từng gia đình của Haldane & Smith là "tối ưu về lý thuyết khi tham số phiền nhiễu α khác nhau giữa các gia đình". Đây là tiền thân của hồi quy logistic có điều kiện cho dữ liệu phân tầng. | §4.5 |
| **Phenylketonuria example** | Ví dụ bệnh phenylketon niệu | Dữ liệu của Haldane & Smith: trẻ em trong 51 gia đình được phân loại bình thường hoặc mắc phenylketon niệu, với *x* là thứ tự sinh. | §4.5 |
| **Birth-order effect** | Hiệu ứng thứ tự sinh | Ví dụ ở §1: nguy cơ mắc một dị tật di truyền có phụ thuộc vào thứ tự sinh của trẻ hay không, sau khi đã loại trừ ảnh hưởng của tuổi mẹ. | §1 |
| **Test with α known** | Kiểm định khi biết α | Khi $H_0$ cố định một xác suất đã biết $\theta_0$, dùng phân phối *không điều kiện* của $X$: $E(X)=\theta_0\sum x_i$ (36) và $V(X)=\theta_0(1-\theta_0)\sum x_i^2$ (37). Với mẫu lớn, ước lượng $\theta_0$ từ dữ liệu cho kết quả tương đương kiểm định có điều kiện. | §4.6 |
| **Test for trend from a known value** | Kiểm định xu hướng so với giá trị đã biết | Ví dụ 3 (kết quả F S F S S F S S S): tổng hạng là 35, so với trung bình 22½ và phương sai 285/4. Xấp xỉ chuẩn cho mức ý nghĩa hai phía khoảng 15½%; giá trị chính xác là 41/256 = 0,160. | §4.6 |

## 9. Dependence on the past: Markov chains and runs

| Term (EN) | Vietnamese | Định nghĩa | Where |
|---|---|---|---|
| **Regression on the preceding trial** | Hồi quy theo phép thử trước | Xác suất ra 1 phụ thuộc vào $Y_{i-1}$: $\Pr(1\mid 1)=e^{\alpha+\beta}/(1+e^{\alpha+\beta})$ (45) và $\Pr(1\mid 0)=e^{\alpha}/(1+e^{\alpha})$ (46). | §6 |
| **Simple Markov chain / transition probabilities** | Xích Markov đơn / xác suất chuyển | (45)–(46) "chỉ là một cách viết đặc biệt cho xác suất chuyển" của xích Markov hai trạng thái. | §6 |
| **Conditional likelihood given $Y_1$** | Hợp lý có điều kiện theo $Y_1$ | Hàm hợp lý (47) được lấy với điều kiện kết quả đầu tiên, vì phân phối của $Y_1$ không được xác định. | §6 eq. (47) |
| **End effects** | Hiệu ứng biên | Quan sát đầu và cuối làm phức tạp tính đủ: $Y_1$ và $Y_n$ phải được thêm vào cùng với $\sum Y_i$ và $\sum Y_iY_{i-1}$. | §6 |
| **Circular set-up / circular definition** | Thiết lập vòng | Đặt $Y_0 = Y_n$ để loại hiệu ứng biên. Khi đó $Y$ và $X=\sum Y_iY_{i-1}$ là thống kê đủ đồng thời (48)–(49). Cox ưu tiên thống kê thật, không vòng, trừ khi có lý do vật lý cho cách định nghĩa vòng. | §6 |
| **Run / number of runs W** | Chuỗi liên tiếp / số chuỗi | Một chuỗi (run) là khối dài nhất gồm các kết quả giống nhau liên tiếp. Trong dãy vòng, $w = 2(y-x)$. Phân phối dưới giả thuyết không là số chuỗi trong một cách sắp xếp ngẫu nhiên *y* số 1 và *n − y* số 0 (50). | §6 |
| **Wald–Wolfowitz runs test** | Kiểm định chuỗi Wald–Wolfowitz | *W* về hình thức trùng với thống kê chuỗi hai mẫu của họ. Kiểm định này vững trước mọi đối thuyết nhưng kém hiệu quả trước các đối thuyết cụ thể. $E(W)\sim 2y(n-y)/n$ (51) và $V(W)\sim 4y^2(n-y)^2/n^3$ (52). Với β nhỏ, trung bình là $2y(n-y)/n - \tfrac12\beta V(W)$ (53). | §6 |
| **Consistent test** | Kiểm định vững | Lực lượng tiến về 1 khi *n* → ∞ với mọi đối thuyết cố định. | §6 |
| **Efficiency (of a test)** | Hiệu quả | Mức lực lượng của một kiểm định trước một đối thuyết cụ thể, so với kiểm định tốt nhất. | §4.3, §6 |
| **Serial correlation coefficient** | Hệ số tự tương quan | Khái niệm tương tự trong chuỗi thời gian, cũng gặp vấn đề định nghĩa vòng như trên. | §6 |
| **Swed–Eisenhart tables** | Bảng Swed–Eisenhart | Bảng tra cho kiểm định chuỗi với mẫu nhỏ. Cách khác: Stevens đưa về bảng 2×2 bằng $W_1$, số chuỗi gồm toàn số 1. | §6 |
| **Markov chains with more than two states** | Xích Markov nhiều trạng thái | Các tác giả được trích dẫn: Bartlett 1951, Hoel 1954, Good 1955, Anderson & Goodman 1957. | §6 |

## 10. Cumulative score and learning models

| Term (EN) | Vietnamese | Định nghĩa | Where |
|---|---|---|---|
| **Cumulative score as independent variable** | Điểm tích lũy làm biến độc lập | Xác suất ra 1 phụ thuộc vào $v$, số kết quả 1 tính đến thời điểm đó: $e^{\alpha+\beta v}/(1+e^{\alpha+\beta v})$ (54). | §7 |
| **Gaps $r_0, r_1, \dots, r_y$** | Khoảng cách giữa các kết quả 1 | $r_0$ là vị trí của số 1 đầu tiên; $r_k$ là khoảng cách từ số 1 thứ k đến số 1 kế tiếp; $r_y = n - r_0 - \dots - r_{y-1}$. Để có thống kê đủ cần toàn bộ dãy khoảng cách (55). | §7 eq. (55) |
| **Statistic R** | Thống kê R | $R=\sum k r_k$, tối ưu cục bộ khi β nhỏ. *R* lớn ứng với β âm. Dưới $H_0$, $E_p(R)=\tfrac12(y'-1)n$ (56) và $V_p(R)=\dfrac{y'(y'+1)}{12}\sum(r_i-\bar r)^2$ (57). | §7 |
| **Permutation distribution** | Phân phối hoán vị | Dưới $H_0$, mọi thứ tự của các $r$ khác 0 đều đồng khả năng. *R* được so với phân phối hoán vị đó. | §7 |
| **Learning experiment** | Thí nghiệm học tập | Ví dụ 4: 20 phép thử với R = 72, so với kỳ vọng 90 và phương sai 110. Độ lệch chuẩn hóa là 1,67, có ý nghĩa ở mức khoảng 9½%. | §7 |
| **Stochastic models for learning** | Mô hình học ngẫu nhiên | Bush & Mosteller (1955), Cane, Audley. Cox lập luận rằng để *rút gọn dữ liệu*, một mô hình logistic đơn giản, linh hoạt tốt hơn một mô hình cơ chế phức tạp. | §7 |
| **Audley's model** | Mô hình Audley | $(\alpha+\beta v)/(1+\beta v)$ (59). Mô hình yêu cầu β ≥ 0, trong khi dạng logistic không cần điều kiện này. | §7 |
| **Failure rates under modification** | Tỷ lệ hỏng khi cải tiến | Ứng dụng công nghiệp: một cơ cấu được cải tiến sau mỗi lần hỏng, và −β là tốc độ giảm xác suất hỏng. | §7 |
| **General logistic model with several history variables** | Mô hình logistic tổng quát | Logit tuyến tính theo (i) số phản hồi đúng (hoặc được thưởng) trước đó, (ii) số phản hồi sai (bị phạt), và (iii) kết quả phép thử trước. Được ước lượng bằng ML với bảng tra của Berkson. | §7 p.231 |
| **Grouping of independent variables** | Gộp nhóm biến | Cần cho phương pháp logit χ² cực tiểu, để có ít ô toàn 0 hoặc toàn 1. Cùng ý tưởng với chia khoảng (binning) hay WoE trong chấm điểm tín dụng. | §7 |
| **Data reduction to a manageable and revealing form** | Rút gọn dữ liệu | Mục đích Cox nêu ra khi dùng mô hình logistic: một bản tóm tắt đơn giản, dễ diễn giải, chứ không phải mô tả cơ chế "thật". | §7 p.230 |

---

## 11. Notation summary

| Symbol | Ý nghĩa |
|---|---|
| $Y_i$ | Kết quả 0/1 của phép thử *i* |
| $x_i,\ u_i$ | Biến độc lập (biến giải thích) |
| $\theta_i$ | $\Pr(Y_i = 1)$ |
| α, β, γ | Hệ số chặn (tham số phiền nhiễu), độ dốc theo *x*, độ dốc theo *u* |
| $Y=\sum Y_i$ | Tổng số kết quả 1 (giá trị quan sát *y*) |
| $X=\sum x_iY_i$ | Thống kê đủ cho β |
| $X_{0,y},\ X_{\beta,y}$ | *X* với điều kiện *Y = y*, khi β = 0 / khi β là giá trị thật |
| $m_{0,y},\ \sigma^2_{0,y}$ | Trung bình và phương sai của *X* dưới giả thuyết không |
| $m_r,\ s_r$ | Mô-men trung tâm tổng thể; tổng lũy thừa |
| $f = y/n$ | Tỷ lệ lấy mẫu |
| $\gamma_1,\ \gamma_2$ | Hệ số bất đối xứng và độ nhọn |
| $M(s),\ K(s)$ | Hàm sinh mô-men / hàm sinh tích lũy |
| $\kappa_{ij},\ \kappa^{ij}$ | Tích lũy hỗn hợp và ma trận nghịch đảo |
| $N_{x,y}$ | Số tập con *y* phần tử của các $x_i$ có tổng bằng *x* |
| ψ | $e^\beta$, tỷ số odds |
| *W*, $W_1$ | Tổng số chuỗi; số chuỗi gồm toàn số 1 |
| $r_k$, *R* | Khoảng cách giữa các số 1; $R=\sum k r_k$ |

## 12. Worked examples in the paper

| # | Section | Dữ liệu | Điều được minh họa |
|---|---|---|---|
| 1 | §4.2 | Cornfield (1956): người hút thuốc / không hút thuốc, nhóm chứng / bệnh nhân ung thư phổi (32/60/11/3, n = 106) | Giới hạn tin cậy cho tỷ số odds. Xấp xỉ chuẩn cho kết quả tốt bất ngờ nhưng có thể cho giá trị vô lý khi một ô có tần số nhỏ, và thêm số hạng bậc cao không giúp ích. |
| 2 | §4.3 | x = −3…3, mỗi giá trị 8 phép thử, 23 kết quả 1 trên 56 | Kiểm định xu hướng 2×k với hiệu chỉnh cumulant; các khai triển chỉ dùng được khi β nhỏ. |
| 3 | §4.6 | 9 phép thử học tập F S F S S F S S S | Kiểm định khi biết α; xấp xỉ chuẩn (~15½%) so với giá trị chính xác (16,0%). |
| 4 | §7 | 20 phép thử học tập | Kiểm định điểm tích lũy bằng *R* (mức ý nghĩa ≈ 9½%). |

## 13. People and works cited (for follow-up)

**Logistic and estimation:**
- Berkson 1955, 1957: maximum likelihood vs minimum χ² for the logistic; ML tables.
- Garwood 1941: ML for dosage–mortality curves; sufficiency of (X, Y).
- Silverstone 1957: estimating the logistic curve.

**Contingency tables:**
- Fisher 1935: exact test for the 2×2 table.
- Cornfield 1956: retrospective studies and odds-ratio limits.
- Patnaik 1948: power of the 2×2 test.
- Stevens 1939, 1951.
- Yates 1948, Armitage 1955, Cochran 1955: trend in 2×k tables.

**Finite-population theory:**
- Irwin & Kendall 1944.
- Madow 1948, Hoeffding 1951, Motoo 1957: limit theorems.

**Rank and run tests:**
- Haldane & Smith 1948: birth-order / rank-sum test.
- Haldane 1940.
- Bennett 1956.
- Wald & Wolfowitz 1940.
- Swed & Eisenhart 1943.
- Siegel 1957, Birnbaum 1956: Wilcoxon / Mann–Whitney.

**Markov chains:** Bartlett 1951, Hoel 1954, Good 1955, Anderson & Goodman 1957.

**Learning models:** Bush & Mosteller 1955, Cane 1956, Audley 1956–57, Audley & Jonckheere 1956.

**Other:**
- Cox 1955: series of events.
- Pearson & Merrington 1951: percentage points of Pearson curves.

---

## 14. Relevance to this project (credit scoring with LR)

These points connect the paper to the report's chapter 2 (§2.3 Logistic Regression).

1. **Model form.** The project's LR is Cox's equation (1) with many variables: $\operatorname{logit}\Pr(\text{default})=\alpha+\beta^\top x$. The sigmoid in Hình (lr_sigmoid) is equations (2)–(3).
2. **Why logistic.** The output must be a probability in [0, 1]. The logit is linear and easy to interpret. This is Cox's argument from §2, and it is the argument the report can cite.
3. **Interpreting coefficients.** $e^{\beta_j}$ is the **odds ratio** for a one-unit increase in feature *j* (eq. 34). This explains why LR is the interpretable baseline in credit scoring.
4. **Fitting.** Cox's method 3 (maximum likelihood, iterative weighted least squares) is what scikit-learn does. Minimizing log-loss is maximizing likelihood (23). The project's default adds an L2 penalty, which Cox does not have.
5. **Grouping variables.** Cox's grouping for minimum logit χ² is a historical parallel to binning and WoE in scorecards.
6. **Wilcoxon link.** Cox shows that the Wilcoxon rank-sum test is optimal against logistic alternatives in the serial-order setting. This is a different use from the project's Wilcoxon signed-rank test with Holm correction (Demšar 2006), so do not conflate the two.
7. **What the paper does not cover.** It does not discuss regularization, classification thresholds, AUC, calibration, or high-dimensional data. Do not cite it for those.

**Suggested citation sentence (VI):**
> *Hồi quy logistic được Cox (1958) đề xuất như một mô hình hồi quy cho biến phản hồi nhị phân, trong đó logit của xác suất là hàm tuyến tính của các biến giải thích.*
