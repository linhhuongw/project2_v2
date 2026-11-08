# Glossary — Cover & Hart (1967), *Nearest Neighbor Pattern Classification*

**Source:** T. M. Cover & P. E. Hart (1967). "Nearest Neighbor Pattern Classification". *IEEE Transactions on Information Theory*, IT-13(1), 21–27. Stanford University / Stanford Research Institute; received February 1966. File: `references/project2/refs/knn_cover_1967.pdf` (7 pages; page 21 opens with the end of another article and page 27 continues into an unrelated article by J. L. Brown).

**What the paper does:** The **nearest neighbor (NN) rule** labels a new point with the class of the closest labelled point. It uses no knowledge of the underlying distributions. Cover & Hart compare it with the best possible rule, the **Bayes rule**, which knows the distributions. Main results:
- In the large-sample limit, the NN error $R$ satisfies $R^* \le R \le R^*(2 - MR^*/(M-1))$ for *M* classes, so it is **at most twice the Bayes error** $R^*$, and these bounds are as tight as possible.
- "Half the classification information in an infinite sample set is contained in the nearest neighbor."
- The 1-NN rule is **admissible** among k-NN rules: no k-NN rule (k ≠ 1) is better for every distribution.
- For k-NN, the large-sample error decreases with *k* towards $R^*$.

The paper is the standard theoretical reference for **k-nearest neighbors (KNN)**. The rule itself goes back to Fix & Hodges (1951).

**How to read this file:** Terms are grouped by topic, in roughly the order the paper uses them.
- Each entry gives the English term, a suggested Vietnamese term, a definition in Vietnamese (định nghĩa), the formula where there is one, and where it appears (section / journal page).
- The Vietnamese terms are suggestions for the report (`report/sections_v1`); check them against `tu_viet_tat.md` and chapter 2.
- Equation numbers (1)–(49) are the paper's own numbers. Cover & Hart write θ for the class label, η for class probabilities and $x'_n$ for the nearest neighbour.

---

## 1. The classification problem

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Classification problem** | Bài toán phân loại | Gán một quan sát *x* vào một trong *M* lớp (category), dựa trên các mẫu đã biết lớp. | §I p.21 |
| **Two extremes of knowledge** | Hai thái cực hiểu biết | (1) Biết đầy đủ phân phối chung của *x* và lớp θ: phân tích Bayes cho quy tắc tối ưu. (2) Không biết gì về phân phối, chỉ có *n* mẫu đã gán nhãn: bài toán **phi tham số**, không có quy tắc tối ưu cho mọi phân phối. | §I p.21 |
| **Category θ / measurement x** | Lớp θ / phép đo *x* | θ ∈ {1, …, *M*} là chỉ số lớp của cá thể; *x* là kết quả các phép đo trên cá thể đó (vectơ đặc trưng). | §II p.22 |
| **Sample set (classified samples)** | Tập mẫu đã phân loại | *n* cặp $(x_1,\theta_1),\dots,(x_n,\theta_n)$ đã biết nhãn, chính là **dữ liệu huấn luyện**. | §I, §II |
| **Independent identically distributed (i.i.d.)** | Độc lập cùng phân phối | Các cặp $(x_i,\theta_i)$ và cặp mới (x, θ) độc lập và có cùng phân phối; bên trong mỗi cặp, *x* và θ thì phụ thuộc nhau. Đây là giả định của mọi kết quả trong bài. | §I, §VI |
| **Metric space / metric d** | Không gian metric / hàm khoảng cách *d* | Không gian của *x* có định nghĩa khoảng cách *d*. Kết quả về giới hạn sai số **không phụ thuộc metric** được chọn. | §II, §V |
| **Separable metric space** | Không gian metric khả ly | Không gian có một tập con đếm được trù mật; điều kiện kỹ thuật để chứng minh láng giềng gần nhất hội tụ về *x*. Mọi không gian Euclid đều thỏa mãn. | §V p.23 |
| **Heuristic: close points have the same class** | Giả định điểm gần nhau cùng lớp | Các quan sát gần nhau (theo metric phù hợp) có cùng lớp, hoặc ít nhất có phân phối hậu nghiệm gần giống nhau. Đây là trực giác đằng sau NN. | §I p.21 |
| **Nonparametric / distribution-free** | Phi tham số / không phụ thuộc phân phối | Quy tắc NN không cần giả định gì về phân phối nền của dữ liệu. | §I, §IV |

## 2. The nearest neighbor rule

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Nearest neighbor $x'_n$** | Láng giềng gần nhất | Điểm $x'_n\in\{x_1,\dots,x_n\}$ thỏa $\min_i d(x_i,x)=d(x'_n,x)$ (1). | §II eq. (1) |
| **Nearest neighbor (NN) rule** | Luật láng giềng gần nhất | Gán *x* vào lớp $\theta'_n$ của láng giềng gần nhất. Mắc lỗi khi $\theta'_n\ne\theta$. Chỉ dùng nhãn của một điểm; *n* − 1 nhãn còn lại bị bỏ qua. | §II p.22 |
| **Ties** | Trường hợp hòa | Nếu có nhiều láng giềng cách đều, chọn lớp phổ biến nhất trong số đó. | §II footnote 1 |
| **k-NN rule (majority vote)** | Luật k láng giềng gần nhất | Gán *x* vào lớp chiếm đa số trong *k* láng giềng gần nhất. Đây là **KNN** dùng trong đồ án. | §I, §III, §VIII |
| **Choice of k** | Chọn *k* | *k* lớn giảm xác suất ra quyết định khác Bayes (bỏ phiếu ổn định hơn); *k* nhỏ (so với *n*) giữ các láng giềng đủ gần *x* để ước lượng đúng xác suất hậu nghiệm. Đây là một dạng đánh đổi bias – variance. | §III p.22 |
| **$k_n$-NN rule, consistency (Fix & Hodges)** | Luật $k_n$-NN, tính vững | Fix & Hodges (1951) chứng minh luật $k_n$-NN **vững** (sai số tiến về sai số Bayes) khi $k_n\to\infty$ và $k_n/n\to0$. | §I p.22 |
| **Empirical Bayes rule / proximity algorithm** | Luật Bayes thực nghiệm / thuật toán lân cận | Các tên gọi khác của NN trong công trình trước đó (Johns; Sebestyen). | §I p.22 |
| **Density estimation by k-NN** | Ước lượng mật độ bằng k-NN | Loftsgaarden & Quesenberry (1965): một biến thể của $k_n$-NN cho ước lượng vững của hàm mật độ. | §I p.22 |

## 3. Bayes procedure (the benchmark)

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Class-conditional density $f_i(x)$** | Mật độ có điều kiện theo lớp | Mật độ của *x* khi cá thể thuộc lớp *i*. | §IV p.23 |
| **Prior probabilities $\eta_i$** | Xác suất tiên nghiệm | $\eta_i\ge0$, $\sum\eta_i=1$: tỷ lệ các lớp. | §IV |
| **Posterior probability $\hat\eta_i(x)$** | Xác suất hậu nghiệm | Theo định lý Bayes: $\hat\eta_i(x)=\eta_i f_i(x)/\sum_j\eta_j f_j(x)$ (4). Quan sát *x* biến vectơ tiên nghiệm η thành vectơ hậu nghiệm $\hat\eta(x)$. | §IV eq. (4) |
| **Loss function L(i, j)** | Hàm mất mát | Thiệt hại khi gán một cá thể lớp *i* vào lớp *j*. | §IV |
| **0–1 loss** | Mất mát 0–1 | $L(i,j)=0$ nếu $i=j$, $=1$ nếu $i\ne j$ (12): đếm một lỗi cho mỗi lần phân loại sai, nên rủi ro bằng **xác suất sai**. | §VI eq. (12) |
| **Conditional loss $r_j(x)$** | Mất mát có điều kiện | Mất mát kỳ vọng khi gán *x* vào lớp *j*: $r_j(x)=\sum_i\hat\eta_i(x)L(i,j)$ (5). | §IV eq. (5) |
| **Bayes decision rule δ\*** | Quy tắc quyết định Bayes | Chọn lớp *j* có $r_j(x)$ nhỏ nhất. Với mất mát 0–1, đó là lớp có **xác suất hậu nghiệm lớn nhất**. | §IV |
| **Conditional Bayes risk $r^*(x)$** | Rủi ro Bayes có điều kiện | $r^*(x)=\min_j r_j(x)$ (6). Hai lớp: $r^*(x)=\min\{\hat\eta_1(x),1-\hat\eta_1(x)\}$ (20). | §IV eq. (6), §VI |
| **Bayes risk R\*** | Rủi ro Bayes (sai số Bayes) | $R^*=E[r^*(x)]$ (7): sai số nhỏ nhất mà **bất kỳ** quy tắc nào có thể đạt được, kể cả khi biết phân phối. Là "giới hạn của sự hoàn hảo" để so sánh. | §IV eq. (7) |
| **Compound density f(x)** | Mật độ hỗn hợp | $f(x)=\sum_i\eta_i f_i(x)$: mật độ của *x* khi không biết lớp. | §IV |

## 4. Convergence and NN risk

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Convergence of the nearest neighbor (Lemma)** | Sự hội tụ của láng giềng gần nhất | Nếu *x*, $x_1,x_2,\dots$ i.i.d. trong không gian metric khả ly thì $x'_n\to x$ với xác suất 1 khi *n* → ∞. Đúng với mọi độ đo xác suất trên không gian Euclid. | §V p.23 |
| **With probability one** | Với xác suất 1 (hầu chắc chắn) | Dạng hội tụ mạnh dùng trong chứng minh. | §V |
| **Sphere $S_x(r)$** | Hình cầu bán kính *r* | $\{z: d(z,x)\le r\}$; dùng để chứng minh bổ đề hội tụ. | §V |
| **n-sample NN risk R(n)** | Rủi ro NN với *n* mẫu | $R(n)=E[L(\theta,\theta'_n)]$ (10). | §VI eq. (10) |
| **Large-sample NN risk R** | Rủi ro NN mẫu lớn | $R=\lim_{n\to\infty}R(n)$ (11). | §VI eq. (11) |
| **Conditional NN risk $r(x,x'_n)$** | Rủi ro NN có điều kiện | Hai lớp: $r(x,x'_n)=\hat\eta_1(x)\hat\eta_2(x'_n)+\hat\eta_2(x)\hat\eta_1(x'_n)$ (15); hội tụ về $r(x)=2\hat\eta_1(x)\hat\eta_2(x)=2r^*(x)(1-r^*(x))$ (19), (21). | §VI |
| **Conditional independence of θ and θ'** | Độc lập có điều kiện của hai nhãn | Khi biết *x* và $x'_n$, nhãn của chúng độc lập; dùng để viết (14). | §VI eq. (14) |
| **Continuity point / point of nonzero measure** | Điểm liên tục / điểm có độ đo dương | Điều kiện của định lý: *x* là điểm liên tục của các mật độ, hoặc có xác suất dương. Thỏa với mật độ liên tục từng khúc và phân phối rời rạc. | §VI Theorem |
| **Dominated convergence theorem** | Định lý hội tụ bị chặn | Cho phép đổi thứ tự giới hạn và kỳ vọng vì mất mát bị chặn bởi 1. | §VI eq. (23) |

## 5. Main results: error bounds

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Two-class NN bound (Theorem 1)** | Cận sai số NN hai lớp | $R^*\le R\le 2R^*(1-R^*)$ (13); các cận này **chặt nhất có thể**. | §VI eq. (13) |
| **Exact two-class identity** | Đẳng thức hai lớp | $R=2R^*(1-R^*)-2\,\mathrm{Var}\,r^*(x)$ (25). Cận trên đạt khi $r^*(x)$ không đổi theo *x*. | §VI eq. (25) |
| **M-class NN bound (Theorem 2)** | Cận sai số NN *M* lớp | $R^*\le R\le R^*\big(2-\tfrac{M}{M-1}R^*\big)$ (29), (38). | §VI eq. (29) |
| **"At most twice the Bayes error"** | Không quá hai lần sai số Bayes | Hệ quả: $R\le 2R^*$, nên mọi quy tắc khác dùng tập mẫu vô hạn chỉ có thể giảm sai số **tối đa một nửa** so với NN. | Abstract, §IX |
| **Half the classification information** | Một nửa thông tin phân loại | Cách diễn đạt của tác giả: một nửa thông tin phân loại trong tập mẫu vô hạn nằm ở láng giềng gần nhất. | Abstract, §IX |
| **Extreme cases** | Trường hợp biên | $R^*=0\Leftrightarrow R=0$ (chắc chắn hoàn toàn); $R^*=\tfrac12\Leftrightarrow R=\tfrac12$ (bất định hoàn toàn): khi đó NN đạt đúng sai số Bayes. | §VI p.24 |
| **No-information experiment** | Thí nghiệm không thông tin | Mọi $f_i$ bằng nhau; cận trên của định lý *M* lớp đạt được trong trường hợp này. | §VI p.26 |
| **Cauchy–Schwarz inequality** | Bất đẳng thức Cauchy–Schwarz | Dùng để chứng minh cận *M* lớp, (33). | §VI |
| **Large-sample vs finite-sample** | Mẫu lớn / mẫu hữu hạn | Các cận là kết quả **tiệm cận** (*n* → ∞); với *n* hữu hạn chỉ có ví dụ cụ thể ở §VII. | §VI, §VII |

## 6. Admissibility, example and k-NN

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Admissible / undominated rule** | Quy tắc chấp nhận được / không bị trội | Không có quy tắc nào khác trong cùng lớp có sai số thấp hơn **với mọi phân phối**. 1-NN là chấp nhận được trong họ k-NN. | §III |
| **Admissibility example** | Ví dụ về tính chấp nhận được | Hai lớp đều nhau, mật độ đều trên hai đĩa đơn vị tâm (−3, 0) và (3, 0): 1-NN sai với xác suất $(1/2)^n$, mọi k-NN với *k* = 2$k_0$ + 1 > 1 sai nhiều hơn. Tổng quát, 1-NN tốt hơn mọi k-NN khi giá (support) của các mật độ lớp tách rời nhau như trong ví dụ này. | §III, Fig. 1 |
| **Triangular-density example** | Ví dụ mật độ tam giác | Hai lớp có mật độ tam giác trên [0, 1], tiên nghiệm bằng nhau: $R(n)=\tfrac13+\tfrac{1}{(n+1)(n+2)}$ (41), $R=\tfrac13$ (42), $R^*=\tfrac14$ (43), nên $\tfrac14<\tfrac13\le 2\cdot\tfrac14\cdot\tfrac34$. $R(1)=\tfrac12$; $R(n)$ hội tụ về giới hạn với tốc độ khoảng $1/n^2$. | §VII, Fig. 2 |
| **Conditional k-NN risk $r_k(x)$** | Rủi ro k-NN có điều kiện | Với hai lớp và *k* lẻ, mỗi láng giềng "bỏ phiếu" độc lập có điều kiện; $r_k(x)$ là xác suất đa số phiếu sai, (45)–(46). | §VIII |
| **Monotone in k** | Đơn điệu theo *k* | $r_k(x)$ giảm khi *k* tăng, về $\min\{\hat\eta_1,1-\hat\eta_1\}$; nên cận trên của $R_k$ cũng giảm về $R^*$. | §VIII |
| **Least concave majorant $\bar\rho_k$** | Hàm lõm nhỏ nhất trội | Hàm lõm nhỏ nhất lớn hơn $\rho_k(r^*)$; cùng bất đẳng thức Jensen cho cận trên chặt $R_k\le\bar\rho_k(R^*)$ (48). | §VIII eq. (47)–(48) |
| **Jensen's inequality** | Bất đẳng thức Jensen | $E[g(X)]\le g(E[X])$ với *g* lõm; dùng để chuyển cận có điều kiện thành cận toàn cục. | §VIII |
| **Chain of k-NN bounds** | Chuỗi cận k-NN | $R^*\le R_k\le\bar\rho_k(R^*)\le\bar\rho_{k-1}(R^*)\le\dots\le\bar\rho_1(R^*)=2R^*(1-R^*)$ (49). | §VIII eq. (49) |

---

## 7. Notation summary

| Symbol | Ý nghĩa |
|---|---|
| $x$, $x_i$ | Phép đo (vectơ đặc trưng) của quan sát mới / mẫu thứ *i* |
| θ, $\theta_i$ | Lớp của quan sát mới / mẫu thứ *i* |
| $x'_n$, $\theta'_n$ | Láng giềng gần nhất trong *n* mẫu và lớp của nó |
| *M* | Số lớp |
| *d* | Hàm khoảng cách (metric) |
| $f_i(x)$, $f(x)$ | Mật độ theo lớp *i*; mật độ hỗn hợp |
| $\eta_i$, $\hat\eta_i(x)$ | Xác suất tiên nghiệm; xác suất hậu nghiệm |
| *L*(i, j) | Hàm mất mát |
| $r^*(x)$, $R^*$ | Rủi ro Bayes có điều kiện; rủi ro (sai số) Bayes |
| $R(n)$, *R* | Rủi ro NN với *n* mẫu; rủi ro NN mẫu lớn |
| $R_k$ | Rủi ro k-NN mẫu lớn |

## 8. Key results

| Result | Value |
|---|---|
| Two classes | $R^*\le R\le 2R^*(1-R^*)$ |
| *M* classes | $R^*\le R\le R^*(2-MR^*/(M-1))$ |
| Simple corollary | $R\le 2R^*$ |
| k-NN (two classes, odd *k*) | $R^*\le R_k\le\dots\le 2R^*(1-R^*)$, decreasing in *k* |
| Triangular example | $R^*=1/4$, $R=1/3$, $R(n)=1/3+1/[(n+1)(n+2)]$ |

## 9. People and works cited

- **Fix & Hodges (1951, 1952):** first formulation of the $k_n$-NN rule; consistency when $k_n\to\infty$, $k_n/n\to0$; small-sample study under normal statistics.
- **Johns (1961):** NN as an empirical Bayes rule.
- **Kanal (1963), Sebestyen (1962), Nilsson (1965):** NN for pattern recognition ("proximity algorithm").
- **Loftsgaarden & Quesenberry (1965):** consistent density estimation with k-NN.

---

## 10. Relevance to this project (credit scoring with KNN)

These points connect the paper to the report's chapter 2 (KNN section).

1. **Same rule.** The project's `KNeighborsClassifier` (k = 5, uniform weights, Euclidean distance) is Cover & Hart's k-NN majority vote, with *M* = 2 classes (good/bad).
2. **Theory is asymptotic.** The "at most twice the Bayes error" result holds as *n* → ∞ with a fixed number of features. It says nothing about finite samples or many features, so do not cite it as a guarantee of KNN performance on the credit datasets.
3. **The metric matters in practice.** The bounds do not depend on the metric, but only in the limit. With finite data, Euclidean distance is dominated by features with large scales. This is why tier B.1 standardises features before KNN.
4. **Probabilities are coarse.** With k = 5 and uniform weights, the predicted probability is the share of bad neighbours, so it takes only the values 0, 0.2, …, 1. This limits ranking (many ties for AUC) and calibration (Brier score). The paper only studies classification error, not probability quality.
5. **Choice of k.** §III explains the trade-off: larger *k* stabilises the vote, smaller *k* keeps neighbours local. The project uses the library default *k* = 5 without tuning.
6. **Link to the other papers.** The Bayes rule and posterior probability $\hat\eta_i(x)$ are the same ideas as Maron's (1961) $P(C_j\mid W)$. The Bayes error $R^*$ is the lower limit for every model in the project.
7. **What the paper does not cover.** It does not discuss distance-weighted voting, high-dimensional data, computational cost, imbalanced classes or probability calibration. Do not cite it for those.

**Suggested citation sentence (VI):**
> *Cover và Hart (1967) chứng minh rằng khi số mẫu tiến tới vô cùng, sai số của luật láng giềng gần nhất không vượt quá hai lần sai số Bayes, tức sai số nhỏ nhất mà bất kỳ bộ phân loại nào có thể đạt được.*
