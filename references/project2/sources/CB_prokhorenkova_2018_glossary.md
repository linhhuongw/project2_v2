# Glossary — Prokhorenkova et al. (2018), *CatBoost: Unbiased Boosting with Categorical Features*

**Source:** L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush & A. Gulin (2018). "CatBoost: unbiased boosting with categorical features". *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*, Montréal. Yandex; Moscow Institute of Physics and Technology. File: `references/project2/refs/catboost_prokhorenkova_2018.pdf` (11 pages; the supplementary material with proofs, Sections A–G, is not included).

**What the paper does:** The authors show that every existing gradient boosting implementation suffers from a **prediction shift**. Gradients for a training example are computed by a model that was trained on that same example's target, which is a special kind of **target leakage**. The same leakage appears when categorical features are replaced by **target statistics** (mean target per category). Their single remedy is the **ordering principle**: use a random permutation as artificial "time", and let each example only use information from examples before it. This gives:
- **ordered target statistics** for categorical features;
- **ordered boosting**, a version of gradient boosting whose residuals are computed by models that have not seen the example.

The paper also describes how CatBoost builds **oblivious trees**, combines categorical features, and uses a Bayesian bootstrap. On nine datasets, CatBoost (Ordered mode) beats tuned XGBoost and LightGBM in log-loss and zero-one loss. Ordered mode helps most on **small datasets**.

**How to read this file:** Terms are grouped by topic, in roughly the order the paper uses them.
- Each entry gives the English term, a suggested Vietnamese term, a definition in Vietnamese (định nghĩa), the formula where there is one, and where it appears (section / page).
- The Vietnamese terms are suggestions for the report (`report/sections_v1`); check them against `tu_viet_tat.md` and chapter 2.
- Equation numbers (1)–(6), properties P1–P2, Theorem 1 and Algorithms 1–2 are the paper's own.

---

## 1. Background: gradient boosting and trees

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Expected loss** | Mất mát kỳ vọng | $\mathcal L(F)=\mathbb E\,L(y,F(x))$, với $(x,y)$ là một mẫu kiểm tra lấy độc lập với tập huấn luyện *D*; *L* là hàm mất mát trơn. | §2 p.2 |
| **Gradient boosting as functional gradient descent** | Gradient boosting là hạ gradient trong không gian hàm | $F^t=F^{t-1}+\alpha h^t$, với $h^t=\arg\min_{h\in H}\mathbb E\,L(y,F^{t-1}(x)+h(x))$ (1); giải bằng bước Newton (xấp xỉ bậc hai) hoặc bước gradient âm. | §1, §2 eq. (1) |
| **Step size α** | Độ dài bước α | Hệ số nhân của mỗi bộ dự đoán cơ sở (learning rate). | §2 |
| **Gradient step (least squares)** | Bước gradient (bình phương tối thiểu) | $h^t=\arg\min_h\mathbb E(-g^t(x,y)-h(x))^2$ (2), với $g^t(x,y)=\partial L(y,s)/\partial s\big|_{s=F^{t-1}(x)}$. | §2 eq. (2) |
| **Binary decision tree** | Cây quyết định nhị phân | Chia đệ quy không gian đặc trưng thành các vùng rời nhau; mỗi thuộc tính chia là biến nhị phân $a=\mathbf 1\{x^k>t\}$; $h(x)=\sum_j b_j\mathbf 1_{\{x\in R_j\}}$ (3). Trong boosting, cây giải một bài toán **hồi quy** trên gradient âm. | §2 eq. (3) |
| **Base predictor** | Bộ dự đoán cơ sở | Hàm $h^t$ thêm vào ở mỗi vòng; trong CatBoost là cây quyết định nhị phân. | §2 |

## 2. Categorical features and target statistics

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Categorical feature** | Biến phân loại | Biến có tập giá trị rời rạc (hạng mục) không so sánh được với nhau. | §3.1 |
| **One-hot encoding** | Mã hóa one-hot | Thêm một biến nhị phân cho mỗi hạng mục; không khả thi với biến có **nhiều hạng mục** (high cardinality) như mã người dùng. | §3.1 |
| **High cardinality** | Số hạng mục lớn | Biến phân loại có rất nhiều giá trị khác nhau. | §3.1 |
| **Target statistic (TS)** | Thống kê mục tiêu | Thay hạng mục $x_k^i$ bằng một số $\hat x_k^i\approx\mathbb E(y\mid x^i=x_k^i)$. Chỉ cần lưu một số cho mỗi hạng mục; với chia nhị phân, điểm chia tối ưu theo log-loss/Gini/MSE nằm trong các ngưỡng của TS. | §3.1–3.2 |
| **Gradient statistics for categories (LightGBM)** | Thống kê gradient theo hạng mục | LightGBM đổi biến phân loại thành thống kê gradient ở mỗi bước; tốn thời gian và bộ nhớ, và gom các hạng mục hiếm vào một cụm nên mất thông tin. | §3.1 |
| **Greedy TS** | TS tham lam | Trung bình *y* trên các mẫu cùng hạng mục, làm trơn bằng prior *p*: $\hat x_k^i=\frac{\sum_j\mathbf 1\{x_j^i=x_k^i\}y_j+a\,p}{\sum_j\mathbf 1\{x_j^i=x_k^i\}+a}$ (4). Mắc **rò rỉ nhãn** vì dùng chính $y_k$. | §3.2 eq. (4) |
| **Prior p, smoothing parameter a** | Prior *p*, tham số làm trơn *a* | *p* thường là trung bình nhãn của toàn bộ dữ liệu; *a* > 0 điều chỉnh mức kéo về *p*, giảm nhiễu cho hạng mục ít mẫu. | §3.2 |
| **Target leakage** | Rò rỉ nhãn | Đặc trưng (hoặc gradient) của mẫu *k* được tính bằng chính nhãn $y_k$, nên trên tập huấn luyện nó mang thông tin về nhãn mà trên dữ liệu mới thì không. | §1, §3.2, §4.1 |
| **Conditional shift** | Dịch chuyển có điều kiện | Phân phối của $\hat x^i\mid y$ khác nhau giữa tập huấn luyện và tập kiểm tra. Ví dụ cực đoan: mọi hạng mục duy nhất và $P(y=1)=0.5$ — greedy TS cho phép phân loại đúng hoàn toàn trên tập huấn luyện nhưng chỉ đúng 50% trên tập kiểm tra. | §3.2 |
| **Property P1** | Tính chất P1 | $\mathbb E(\hat x^i\mid y=v)=\mathbb E(\hat x_k^i\mid y_k=v)$: TS phải có cùng phân phối có điều kiện trên mẫu huấn luyện và mẫu kiểm tra. | §3.2 |
| **Property P2** | Tính chất P2 | Dùng hiệu quả **toàn bộ** dữ liệu huấn luyện cho cả việc tính TS và học mô hình. | §3.2 |
| **TS on a subset $D_k$** | TS trên tập con $D_k$ | Ý chung để tránh dịch chuyển: tính TS của $x_k$ trên tập $D_k\subset D\setminus\{x_k\}$ (5). | §3.2 eq. (5) |
| **Holdout TS** | TS giữ lại | Chia dữ liệu làm hai: một phần tính TS, phần kia huấn luyện. Thỏa P1 nhưng vi phạm P2. | §3.2 |
| **Leave-one-out TS** | TS bỏ một | $D_k=D\setminus x_k$; ngạc nhiên là vẫn rò rỉ: với một biến hằng, TS của mẫu phụ thuộc $y_k$ nên vẫn chia hoàn hảo được tập huấn luyện. | §3.2 |
| **Ordered TS** | TS có thứ tự | Lấy một hoán vị ngẫu nhiên σ làm "thời gian nhân tạo"; với mẫu huấn luyện dùng $D_k=\{x_j:\sigma(j)<\sigma(k)\}$, với mẫu kiểm tra dùng toàn bộ *D*. Thỏa cả P1 và P2. Dùng nhiều hoán vị cho các vòng khác nhau để giảm phương sai ở đầu hoán vị. | §3.2 |
| **Online learning inspiration** | Cảm hứng từ học trực tuyến | Trong học trực tuyến, mẫu đến theo thời gian và TS chỉ dựa trên lịch sử đã quan sát; nguyên tắc thứ tự mô phỏng điều này cho dữ liệu offline. | §3.2 |

## 3. Prediction shift and ordered boosting

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Prediction shift** | Dịch chuyển dự đoán | Phân phối của $F^{t-1}(x_k)\mid x_k$ trên mẫu huấn luyện lệch khỏi $F^{t-1}(x)\mid x$ trên mẫu kiểm tra, vì gradient ở mỗi bước được tính bằng nhãn của chính các mẫu mà mô hình đã học. Có trong **mọi** cài đặt gradient boosting trước đó. | §1, §4.1 |
| **Chain of shifts** | Chuỗi dịch chuyển | (1) Phân phối gradient $g^t(x_k,y_k)\mid x_k$ trên mẫu huấn luyện lệch so với mẫu kiểm tra; (2) bộ dự đoán cơ sở $h^t$ ước lượng theo (6) bị chệch so với nghiệm của (2); (3) khả năng tổng quát của $F^t$ bị ảnh hưởng. | §4.1 |
| **Empirical gradient step** | Bước gradient thực nghiệm | Trong thực tế (2) được xấp xỉ trên cùng tập *D*: $h^t=\arg\min_h\frac1n\sum_k(-g^t(x_k,y_k)-h(x_k))^2$ (6). | §4.1 eq. (6) |
| **Iterated bagging / subsampling** | Bagging lặp / lấy mẫu con | Các cách đã có (Breiman 2001; Friedman 2002) chỉ giảm nhẹ vấn đề: phần dư out-of-bag vẫn bị dịch chuyển, lấy mẫu con chỉ là cách heuristic. | §4.1 |
| **Theorem 1 (bias of boosting)** | Định lý 1 (độ chệch của boosting) | Hồi quy bình phương, hai biến Bernoulli, $y=c_1x^1+c_2x^2$, hai bước với cây gốc (stump), α = 1. Dùng hai mẫu độc lập cho hai bước: $\mathbb E F^2(x)=f^*(x)+O(2^{-n})$ (không chệch). Dùng cùng một mẫu: $\mathbb E F^2(x)=f^*(x)-\frac{1}{n-1}c_2(x^2-\tfrac12)+O(2^{-n})$, tức có **độ chệch tỷ lệ nghịch với cỡ mẫu *n***. | §4.1 Thm. 1 |
| **Decision stump** | Cây gốc (stump) | Cây độ sâu 1, dùng trong phân tích lý thuyết. | §4.1 |
| **Ordering principle** | Nguyên tắc thứ tự | Ý tưởng trung tâm của bài: mỗi mẫu chỉ được dùng thông tin từ các mẫu đứng trước nó trong một hoán vị ngẫu nhiên. Giải quyết cả rò rỉ của TS lẫn prediction shift. | §1, §3.2, §4.2 |
| **Ordered boosting (Algorithm 1)** | Boosting có thứ tự | Duy trì *n* mô hình hỗ trợ $M_1,\dots,M_n$, $M_i$ chỉ học trên *i* mẫu đầu của hoán vị σ; phần dư của mẫu *j* tính bằng $M_{j-1}$: $r_i=y_i-M_{\sigma(i)-1}(x_i)$. Dạng nguyên thủy không khả thi vì tốn gấp *n* lần. | §4.2, Fig. 1 |
| **Supporting models $M_i$** | Mô hình hỗ trợ | Các mô hình trung gian chỉ học trên tiền tố của hoán vị, dùng để tính phần dư không chệch. | §4.2 |
| **Same permutation for TS and boosting** | Cùng hoán vị cho TS và boosting | Phải lấy $\sigma_{cat}=\sigma_{boost}$ để nhãn $y_i$ không được dùng khi học $M_i$, kể cả qua TS lẫn qua gradient. | §4.2 |

## 4. CatBoost implementation

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Ordered vs Plain mode** | Chế độ Ordered và Plain | **Plain**: GBDT tiêu chuẩn, kèm ordered TS sẵn có nếu có biến phân loại. **Ordered**: cài đặt hiệu quả của Algorithm 1. | §5 p.6 |
| **s + 1 random permutations** | *s* + 1 hoán vị ngẫu nhiên | $\sigma_1..\sigma_s$ dùng để đánh giá phép chia (cấu trúc cây); $\sigma_0$ dùng để tính giá trị lá của cây cuối. Nhiều hoán vị giảm phương sai do các mẫu ở đầu hoán vị có "lịch sử" ngắn. | §5 |
| **Oblivious decision tree (decision table)** | Cây quyết định đối xứng (bảng quyết định) | Mọi nút ở cùng một tầng dùng **chung một tiêu chí chia**. Cây cân bằng, ít overfit, và dự đoán rất nhanh. Là bộ dự đoán cơ sở của CatBoost. | §5 p.6 |
| **Building a tree (Algorithm 2)** | Dựng cây trong CatBoost | Mỗi vòng chọn ngẫu nhiên một hoán vị $\sigma_r$; tính TS theo hoán vị đó; với mỗi phép chia ứng viên, ước lượng giá trị lá $\Delta(i)$ (Ordered: chỉ trung bình gradient của các mẫu **đứng trước** trong cùng lá); chọn phép chia cực tiểu $loss(T_c)$; sau đó cập nhật mọi mô hình hỗ trợ. | §5, Alg. 2 |
| **Cosine similarity split criterion** | Tiêu chí chia theo độ tương đồng cosine | $loss(T_c)=\cos(\Delta,G)$ đo mức xấp xỉ vectơ gradient *G* bằng vectơ giá trị lá Δ khi đánh giá phép chia (theo cách viết của thuật toán). | §5, Alg. 2 |
| **Choosing leaf values** | Chọn giá trị lá | Khi cấu trúc cây đã cố định, giá trị lá của mô hình cuối tính theo thủ tục gradient boosting chuẩn, như nhau cho cả hai chế độ; khi dự đoán dữ liệu mới, TS tính trên toàn bộ dữ liệu huấn luyện. | §5 |
| **Complexity trick** | Mẹo giảm độ phức tạp | Chỉ lưu dự đoán hỗ trợ $M'_{r,j}=M_{r,2^j}$ với $j=1..\lceil\log_2n\rceil$, giảm từ $O(sn^2)$ xuống $O(sn)$. Ordered boosting có cùng độ phức tạp tiệm cận với GBDT chuẩn có ordered TS. | §5, Table 1 |
| **Feature combinations** | Tổ hợp đặc trưng | Ghép tham lam các biến phân loại (và tổ hợp) đã dùng trong cây hiện tại với mọi biến phân loại khác, đổi sang TS ngay khi cần; nắm bắt phụ thuộc bậc cao (ví dụ mã người dùng × chủ đề quảng cáo). | §5, §6 |
| **Bayesian bootstrap** | Bootstrap Bayes | Lấy mẫu con kiểu Friedman (2002) được cài bằng trọng số ngẫu nhiên $w_i=a_i^t$ (Rubin 1981) nhân vào gradient khi dựng cây. | §5 p.7 |
| **Discarding early examples** | Bỏ các mẫu đầu hoán vị | Mẫu có $\sigma_r(i)$ nhỏ có gradient phương sai cao nên được bỏ khỏi phép tính cosine khi đánh giá phép chia. | §5 p.8 |

## 5. Experiments

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Datasets** | Bộ dữ liệu | Adult, Amazon, Click, Epsilon, KDD Appetency, Churn, Internet, Upselling, Kick. Tinh chỉnh tham số trên 4/5 dữ liệu, kiểm tra trên 1/5; mọi thuật toán dùng ordered TS cho biến phân loại. | §6, Table 2 |
| **Log-loss and zero-one loss** | Log-loss và mất mát 0–1 | Hai thước đo chính; CatBoost tốt hơn XGBoost và LightGBM trên mọi bộ (ví dụ Amazon: +17% log-loss cho hai thư viện kia). | §6, Table 2 |
| **Paired one-tailed t-test** | Kiểm định t ghép cặp một phía | Kiểm tra ý nghĩa thống kê; cải thiện có ý nghĩa (p ≤ 0,01) trừ Appetency, Churn, Upselling. | §6 |
| **Ordered vs Plain results** | Kết quả Ordered so với Plain | Ordered tốt hơn rõ nhất trên dữ liệu **nhỏ** (Adult, Internet, dưới 40K mẫu); khi lọc ngẫu nhiên để thu nhỏ dữ liệu, Plain càng kém. Phù hợp với Định lý 1 (độ chệch ~ 1/*n*). | §6, Table 3, Fig. 2 |
| **Running time** | Thời gian chạy | Trên Epsilon, CatBoost Plain và LightGBM nhanh nhất; Ordered chậm hơn khoảng 1,7 lần. | §6 |
| **Comparison of TS methods** | So sánh các loại TS | Ordered TS tốt nhất; greedy TS kém nhất (tới +57% log-loss); holdout tốt nhất trong các phương án còn lại nhưng thua do dùng ít dữ liệu; leave-one-out có thể rất tệ khi mọi hạng mục đều phổ biến (Adult). | §6, Table 4 |
| **Number of permutations s** | Số hoán vị *s* | Tăng *s* giảm nhẹ log-loss (0,19% với *s* = 3, 0,38% với *s* = 9 so với *s* = 1). | §6 |
| **Max combined features $c_{max}$** | Số biến tối đa trong tổ hợp | Tăng từ 1 lên 2 cải thiện log-loss trung bình 1,86% (tới 11,3%); lên 3 được 2,04%; thêm nữa không đáng kể. | §6 |

---

## 6. Notation summary

| Symbol | Ý nghĩa |
|---|---|
| $D=\{(x_k,y_k)\}$ | Tập huấn luyện *n* mẫu |
| $x_k^i$, $\hat x_k^i$ | Giá trị biến phân loại *i* của mẫu *k*; TS thay thế |
| $F^t$, $h^t$ | Mô hình sau vòng *t*; bộ dự đoán cơ sở vòng *t* |
| α | Độ dài bước (learning rate) |
| $g^t(x,y)$ | Gradient của mất mát tại $F^{t-1}(x)$ |
| *p*, *a* | Prior và tham số làm trơn của TS |
| σ, $\sigma_r$ | Hoán vị ngẫu nhiên ("thời gian nhân tạo") |
| $D_k$ | Tập mẫu dùng để tính TS cho mẫu *k* |
| $M_i$, $M_{r,j}$ | Mô hình hỗ trợ học trên *i* (hoặc *j*) mẫu đầu của hoán vị |
| *s* | Số hoán vị dùng để dựng cây |
| $\Delta(i)$, *G* | Giá trị lá ước lượng cho mẫu *i*; vectơ gradient |
| $c_{max}$ | Số biến phân loại tối đa trong một tổ hợp |

## 7. Key results

| Item | Value |
|---|---|
| Theorem 1 bias (same data each step) | $-\frac{1}{n-1}c_2(x^2-\frac12)$, shrinking as 1/*n* |
| CatBoost vs LightGBM / XGBoost | Better log-loss on all 9 datasets (+0.1% to +17% for baselines) |
| Plain vs Ordered | Ordered clearly better on Adult (+1.1% log-loss for Plain) and Internet (+3.9%) |
| Greedy TS vs ordered TS | Up to +57% log-loss (Upselling) |
| Ordered mode speed | ≈ 1.7× slower than Plain on Epsilon |

## 8. People and works cited (selected)

- **Friedman (2001) [12]:** gradient boosting. **Friedman (2002) [13]:** stochastic gradient boosting. **Friedman, Hastie & Tibshirani (2000, 2001) [10, 11].**
- **Chen & Guestrin (2016) [8]:** XGBoost. **Ke et al. (2017) [16]:** LightGBM.
- **Breiman (1996) [2]:** out-of-bag estimation. **Breiman (2001) [3]:** iterated bagging. **Breiman et al. (1984) [4]:** CART.
- **Micci-Barreca (2001) [25]:** target statistics for high-cardinality categories. **Cestnik (1990) [6]:** probability estimation with priors.
- **Ferov & Modrý (2016) [9]; Gulin et al. (2011) [14]; Lou & Obukhov (2017) [23]:** oblivious trees / decision tables.
- **Rubin (1981) [28]:** Bayesian bootstrap. **Zhang et al. (2013) [30]:** target and conditional shift.
- **Mason et al. (2000) [24]:** boosting as gradient descent. **Caruana & Niculescu-Mizil (2006) [5]:** empirical comparison of classifiers.

---

## 9. Relevance to this project (credit scoring with CatBoost)

These points connect the paper to the report's chapter 2 (§2.4.6 CatBoost).

1. **What the project actually runs.** `CatBoostClassifier` with library defaults (CatBoost 1.2.10): **`boosting_type = Plain`**, `grow_policy = SymmetricTree` (oblivious trees), depth 6, `l2_leaf_reg = 3`, 1,000 trees, learning rate chosen automatically. A runtime check with 500, 5,000 and 60,000 rows gave `Plain` every time, so ordered boosting was **never** used, even on the small datasets.
2. **No target statistics.** The GBDT matrices pass categorical variables as integer codes without declaring `cat_features`, so CatBoost treats them as numbers. Ordered TS (§3.2) and feature combinations (§5) were not used either.
3. **What did apply.** Oblivious trees, the L2 leaf regularisation and the large number of trees with a small learning rate are the CatBoost-specific mechanisms behind the project's CatBoost results. The report (§2.4.6, §3.6, §4.7) already says this; keep it consistent.
4. **Prediction shift is still relevant to explain.** It is a property of all the GBDT models in the project (XGBoost, LightGBM, CatBoost Plain). Theorem 1 says the bias shrinks like 1/*n*, so it matters most on the small datasets (German credit, South German, HMEQ). Do not claim CatBoost avoided it in these experiments.
5. **Possible extension.** Running CatBoost with `boosting_type = "Ordered"` and real `cat_features` on the small datasets would test the paper's claim directly. This is optional future work, not something the current results show.
6. **Target leakage link.** The paper's "target leakage" is the same concern as the project's data leakage controls (fitting preprocessing on train only, the HMEQ missing-value leakage checks). Both are about information from the label reaching the features.
7. **Evaluation style.** The paper reports log-loss and zero-one loss with a paired t-test; the project uses AUC/Brier with Friedman/Wilcoxon–Holm tests across datasets (Demšar 2006). The paper's per-dataset t-tests are not comparable to the project's cross-dataset tests.
8. **What the paper does not cover.** Probability calibration, AUC, class imbalance, or the GPU implementation (see Dorogush et al. 2018).

**Suggested citation sentence (VI):**
> *Prokhorenkova và cộng sự (2018) chỉ ra rằng mọi cài đặt gradient boosting đều mắc dịch chuyển dự đoán (prediction shift) do gradient của mỗi mẫu được tính bằng mô hình đã học trên chính nhãn của mẫu đó, và đề xuất ordered boosting cùng ordered target statistics dựa trên một hoán vị ngẫu nhiên của dữ liệu huấn luyện để loại bỏ rò rỉ này.*
