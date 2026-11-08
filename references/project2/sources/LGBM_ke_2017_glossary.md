# Glossary — Ke et al. (2017), *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*

**Source:** G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye & T.-Y. Liu (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, Long Beach, CA. Microsoft Research, Peking University, Microsoft Redmond. File: `references/project2/refs/lightgbm_ke_el_al_2017.pdf` (9 pages; proofs and parameter settings are in a supplementary file not included).

**What the paper does:** Conventional GBDT must scan every instance for every feature to find split points, so its cost grows with both the number of rows and the number of features. LightGBM builds on the **histogram-based algorithm** and adds two techniques:
- **Gradient-based One-Side Sampling (GOSS)** reduces the number of rows. It keeps all instances with large gradients, samples randomly from those with small gradients, and re-weights the sample so the information gain stays almost unbiased.
- **Exclusive Feature Bundling (EFB)** reduces the number of features. It merges sparse features that rarely are non-zero at the same time into one feature. Finding the optimal bundling is NP-hard, so a greedy graph-colouring algorithm is used.

On five large datasets, LightGBM trains up to more than 20× faster than the same algorithm without GOSS and EFB, with almost the same accuracy. GOSS beats Stochastic Gradient Boosting at the same sampling ratio.

**How to read this file:** Terms are grouped by topic, in roughly the order the paper uses them.
- Each entry gives the English term, a suggested Vietnamese term, a definition in Vietnamese (định nghĩa), the formula where there is one, and where it appears (section / page of the NIPS paper).
- The Vietnamese terms are suggestions for the report (`report/sections_v1`); check them against `tu_viet_tat.md` and chapter 2.
- Equation numbers (1)–(2), Definition 3.1, Theorems 3.2 and 4.1 and Algorithms 1–4 are the paper's own.

---

## 1. GBDT and its cost

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Gradient boosting decision tree (GBDT)** | Cây quyết định gradient boosting | Tập hợp cây quyết định huấn luyện tuần tự; mỗi vòng khớp cây với **gradient âm** (còn gọi là phần dư) của hàm mất mát (Friedman 2001). | §1, §2.1 p.2 |
| **Information gain (split)** | Độ lợi thông tin của phép chia | Mức cải thiện khi chia một nút; GBDT phải ước lượng nó cho mọi điểm chia khả dĩ. Đây là bước tốn thời gian nhất. | §1, §2.1 |
| **Accuracy–efficiency trade-off** | Đánh đổi độ chính xác – hiệu năng | Với dữ liệu lớn về cả số hàng và số cột, GBDT thông thường rất chậm vì chi phí tỷ lệ với cả hai. | §1 |
| **Pre-sorted algorithm** | Thuật toán sắp xếp trước | Liệt kê mọi điểm chia trên các giá trị đặc trưng đã sắp xếp sẵn; tìm được điểm chia tối ưu nhưng chậm và tốn bộ nhớ. scikit-learn và R gbm dùng cách này. | §2.1, §2.2 |
| **Histogram-based algorithm (Algorithm 1)** | Thuật toán dựa trên histogram | Gom giá trị liên tục vào các **bin rời rạc**, xây histogram của mỗi đặc trưng rồi tìm điểm chia trên histogram. Chi phí $O(\#data\times\#feature)$ để xây histogram và $O(\#bin\times\#feature)$ để tìm điểm chia. Nhanh và ít tốn bộ nhớ hơn; là nền của LightGBM. | §2.1, Alg. 1 |
| **Bin** | Bin (khoảng rời rạc) | Một khoảng giá trị của đặc trưng sau khi rời rạc hóa; số bin thường nhỏ hơn nhiều so với số mẫu. | §2.1 |
| **Feature histogram** | Histogram đặc trưng | Với mỗi bin, tổng gradient và số mẫu rơi vào bin đó; dùng để tính gain của mọi ngưỡng giữa các bin. | Alg. 1 |
| **Two ways to speed up** | Hai hướng tăng tốc | Vì xây histogram chiếm phần lớn chi phí, giảm **#data** (GOSS) hoặc **#feature** (EFB) sẽ tăng tốc đáng kể. | §2.1 |

## 2. Related approaches

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Down-sampling for boosting** | Lấy mẫu giảm cho boosting | Các phương pháp lọc mẫu theo trọng số (FHT00, Dubout & Fleuret, Appel et al.) dựa trên AdaBoost; không áp dụng trực tiếp cho GBDT vì GBDT **không có trọng số mẫu tự nhiên**. | §1, §2.2 |
| **Stochastic Gradient Boosting (SGB)** | Gradient boosting ngẫu nhiên | Friedman (2002): mỗi vòng học trên một tập con ngẫu nhiên; áp dụng được cho GBDT nhưng thường làm giảm độ chính xác. | §2.2, §5.2 |
| **Filtering weak features** | Lọc đặc trưng yếu | Giảm chiều bằng PCA hoặc projection pursuit; dựa vào giả định có nhiều đặc trưng dư thừa, không phải lúc nào cũng đúng. | §2.2 |
| **Sparse optimization** | Tối ưu cho dữ liệu thưa | Thuật toán sắp xếp trước có thể bỏ qua giá trị 0 (XGBoost); thuật toán histogram thì không, vì phải lấy bin của mọi mẫu kể cả giá trị 0. EFB giải quyết vấn đề này. | §2.2 |
| **XGBoost as baseline** | XGBoost làm chuẩn so sánh | XGBoost hỗ trợ cả hai thuật toán (xgb_exa, xgb_his) và vượt các công cụ khác, nên được chọn làm chuẩn so sánh. | §2.2, §5.1 |

## 3. Gradient-based One-Side Sampling (GOSS)

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Gradient as importance signal** | Gradient như tín hiệu quan trọng | GBDT không có trọng số mẫu, nhưng **độ lớn gradient** cho biết mẫu đã được học tốt (gradient nhỏ) hay chưa (gradient lớn, *under-trained*). "Lớn/nhỏ" là theo giá trị tuyệt đối. | §1, §3.1 |
| **Under-trained instances** | Mẫu chưa được học tốt | Mẫu có |gradient| lớn; đóng góp nhiều hơn vào information gain. | §1 |
| **GOSS (Algorithm 2)** | Lấy mẫu một phía theo gradient | Sắp xếp mẫu theo |gradient| giảm dần; giữ **toàn bộ** *a* × 100% mẫu đầu (tập *A*); lấy ngẫu nhiên *b* × 100% từ phần còn lại (tập *B*); khi tính gain, nhân gradient của *B* với hằng số $(1-a)/b$ để giữ gần nguyên phân phối dữ liệu. | §3.1, Alg. 2 |
| **Sampling ratios a, b** | Tỷ lệ lấy mẫu *a*, *b* | *a*: tỷ lệ mẫu gradient lớn được giữ; *b*: tỷ lệ mẫu gradient nhỏ được lấy ngẫu nhiên. Thí nghiệm dùng *a* = *b* = 0,05 hoặc 0,1. | §3.1, §5.1 |
| **Amplification factor (1 − a)/b** | Hệ số khuếch đại | Đưa tổng gradient trên *B* về đúng quy mô của toàn bộ phần gradient nhỏ $A^c$. | §3.1, §3.2 |
| **One-side** | Một phía | Chỉ lấy mẫu giảm ở phía gradient nhỏ; phía gradient lớn giữ nguyên. | §3.1 |
| **Variance gain $V_{j\mid O}(d)$ (Definition 3.1)** | Độ lợi phương sai | Gain của phép chia đặc trưng *j* tại điểm *d* trong nút *O*: $V_{j\mid O}(d)=\frac{1}{n_O}\Big(\frac{(\sum_{x_{ij}\le d}g_i)^2}{n^j_{l\mid O}(d)}+\frac{(\sum_{x_{ij}>d}g_i)^2}{n^j_{r\mid O}(d)}\Big)$. Phân tích trong bài dùng dạng chỉ có gradient bậc một. | §3.2 Def. 3.1 |
| **Estimated variance gain $\tilde V_j(d)$** | Độ lợi phương sai ước lượng | Cùng công thức nhưng chỉ trên $A\cup B$, với gradient của *B* nhân $(1-a)/b$ (1). | §3.2 eq. (1) |
| **Approximation error bound (Theorem 3.2)** | Cận sai số xấp xỉ | Với xác suất ≥ 1 − δ, $\mathcal E(d)=|\tilde V_j(d)-V_j(d)|$ bị chặn bởi (2). Tỷ lệ xấp xỉ tiệm cận là $O\big(\frac{1}{n_l^j(d)}+\frac{1}{n_r^j(d)}+\frac{1}{\sqrt n}\big)$; nếu phép chia không quá lệch, sai số giảm về 0 với tốc độ $O(\sqrt n)$. | §3.2 eq. (2) |
| **Random sampling as special case** | Lấy mẫu ngẫu nhiên là trường hợp riêng | Lấy mẫu ngẫu nhiên tương ứng *a* = 0; GOSS tốt hơn trong nhiều trường hợp khi điều kiện $C_{0,\beta}>C_{a,\beta-a}$ thỏa. | §3.2 |
| **Generalization with GOSS** | Tổng quát hóa với GOSS | Sai số tổng quát của GOSS gần với khi dùng toàn bộ dữ liệu nếu xấp xỉ tốt; lấy mẫu còn làm tăng **độ đa dạng** của các bộ học cơ sở, có thể giúp tổng quát hóa. | §3.2 |

## 4. Exclusive Feature Bundling (EFB)

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Sparse feature space** | Không gian đặc trưng thưa | Dữ liệu nhiều chiều thường rất thưa, ví dụ đặc trưng one-hot. | §1, §4 |
| **Mutually exclusive features** | Đặc trưng loại trừ lẫn nhau | Các đặc trưng **không bao giờ (hoặc hiếm khi) cùng khác 0** trên một mẫu, như các cột one-hot của cùng một biến. | §1, §4 |
| **Exclusive feature bundle** | Bó đặc trưng loại trừ | Một đặc trưng mới gộp nhiều đặc trưng loại trừ lẫn nhau; histogram xây từ bó cho kết quả như từ từng đặc trưng. Chi phí xây histogram giảm từ $O(\#data\times\#feature)$ xuống $O(\#data\times\#bundle)$. | §4 |
| **NP-hardness (Theorem 4.1)** | Tính NP-khó | Chia đặc trưng thành **ít bó nhất** là NP-khó, chứng minh bằng quy về bài toán **tô màu đồ thị**. | §4 Thm. 4.1 |
| **Graph colouring reduction** | Quy về tô màu đồ thị | Mỗi đặc trưng là một đỉnh; nối cạnh giữa hai đặc trưng không loại trừ nhau; mỗi màu là một bó. | §1, §4 |
| **Greedy Bundling (Algorithm 3)** | Gộp bó tham lam | Xây đồ thị có trọng số cạnh = số xung đột; sắp đặc trưng theo bậc giảm dần; đưa mỗi đặc trưng vào bó có ít xung đột (≤ *K*) hoặc tạo bó mới. Độ phức tạp $O(\#feature^2)$, chạy một lần trước khi huấn luyện. Phiên bản nhanh hơn sắp theo số giá trị khác 0 thay vì bậc. | §4, Alg. 3 |
| **Conflict / max conflict rate γ** | Xung đột / tỷ lệ xung đột tối đa γ | Cho phép một tỷ lệ nhỏ mẫu mà hai đặc trưng trong bó cùng khác 0; ảnh hưởng tới độ chính xác tối đa $O([(1-\gamma)n]^{-2/3})$. Thí nghiệm dùng γ = 0. | §4, §5.1 |
| **Merge Exclusive Features (Algorithm 4)** | Gộp đặc trưng loại trừ | Cộng **độ lệch (offset)** vào bin của từng đặc trưng để chúng nằm ở các khoảng bin khác nhau trong bó; ví dụ A ∈ [0, 10), B ∈ [0, 20) → B + 10 ∈ [10, 30), bó ∈ [0, 30). | §4, Alg. 4 |
| **Non-zero data table** | Bảng dữ liệu khác 0 | Tối ưu cơ bản: lưu cho mỗi đặc trưng danh sách mẫu khác 0 để xây histogram với chi phí $O(\#non\_zero\_data)$; được cài như chức năng cơ bản trong LightGBM. | §4 |

## 5. Tree growth and experiments

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Leaf-wise (best-first) tree growth** | Phát triển cây theo lá (best-first) | Luôn chia lá cho gain lớn nhất (Shi 2007); dùng cho xgb_his, lgb_baseline và LightGBM trong thí nghiệm. Bài không trình bày đây là đóng góp mới. | §5.1 |
| **Layer-wise (level-wise) growth** | Phát triển theo tầng | Chia mọi nút ở cùng độ sâu; xgb_exa chỉ hỗ trợ cách này nên được chỉnh tham số để tạo cây tương tự. | §5.1 |
| **lgb_baseline / EFB_only / LightGBM** | Các biến thể so sánh | lgb_baseline: LightGBM không GOSS, không EFB; EFB_only: chỉ thêm EFB; LightGBM: thêm cả GOSS và EFB. | §5.1, Table 2 |
| **Datasets** | Bộ dữ liệu | Allstate (12M, 4.228 đặc trưng, thưa), Flight Delay (10M, 700, thưa), LETOR (2M, 136, dày, xếp hạng), KDD10 (19M, 29M đặc trưng), KDD12 (119M, 54M đặc trưng). | §5, Table 1 |
| **Speed-up** | Mức tăng tốc | So với lgb_baseline: 21×, 6×, 1,6×, 14×, 13× trên năm bộ dữ liệu; nhanh hơn xgb_his tới 9× (Allstate); xgb_his hết bộ nhớ trên KDD10/KDD12. | §5.1, Table 2 |
| **Accuracy (AUC / NDCG@10)** | Độ chính xác | Gần như bằng các chuẩn so sánh, ví dụ Allstate AUC 0,6093, Flight Delay 0,7846. | §5.1, Table 3 |
| **GOSS vs SGB** | GOSS so với SGB | Cùng tỷ lệ lấy mẫu, GOSS luôn tốt hơn SGB trên LETOR (ví dụ 0,5275 so với 0,5239 ở tỷ lệ 0,2). GOSS tự nó tăng tốc gần 2× khi dùng 10–20% dữ liệu; không tuyến tính vì vẫn phải dự đoán và tính gradient trên toàn bộ dữ liệu. | §5.2, Table 4 |
| **EFB contribution** | Đóng góp của EFB | Tăng tốc lớn trên dữ liệu lớn nhờ gộp nhiều đặc trưng thưa (one-hot và loại trừ ngầm), tăng tính cục bộ bộ nhớ và tỷ lệ cache hit. | §5.3 |
| **Early stopping** | Dừng sớm | Dùng trong thí nghiệm GOSS/SGB để đảm bảo mọi cấu hình hội tụ. | §5.2 |

---

## 6. Notation summary

| Symbol | Ý nghĩa |
|---|---|
| $g_i$ | Gradient âm của mất mát tại mẫu *i* |
| *a*, *b* | Tỷ lệ mẫu gradient lớn được giữ; tỷ lệ mẫu gradient nhỏ được lấy ngẫu nhiên |
| *A*, $A^c$, *B* | Tập gradient lớn; phần còn lại; mẫu ngẫu nhiên từ $A^c$ |
| $(1-a)/b$ | Hệ số khuếch đại gradient của *B* |
| $V_j(d)$, $\tilde V_j(d)$ | Độ lợi phương sai thật; ước lượng bằng GOSS |
| $n_l^j(d)$, $n_r^j(d)$ | Số mẫu bên trái/phải khi chia đặc trưng *j* tại *d* |
| $\mathcal E(d)$ | Sai số xấp xỉ của GOSS |
| γ (EFB) | Tỷ lệ xung đột tối đa trong một bó |
| *K* (Alg. 3) | Số xung đột tối đa cho phép |
| #data, #feature, #bin, #bundle | Số mẫu, số đặc trưng, số bin, số bó |

## 7. Key results

| Item | Value |
|---|---|
| Max speed-up vs lgb_baseline | 21× (Allstate) |
| Max speed-up vs xgb_his | 9× (Allstate) |
| GOSS alone | ≈ 2× with 10–20% of data |
| Accuracy | Same as baselines within ±0.0005 AUC on most datasets |
| GOSS vs SGB (LETOR, ratio 0.2) | 0.5275 vs 0.5239 NDCG@10 |

## 8. People and works cited (selected)

- **Friedman (2001) [1]:** GBDT. **Friedman (2002) [20]:** stochastic gradient boosting.
- **Chen & Guestrin (2016) [13]:** XGBoost (baseline). **Tyree et al. (2011) [14]:** pGBRT, histogram-based.
- **Friedman, Hastie & Tibshirani (2000) [5]; Dubout & Fleuret (2011) [6]; Appel et al. (2013) [7]:** sampling for boosting.
- **Mehta et al. (1996, SLIQ) [8]; Shafer et al. (1996, SPRINT) [9]:** pre-sorted algorithm. **Ranka & Singh (1998) [10]; Jin & Agrawal (2003) [11]; Li et al. (2007) [12]:** histogram-based.
- **Shi (2007) [32]:** best-first (leaf-wise) decision tree learning.
- **Jensen & Toft (2011) [25]:** graph colouring. **Jolliffe (2002) [22]:** PCA.
- **Pedregosa et al. (2011) [15]:** scikit-learn. **Ridgeway (2007) [16]:** R gbm.

---

## 9. Relevance to this project (credit scoring with LightGBM)

These points connect the paper to the report's chapter 2 (§2.4.5 LightGBM).

1. **What the project actually runs.** `LGBMClassifier` with library defaults (LightGBM 4.7.0): `boosting_type = gbdt`, 100 trees, 31 leaves, learning rate 0.1, at least 20 rows per leaf, `max_depth = -1`, histogram split finding, leaf-wise growth. **GOSS is off** (it needs `data_sample_strategy = goss`); **EFB is on** by default.
2. **What carries over from the paper.** Histogram split finding and leaf-wise growth are what shape the project's models. GOSS and EFB are speed techniques; GOSS is unused, and EFB matters little because the GBDT matrices use integer category codes rather than wide one-hot columns.
3. **Gain formula.** The paper analyses a first-order "variance gain" (Definition 3.1). The library's split gain actually uses gradients and Hessians with L2 regularisation, like XGBoost's. Present the XGBoost formula as the general one and cite this paper for GOSS/EFB.
4. **Leaf-wise growth and overfitting.** Leaf-wise trees can grow deep and unbalanced. With small datasets such as German credit (1,000 rows) this can overfit; `num_leaves = 31` and `min_child_samples = 20` are the main limits. The paper itself does not discuss overfitting from leaf-wise growth; cite it only for the use of best-first growth (Shi 2007).
5. **Scale.** The paper's datasets have millions of rows and up to 54 million features. The credit datasets are far smaller, so LightGBM's speed advantage shows in the runtime table but is not the reason for any accuracy difference.
6. **Link to Friedman (2001).** GOSS is close in spirit to Friedman's influence trimming (dropping observations whose loss barely changes), but uses gradient size and re-weighting. GOSS is compared with Friedman's (2002) stochastic gradient boosting.
7. **What the paper does not cover.** Categorical feature handling (native in the library, not in the paper), probability calibration, prediction shift, or regularisation parameters. Cite the CatBoost and XGBoost papers for those.

**Suggested citation sentence (VI):**
> *LightGBM (Ke và cộng sự, 2017) xây dựng trên thuật toán tìm điểm chia bằng histogram và đề xuất hai kỹ thuật tăng tốc: GOSS giữ lại các mẫu có gradient lớn và lấy mẫu ngẫu nhiên các mẫu có gradient nhỏ, EFB gộp các đặc trưng thưa hiếm khi cùng khác 0 thành một đặc trưng.*
