# Glossary — Dorogush, Ershov & Gulin (2018), *CatBoost: Gradient Boosting with Categorical Features Support*

**Source:** A. V. Dorogush, V. Ershov & A. Gulin (2018). "CatBoost: gradient boosting with categorical features support". arXiv:1810.11363v1 [cs.LG], 24 October 2018 (workshop paper). Yandex. File: `references/project2/refs/catboost_dorogush_2018.pdf` (7 pages).

**What the paper does:** This is the short, engineering-oriented companion to Prokhorenkova et al. (2018). It presents the CatBoost library and covers:
- how categorical features are handled **during training**: one-hot for low-cardinality features, **ordered target statistics** with a prior, counter statistics, and greedy **feature combinations** (including categorical × numerical splits);
- the scheme for **fighting gradient bias**: separate supporting models $M_i$ that never see example *i*, sharing tree structures;
- the **fast scorer** that exploits **oblivious trees**, where a leaf index is a binary vector;
- the **GPU training** implementation: histogram split search without atomic operations, feature packing, perfect hashing for categories, multi-GPU feature-parallel learning;
- experiments: tuned CatBoost beats tuned LightGBM, XGBoost and H2O in log-loss on 8 datasets; GPU training up to 15× faster than CPU; scoring about 25× faster than XGBoost and 60× faster than LightGBM.

Its reference [5] is the arXiv (2017) version of Prokhorenkova et al. (2018), which holds the formal analysis.

**How to read this file:** Terms are grouped by topic, in roughly the order the paper uses them.
- Each entry gives the English term, a suggested Vietnamese term, a definition in Vietnamese (định nghĩa), the formula where there is one, and where it appears (section / page).
- The Vietnamese terms are suggestions for the report (`report/sections_v1`); check them against `tu_viet_tat.md` and chapter 2.
- Equation (1) and Algorithm 1 are the paper's own. Many terms are shared with the Prokhorenkova et al. glossary (`CB_prokhorenkova_2018_glossary.md`); this file focuses on what is new or described differently here.

---

## 1. Motivation

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Gradient boosting as gradient descent in function space** | Gradient boosting là hạ gradient trong không gian hàm | Ghép tham lam các mô hình yếu (bộ dự đoán cơ sở) thành mô hình mạnh; phương pháp hàng đầu cho dữ liệu có đặc trưng không đồng nhất, nhiễu, phụ thuộc phức tạp. | §1 p.1 |
| **Heterogeneous features** | Đặc trưng không đồng nhất | Dữ liệu trộn biến số và biến phân loại có thang đo và ý nghĩa khác nhau, như dữ liệu bảng trong tín dụng. | §1 |
| **Handling categories during training** | Xử lý biến phân loại trong lúc huấn luyện | Thay vì đổi biến phân loại sang số ở bước tiền xử lý, CatBoost xử lý chúng ngay trong quá trình huấn luyện. | §1 |
| **New leaf-value schema** | Cách tính giá trị lá mới | Khi chọn cấu trúc cây, CatBoost tính giá trị lá theo cách mới để giảm overfitting (xem §3). | §1 |

## 2. Categorical features

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Categorical feature** | Biến phân loại | Biến có tập giá trị rời rạc không nhất thiết so sánh được với nhau (mã người dùng, tên thành phố); không dùng trực tiếp được trong cây quyết định nhị phân. | §1, §2 |
| **One-hot encoding (in training)** | Mã hóa one-hot trong lúc huấn luyện | Dùng cho biến ít hạng mục; mỗi hạng mục thành một biến nhị phân. CatBoost làm việc này trong lúc huấn luyện, hiệu quả hơn làm ở tiền xử lý. (Thư viện điều khiển bằng tham số `one_hot_max_size`.) | §2 |
| **Average label per category** | Trung bình nhãn theo hạng mục | Cách đơn giản nhất: thay hạng mục bằng trung bình nhãn trên toàn tập huấn luyện. **Gây overfitting**: hạng mục chỉ có một mẫu sẽ nhận đúng nhãn của mẫu đó. | §2 |
| **Iverson brackets [·]** | Ký hiệu Iverson | [điều kiện] = 1 nếu đúng, 0 nếu sai. | §2 |
| **Holdout statistics** | Thống kê trên phần giữ lại | Chia dữ liệu: một phần tính thống kê, một phần huấn luyện. Giảm overfit nhưng giảm lượng dữ liệu cho cả hai việc. | §2 |
| **Ordered target statistic (eq. 1)** | Thống kê mục tiêu có thứ tự | Hoán vị ngẫu nhiên σ; hạng mục của mẫu ở vị trí *p* được thay bằng $\dfrac{\sum_{j=1}^{p-1}[x_{\sigma_j,k}=x_{\sigma_p,k}]Y_{\sigma_j}+a\cdot P}{\sum_{j=1}^{p-1}[x_{\sigma_j,k}=x_{\sigma_p,k}]+a}$: chỉ dùng nhãn của các mẫu **đứng trước** cùng hạng mục, cộng prior. Dùng được toàn bộ dữ liệu mà giảm overfitting. | §2 eq. (1) |
| **Prior P and its weight a** | Prior *P* và trọng số *a* | Giảm nhiễu cho hạng mục hiếm. Hồi quy: thường là trung bình nhãn; phân loại nhị phân: xác suất tiên nghiệm của lớp dương. CatBoost thử nhiều prior và tạo một đặc trưng cho mỗi prior. | §2 |
| **Several permutations** | Nhiều hoán vị | Dùng nhiều hoán vị cho thống kê hiệu quả hơn, nhưng dùng thẳng sẽ gây overfit; CatBoost tránh được nhờ cách tính giá trị lá ở §3. | §2 |
| **Feature combinations** | Tổ hợp đặc trưng | Tổ hợp của nhiều biến phân loại là một biến mới (ví dụ mã người dùng × thể loại nhạc). Số tổ hợp tăng theo hàm mũ nên CatBoost chọn tham lam: không tổ hợp ở phép chia đầu; các phép chia sau ghép mọi biến/tổ hợp đã có trong cây với mọi biến phân loại; đổi sang số ngay khi cần. | §2 |
| **Numerical × categorical combinations** | Tổ hợp biến số và biến phân loại | Mọi phép chia đã chọn trong cây được coi như biến phân loại hai giá trị và đưa vào tổ hợp như biến phân loại. | §2 |
| **Counter statistic (frequency encoding)** | Thống kê đếm (mã hóa tần suất) | Thay hạng mục bằng **số lần xuất hiện** của nó trong dữ liệu; đơn giản nhưng mạnh, cũng tính cho tổ hợp. | §2 |

## 3. Fighting gradient bias

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Biased pointwise gradient estimates** | Ước lượng gradient từng điểm bị chệch | Gradient ở mỗi bước được ước lượng trên chính các điểm dữ liệu mà mô hình hiện tại đã học, nên phân phối gradient ước lượng trong một vùng không gian đặc trưng lệch khỏi phân phối thật, gây overfitting. Prokhorenkova et al. gọi đây là **prediction shift**. | §3 p.3 |
| **Two phases of building a tree** | Hai pha dựng cây | (1) Chọn cấu trúc cây: thử các phép chia, đặt giá trị lá tạm, chấm điểm, chọn phép chia tốt nhất; (2) đặt giá trị lá sau khi cấu trúc đã cố định. Giá trị lá ở cả hai pha thường là xấp xỉ gradient hoặc bước Newton. CatBoost dùng cách cải tiến cho pha 1 và cách GBDT truyền thống cho pha 2. | §3 |
| **Unbiased gradient** | Gradient không chệch | Để $g^i(X_k,Y_k)$ không chệch theo mô hình $F^i$, $F^i$ phải được học **không có** mẫu $X_k$. | §3 |
| **Separate model $M_k$ per example (Algorithm 1)** | Mô hình riêng cho mỗi mẫu | Với mỗi mẫu $X_k$, duy trì một mô hình $M_k$ không bao giờ được cập nhật bằng gradient của chính mẫu đó; dùng $M_k$ để ước lượng gradient của $X_k$ khi chấm điểm cây. | §3, Alg. 1 |
| **Shared tree structures (relaxation)** | Dùng chung cấu trúc cây | Nới lỏng trong cài đặt thực: mọi $M_i$ dùng chung cấu trúc cây, chỉ khác giá trị lá. | §3 |
| **s random permutations** | *s* hoán vị ngẫu nhiên | Cùng các hoán vị dùng cho thống kê biến phân loại; mỗi cây lấy ngẫu nhiên một hoán vị để tính gradient; dùng các hoán vị khác nhau cho các mô hình khác nhau nên không gây overfit. | §3 |
| **O(sn²) → O(sn) trick** | Mẹo giảm từ O(sn²) xuống O(sn) | Thay vì lưu mọi $M_i(X_j)$, chỉ lưu $M'_i(X_j)$, xấp xỉ cho mẫu *j* dựa trên $2^i$ mẫu đầu, với $i\le\log_2n$, $j<2^{i+1}$; tổng số giá trị < 4*n*. Gradient của mẫu $X_k$ dùng $M'_{\lfloor\log_2k\rfloor}(X_k)$. | §3 |
| **Speed cost of bias fighting** | Chi phí tốc độ | Theo bài (2018), CatBoost **mặc định** dùng cơ chế chống chệch này và chậm hơn boosting cổ điển 2–3 lần; bản GPU có chế độ cổ điển cho ai cần tốc độ. (Ở CatBoost 1.2.10 dùng trong đồ án, mặc định đã là `Plain`.) | §6 p.6 |

## 4. Fast scorer and oblivious trees

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Oblivious tree** | Cây đối xứng (oblivious) | Cùng một tiêu chí chia cho cả một tầng của cây; cây cân bằng và ít overfit. Đã được dùng thành công trong boosting (Ferov & Modrý; Gulin et al. – YetiRank). | §4 p.4 |
| **Leaf index as a binary vector** | Chỉ số lá là vectơ nhị phân | Với cây đối xứng độ sâu *d*, chỉ số lá = $\sum_{i=0}^{d-1}2^i\cdot B(x,f(t,i))$, trong đó $B(x,f)$ là giá trị đặc trưng nhị phân *f* của mẫu *x* ở tầng *i*. Giá trị lá lưu trong vectơ dài $2^d$. | §4 |
| **Binarization of features** | Nhị phân hóa đặc trưng | Biến số, thống kê và đặc trưng one-hot đều được đổi sang các đặc trưng nhị phân trước khi tính dự đoán; lưu liên tục trong vectơ *B*. | §4 |
| **Data-parallel scoring** | Chấm điểm song song theo dữ liệu | Xây các vectơ chỉ số lá song song, tăng tốc tới 3×. | §4 |
| **Scorer comparison** | So sánh tốc độ dự đoán | 8.000 cây trên Epsilon, 1 luồng: CatBoost 2,4 s, XGBoost 78 s (×32,5), LightGBM 122 s (×50,8); 32 luồng: 231 ms, 4,5 s, 17,1 s. Khoảng 25× nhanh hơn XGBoost và 60× nhanh hơn LightGBM. | §6, Table 3 |

## 5. GPU training

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Feature discretization into bins** | Rời rạc hóa đặc trưng thành bin | Chia biến số thành số bin cố định (tham số của thuật toán) để giảm bộ nhớ; cho phép tìm điểm chia bằng **histogram**. | §5 p.4 |
| **Feature packing** | Đóng gói đặc trưng | Gộp nhiều đặc trưng vào một số nguyên 32 bit: 1 bit cho biến nhị phân (32 biến/số), 4 bit cho biến ≤ 15 bin (8 biến/số), 8 bit cho biến khác (tối đa 255 bin, 4 biến/số). | §5 |
| **Histogram without atomic operations** | Histogram không dùng phép toán nguyên tử | LightGBM và XGBoost trên GPU dựa vào phép toán nguyên tử (atomic), dễ lập trình nhưng chậm. CatBoost dựng histogram từng phần cho mỗi **warp** (nhóm 32 luồng) trong bộ nhớ chia sẻ, mỗi luồng lần lượt cập nhật các histogram khác nhau để tránh xung đột. | §5 |
| **Warp / shared memory / occupancy** | Warp / bộ nhớ chia sẻ / mức chiếm dụng | Khái niệm phần cứng GPU NVIDIA; CatBoost dùng hết bộ nhớ chia sẻ nên không đạt 100% occupancy và bù bằng loop unrolling. | §5 |
| **Perfect hash for categories** | Băm hoàn hảo cho hạng mục | Lưu giá trị biến phân loại bằng hàm băm hoàn hảo nén bit trong RAM của CPU, truyền sang GPU khi cần; xây bằng radix sort. | §5 |
| **Segmented scan** | Quét phân đoạn | Phép toán GPU tính tổng tiền tố trong từng nhóm, dùng để tính thống kê có thứ tự cho tổ hợp đặc trưng. | §5 |
| **Feature-parallel multi-GPU learning** | Học song song theo đặc trưng trên nhiều GPU | Vì cần nhiều hoán vị và tính thống kê biến phân loại trong lúc huấn luyện, CatBoost chia công việc theo đặc trưng chứ không theo mẫu. | §5 |

## 6. Experiments

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Quality comparison (tuned)** | So sánh chất lượng (đã tinh chỉnh) | Log-loss trên 8 bộ (Adult, Amazon, Appetency, Click, Internet, KDD98, KDD Churn, Kick); tinh chỉnh trên 4/5, kiểm tra trên 1/5, trung bình 5 lần chạy với hạt giống khác nhau. CatBoost tốt nhất trên mọi bộ (ví dụ Amazon: LightGBM +18,8%, XGBoost +18,6%, H2O +18,1%). | §6, Table 1 |
| **Default CatBoost vs tuned baselines** | CatBoost mặc định so với đối thủ đã tinh chỉnh | Theo kho GitHub của tác giả, CatBoost với tham số mặc định thắng XGBoost và H2O đã tinh chỉnh trên mọi bộ và LightGBM trên mọi bộ trừ một. | §6 |
| **GPU vs CPU training** | Huấn luyện GPU so với CPU | Epsilon (400K mẫu, 2.000 đặc trưng) và Criteo (36M mẫu, 26 biến phân loại): từ 1,3× (K40) tới 15× (V100) nhanh hơn CPU 32 luồng. | §6, Table 2 |
| **Tree construction time (GPU)** | Thời gian dựng cây trên GPU | Epsilon, độ sâu 6 (LightGBM 64 lá), 15 bin, chế độ boosting cổ điển: CatBoost 17,9 ms/cây, LightGBM 40 ms, XGBoost 488 ms. | §6 |
| **Hard to compare speed fairly** | Khó so sánh tốc độ công bằng | Mỗi thư viện có đánh đổi chất lượng/tốc độ riêng và nhiều tham số ảnh hưởng không rõ ràng; chỉ so được thời gian dựng mô hình có cùng kích thước. | §6 |

---

## 7. Notation summary

| Symbol | Ý nghĩa |
|---|---|
| $D=\{(X_i,Y_i)\}$ | Tập huấn luyện *n* mẫu, *m* đặc trưng |
| $x_{i,k}$ | Giá trị đặc trưng *k* của mẫu *i* |
| σ | Hoán vị ngẫu nhiên của dữ liệu |
| *P*, *a* | Prior và trọng số của prior |
| $F^i$, $g^i(X_k,Y_k)$ | Mô hình sau *i* cây; gradient của mẫu *k* |
| $M_i$, $M'_i$ | Mô hình hỗ trợ không thấy mẫu *i*; phiên bản rút gọn dựa trên $2^i$ mẫu đầu |
| *s* | Số hoán vị |
| *d* | Độ sâu cây đối xứng |
| $B(x,f)$ | Giá trị đặc trưng nhị phân *f* của mẫu *x* |

## 8. People and works cited

- **Dorogush, Gulin, Gusev, Prokhorenkova & Vorobev (2017) [5]:** arXiv version of the CatBoost unbiased-boosting paper (Prokhorenkova et al. 2018).
- **Friedman (2001) [8]; Friedman (2002) [9]:** gradient boosting; stochastic gradient boosting.
- **Breiman (1996) [1]:** out-of-bag estimation (biased gradients discussed).
- **Chen & Guestrin (2016) [4]:** XGBoost.
- **Micci-Barreca (2001) [14]; Cestnik (1990) [3]:** target statistics, priors.
- **Kohavi & Li (1995) [12]; Langley & Sage (1994) [13]:** oblivious decision trees. **Ferov & Modrý (2016) [7]; Gulin et al. (2011) [10]:** boosted oblivious trees.
- **Zhang, Si & Hsieh (2017) [11]:** GPU tree boosting. **Merrill (2016) [6]; Sengupta et al. (2008) [16]:** GPU scan primitives.
- **Caruana & Niculescu-Mizil (2006) [2]:** empirical comparison of supervised learning algorithms.

---

## 9. Relevance to this project (credit scoring with CatBoost)

These points connect the paper to the report's chapter 2 (§2.4.6 CatBoost) and complement the Prokhorenkova et al. glossary.

1. **Defaults have changed since 2018.** The paper says CatBoost uses the bias-fighting (ordered) scheme by default. CatBoost 1.2.10, used in the project, defaults to `boosting_type = Plain` (checked at runtime on 500 to 60,000 rows). When citing the paper, do not imply that the project's CatBoost used this scheme.
2. **Oblivious trees are what applied.** §4 explains the main structural difference that did affect the project: `SymmetricTree` (oblivious) trees of depth 6, which are balanced and fast to score. This supports the report's explanation for CatBoost's behaviour.
3. **Categorical handling was not exercised.** The project passes categories as integer codes without `cat_features`, so one-hot-in-training, ordered TS, counters and feature combinations (§2) were all unused. If categorical support is added later, eq. (1) is the formula to describe.
4. **Prior for binary classification.** The paper notes that for binary tasks the prior is usually the positive-class probability; for credit data that is the default rate. Useful if the report discusses target encoding.
5. **Speed numbers.** GPU and scorer benchmarks (§5–6) are not relevant to the project's CPU runs on small datasets; the project's runtime table measures CPU training.
6. **Which CatBoost paper to cite.** Cite Prokhorenkova et al. (2018) for prediction shift, ordered boosting and ordered TS (the theory); cite Dorogush et al. (2018) for the library, categorical feature combinations, oblivious-tree scoring and GPU training.
7. **What the paper does not cover.** Formal analysis of the bias (in [5]), calibration, AUC, class imbalance.

**Suggested citation sentence (VI):**
> *Dorogush và cộng sự (2018) giới thiệu thư viện CatBoost, xử lý biến phân loại ngay trong lúc huấn luyện bằng thống kê mục tiêu tính theo một hoán vị ngẫu nhiên và các tổ hợp đặc trưng, đồng thời dùng cây đối xứng (oblivious tree) giúp dự đoán nhanh và giảm overfitting.*
