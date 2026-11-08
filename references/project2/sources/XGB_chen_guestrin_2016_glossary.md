# Glossary — Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*

**Source:** T. Chen & C. Guestrin (2016). "XGBoost: A Scalable Tree Boosting System". *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, 785–794. University of Washington. DOI 10.1145/2939672.2939785. File: `references/project2/refs/xgboost_chen&guestrin_2016.pdf` (10 pages, pp. 785–794).

**What the paper does:** XGBoost is gradient tree boosting (Friedman 2001) built as a fast, scalable system. The paper contributes:
- a **regularized objective** $\sum l + \sum(\gamma T+\tfrac12\lambda\lVert w\rVert^2)$ solved with a **second-order (Newton) approximation**, giving closed-form **optimal leaf weights** and a **split gain** formula;
- **shrinkage** and **column subsampling** against overfitting;
- split finding by the **exact greedy** algorithm or an **approximate** algorithm with candidate splits from a new **weighted quantile sketch**;
- **sparsity-aware split finding** that learns a **default direction** for missing values;
- system design: **column blocks** for parallel learning, **cache-aware** access, **block compression and sharding** for out-of-core training.

On Higgs-1M it matches scikit-learn's accuracy while being more than 10× faster, and it scales to 1.7 billion rows.

**How to read this file:** Terms are grouped by topic, in roughly the order the paper uses them.
- Each entry gives the English term, a suggested Vietnamese term, a definition in Vietnamese (định nghĩa), the formula where there is one, and where it appears (section / page).
- The Vietnamese terms are suggestions for the report (`report/sections_v1`); check them against `tu_viet_tat.md` and chapter 2.
- Equation numbers (1)–(9) and Algorithms 1–3 are the paper's own. The paper uses *K* for the number of trees, *T* for the number of leaves, *m* for the number of features and *t* for the boosting iteration.

---

## 1. Tree ensemble model and regularized objective

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Gradient tree boosting (GBM, GBRT)** | Gradient boosting cây | Phương pháp của Friedman (2001); còn gọi là gradient boosting machine hoặc gradient boosted regression tree. XGBoost là một hệ thống cài đặt phương pháp này. | §1 p.785 |
| **Tree ensemble model** | Mô hình tập hợp cây | $\hat y_i=\phi(x_i)=\sum_{k=1}^K f_k(x_i)$, $f_k\in\mathcal F$ (1): dự đoán là **tổng điểm** của *K* cây. | §2.1 eq. (1), Fig. 1 |
| **Space of regression trees $\mathcal F$ (CART)** | Không gian cây hồi quy | $\mathcal F=\{f(x)=w_{q(x)}\}$, với $q:\mathbb R^m\to T$ là cấu trúc cây (ánh xạ mẫu vào chỉ số lá) và $w\in\mathbb R^T$ là trọng số lá. | §2.1 |
| **Tree structure q** | Cấu trúc cây *q* | Hàm đưa mỗi mẫu tới một lá theo các quy tắc chia. | §2.1 |
| **Leaf weight / leaf score $w_j$** | Trọng số lá / điểm lá | Giá trị liên tục tại lá *j*; khác cây quyết định thông thường (cho nhãn), cây hồi quy cho **điểm số**. | §2.1 |
| **Regularized objective** | Hàm mục tiêu có điều chuẩn | $\mathcal L(\phi)=\sum_i l(\hat y_i,y_i)+\sum_k\Omega(f_k)$, $\Omega(f)=\gamma T+\tfrac12\lambda\lVert w\rVert^2$ (2). Khi các tham số điều chuẩn bằng 0, quay về gradient boosting truyền thống. | §2.1 eq. (2) |
| **Differentiable convex loss l** | Hàm mất mát lồi khả vi | Đo chênh lệch giữa dự đoán và nhãn; với phân loại nhị phân là log-loss. | §2.1 |
| **Complexity penalty Ω** | Phạt độ phức tạp Ω | γ phạt **số lá** *T*; λ phạt **bình phương trọng số lá** (L2). Giúp làm trơn trọng số đã học, tránh overfitting; mô hình có xu hướng chọn các hàm đơn giản mà vẫn dự đoán tốt. | §2.1 |
| **Regularized greedy forest (RGF)** | Rừng tham lam có điều chuẩn | Zhang & Johnson (2014): mô hình có điều chuẩn tương tự; mục tiêu của XGBoost đơn giản hơn và dễ song song hóa hơn. | §2.1, §5 |

## 2. Second-order gradient boosting

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Additive training** | Huấn luyện cộng dần | Ở vòng *t*, thêm hàm $f_t$ làm giảm nhiều nhất $\mathcal L^{(t)}=\sum_i l(y_i,\hat y_i^{(t-1)}+f_t(x_i))+\Omega(f_t)$. | §2.2 |
| **Second-order approximation** | Xấp xỉ bậc hai | Khai triển Taylor bậc hai của mất mát quanh $\hat y^{(t-1)}$: $\mathcal L^{(t)}\simeq\sum_i[l(y_i,\hat y^{(t-1)})+g_if_t(x_i)+\tfrac12h_if_t^2(x_i)]+\Omega(f_t)$. Ý tưởng bậc hai bắt nguồn từ Friedman, Hastie & Tibshirani (2000). | §2.2 |
| **Gradient statistics $g_i$, $h_i$** | Thống kê gradient bậc một, bậc hai | $g_i=\partial_{\hat y^{(t-1)}}l(y_i,\hat y^{(t-1)})$, $h_i=\partial^2_{\hat y^{(t-1)}}l(y_i,\hat y^{(t-1)})$. Với log-loss trên thang log-odds: $g_i=p_i-y_i$, $h_i=p_i(1-p_i)$. | §2.2 |
| **Simplified objective** | Mục tiêu rút gọn | Bỏ hằng số: $\tilde{\mathcal L}^{(t)}=\sum_i[g_if_t(x_i)+\tfrac12h_if_t^2(x_i)]+\Omega(f_t)$ (3). | §2.2 eq. (3) |
| **Instance set of leaf $I_j$** | Tập mẫu của lá $I_j$ | $I_j=\{i\mid q(x_i)=j\}$: các mẫu rơi vào lá *j*; ký hiệu $G_j=\sum_{i\in I_j}g_i$, $H_j=\sum_{i\in I_j}h_i$. | §2.2 eq. (4) |
| **Optimal leaf weight $w_j^*$** | Trọng số lá tối ưu | Với cấu trúc *q* cố định: $w_j^*=-\dfrac{\sum_{i\in I_j}g_i}{\sum_{i\in I_j}h_i+\lambda}$ (5). Khi λ = 0, đây chính là bước Newton cho giá trị lá của Friedman. | §2.2 eq. (5) |
| **Structure score** | Điểm cấu trúc | Giá trị mục tiêu tối ưu của một cấu trúc cây: $\tilde{\mathcal L}^{(t)}(q)=-\tfrac12\sum_{j=1}^T\dfrac{(\sum_{i\in I_j}g_i)^2}{\sum_{i\in I_j}h_i+\lambda}+\gamma T$ (6). Giống độ vẩn đục của cây quyết định nhưng dùng được cho nhiều hàm mục tiêu hơn. | §2.2 eq. (6), Fig. 2 |
| **Greedy tree growing** | Phát triển cây tham lam | Không thể liệt kê mọi cấu trúc cây; bắt đầu từ một lá và thêm nhánh lần lượt. | §2.2 |
| **Split gain (loss reduction)** | Độ lợi của phép chia | $\mathcal L_{split}=\tfrac12\Big[\dfrac{G_L^2}{H_L+\lambda}+\dfrac{G_R^2}{H_R+\lambda}-\dfrac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big]-\gamma$ (7). γ đóng vai trò ngưỡng tối thiểu: phép chia có gain ≤ 0 không có lợi. | §2.2 eq. (7) |

## 3. Other overfitting controls

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Shrinkage η** | Co rút η (learning rate) | Nhân trọng số của mỗi cây mới thêm với η; giảm ảnh hưởng của từng cây và để chỗ cho các cây sau cải thiện mô hình. Bài trích Friedman [11] (*Stochastic gradient boosting*, 2002), dù co rút đã có trong Friedman (2001). | §2.3 |
| **Column (feature) subsampling** | Lấy mẫu con theo cột | Mỗi cây (hoặc mỗi lần chia) chỉ dùng một tập con đặc trưng; mượn từ Random Forest (Breiman 2001). Theo phản hồi người dùng, giảm overfit tốt hơn lấy mẫu theo hàng, và tăng tốc tính toán. | §2.3, §5 |
| **Row subsampling** | Lấy mẫu con theo hàng | Mỗi cây học trên một tập con mẫu (stochastic gradient boosting); XGBoost cũng hỗ trợ. | §2.3 |

## 4. Split finding algorithms

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Exact greedy algorithm (Algorithm 1)** | Thuật toán tham lam chính xác | Liệt kê **mọi** điểm chia khả dĩ trên mọi đặc trưng: sắp xếp dữ liệu theo giá trị đặc trưng, duyệt tuần tự và tích lũy $G_L,H_L$ để tính điểm (7). scikit-learn, R gbm và XGBoost một máy đều hỗ trợ. | §3.1 |
| **Approximate algorithm (Algorithm 2)** | Thuật toán xấp xỉ | Đề xuất các **điểm chia ứng viên** theo phân vị của mỗi đặc trưng, gom các giá trị liên tục vào các **bucket** theo các điểm đó, cộng dồn thống kê gradient và chỉ tìm điểm chia tốt nhất trong các ứng viên. Cần khi dữ liệu không vừa bộ nhớ hoặc phân tán. | §3.2 |
| **Global vs local proposal** | Đề xuất toàn cục / cục bộ | Toàn cục: đề xuất ứng viên một lần cho cả cây; cục bộ: đề xuất lại sau mỗi lần chia. Cục bộ cần ít ứng viên hơn; toàn cục chính xác ngang nếu có đủ ứng viên. | §3.2, Fig. 3 |
| **Approximation factor ε** | Hệ số xấp xỉ ε | Độ chính xác của sketch; tương ứng khoảng 1/ε điểm ứng viên. | §3.2–3.3 |
| **Approximate histograms of gradient statistics** | Histogram xấp xỉ của thống kê gradient | Một cách khác là xây trực tiếp histogram của gradient (Tyree et al. 2011); đây là ý tưởng của phương pháp **histogram** (`tree_method = hist`) dùng trong các phiên bản XGBoost về sau và trong LightGBM. | §3.2 |
| **Weighted quantile sketch** | Phác thảo phân vị có trọng số | Tìm các điểm chia ứng viên sao cho mỗi khoảng chứa khoảng ε tổng trọng số, với **trọng số là $h_i$**; dùng hàm hạng $r_k(z)=\frac{1}{\sum h}\sum_{x<z}h$ (8) và điều kiện (9). Lý do dùng $h_i$: mục tiêu (3) chính là bình phương có trọng số với nhãn $g_i/h_i$ và trọng số $h_i$. Là đóng góp thuật toán mới của bài, có chứng minh trong phụ lục bổ sung. | §3.3 eq. (8)–(9) |
| **Quantile summary / sketch** | Tóm tắt phân vị | Bài toán kinh điển trong cơ sở dữ liệu (Greenwald & Khanna; Zhang & Wang) nhưng chỉ cho dữ liệu không trọng số. | §3.3, §5 |

## 5. Sparsity-aware split finding

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Sparse input** | Dữ liệu thưa | Nguyên nhân: (1) giá trị thiếu; (2) nhiều giá trị 0 trong thống kê; (3) sản phẩm phụ của kỹ thuật đặc trưng như **one-hot encoding**. | §3.4 |
| **Default direction** | Hướng mặc định | Mỗi nút có một hướng (trái/phải) cho mẫu bị thiếu giá trị đặc trưng dùng để chia; hướng tối ưu được **học từ dữ liệu**. | §3.4, Fig. 4 |
| **Sparsity-aware split finding (Algorithm 3)** | Tìm điểm chia nhận biết dữ liệu thưa | Chỉ duyệt các mẫu **không thiếu** $I_k$; thử đưa toàn bộ mẫu thiếu sang phải rồi sang trái, giữ hướng có gain lớn hơn. Độ phức tạp tuyến tính theo số phần tử khác thiếu; nhanh hơn 50 lần so với cách thông thường trên Allstate-10K. | §3.4, Fig. 5 |
| **Unified handling of sparsity** | Xử lý thống nhất mọi dạng thưa | Theo tác giả, đây là cách tiếp cận thống nhất đầu tiên cho mọi dạng dữ liệu thưa trong học cây. | §3.4, §5 |

## 6. System design

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Column block** | Khối cột | Dữ liệu lưu trong các khối bộ nhớ theo định dạng cột nén (CSC), mỗi cột **đã sắp xếp sẵn** theo giá trị đặc trưng; chỉ cần sắp xếp một lần trước khi huấn luyện, sau đó tìm điểm chia bằng một lần quét tuyến tính. Hỗ trợ song song hóa theo cột và lấy mẫu cột. | §4.1, Fig. 6 |
| **Compressed column (CSC) format** | Định dạng cột nén | Cách lưu ma trận thưa theo cột. | §4.1 |
| **Time complexity** | Độ phức tạp thời gian | Thuật toán chính xác gốc: $O(Kd\lVert x\rVert_0\log n)$; với khối cột: $O(Kd\lVert x\rVert_0+\lVert x\rVert_0\log n)$, tiết kiệm một thừa số log *n* (*d* là độ sâu tối đa, $\lVert x\rVert_0$ là số phần tử khác thiếu). | §4.1 |
| **Cache-aware access / prefetching** | Truy cập nhận biết bộ nhớ đệm / nạp trước | Thống kê gradient được đọc theo thứ tự đặc trưng nên truy cập bộ nhớ không liên tục; mỗi luồng nạp trước vào bộ đệm riêng rồi cộng dồn theo lô nhỏ, nhanh gấp đôi với dữ liệu lớn. | §4.2, Fig. 7–8 |
| **Block size** | Kích thước khối | Số mẫu tối đa trong một khối; quá nhỏ làm song song hóa kém, quá lớn gây cache miss; $2^{16}$ mẫu là cân bằng tốt. | §4.2, Fig. 9 |
| **Out-of-core computation** | Tính toán ngoài bộ nhớ | Chia dữ liệu thành các khối lưu trên đĩa, dùng luồng riêng để nạp trước khi tính toán. | §4.3 |
| **Block compression** | Nén khối | Nén theo cột và giải nén song song khi nạp; chỉ số hàng lưu bằng độ lệch 16 bit; tỷ lệ nén khoảng 26–29%; tăng tốc khoảng 3 lần. | §4.3, Fig. 11 |
| **Block sharding** | Phân mảnh khối | Phân dữ liệu lên nhiều đĩa luân phiên, mỗi đĩa một luồng nạp; tăng tốc thêm khoảng 2 lần. | §4.3 |
| **Distributed XGBoost** | XGBoost phân tán | Chạy trên Hadoop, MPI, Sun Grid Engine, Spark, Flink qua thư viện allreduce rabit; nhanh hơn Spark MLLib hơn 10 lần mỗi vòng. | §6.1, §6.6 |

## 7. Experiments

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Datasets** | Bộ dữ liệu | Allstate (10M, 4.227 đặc trưng, bảo hiểm, thưa do one-hot), Higgs Boson (10M, 28), Yahoo! LTRC (473K, 700, xếp hạng), Criteo (1,7 tỷ, 67, tỷ lệ click). | §6.2, Table 2 |
| **Common setting** | Thiết lập chung | Độ sâu tối đa 8, shrinkage 0,1, không lấy mẫu cột (trừ khi ghi rõ). | §6.2 |
| **Higgs-1M classification** | Phân loại Higgs-1M | 500 cây: XGBoost AUC 0,8304, 0,684 s/cây; scikit-learn AUC 0,8302, 28,5 s/cây; R gbm AUC 0,6224 (chỉ mở rộng một nhánh). Lấy mẫu cột 0,5 hơi kém hơn trên dữ liệu này. | §6.3, Table 3 |
| **Learning to rank (NDCG@10)** | Học xếp hạng | Trên Yahoo! LTRC, XGBoost nhanh hơn pGBRT; lấy mẫu cột vừa nhanh hơn vừa tốt hơn chút (0,7913). | §6.4, Table 4 |
| **Kaggle / KDDCup adoption** | Mức độ sử dụng | 17/29 lời giải thắng cuộc trên blog Kaggle năm 2015 dùng XGBoost; mọi đội top-10 KDDCup 2015 đều dùng. | §1 |

---

## 8. Notation summary

| Symbol | Ý nghĩa |
|---|---|
| $\hat y_i$, $\hat y_i^{(t)}$ | Dự đoán cho mẫu *i*; dự đoán sau vòng *t* |
| $f_k$, $f_t$ | Cây thứ *k*; cây thêm ở vòng *t* |
| *q*, $w$, *T* | Cấu trúc cây; vectơ trọng số lá; số lá |
| *l*, $\Omega$ | Hàm mất mát; phạt độ phức tạp |
| γ, λ | Phạt mỗi lá; phạt L2 trên trọng số lá |
| $g_i$, $h_i$ | Đạo hàm bậc một, bậc hai của mất mát |
| $I_j$, $G_j$, $H_j$ | Tập mẫu của lá *j*; tổng $g$, tổng $h$ trong lá |
| $I_L$, $I_R$ | Tập mẫu nút trái/phải sau khi chia |
| η | Shrinkage (learning rate) |
| ε | Hệ số xấp xỉ của quantile sketch |
| *K*, *m*, *n*, *d* | Số cây; số đặc trưng; số mẫu; độ sâu tối đa |

## 9. Key formulas

| Formula | Meaning (VI) |
|---|---|
| $\hat y_i=\sum_k f_k(x_i)$ | Dự đoán là tổng điểm của các cây |
| $\mathcal L=\sum l+\sum(\gamma T+\tfrac12\lambda\lVert w\rVert^2)$ | Mục tiêu có điều chuẩn |
| $w_j^*=-G_j/(H_j+\lambda)$ | Trọng số lá tối ưu |
| $-\tfrac12\sum_j G_j^2/(H_j+\lambda)+\gamma T$ | Điểm cấu trúc của cây |
| $\tfrac12[G_L^2/(H_L+\lambda)+G_R^2/(H_R+\lambda)-G^2/(H+\lambda)]-\gamma$ | Độ lợi của phép chia |

## 10. People and works cited (selected)

- **Friedman (2001) [10]:** gradient boosting. **Friedman (2002) [11]:** stochastic gradient boosting, shrinkage. **Friedman, Hastie & Tibshirani (2000) [12]:** additive logistic regression, second-order boosting.
- **Breiman (2001) [4]:** Random Forest, source of column subsampling.
- **Zhang & Johnson (2014) [25]:** regularized greedy forest.
- **Tyree et al. (2011) [22], pGBRT:** parallel boosted regression trees with histograms. **Panda et al. (2009) [19], PLANET; Ye et al. (2009) [23].**
- **Greenwald & Khanna (2001) [14]; Zhang & Wang (2007) [24]:** quantile summaries.
- **Burges (2010) [5]:** LambdaMART. **Li (2010) [16]:** robust LogitBoost.
- **Pedregosa et al. (2011) [20]:** scikit-learn. **Ridgeway [21]:** R gbm. **Meng et al. (2016) [18]:** Spark MLLib.

---

## 11. Relevance to this project (credit scoring with XGBoost)

These points connect the paper to the report's chapter 2 (§2.4.4 XGBoost).

1. **What the project actually runs.** `XGBClassifier` with library defaults (XGBoost 3.3.0): 100 trees, max depth 6, η = 0.3, λ = 1, γ = 0, `tree_method = hist`, depth-wise growth. The objective is the paper's (2) with log-loss, and leaf weights/gains are (5)–(7).
2. **Formulas in the report.** $g_i=p_i-y_i$, $h_i=p_i(1-p_i)$, $w^*=-G/(H+\lambda)$ and the Gain formula in §2.4.4 come from this paper. With λ = 0 the leaf weight reduces to Friedman's (2001) Newton step.
3. **Histogram, not weighted quantile sketch.** The paper's main approximate method is the weighted quantile sketch; the project's default `hist` method builds gradient histograms over pre-binned features, an approach the paper only mentions (§3.2) and that LightGBM popularised. Describe XGBoost's split finding accordingly.
4. **Level-wise growth is an implementation default.** The paper grows trees greedily up to a maximum depth but does not discuss level-wise vs leaf-wise. XGBoost's default `grow_policy = depthwise` is what makes it "level-wise" in Bảng 2.2.
5. **Missing values not exercised.** Tier B.1 imputes all missing values before training, so the default-direction mechanism of §3.4 is never used in the experiments. One-hot columns for classic models are sparse, but the GBDT matrices use integer category codes.
6. **Regularization at defaults.** Only λ = 1 is active by default (γ = 0, no row/column subsampling). Overfitting is controlled mainly by depth 6, η and the number of trees.
7. **Scale is not the point here.** Most of the paper (§4, §6.5–6.6) is about systems for very large data. The credit datasets are small to medium, so those optimisations matter for speed, not for the comparison of accuracy.
8. **What the paper does not cover.** Categorical feature handling (`enable_categorical` came later), leaf-wise growth, GOSS/EFB, prediction shift, or calibration. Cite LightGBM/CatBoost papers for those.

**Suggested citation sentence (VI):**
> *XGBoost (Chen và Guestrin, 2016) bổ sung vào gradient boosting một hàm mục tiêu có điều chuẩn trên số lá và trọng số lá, dùng xấp xỉ bậc hai của hàm mất mát để tính trọng số lá tối ưu $w^*=-G/(H+\lambda)$ và độ lợi của mỗi phép chia, đồng thời học hướng mặc định cho giá trị thiếu.*
