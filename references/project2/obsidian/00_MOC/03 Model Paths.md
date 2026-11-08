---
type: moc
---

# Lộ trình theo mô hình — học xuôi và ngược

*Sinh tự động bởi `scripts/build_knowledge_tree.py` — đừng sửa tay.*

- **Ngược** (mô hình → nền tảng): mở ghi chú mô hình, xem *Lộ trình ngược*, hoặc đọc mục 3 dưới đây.
- **Xuôi** (nền tảng → mô hình): mở một khái niệm, xem *➡ Mở ra* và *🎯 Dùng trong mô hình*.
- Vai trò: ▫️ kiến thức trước · 🟦 nền · 🟨 kế thừa · 🟥 mới · 🎯 mô hình · ⚙️ đồ án.
- Mũi tên trong sơ đồ: A → B nghĩa là cần hiểu A trước B.

## 1. Bài báo kế thừa nhau thế nào

Mũi tên A → B: B trích dẫn A (lấy từ danh sách tài liệu tham khảo của B).

```mermaid
flowchart LR
    p0["Cox 1958<br/>Logistic Regression"]
    p1["Maron 1961<br/>Naive Bayes"]
    p2["Cover Hart 1967<br/>KNN"]
    p3["Breiman 2001<br/>Random Forest"]
    p4["Friedman 2001<br/>Gradient Boosting"]
    p5["Chen Guestrin 2016<br/>XGBoost"]
    p6["Ke 2017<br/>LightGBM"]
    p7["Dorogush 2018<br/>CatBoost"]
    p8["Prokhorenkova 2018<br/>CatBoost"]
    p4 --> p5
    p3 --> p5
    p4 --> p6
    p5 --> p6
    p4 --> p7
    p5 --> p7
    p8 --> p7
    p4 --> p8
    p5 --> p8
    p6 --> p8
    class p0,p1,p2,p3,p4,p5,p6,p7,p8 internal-link
```

| Năm | Bài báo | Mô hình | Trích dẫn | Được trích bởi |
|---|---|---|---|---|
| 1958 | [[Cox 1958]] | [[Logistic Regression]] | — | — |
| 1961 | [[Maron 1961]] | [[Naive Bayes]] | — | — |
| 1967 | [[Cover Hart 1967]] | [[KNN]] | — | — |
| 2001 | [[Breiman 2001]] | [[Random Forest]] | — | [[Chen Guestrin 2016]] |
| 2001 | [[Friedman 2001]] | [[Gradient Boosting]] | — | [[Chen Guestrin 2016]] · [[Ke 2017]] · [[Dorogush 2018]] · [[Prokhorenkova 2018]] |
| 2016 | [[Chen Guestrin 2016]] | [[XGBoost]] | [[Friedman 2001]] · [[Breiman 2001]] | [[Ke 2017]] · [[Dorogush 2018]] · [[Prokhorenkova 2018]] |
| 2017 | [[Ke 2017]] | [[LightGBM]] | [[Friedman 2001]] · [[Chen Guestrin 2016]] | [[Prokhorenkova 2018]] |
| 2018 | [[Dorogush 2018]] | [[CatBoost]] | [[Friedman 2001]] · [[Chen Guestrin 2016]] · [[Prokhorenkova 2018]] | — |
| 2018 | [[Prokhorenkova 2018]] | [[CatBoost]] | [[Friedman 2001]] · [[Chen Guestrin 2016]] · [[Ke 2017]] | [[Dorogush 2018]] |

## 2. Kiến thức dùng chung giữa các mô hình

| Khái niệm                         | Tầng | Logistic Regression | Naive Bayes | KNN | Random Forest | XGBoost | LightGBM | CatBoost | Số mô hình |
| --------------------------------- | ---- | ------------------- | ----------- | --- | ------------- | ------- | -------- | -------- | ---------- |
| [[Feature]]                       | `L1` | ✓                   | ✓           | ✓   | ✓             | ✓       | ✓        | ✓        | 7          |
| [[Supervised learning]]           | `L1` | ✓                   | ✓           | ✓   | ✓             | ✓       | ✓        | ✓        | 7          |
| [[Conditional probability]]       | `L0` |                     | ✓           | ✓   | ✓             | ✓       | ✓        |          | 5          |
| [[Generalization]]                | `L1` | ✓                   |             |     | ✓             | ✓       | ✓        | ✓        | 5          |
| [[Overfitting]]                   | `L1` | ✓                   |             |     | ✓             | ✓       | ✓        | ✓        | 5          |
| [[Bernoulli distribution]]        | `L0` | ✓                   |             |     |               | ✓       | ✓        | ✓        | 4          |
| [[Gradient descent]]              | `L0` | ✓                   |             |     |               | ✓       | ✓        | ✓        | 4          |
| [[Likelihood]]                    | `L0` | ✓                   |             |     |               | ✓       | ✓        | ✓        | 4          |
| [[Logistic function]]             | `L0` | ✓                   |             |     |               | ✓       | ✓        | ✓        | 4          |
| [[Logit]]                         | `L0` | ✓                   |             |     |               | ✓       | ✓        | ✓        | 4          |
| [[Newton method]]                 | `L0` | ✓                   |             |     |               | ✓       | ✓        | ✓        | 4          |
| [[Odds]]                          | `L0` | ✓                   |             |     |               | ✓       | ✓        | ✓        | 4          |
| [[Log-loss]]                      | `L1` | ✓                   |             |     |               | ✓       | ✓        | ✓        | 4          |
| [[Loss function]]                 | `L1` | ✓                   |             |     |               | ✓       | ✓        | ✓        | 4          |
| [[Decision tree]]                 | `L2` |                     |             |     | ✓             | ✓       | ✓        | ✓        | 4          |
| [[Tree ensembles]]                | `L2` |                     |             |     | ✓             | ✓       | ✓        | ✓        | 4          |
| [[Information gain]]              | `L0` |                     |             |     | ✓             | ✓       | ✓        |          | 3          |
| [[Prior probability]]             | `L0` |                     | ✓           | ✓   |               |         |          | ✓        | 3          |
| [[Regularization]]                | `L0` | ✓                   |             |     |               | ✓       | ✓        |          | 3          |
| [[Shannon entropy]]               | `L0` |                     |             |     | ✓             | ✓       | ✓        |          | 3          |
| [[Boosting]]                      | `L2` |                     |             |     |               | ✓       | ✓        | ✓        | 3          |
| [[Split criterion]]               | `L2` |                     |             |     | ✓             | ✓       | ✓        |          | 3          |
| [[Tree growth strategy]]          | `L2` |                     |             |     |               | ✓       | ✓        | ✓        | 3          |
| [[Gradient Boosting]]             | `L3` |                     |             |     |               | ✓       | ✓        | ✓        | 3          |
| [[Pseudo-residual]]               | `L3` |                     |             |     |               | ✓       | ✓        | ✓        | 3          |
| [[Shrinkage]]                     | `L3` |                     |             |     |               | ✓       | ✓        | ✓        | 3          |
| [[Bayes theorem]]                 | `L0` |                     | ✓           | ✓   |               |         |          |          | 2          |
| [[Posterior probability]]         | `L0` |                     | ✓           | ✓   |               |         |          |          | 2          |
| [[Sampling]]                      | `L0` |                     |             |     | ✓             |         | ✓        |          | 2          |
| [[Categorical encoding]]          | `L1` |                     |             |     |               |         | ✓        | ✓        | 2          |
| [[Feature scaling]]               | `L1` | ✓                   |             | ✓   |               |         |          |          | 2          |
| [[Log-loss gradient and Hessian]] | `L1` |                     |             |     |               | ✓       | ✓        |          | 2          |
| [[Histogram-based split]]         | `L3` |                     |             |     |               | ✓       | ✓        |          | 2          |
| [[Random feature subspace]]       | `L3` |                     |             |     | ✓             | ✓       |          |          | 2          |
| [[Regularized objective]]         | `L3` |                     |             |     |               | ✓       | ✓        |          | 2          |
| [[Split gain]]                    | `L3` |                     |             |     |               | ✓       | ✓        |          | 2          |

## 3. Lộ trình ngược của từng mô hình

### [[Logistic Regression]]

*Hồi quy logistic dự đoán xác suất vỡ nợ thế nào?* — 20 bước. Nền: [[Linear probabilistic models]] · Kế thừa: [[Maximum likelihood]] · [[Log-loss]] · Mới: [[Logistic function]] · [[Odds ratio]]

[[Bernoulli distribution]] → [[Gradient descent]] → [[Likelihood]] → [[Newton method]] → [[Odds]] → [[Logit]] → [[Supervised learning]] → [[Feature]] → [[Feature scaling]] → [[Generalization]] → [[Loss function]] → [[Overfitting]] → [[Regularization]] → [[Maximum likelihood]] → [[Logistic function]] → [[Linear probabilistic models]] → [[Log-loss]] → [[Odds ratio]] → [[Logistic Regression]] → [[LR config]]

```mermaid
flowchart TD
    n0["Bernoulli distribution"]
    n1["Feature"]
    n2["Feature scaling"]
    n3["Generalization"]
    n4["Gradient descent"]
    n5["LR config"]
    n6["Likelihood"]
    n7["Linear probabilistic models"]
    n8["Log-loss"]
    n9["Logistic Regression"]
    n10["Logistic function"]
    n11["Logit"]
    n12["Loss function"]
    n13["Maximum likelihood"]
    n14["Newton method"]
    n15["Odds"]
    n16["Odds ratio"]
    n17["Overfitting"]
    n18["Regularization"]
    n19["Supervised learning"]
    n19 --> n1
    n1 --> n2
    n19 --> n3
    n18 --> n5
    n2 --> n5
    n9 --> n5
    n0 --> n6
    n10 --> n7
    n1 --> n7
    n6 --> n8
    n12 --> n8
    n10 --> n8
    n7 --> n9
    n13 --> n9
    n8 --> n9
    n10 --> n9
    n16 --> n9
    n11 --> n10
    n15 --> n11
    n19 --> n12
    n6 --> n13
    n14 --> n13
    n4 --> n14
    n15 --> n16
    n3 --> n17
    n17 --> n18
    class n16,n10 new
    class n12,n18,n19,n15,n0,n17,n2,n6,n14,n3,n1,n4,n11 prereq
    class n8,n13 inherited
    class n9 model
    class n5 config
    class n7 base
    class n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19 internal-link
    classDef prereq fill:#f1f1f1,stroke:#9a9a9a
    classDef base fill:#cfe8ff,stroke:#0090ff
    classDef inherited fill:#fff1b8,stroke:#d4a800
    classDef new fill:#ffd6d6,stroke:#e5484d
    classDef model fill:#d6409f,color:#fff,stroke:#a02070
    classDef config fill:#e4defc,stroke:#6e56cf
```

### [[Naive Bayes]]

*Naive Bayes kết hợp xác suất của từng đặc trưng thế nào?* — 15 bước. Nền: [[Generative models]] · Kế thừa: [[Bayes theorem]] · [[Independence assumption]] · [[Posterior probability]] · Mới: [[Naive Bayes formula]] · [[Smoothing]]

[[Conditional probability]] → [[Contingency table]] → [[Empty cells and small counts]] → [[Gaussian distribution]] → [[Prior probability]] → [[Supervised learning]] → [[Feature]] → [[Bayes theorem]] → [[Generative models]] → [[Independence assumption]] → [[Posterior probability]] → [[Smoothing]] → [[Naive Bayes formula]] → [[Naive Bayes]] → [[NB config]]

```mermaid
flowchart TD
    n0["Bayes theorem"]
    n1["Conditional probability"]
    n2["Contingency table"]
    n3["Empty cells and small counts"]
    n4["Feature"]
    n5["Gaussian distribution"]
    n6["Generative models"]
    n7["Independence assumption"]
    n8["NB config"]
    n9["Naive Bayes"]
    n10["Naive Bayes formula"]
    n11["Posterior probability"]
    n12["Prior probability"]
    n13["Smoothing"]
    n14["Supervised learning"]
    n1 --> n0
    n12 --> n0
    n2 --> n3
    n14 --> n4
    n0 --> n6
    n4 --> n6
    n1 --> n7
    n5 --> n8
    n9 --> n8
    n6 --> n9
    n0 --> n9
    n7 --> n9
    n11 --> n9
    n10 --> n9
    n13 --> n9
    n0 --> n10
    n7 --> n10
    n0 --> n11
    n3 --> n13
    class n4,n2,n14,n5,n3,n1,n12 prereq
    class n9 model
    class n11,n7,n0 inherited
    class n8 config
    class n6 base
    class n10,n13 new
    class n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14 internal-link
    classDef prereq fill:#f1f1f1,stroke:#9a9a9a
    classDef base fill:#cfe8ff,stroke:#0090ff
    classDef inherited fill:#fff1b8,stroke:#d4a800
    classDef new fill:#ffd6d6,stroke:#e5484d
    classDef model fill:#d6409f,color:#fff,stroke:#a02070
    classDef config fill:#e4defc,stroke:#6e56cf
```

### [[KNN]]

*KNN phân loại theo các láng giềng gần nhất thế nào?* — 13 bước. Nền: [[Distance-based models]] · Kế thừa: [[Posterior probability]] · Mới: [[Nearest neighbor error bound]]

[[Conditional probability]] → [[Euclidean distance]] → [[Prior probability]] → [[Bayes theorem]] → [[Supervised learning]] → [[Feature]] → [[Feature scaling]] → [[Distance-based models]] → [[Posterior probability]] → [[Bayes error]] → [[Nearest neighbor error bound]] → [[KNN]] → [[KNN config]]

```mermaid
flowchart TD
    n0["Bayes error"]
    n1["Bayes theorem"]
    n2["Conditional probability"]
    n3["Distance-based models"]
    n4["Euclidean distance"]
    n5["Feature"]
    n6["Feature scaling"]
    n7["KNN"]
    n8["KNN config"]
    n9["Nearest neighbor error bound"]
    n10["Posterior probability"]
    n11["Prior probability"]
    n12["Supervised learning"]
    n10 --> n0
    n2 --> n1
    n11 --> n1
    n4 --> n3
    n5 --> n3
    n12 --> n5
    n5 --> n6
    n3 --> n7
    n10 --> n7
    n9 --> n7
    n6 --> n8
    n7 --> n8
    n0 --> n9
    n3 --> n9
    n1 --> n10
    class n4,n0,n5,n6,n12,n2,n11,n1 prereq
    class n8 config
    class n9 new
    class n7 model
    class n3 base
    class n10 inherited
    class n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12 internal-link
    classDef prereq fill:#f1f1f1,stroke:#9a9a9a
    classDef base fill:#cfe8ff,stroke:#0090ff
    classDef inherited fill:#fff1b8,stroke:#d4a800
    classDef new fill:#ffd6d6,stroke:#e5484d
    classDef model fill:#d6409f,color:#fff,stroke:#a02070
    classDef config fill:#e4defc,stroke:#6e56cf
```

### [[Random Forest]]

*Random Forest kết hợp các cây ngẫu nhiên thế nào?* — 20 bước. Nền: [[Bagging]] · Kế thừa: [[Split criterion]] · [[Bias-variance tradeoff]] · Mới: [[Random feature subspace]] · [[Out-of-bag error]] · [[Strength and correlation]]

[[Conditional probability]] → [[Sampling]] → [[Bootstrap]] → [[Shannon entropy]] → [[Information gain]] → [[Supervised learning]] → [[Feature]] → [[Generalization]] → [[Overfitting]] → [[Decision tree]] → [[Tree ensembles]] → [[Bagging]] → [[Bias-variance tradeoff]] → [[Split criterion]] → [[Gini impurity]] → [[Out-of-bag error]] → [[Random feature subspace]] → [[Strength and correlation]] → [[Random Forest]] → [[RF config]]

```mermaid
flowchart TD
    n0["Bagging"]
    n1["Bias-variance tradeoff"]
    n2["Bootstrap"]
    n3["Conditional probability"]
    n4["Decision tree"]
    n5["Feature"]
    n6["Generalization"]
    n7["Gini impurity"]
    n8["Information gain"]
    n9["Out-of-bag error"]
    n10["Overfitting"]
    n11["RF config"]
    n12["Random Forest"]
    n13["Random feature subspace"]
    n14["Sampling"]
    n15["Shannon entropy"]
    n16["Split criterion"]
    n17["Strength and correlation"]
    n18["Supervised learning"]
    n19["Tree ensembles"]
    n19 --> n0
    n2 --> n0
    n10 --> n1
    n14 --> n2
    n5 --> n4
    n18 --> n5
    n18 --> n6
    n16 --> n7
    n15 --> n8
    n3 --> n8
    n2 --> n9
    n6 --> n9
    n6 --> n10
    n7 --> n11
    n12 --> n11
    n0 --> n12
    n16 --> n12
    n1 --> n12
    n13 --> n12
    n9 --> n12
    n17 --> n12
    n4 --> n13
    n4 --> n16
    n8 --> n16
    n0 --> n17
    n6 --> n17
    n4 --> n19
    class n4,n18,n3,n2,n14,n10,n19,n6,n5,n8,n7,n15 prereq
    class n1,n16 inherited
    class n12 model
    class n17,n13,n9 new
    class n11 config
    class n0 base
    class n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19 internal-link
    classDef prereq fill:#f1f1f1,stroke:#9a9a9a
    classDef base fill:#cfe8ff,stroke:#0090ff
    classDef inherited fill:#fff1b8,stroke:#d4a800
    classDef new fill:#ffd6d6,stroke:#e5484d
    classDef model fill:#d6409f,color:#fff,stroke:#a02070
    classDef config fill:#e4defc,stroke:#6e56cf
```

### [[XGBoost]]

*XGBoost thêm điều chuẩn và xấp xỉ bậc hai vào gradient boosting thế nào?* — 35 bước. Nền: [[Gradient Boosting]] · Kế thừa: [[Regularization]] · [[Random feature subspace]] · Mới: [[Regularized objective]] · [[Split gain]] · [[Weighted quantile sketch]] · [[Sparsity-aware split finding]]

[[Bernoulli distribution]] → [[Conditional probability]] → [[Gradient descent]] → [[Likelihood]] → [[Newton method]] → [[Odds]] → [[Logit]] → [[Logistic function]] → [[Shannon entropy]] → [[Information gain]] → [[Supervised learning]] → [[Feature]] → [[Generalization]] → [[Loss function]] → [[Log-loss]] → [[Log-loss gradient and Hessian]] → [[Overfitting]] → [[Decision tree]] → [[Split criterion]] → [[Tree ensembles]] → [[Boosting]] → [[Tree growth strategy]] → [[Histogram-based split]] → [[Level-wise growth]] → [[Pseudo-residual]] → [[Shrinkage]] → [[Gradient Boosting]] → [[Regularization]] → [[Random feature subspace]] → [[Regularized objective]] → [[Split gain]] → [[Sparsity-aware split finding]] → [[Weighted quantile sketch]] → [[XGBoost]] → [[XGBoost config]]

```mermaid
flowchart TD
    n0["Bernoulli distribution"]
    n1["Boosting"]
    n2["Conditional probability"]
    n3["Decision tree"]
    n4["Feature"]
    n5["Generalization"]
    n6["Gradient Boosting"]
    n7["Gradient descent"]
    n8["Histogram-based split"]
    n9["Information gain"]
    n10["Level-wise growth"]
    n11["Likelihood"]
    n12["Log-loss"]
    n13["Log-loss gradient and Hessian"]
    n14["Logistic function"]
    n15["Logit"]
    n16["Loss function"]
    n17["Newton method"]
    n18["Odds"]
    n19["Overfitting"]
    n20["Pseudo-residual"]
    n21["Random feature subspace"]
    n22["Regularization"]
    n23["Regularized objective"]
    n24["Shannon entropy"]
    n25["Shrinkage"]
    n26["Sparsity-aware split finding"]
    n27["Split criterion"]
    n28["Split gain"]
    n29["Supervised learning"]
    n30["Tree ensembles"]
    n31["Tree growth strategy"]
    n32["Weighted quantile sketch"]
    n33["XGBoost"]
    n34["XGBoost config"]
    n30 --> n1
    n4 --> n3
    n29 --> n4
    n29 --> n5
    n1 --> n6
    n7 --> n6
    n16 --> n6
    n12 --> n6
    n17 --> n6
    n20 --> n6
    n25 --> n6
    n27 --> n8
    n24 --> n9
    n2 --> n9
    n31 --> n10
    n0 --> n11
    n11 --> n12
    n16 --> n12
    n14 --> n12
    n12 --> n13
    n17 --> n13
    n15 --> n14
    n18 --> n15
    n29 --> n16
    n7 --> n17
    n5 --> n19
    n7 --> n20
    n16 --> n20
    n3 --> n21
    n19 --> n22
    n22 --> n23
    n16 --> n23
    n19 --> n25
    n28 --> n26
    n3 --> n27
    n9 --> n27
    n23 --> n28
    n13 --> n28
    n27 --> n28
    n3 --> n30
    n3 --> n31
    n28 --> n32
    n6 --> n33
    n22 --> n33
    n21 --> n33
    n23 --> n33
    n28 --> n33
    n32 --> n33
    n26 --> n33
    n8 --> n34
    n10 --> n34
    n33 --> n34
    class n8,n25,n13,n29,n11,n30,n20,n14,n16,n3,n12,n18,n2,n0,n31,n17,n19,n5,n1,n4,n9,n7,n15,n27,n24,n10 prereq
    class n34 config
    class n22,n21 inherited
    class n33 model
    class n6 base
    class n28,n32,n23,n26 new
    class n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25,n26,n27,n28,n29,n30,n31,n32,n33,n34 internal-link
    classDef prereq fill:#f1f1f1,stroke:#9a9a9a
    classDef base fill:#cfe8ff,stroke:#0090ff
    classDef inherited fill:#fff1b8,stroke:#d4a800
    classDef new fill:#ffd6d6,stroke:#e5484d
    classDef model fill:#d6409f,color:#fff,stroke:#a02070
    classDef config fill:#e4defc,stroke:#6e56cf
```

### [[LightGBM]]

*LightGBM làm gradient boosting nhanh hơn thế nào?* — 36 bước. Nền: [[Gradient Boosting]] · Kế thừa: [[Split gain]] · [[Histogram-based split]] · [[Leaf-wise growth]] · Mới: [[GOSS]] · [[EFB]]

[[Bernoulli distribution]] → [[Conditional probability]] → [[Gradient descent]] → [[Likelihood]] → [[Newton method]] → [[Odds]] → [[Logit]] → [[Logistic function]] → [[Sampling]] → [[Shannon entropy]] → [[Information gain]] → [[Supervised learning]] → [[Feature]] → [[Categorical encoding]] → [[Generalization]] → [[Loss function]] → [[Log-loss]] → [[Log-loss gradient and Hessian]] → [[Overfitting]] → [[Regularization]] → [[Decision tree]] → [[Split criterion]] → [[Tree ensembles]] → [[Boosting]] → [[Tree growth strategy]] → [[Pseudo-residual]] → [[Regularized objective]] → [[Shrinkage]] → [[Gradient Boosting]] → [[Histogram-based split]] → [[Split gain]] → [[Leaf-wise growth]] → [[EFB]] → [[GOSS]] → [[LightGBM]] → [[LightGBM config]]

```mermaid
flowchart TD
    n0["Bernoulli distribution"]
    n1["Boosting"]
    n2["Categorical encoding"]
    n3["Conditional probability"]
    n4["Decision tree"]
    n5["EFB"]
    n6["Feature"]
    n7["GOSS"]
    n8["Generalization"]
    n9["Gradient Boosting"]
    n10["Gradient descent"]
    n11["Histogram-based split"]
    n12["Information gain"]
    n13["Leaf-wise growth"]
    n14["LightGBM"]
    n15["LightGBM config"]
    n16["Likelihood"]
    n17["Log-loss"]
    n18["Log-loss gradient and Hessian"]
    n19["Logistic function"]
    n20["Logit"]
    n21["Loss function"]
    n22["Newton method"]
    n23["Odds"]
    n24["Overfitting"]
    n25["Pseudo-residual"]
    n26["Regularization"]
    n27["Regularized objective"]
    n28["Sampling"]
    n29["Shannon entropy"]
    n30["Shrinkage"]
    n31["Split criterion"]
    n32["Split gain"]
    n33["Supervised learning"]
    n34["Tree ensembles"]
    n35["Tree growth strategy"]
    n34 --> n1
    n6 --> n2
    n6 --> n4
    n2 --> n5
    n11 --> n5
    n33 --> n6
    n28 --> n7
    n32 --> n7
    n33 --> n8
    n1 --> n9
    n10 --> n9
    n21 --> n9
    n17 --> n9
    n22 --> n9
    n25 --> n9
    n30 --> n9
    n31 --> n11
    n29 --> n12
    n3 --> n12
    n35 --> n13
    n32 --> n13
    n9 --> n14
    n32 --> n14
    n11 --> n14
    n13 --> n14
    n7 --> n14
    n5 --> n14
    n14 --> n15
    n0 --> n16
    n16 --> n17
    n21 --> n17
    n19 --> n17
    n17 --> n18
    n22 --> n18
    n20 --> n19
    n23 --> n20
    n33 --> n21
    n10 --> n22
    n8 --> n24
    n10 --> n25
    n21 --> n25
    n24 --> n26
    n26 --> n27
    n21 --> n27
    n24 --> n30
    n4 --> n31
    n12 --> n31
    n27 --> n32
    n18 --> n32
    n31 --> n32
    n4 --> n34
    n4 --> n35
    class n11,n32,n13 inherited
    class n30,n18,n26,n33,n16,n34,n25,n2,n19,n21,n4,n17,n23,n3,n0,n35,n28,n22,n24,n8,n1,n6,n12,n27,n10,n20,n31,n29 prereq
    class n5,n7 new
    class n9 base
    class n14 model
    class n15 config
    class n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25,n26,n27,n28,n29,n30,n31,n32,n33,n34,n35 internal-link
    classDef prereq fill:#f1f1f1,stroke:#9a9a9a
    classDef base fill:#cfe8ff,stroke:#0090ff
    classDef inherited fill:#fff1b8,stroke:#d4a800
    classDef new fill:#ffd6d6,stroke:#e5484d
    classDef model fill:#d6409f,color:#fff,stroke:#a02070
    classDef config fill:#e4defc,stroke:#6e56cf
```

### [[CatBoost]]

*CatBoost chống dịch chuyển dự đoán và xử lý biến phân loại thế nào?* — 30 bước. Nền: [[Gradient Boosting]] · Kế thừa: [[Categorical encoding]] · [[Data leakage]] · Mới: [[Prediction shift]] · [[Ordered boosting]] · [[Ordered target statistics]] · [[Oblivious tree]]

[[Bernoulli distribution]] → [[Gradient descent]] → [[Likelihood]] → [[Newton method]] → [[Odds]] → [[Logit]] → [[Logistic function]] → [[Prior probability]] → [[Supervised learning]] → [[Feature]] → [[Generalization]] → [[Loss function]] → [[Log-loss]] → [[Overfitting]] → [[Train-test split]] → [[Decision tree]] → [[Tree ensembles]] → [[Boosting]] → [[Tree growth strategy]] → [[Pseudo-residual]] → [[Shrinkage]] → [[Gradient Boosting]] → [[Categorical encoding]] → [[Data leakage]] → [[Oblivious tree]] → [[Ordered target statistics]] → [[Prediction shift]] → [[Ordered boosting]] → [[CatBoost]] → [[CatBoost config]]

```mermaid
flowchart TD
    n0["Bernoulli distribution"]
    n1["Boosting"]
    n2["CatBoost"]
    n3["CatBoost config"]
    n4["Categorical encoding"]
    n5["Data leakage"]
    n6["Decision tree"]
    n7["Feature"]
    n8["Generalization"]
    n9["Gradient Boosting"]
    n10["Gradient descent"]
    n11["Likelihood"]
    n12["Log-loss"]
    n13["Logistic function"]
    n14["Logit"]
    n15["Loss function"]
    n16["Newton method"]
    n17["Oblivious tree"]
    n18["Odds"]
    n19["Ordered boosting"]
    n20["Ordered target statistics"]
    n21["Overfitting"]
    n22["Prediction shift"]
    n23["Prior probability"]
    n24["Pseudo-residual"]
    n25["Shrinkage"]
    n26["Supervised learning"]
    n27["Train-test split"]
    n28["Tree ensembles"]
    n29["Tree growth strategy"]
    n28 --> n1
    n9 --> n2
    n4 --> n2
    n5 --> n2
    n22 --> n2
    n19 --> n2
    n20 --> n2
    n17 --> n2
    n2 --> n3
    n7 --> n4
    n27 --> n5
    n7 --> n6
    n26 --> n7
    n26 --> n8
    n1 --> n9
    n10 --> n9
    n15 --> n9
    n12 --> n9
    n16 --> n9
    n24 --> n9
    n25 --> n9
    n0 --> n11
    n11 --> n12
    n15 --> n12
    n13 --> n12
    n14 --> n13
    n18 --> n14
    n26 --> n15
    n10 --> n16
    n29 --> n17
    n22 --> n19
    n4 --> n20
    n5 --> n20
    n23 --> n20
    n8 --> n21
    n24 --> n22
    n5 --> n22
    n10 --> n24
    n15 --> n24
    n21 --> n25
    n8 --> n27
    n6 --> n28
    n6 --> n29
    class n15,n25,n6,n12,n26,n18,n29,n0,n27,n21,n16,n11,n28,n24,n8,n1,n7,n23,n10,n14,n13 prereq
    class n2 model
    class n5,n4 inherited
    class n20,n22,n19,n17 new
    class n9 base
    class n3 config
    class n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25,n26,n27,n28,n29 internal-link
    classDef prereq fill:#f1f1f1,stroke:#9a9a9a
    classDef base fill:#cfe8ff,stroke:#0090ff
    classDef inherited fill:#fff1b8,stroke:#d4a800
    classDef new fill:#ffd6d6,stroke:#e5484d
    classDef model fill:#d6409f,color:#fff,stroke:#a02070
    classDef config fill:#e4defc,stroke:#6e56cf
```

