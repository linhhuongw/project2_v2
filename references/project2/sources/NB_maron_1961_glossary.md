# Glossary — Maron (1961), *Automatic Indexing: An Experimental Inquiry*

**Source:** M. E. Maron (1961). "Automatic Indexing: An Experimental Inquiry". *Journal of the ACM*, 8(3), 404–417. The RAND Corporation, Santa Monica, California; received January 1961. File: `references/project2/refs/naive_bayes_maron_1961.pdf` (14 pages, pp. 404–417, including Appendices A–C).

**What the paper does:** The task is to have a computer read a document and decide which of 32 subject categories it belongs to, using only the presence of selected "clue words". Maron:
- picks 90 clue words from 260 training abstracts;
- estimates $P(C_j)$ and $P(W_i \mid C_j)$ by counting;
- combines them with **Bayes' theorem** under the assumption that clue words are **independent within a category**;
- tests the method on the same 260 documents (84.6% top-1 correct) and on 145 unseen documents (51.8% top-1 for documents with ≥ 2 clue words).

The paper is usually cited as the first published use of what is now called the **Naive Bayes classifier** (applied to text classification).

**How to read this file:** Terms are grouped by topic, in roughly the order the paper uses them.
- Each entry gives the English term, a suggested Vietnamese term, a definition in Vietnamese (định nghĩa), the formula where there is one, and where it appears (section § / journal page).
- The Vietnamese terms are suggestions for the report (`report/sections_v1`); check them against `tu_viet_tat.md` and chapter 2.
- Equation numbers (1)–(9) are Maron's own numbers. Maron writes $P(W_k \cdot W_m)$ with a dot for "and"; here it is written the same way.

---

## 1. The problem: automatic indexing

| Term (EN)                                                     | Vietnamese                                           | Định nghĩa / công thức                                                                                                                                                                                                      | Where            |
| ------------------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| **Automatic indexing**                                        | Đánh chỉ mục tự động                                 | Bài toán quyết định một cách máy móc xem một tài liệu thuộc chủ đề (lĩnh vực tri thức) nào, tức là xác định tài liệu "nói về" cái gì. Ngày nay gọi là **phân loại văn bản** (text classification).                          | §1.1 p.404       |
| **Classifying (indexing) documents**                          | Phân loại tài liệu                                   | Gán mỗi tài liệu vào một hoặc nhiều danh mục chủ đề dựa trên nội dung của nó.                                                                                                                                               | Abstract, §1.1   |
| **Subject category**                                          | Danh mục chủ đề                                      | Một lớp nội dung (ví dụ "Information theory", "Boolean algebra"). Trong thí nghiệm có 32 danh mục (Phụ lục B). Tương ứng với **lớp** (class) trong bài toán phân loại.                                                      | §1.1, §4, App. B |
| **Not exclusive, not independent, but exhaustive categories** | Danh mục không loại trừ, không độc lập nhưng bao phủ | Các danh mục có thể chồng lấn nhau, nhưng mọi tài liệu đều phải thuộc ít nhất một danh mục.                                                                                                                                 | §1.1             |
| **Categories not defined extensionally**                      | Danh mục không được định nghĩa theo ngoại diên       | Danh mục không được xác định bằng cách liệt kê mọi tài liệu thuộc nó. Ngược lại, từ một khái niệm ít nhiều rõ ràng về danh mục, ta phải quyết định một tài liệu bất kỳ có thuộc về nó hay không. Đây là lý do bài toán khó. | §1.1             |
| **Two parts of classification**                               | Hai phần của bài toán phân loại                      | (1) Chọn các khía cạnh liên quan của đối tượng làm **bằng chứng** (tương ứng với chọn đặc trưng); (2) dùng bằng chứng đó để **dự đoán** danh mục đúng (tương ứng với mô hình phân loại).                                    | §1.1             |
| **Forming new concepts / new categories**                     | Hình thành khái niệm mới                             | Maron cho rằng phân loại đúng là dấu hiệu của trí thông minh; tạo ra danh mục mới còn là dấu hiệu cao hơn.                                                                                                                  | §1.1             |
| **Problem of indexing information (semantics)**               | Bài toán đánh chỉ mục thông tin (ngữ nghĩa)          | Từ và câu chỉ "đại diện" cho sự vật, không có quan hệ một-một giữa từ và sự kiện, và không có quy tắc rõ ràng nào cho biết tổ hợp câu nào nói về chủ đề nào. Phương pháp thống kê cho phép **bỏ qua** vấn đề ý nghĩa.       | §1.2             |
| **Statistical technique (two steps)**                         | Kỹ thuật thống kê (hai bước)                         | (1) Xác định quan hệ xác suất giữa từng từ mang nội dung và các danh mục chủ đề; (2) dùng các quan hệ đó để dự đoán danh mục của tài liệu chứa các từ ấy.                                                                   | §1.2             |
| **Automatic dissemination / standing requests**               | Phân phối tự động / yêu cầu thường trực              | Ứng dụng: máy đọc tài liệu đến hằng ngày và chuyển cho đúng người dùng, dựa trên các yêu cầu thông tin cố định của họ.                                                                                                      | §1.3             |
| **Mechanized library system / retrieval**                     | Hệ thống thư viện tự động hóa / truy hồi             | Đánh chỉ mục tự động giúp tài liệu được nhận dạng để truy hồi sau này.                                                                                                                                                      | §1.3             |
| **Information explosion**                                     | Bùng nổ thông tin                                    | Lượng tài liệu khoa học tăng nhanh, là động lực cho việc tự động hóa.                                                                                                                                                       | §1.3             |

## 2. Basic notions: words, evidence, hypotheses, predictions

| Term (EN)                                       | Vietnamese                           | Định nghĩa / công thức                                                                                                                                                          | Where                  |
| ----------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| **Fundamental thesis**                          | Luận điểm cơ bản                     | Thống kê về loại, tần suất, vị trí, thứ tự... của các từ được chọn là đủ để dự đoán khá tốt chủ đề của tài liệu. Không xét ý nghĩa của từng từ, chỉ xem chúng mô tả chủ đề nào. | §2.1 p.405             |
| **Clue word**                                   | Từ gợi ý (từ khóa manh mối)          | Một từ trong tài liệu được dùng làm manh mối để dự đoán danh mục. Tương ứng với **đặc trưng** (feature). Thí nghiệm dùng 90 từ (Phụ lục C).                                     | §2.1, §2.2, §5, App. C |
| **Content-bearing words vs logical-type words** | Từ mang nội dung / từ chức năng      | Từ mang nội dung ("electron", "transistor") là manh mối tốt; từ chức năng/logic ("if", "then", "the", "of") không cho biết gì về chủ đề. Ngày nay gọi là **stop words**.        | §2.2, §5.1             |
| **Rare-word inefficiency**                      | Từ hiếm kém hiệu quả                 | Từ mang nội dung rõ ràng nhưng rất hiếm thì là manh mối kém hiệu quả vì ít khi xuất hiện, và thống kê về chúng không đáng tin.                                                  | §2.2, §5.1             |
| **Universe of inquiry / objects**               | Không gian nghiên cứu / đối tượng    | Các đối tượng cơ bản là tập tài liệu đang xét.                                                                                                                                  | §2.2                   |
| **Properties / observables**                    | Thuộc tính / đại lượng quan sát được | Thuộc tính của tài liệu là các từ gợi ý mà nó chứa; đây là những gì ta đo được.                                                                                                 | §2.2                   |
| **Measurement**                                 | Phép đo                              | Danh sách các thuộc tính mà đối tượng có (từ gợi ý nào xuất hiện). Phép đo tinh vi hơn có thể ghi cả tần suất, phân bố, thứ tự. Tương ứng với **vectơ đặc trưng**.              | §2.2                   |
| **Evidence / evidence statement**               | Bằng chứng / mệnh đề bằng chứng      | Thông tin do phép đo cung cấp, ví dụ "Tài liệu $D_1$ chứa các từ $W_1, W_5, W_{11}$".                                                                                           | §2.2                   |
| **Hypotheses (statistical data)**               | Giả thuyết (dữ liệu thống kê)        | Dữ liệu thống kê liên hệ từ gợi ý với danh mục; có nhiều mức giả thuyết, mức sau phức tạp hơn mức trước. Tương ứng với **các tham số đã học** của mô hình.                      | §2.2                   |
| **Prediction**                                  | Dự đoán                              | Kết luận dạng "Tài liệu $D_1$ thuộc danh mục $C_7$ với xác suất *p*", rút ra từ bằng chứng và giả thuyết. Đây là đầu ra **xác suất** của bộ phân loại.                          | §2.2                   |
| **Empirical test**                              | Kiểm nghiệm thực nghiệm              | Thiết kế thí nghiệm: đánh chỉ mục một tập tài liệu, chọn từ gợi ý, tính ma trận từ–danh mục, rồi cho máy đánh chỉ mục một tập tài liệu **khác**.                                | §2.3                   |
| **Clue word vs. category matrix**               | Ma trận từ gợi ý – danh mục          | Ma trận mà phần tử (i, j) là số lần từ gợi ý thứ *i* xuất hiện trong các tài liệu thuộc danh mục thứ *j*. Đây là toàn bộ "mô hình" đã học.                                      | §2.3, §7.3             |

## 3. Corpus and data split

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Corpus** | Kho ngữ liệu | Tập tài liệu dùng cho thí nghiệm: 405 tóm tắt bài báo từ *IRE Transactions on Electronic Computers* (PGEC) năm 1959. | §3 p.407 |
| **Document (abstract)** | Tài liệu (bản tóm tắt) | Mỗi "tài liệu" thực ra là một bản tóm tắt; trung bình 79 từ. Ví dụ ở Phụ lục A. | §3.1, App. A |
| **Degree of discrimination** | Mức độ phân biệt | Các chủ đề được chọn đều trong lĩnh vực máy tính để bài toán **đủ khó**; nếu chủ đề quá xa nhau (âm nhạc so với nhiệt động lực học), phân loại sẽ quá dễ. | §3.1 |
| **Group 1 (population 1)** | Nhóm 1 (tập huấn luyện) | 260 tóm tắt (số tháng 3 và 6/1959), dùng để thu thập thống kê. Tương ứng với **tập huấn luyện**. | §3.2 |
| **Group 2 (population 2)** | Nhóm 2 (tập kiểm tra) | 145 tóm tắt (số tháng 9/1959), **không hề được xem** trong giai đoạn chuẩn bị, kể cả khi chọn từ gợi ý. Tương ứng với **tập kiểm tra độc lập (hold-out)**. | §3.2, §7.3 |
| **Word occurrences / different words** | Lượt xuất hiện từ / số từ khác nhau | Nhóm 1 có 20.515 lượt xuất hiện từ và 3.263 từ khác nhau. | §3.2, §5.1 |
| **Key punching** | Đục lỗ thẻ | Nhập từng từ vào máy bằng thẻ đục lỗ. | §3.2, §5.1 |
| **Uncontaminated population** | Tập không bị nhiễm | Tập dữ liệu không được dùng để thu thống kê (nhóm 2). Kết quả trên tập này mới phản ánh khả năng dự đoán thật. | §7.1 |
| **Avoiding bias** | Tránh thiên lệch | Không xem nhóm 2 khi thiết kế thí nghiệm để kết quả không bị thiên lệch, tương ứng với tránh **rò rỉ dữ liệu** (data leakage) ngày nay. | §7.3 |

## 4. Categories and labelling

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Classification system (PGEC, 10 major categories)** | Hệ phân loại của PGEC | PGEC có sẵn 10 danh mục chính, 5 trong số đó chia thành khoảng 3 danh mục con. Maron không dùng hệ này. | §4.1 |
| **Finer set of 32 categories** | Bộ 32 danh mục chi tiết hơn | Bộ danh mục do tác giả tạo cho thí nghiệm (Phụ lục B). | §4.1, App. B |
| **Sorting / manual indexing** | Phân loại thủ công | Mỗi tài liệu nhóm 1 được đọc kỹ và xếp vào 1–3 danh mục, tạo ra **nhãn** để huấn luyện. | §4.2 |
| **Multi-category documents** | Tài liệu đa danh mục | Khoảng 20% tài liệu thuộc 2 danh mục, rất ít thuộc 3, không có tài liệu nào thuộc hơn 3. Ngày nay gọi là **phân loại đa nhãn** (multi-label). | §4.2 |
| **Most / least "popular" category** | Danh mục phổ biến nhất / ít nhất | 37 tài liệu ở danh mục đông nhất, chỉ 2 ở danh mục ít nhất, tức là các lớp **mất cân bằng**. | §4.2 |
| **Index entry** | Mục chỉ mục | Một lần gán một tài liệu vào một danh mục. Tài liệu thuộc 2 danh mục tạo ra 2 mục chỉ mục. | §6.3 |

## 5. Clue word selection (feature selection)

| Term (EN)                                  | Vietnamese                          | Định nghĩa / công thức                                                                                                                                                                                                                                             | Where        |
| ------------------------------------------ | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------ |
| **Tokenization rules**                     | Quy tắc tách từ                     | Cụm trong ngoặc kép là một từ; cụm có gạch nối là một từ ("analog-to-digital"); dạng số ít và số nhiều là hai từ khác nhau ("circuit"/"circuits"); hai cách viết khác nhau là hai từ ("analogue"/"analog"). Không dùng gộp gốc từ (stemming).                      | §5.1         |
| **Rejecting logical-type words**           | Loại từ chức năng                   | 55 từ chức năng phổ biến nhất ("the", "of", "a"...) chiếm 8.402/20.515 lượt (dưới 2% số từ nhưng hơn 40% lượt xuất hiện), nên bị loại.                                                                                                                             | §5.1         |
| **Rejecting too-common content words**     | Loại từ nội dung quá phổ biến       | Các từ như "computer", "system", "data", "machine" quá chung cho lĩnh vực máy tính nên bị loại.                                                                                                                                                                    | §5.1         |
| **Rejecting rare words**                   | Loại từ hiếm                        | 2.120/3.263 từ (65%) xuất hiện dưới 3 lần, nên bị loại vì kém hiệu quả và thống kê không đáng tin. Còn lại khoảng 1.000 từ có tần suất trung bình.                                                                                                                 | §5.1         |
| **Relative frequency of occurrence**       | Tần suất xuất hiện tương đối        | Tỷ lệ số lần một từ xuất hiện. Từ được giữ lại là những từ có tần suất không quá cao, không quá thấp.                                                                                                                                                              | §5.1         |
| **"Peaked" vs flat distribution**          | Phân phối có đỉnh / phân phối phẳng | Với mỗi từ, xem số lần xuất hiện của nó ở từng danh mục. Nếu dồn vào ít nhất một danh mục ("có đỉnh"), từ đó là manh mối tốt; nếu đều nhau ("phẳng"), từ đó bị loại. Đây là tiêu chí chọn đặc trưng **bằng mắt**.                                                  | §5.1, §5.2   |
| **Criteria of adequacy for a clue word**   | Tiêu chí phù hợp của từ gợi ý       | Một từ tốt phải dẫn đến dự đoán đúng danh mục của tài liệu.                                                                                                                                                                                                        | §5.2         |
| **A priori probability distribution**      | Phân phối xác suất tiên nghiệm      | $P(C_j)$: xác suất một tài liệu bất kỳ thuộc danh mục *j* khi chưa nhìn vào từ nào. Biểu diễn độ bất định ban đầu.                                                                                                                                                 | §5.2, §6.1   |
| **Shannon entropy**                        | Entropy Shannon                     | Đo độ bất định của phân phối danh mục: $H = -\sum_{j=1}^{32} P(C_j)\log_2 P(C_j)$.                                                                                                                                                                                 | §5.2 p.409   |
| **Conditional entropy given a word**       | Entropy có điều kiện theo từ        | Độ bất định sau khi biết từ $W_i$ xuất hiện: $H' = -\sum_{j=1}^{32} P(C_j\mid W_i)\log_2 P(C_j\mid W_i)$.                                                                                                                                                          | §5.2         |
| **Uncertainty removed**                    | Lượng bất định được loại bỏ         | Hiệu giữa *H* và *H'*. Từ $W_1$ là manh mối tốt hơn $W_2$ nếu sự xuất hiện của nó loại bỏ nhiều bất định hơn (có tính thêm xác suất tiên nghiệm của từ). Đây là ý tưởng của **độ lợi thông tin** (information gain), cũng là tiêu chí chia nút của cây quyết định. | §5.2         |
| **Dependence relationships between words** | Quan hệ phụ thuộc giữa các từ       | Nếu xét sự phụ thuộc giữa các từ, việc xếp hạng từ sẽ rất phức tạp, nên Maron không làm.                                                                                                                                                                           | §5.2         |
| **Allowable clue words (90 words)**        | Tập từ gợi ý được chấp nhận         | 90 từ được chọn sao cho mỗi danh mục trong 32 danh mục có ít nhất một từ "có đỉnh" ở đó (Phụ lục C).                                                                                                                                                               | §5.2, App. C |

## 6. The prediction method (Naive Bayes)

| Term (EN)                                       | Vietnamese                                    | Định nghĩa / công thức                                                                                                                                                                                                                                    | Where              |
| ----------------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| **Inverse inference (evidence → hypothesis)**   | Suy luận ngược (từ bằng chứng đến giả thuyết) | Từ các từ gợi ý quan sát được, suy ra danh mục. Phép tính xác suất (định lý Bayes) cho công thức tính đúng.                                                                                                                                               | §6.1 p.409         |
| **Bayes' theorem (one clue word)**              | Định lý Bayes (một từ gợi ý)                  | $P(C_j\mid W_i) = \dfrac{P(C_j)\,P(W_i\mid C_j)}{P(W_i)}$ (1).                                                                                                                                                                                            | §6.1 eq. (1)       |
| **Prior probability $P(C_j)$**                  | Xác suất tiên nghiệm                          | Xác suất một tài liệu được xếp vào danh mục *j* trước khi xét từ nào.                                                                                                                                                                                     | §6.1               |
| **Likelihood $P(W_i\mid C_j)$**                 | Xác suất có điều kiện (hợp lý)                | Xác suất một tài liệu thuộc danh mục *j* chứa từ $W_i$.                                                                                                                                                                                                   | §6.1               |
| **Posterior probability $P(C_j\mid W_i)$**      | Xác suất hậu nghiệm                           | Xác suất tài liệu thuộc danh mục *j* khi biết nó chứa $W_i$.                                                                                                                                                                                              | §5.2, §6.1         |
| **Scaling factor k**                            | Hệ số chuẩn hóa *k*                           | Mẫu số $P(W_i)$ không đổi theo *j*, nên viết $P(C_j\mid W_i) = k\,P(C_j)\,P(W_i\mid C_j)$ (2). *k* được chọn để tổng các xác suất bằng 1.                                                                                                                 | §6.1 eq. (2), §6.2 |
| **Chain rule for two words**                    | Quy tắc nhân cho hai từ                       | $P(C_j\mid W_1 W_2) = \dfrac{P(C_j)\,P(W_1\mid C_j)\,P(W_2\mid C_j W_1)}{P(W_1)\,P(W_2\mid W_1)}$ (3), tức là $k\,P(C_j)\,P(W_1\mid C_j)\,P(W_2\mid C_j W_1)$ (4).                                                                                        | §6.2 eqs. (3)–(4)  |
| **Conditional independence assumption**         | Giả định độc lập có điều kiện                 | Giả định rằng, **trong cùng một danh mục**, hai từ gợi ý bất kỳ độc lập với nhau: $P(W_2\mid C_j W_1) = P(W_2\mid C_j)$, nên (4) trở thành $k\,P(C_j)\,P(W_1\mid C_j)\,P(W_2\mid C_j)$ (5). Đây chính là giả định **"ngây thơ"** (naive) của Naive Bayes. | §6.2 eq. (5)       |
| **"Clearly false" independence**                | Giả định độc lập rõ ràng là sai               | Maron thừa nhận giả định này sai, vì $P(W_k\mid C_j W_j) \ne P(W_k\mid C_j)$ (6), nhưng vẫn dùng để **đơn giản hóa tính toán** (dù làm giảm chất lượng).                                                                                                  | §6.2 eq. (6)       |
| **General Naive Bayes formula**                 | Công thức Naive Bayes tổng quát               | Với tài liệu chứa các từ $W_k, W_m, \dots, W_r$: $P(C_j\mid W_k W_m \cdots W_r) = k\,P(C_j)\,P(W_k\mid C_j)\,P(W_m\mid C_j)\cdots P(W_r\mid C_j)$ (7).                                                                                                    | §6.2 eq. (7)       |
| **Attribute number**                            | Số thuộc tính (điểm hậu nghiệm)               | Tên Maron đặt cho vế trái của (7). Mỗi tài liệu có 32 attribute number, mỗi danh mục một số. Tương ứng với **xác suất hậu nghiệm** mà bộ phân loại trả về.                                                                                                | §6.2               |
| **Normalization to 1**                          | Chuẩn hóa tổng bằng 1                         | *k* được chọn sao cho $\sum_{j=1}^{32} P(C_j\mid W_k \cdots W_r) = 1$ (8).                                                                                                                                                                                | §6.2 eq. (8)       |
| **Mutually exclusive and exhaustive (assumed)** | Giả định loại trừ lẫn nhau và bao phủ         | (8) ngầm giả định các danh mục loại trừ lẫn nhau và bao phủ. Thực tế danh mục **không** loại trừ (một tài liệu có thể thuộc nhiều danh mục), nhưng vẫn được xử lý như loại trừ để đơn giản tính toán.                                                     | §6.2               |
| **Only occurrence used**                        | Chỉ dùng sự xuất hiện                         | Bằng chứng chỉ là *từ nào xuất hiện*, không dùng số lần xuất hiện trong tài liệu.                                                                                                                                                                         | §6.1, §7.4         |

## 7. Parameter estimation and smoothing

| Term (EN) | Vietnamese | Định nghĩa / công thức | Where |
|---|---|---|---|
| **Estimate of $P(C_j)$** | Ước lượng xác suất tiên nghiệm | Số mục chỉ mục của danh mục *j* chia cho tổng số mục chỉ mục. Đây là ước lượng **tần suất tương đối**. | §6.3 p.411 |
| **Estimate of $P(W_i\mid C_j)$** | Ước lượng xác suất có điều kiện | Số lần từ *i* xuất hiện trong các tài liệu thuộc danh mục *j*, chia cho tổng số lần xuất hiện của mọi từ gợi ý trong danh mục *j*. Cách đếm này giống mô hình **Naive Bayes đa thức** (multinomial) ngày nay. | §6.3 |
| **Estimates vs probabilities** | Ước lượng và xác suất | Maron lưu ý các giá trị thu được thực chất chỉ là **ước lượng** của xác suất. | §7.2 footnote 2 |
| **Zero-probability problem** | Vấn đề xác suất bằng 0 | Vì (9) là tích các xác suất, chỉ cần một $P(W_i\mid C_j) = 0$ là toàn bộ attribute number bằng 0. Với kho chỉ 260 tài liệu, ma trận từ–danh mục có nhiều ô rỗng. | §7.3 eq. (9) p.413 |
| **Empty entries / sparse matrix** | Ô rỗng / ma trận thưa | Các ô bằng 0 trong ma trận từ–danh mục. Maron nhận xét nếu cỡ mẫu lớn hơn, số ô khác 0 sẽ tăng đáng kể. | §7.3 |
| **Small-value substitution (0.001)** | Thay giá trị nhỏ (0,001) | Mỗi ô rỗng được thay bằng 0,001: đủ để (9) không bằng 0 nhưng đủ nhỏ để không làm thay đổi thứ tự các attribute number. Đây là tiền thân của **làm trơn** (smoothing), như làm trơn Laplace hay `var_smoothing` trong GaussianNB. Do Paul Baran đề xuất, chỉ áp dụng cho nhóm 2. | §7.3, footnote 3 |

## 8. Evaluation and results

| Term (EN)                             | Vietnamese                                     | Định nghĩa / công thức                                                                                                                                                                | Where            |
| ------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| **Output list**                       | Danh sách đầu ra                               | Với mỗi tài liệu, máy in ra các danh mục kèm attribute number, xếp hạng giảm dần.                                                                                                     | §7.2             |
| **Correct category at the top**       | Danh mục đúng đứng đầu                         | Tiêu chí đánh giá chính: danh mục có attribute number lớn nhất có phải là danh mục đúng không. Ngày nay gọi là **độ chính xác top-1**.                                                | §7.2             |
| **Top-3 accuracy**                    | Độ chính xác top-3                             | Xác suất danh mục đúng nằm trong 3 vị trí đầu: 68/85 = 80% ở nhóm 2.                                                                                                                  | §7.3             |
| **Results on group 1 (training set)** | Kết quả trên nhóm 1                            | 12 tài liệu không chứa từ gợi ý nào và 1 tài liệu lỗi bị loại, còn 247. Top-1 đúng: 209/247 = 84,6%. Với tài liệu có ≥ 2 từ gợi ý: 191/210 = 91%.                                     | §7.2             |
| **More clue words → better indexing** | Nhiều từ gợi ý hơn → chỉ mục tốt hơn           | Bảng I: tỷ lệ đúng tăng theo số từ gợi ý trong tài liệu, từ 48,7% (1 từ) lên 84,9% (2), 85,2% (3), 91,1% (4), 95,7% (5), và 100% (6–8 từ).                                            | §7.2, Table I    |
| **Single-category results (group 1)** | Kết quả với tài liệu một danh mục              | Trong 157 tài liệu thuộc đúng 1 danh mục, có ≥ 2 từ gợi ý: đúng 143/157 = 91,1%.                                                                                                      | §7.2             |
| **Two-category results (group 1)**    | Kết quả với tài liệu hai danh mục              | Trong 50 tài liệu thuộc đúng 2 danh mục: 2 vị trí đầu đều đúng ở 39/50 = 78%. Cả 3 tài liệu thuộc 3 danh mục đều có 3 danh mục đúng ở 3 vị trí đầu.                                   | §7.2             |
| **Results on group 2 (test set)**     | Kết quả trên nhóm 2                            | 145 tài liệu: 20 không có từ gợi ý, 40 chỉ có 1 từ, còn 85 có ≥ 2 từ. Top-1 đúng: 44/85 = 51,8%; top-3: 80%. Tài liệu một danh mục: 33/66 = 50%.                                      | §7.3             |
| **Training vs test performance gap**  | Chênh lệch giữa tập huấn luyện và tập kiểm tra | Maron dự đoán và xác nhận rằng kết quả trên tập dùng để tính thống kê tốt hơn trên tập mới (91% so với 51,8%). Đây là hiện tượng ngày nay gọi là **quá khớp / sai số tổng quát hóa**. | §7.1, §7.2, §7.3 |
| **Chance-level comparison**           | So với mức ngẫu nhiên                          | 50% có vẻ thấp, nhưng xác suất đạt được mức đó hoặc cao hơn hoàn toàn do ngẫu nhiên gần như bằng 0 (có 32 danh mục).                                                                  | §7.3 footnote 4  |
| **Coverage problem**                  | Vấn đề độ phủ                                  | Tài liệu không chứa từ gợi ý nào thì không thể phân loại (12 ở nhóm 1, 20 ở nhóm 2).                                                                                                  | §7.2, §7.3       |

## 9. Proposed improvements and closing ideas

| Term (EN)                                        | Vietnamese                         | Định nghĩa / công thức                                                                                                                                                               | Where            |
| ------------------------------------------------ | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- |
| **Larger clue word list / larger sample**        | Danh sách từ lớn hơn / mẫu lớn hơn | Đề xuất dùng 900 từ thay vì 90, và 2.600 tài liệu thay vì 260, để có thống kê vững hơn.                                                                                              | §7.4             |
| **Higher-order statistics $P(W_k\mid C_j W_i)$** | Thống kê bậc cao                   | Dùng xác suất có điều kiện theo cả danh mục và từ khác, tức là **bỏ giả định độc lập**.                                                                                              | §7.4             |
| **Using word frequency**                         | Dùng tần suất từ                   | Xét không chỉ sự xuất hiện mà cả số lần từ xuất hiện trong tài liệu.                                                                                                                 | §7.4             |
| **Binary (all-or-none) indexing assumption**     | Giả định chỉ mục nhị phân          | Giả định ngầm rằng tài liệu hoặc thuộc hoặc không thuộc một danh mục, không có mức trung gian. Maron cho rằng thực tế không phải vậy.                                                | §8.1 p.414       |
| **Degree of membership / index weight**          | Mức độ thuộc / trọng số chỉ mục    | Quan hệ giữa tài liệu và danh mục có **mức độ**; một tài liệu có thể thuộc danh mục với một trọng số.                                                                                | §8.1             |
| **Automatic probabilistic indexing**             | Đánh chỉ mục xác suất tự động      | Thay chỉ mục nhị phân bằng chỉ mục có trọng số, trong đó attribute number được hiểu là trọng số của nhãn chỉ mục.                                                                    | §8.1             |
| **Relevance number**                             | Số liên quan                       | Con số đo mức liên quan có thể có của một tài liệu đối với người yêu cầu, suy ra bằng thống kê từ trọng số chỉ mục và dữ liệu sử dụng thư viện.                                      | §8.1             |
| **Ranked retrieval**                             | Truy hồi có xếp hạng               | Kết quả tìm kiếm là danh sách tài liệu xếp theo số liên quan. Ý tưởng được phát triển trong Maron & Kuhns (1960).                                                                    | §8.1, footnote 5 |
| **Final inference**                              | Kết luận cuối                      | Việc để máy đọc tài liệu và tự quyết định danh mục là khả thi; không cần đột phá trí tuệ nào. Có thêm dữ liệu thống kê và xét thêm quan hệ giữa các từ thì độ chính xác sẽ tăng dần. | §8.2 p.415       |
| **Abstract vs document**                         | Tóm tắt và toàn văn                | Nói chặt chẽ, kết luận chỉ đúng cho thư viện các bản tóm tắt, nhưng nguyên lý áp dụng được cho toàn văn.                                                                             | §8.2 footnote 6  |

---

## 10. Notation summary

| Symbol | Ý nghĩa |
|---|---|
| $D_i$ | Tài liệu thứ *i* |
| $W_i$ | Từ gợi ý thứ *i* (90 từ) |
| $C_j$ | Danh mục chủ đề thứ *j* (32 danh mục) |
| $P(C_j)$ | Xác suất tiên nghiệm của danh mục *j* |
| $P(W_i\mid C_j)$ | Xác suất từ *i* xuất hiện khi tài liệu thuộc danh mục *j* |
| $P(C_j\mid W_k \cdots W_r)$ | Xác suất hậu nghiệm ("attribute number") |
| $P(W_i)$ | Xác suất xuất hiện của từ *i* (mẫu số, không đổi theo *j*) |
| *k* | Hệ số chuẩn hóa để tổng xác suất bằng 1 |
| *H*, *H'* | Entropy trước và sau khi biết một từ xuất hiện |
| $W_k \cdot W_m$ | Cả hai từ cùng xuất hiện |

## 11. Equations in the paper

| # | Formula | Meaning (VI) |
|---|---|---|
| (1) | $P(C_j\mid W_i) = P(C_j)P(W_i\mid C_j)/P(W_i)$ | Định lý Bayes cho một từ |
| (2) | $P(C_j\mid W_i) = k\,P(C_j)P(W_i\mid C_j)$ | Bỏ mẫu số không đổi |
| (3)–(4) | $k\,P(C_j)P(W_1\mid C_j)P(W_2\mid C_j W_1)$ | Hai từ, quy tắc nhân đầy đủ |
| (5) | $k\,P(C_j)P(W_1\mid C_j)P(W_2\mid C_j)$ | Hai từ, giả định độc lập có điều kiện |
| (6) | $P(W_k\mid C_j W_j) \ne P(W_k\mid C_j)$ | Thừa nhận giả định độc lập là sai |
| (7) / (9) | $k\,P(C_j)\prod P(W\mid C_j)$ | Công thức Naive Bayes tổng quát |
| (8) | $\sum_j P(C_j\mid \cdot) = 1$ | Chuẩn hóa, ngầm coi danh mục loại trừ lẫn nhau |

## 12. Key numbers

| Item | Value |
|---|---|
| Corpus | 405 abstracts (group 1: 260; group 2: 145) |
| Group 1 size | 20,515 word occurrences, 3,263 distinct words, ~79 words per document |
| Categories | 32 (PGEC original: 10) |
| Clue words | 90 (from ~1,000 mid-frequency candidates) |
| Group 1 top-1 | 209/247 = 84.6%; ≥ 2 clue words: 191/210 = 91% |
| Group 2 top-1 (≥ 2 clue words) | 44/85 = 51.8%; top-3 68/85 = 80% |
| Smoothing constant | 0.001 for empty matrix entries |

## 13. People and works cited

- **IRE Transactions on Electronic Computers**, Vol. EC-8, No. 1 (1959), PGEC abstracting service: source of the corpus.
- **Maron, M. E. & Kuhns, J. L. (1960).** "On Relevance, Probabilistic Indexing and Information Retrieval", *J. ACM* 7, 216–244: index weights and relevance ranking.
- **Shannon:** entropy as a measure of uncertainty (no formal reference given).
- **Paul Baran:** suggested the 0.001 substitution for zero probabilities.
- Acknowledged: Sharla Perrine (computations), William Nadeau (word data), Michael Warshaw (categories and indexing of group 1).

---

## 14. Relevance to this project (credit scoring with NB)

These points connect the paper to the report's chapter 2 (Naive Bayes section).

1. **Same decision rule.** The project's NB uses Maron's equation (7): posterior ∝ prior × product of per-feature likelihoods, under conditional independence given the class. In credit scoring the classes are good/bad (2 classes, not 32), and the features are applicant attributes, not words.
2. **Different likelihood model.** Maron estimates $P(W\mid C)$ by counting word occurrences (a multinomial-style model for presence of words). The project uses scikit-learn `GaussianNB` (`src/models/train_classic.py`), which models each feature within each class as a normal distribution. Cite Maron for the **idea** (Bayes' rule + independence), not for the Gaussian variant.
3. **The independence assumption is known to be false.** Maron says so himself (eq. 6). In credit data, features are often strongly correlated (e.g. income and loan amount, or several delinquency counts). This is a principled explanation for weaker NB results in the project.
4. **Probability output and calibration.** Maron treats attribute numbers as probabilities but evaluates only their **ranking** (top-1). The independence assumption tends to push NB posteriors towards 0 or 1, which fits with poor Brier score / calibration even when the ranking (AUC) is acceptable.
5. **Smoothing.** The 0.001 substitution is the ancestor of Laplace smoothing; in `GaussianNB` the analogue is `var_smoothing` (default 1e-9), which keeps variances away from zero.
6. **Hold-out evaluation.** Maron kept group 2 completely unseen, including during feature selection. This matches the project's rule that column screening (B.0) and preprocessing (B.1) are fitted on training data only.
7. **What the paper does not cover.** It does not discuss continuous features, Gaussian likelihoods, binary credit outcomes, AUC, or comparisons with other classifiers. Do not cite it for those.

**Suggested citation sentence (VI):**
> *Bộ phân loại Naive Bayes được Maron (1961) sử dụng lần đầu để phân loại văn bản tự động: xác suất hậu nghiệm của mỗi lớp tỷ lệ với xác suất tiên nghiệm nhân với tích các xác suất có điều kiện của từng đặc trưng, với giả định các đặc trưng độc lập với nhau khi biết lớp.*
