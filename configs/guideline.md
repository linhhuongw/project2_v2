# Phương án chia train/test theo Baesens et al. (2026)

**Viết tắt:** CV = Cross-Validation (kiểm định chéo)

## Paper

**Baesens, B., Goethals, A., Lessmann, S., De Vos, S., Bravo, C., Martens, D., Medina-Olivares, V., Mues, C., Óskarsdóttir, M., vanden Broucke, S., Verdonck, T., & Verbeke, W. (2026).** *Foundation Models for Credit Risk Prediction: A Game Changer?* arXiv:2605.18147, mục 4.3 (Benchmarking protocol).

## Quy tắc split chung

- Đánh giá trên phần test giữ lại trong sơ đồ **5-fold CV**.
- Tinh chỉnh siêu tham số trên phần validation bằng **20% tập train của mỗi fold**.

| Tầng | Tỷ lệ trên toàn bộ dữ liệu | Dùng cho |
|---|---|---|
| Test (1 fold) | 20% | Đánh giá |
| Validation (20% của train fold) | 16% | Tinh chỉnh siêu tham số |
| Train thực | 64% | Huấn luyện, học tham số tiền xử lý |

## Split theo từng bộ dữ liệu

| Dataset | Có trong Baesens | Số dòng dùng để split | Cách split | Test / fold | Validation / fold | Train / fold |
|---|---|---|---|---|---|---|
| Give Me Some Credit | Có | 150.000 | 5-fold CV | ~30.000 | ~24.000 | ~96.000 |
| Taiwan | Có | 30.000 | 5-fold CV | ~6.000 | ~4.800 | ~19.200 |
| Home Credit | Có | 307.511 (`application_train`) | 5-fold CV | ~61.500 | ~49.200 | ~196.800 |
| HMEQ | Có | 5.960 | 5-fold CV | ~1.192 | ~954 | ~3.814 |
| German / South German | Có (German) | 1.000 | 5-fold CV | 200 | 160 | 640 |
| FICO HELOC | Không (đề xuất bổ sung) | 9.870 (bỏ 1 dòng trùng) | 5-fold CV | ~1.974 | ~1.579 | ~6.317 |
| Lending Club | Không (đề xuất bổ sung) | 621.022 (khoản 36 tháng, giải ngân 2007–2015, đã kết thúc) | **5-fold out-of-time, cửa sổ mở rộng** — xem mục dưới | 14.101 → 283.026 | 20% cuối (theo thời gian) của train | 13.946 → 270.397 |

### FICO HELOC — 5-fold CV phân tầng như các bộ khác

- Không có cột thời gian, mỗi dòng là một người nộp đơn độc lập → dùng đúng quy tắc chung (`StratifiedKFold(5)`, validation 20% phân tầng).
- Trước khi split: bỏ 1 dòng trùng lặp (tránh cùng một mẫu nằm ở cả train và test); đổi mã -7/-8/-9 → NaN + cờ chỉ báo (phép biến đổi theo từng dòng, không học từ dữ liệu nên làm trước split được).
- Nhãn gần cân bằng (52% bad) nhưng vẫn stratify để các fold có cùng tỉ lệ. Tỉ lệ này là nhân tạo → không so sánh trực tiếp xác suất/calibration với các bộ khác.

### Lending Club — chia theo thời gian (out-of-time), KHÔNG xáo trộn

Lý do không dùng 5-fold ngẫu nhiên: nhãn phụ thuộc năm giải ngân (bad rate và chính sách cho vay thay đổi theo thời gian), nên xáo trộn sẽ để mô hình “nhìn thấy tương lai” và cho kết quả lạc quan.

**Bước 1 — Chọn tập mẫu có nhãn đầy đủ (tránh thiên lệch do khoản vay chưa kết thúc).** Dữ liệu dừng ở 2018Q4, nên chỉ những khoản đã hết kỳ hạn mới có kết cục không bị cắt xén:

| Kỳ hạn | Năm giải ngân đã kết thúc ~100% | Từ năm này trở đi bị cắt |
|---|---|---|
| 36 tháng | 2007–2015 (2015: 99,9%) | 2016 (72%), 2017 (40%), 2018 (12%) |
| 60 tháng | 2010–2013 | 2014 (83%), 2015 (67%)… |

→ Đề xuất dùng **khoản 36 tháng, giải ngân 2007–2015, đã kết thúc** (621.022 dòng, bad rate ≈ 14%). Không trộn khoản 60 tháng vì chúng chỉ đầy đủ tới 2013, sẽ làm thành phần kỳ hạn giữa train và test lệch nhau. Khoản giải ngân từ 2016 trở đi bị loại khỏi đánh giá (mẫu “đã kết thúc” của các năm này lệch về khoản vỡ nợ sớm).

**Bước 2 — 5 fold theo thời gian với cửa sổ huấn luyện mở rộng (expanding window).** Mỗi fold: test = 1 năm giải ngân, train = mọi năm trước đó; validation = **20% khoản mới nhất** của cửa sổ train (sắp theo `issue_d`, không lấy ngẫu nhiên) — giữ đúng tỉ lệ 20% của quy tắc chung nhưng tôn trọng thứ tự thời gian.

| Fold | Cửa sổ train (năm) | Train thực | Validation (20% mới nhất) | Test (năm) | Test |
|---|---|---:|---:|---|---:|
| 1 | 2007–2010 | ~13.946 | ~3.487 | 2011 | 14.101 |
| 2 | 2007–2011 | ~25.227 | ~6.307 | 2012 | 43.470 |
| 3 | 2007–2012 | ~60.003 | ~15.001 | 2013 | 100.422 |
| 4 | 2007–2013 | ~140.341 | ~35.085 | 2014 | 162.570 |
| 5 | 2007–2014 | ~270.397 | ~67.599 | 2015 | 283.026 |

- Báo cáo trung bình ± độ lệch chuẩn qua 5 fold (so sánh được với các bộ khác), **và** kết quả riêng fold 5 (gần nhất, dữ liệu train nhiều nhất) như con số “triển khai thực tế”.
- Kích thước fold không đều và bad rate trôi theo năm (≈ 10,6% năm 2011 → 14,9% năm 2015) — đây là đặc tính cần đo, không phải lỗi; ghi nhận thêm độ lệch calibration giữa train và test.

**Bước 3 — Cột được phép dùng.** Chỉ giữ biến có tại thời điểm nộp đơn; loại toàn bộ cột sau giải ngân (thanh toán, thu hồi, điểm FICO cập nhật cuối, cờ hardship/settlement, `loan_status`…), cột định danh/văn bản tự do (`id`, `url`, `desc`, `emp_title`, `title`) và `issue_d` (chỉ dùng để chia). `grade`, `sub_grade`, `int_rate` là đầu ra của mô hình rủi ro nội bộ Lending Club — được phép dùng nhưng nên chạy thêm một cấu hình không có chúng.

**Tuỳ chọn (không bắt buộc):** chèn khoảng trống (gap) giữa train và test bằng thời gian quan sát nhãn (36 tháng) để mô phỏng chặt chẽ việc khi triển khai chưa biết kết cục của các khoản gần đây; đổi lại train nhỏ đi đáng kể.

## Thứ tự thực hiện

1. Khai báo dữ liệu (nhãn, loại cột ID và cột rò rỉ, mã đặc biệt → NaN)
2. **Chia 5 fold**
3. Trong mỗi fold: tách 20% train → validation
4. Fit tiền xử lý **chỉ trên train thực** → áp cho validation và test
5. Tinh chỉnh trên validation → đánh giá trên test

## Không được nêu trong paper (tự chọn, ghi rõ trong báo cáo)

| Mục | Giá trị đề xuất |
|---|---|
| Phân tầng theo nhãn | `StratifiedKFold`, `stratify=y` khi tách validation (trừ Lending Club) |
| Seed | Cố định, ví dụ `random_state=42` |
| Xáo trộn | `shuffle=True` (Lending Club: `shuffle=False`, sắp theo `issue_d`) |