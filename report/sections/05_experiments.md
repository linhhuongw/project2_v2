# 5. Thực nghiệm

## 5.1. Thiết lập

| Mục | Giá trị |
|---|---|
| Mô hình | `LogisticRegression(max_iter=2000, random_state=42)`, các tham số khác mặc định (scikit-learn 1.7) |
| Tiền xử lý | Tầng A → chia dữ liệu → B.0 → B.1 nhánh classic (chương 4) |
| Số lần fit | 5 fold × 6 bộ = 30; Lending Club: 2 (train 2013, và fit lại trên 2013–2014) |
| Seed | 42 cho chia dữ liệu, mô hình và bootstrap |
| Phần cứng | Máy tính cá nhân, CPU |

## 5.2. Kích thước các phần dữ liệu

| Bộ dữ liệu | Cách chia | Train | Validation | Test | Tỷ lệ xấu train / val / test (fold 1) |
|---|---|---:|---:|---:|---|
| GMSC | Stratified group 5-fold | ~96.000 | ~24.010 | ~30.000 | 6,70% / 6,67% / 6,63% |
| Taiwan | Stratified group 5-fold | ~19.200 | ~4.800 | ~6.000 | 22,19% / 22,04% / 21,95% |
| Home Credit | Stratified group 5-fold | 196.807 | 49.202 | 61.502 | 8,07% / 7,93% / 8,21% |
| HMEQ | Stratified group 5-fold | 3.814 | 954 | 1.192 | 20,32% / 18,55% / 19,88% |
| South German | Stratified group 5-fold | 640 | 160 | 200 | 30,47% / 27,50% / 30,50% |
| FICO HELOC | Stratified group 5-fold | ~6.318 | ~1.579 | ~1.974 | 52,59% / 48,96% / 52,71% |
| Lending Club | Theo thời gian | 100.422 (2013) | 162.570 (2014) | 282.787 (2015) | 12,33% / 13,73% / 14,88% |

Kích thước các fold dao động nhẹ (vài chục dòng) vì việc chia theo nhóm dòng trùng lặp. Với Lending Club, train nhỏ hơn validation (1,6 lần) và test (2,8 lần) do quy mô cho vay tăng nhanh theo năm; mô hình cuối được fit lại trên train + validation (262.992 dòng).

## 5.3. Thời gian chạy

| Bộ dữ liệu | Số vòng lặp tối đa | Hội tụ | Tổng thời gian fit (giây) |
|---|---:|---|---:|
| GMSC | 35 | Có | 0,5 |
| Taiwan | 68 | Có | 0,3 |
| Home Credit | 82 | Có | 9,3 |
| HMEQ | 38 | Có | < 0,1 |
| South German | 58 | Có | < 0,1 |
| FICO HELOC | 47 | Có | 0,1 |
| Lending Club | 89 | Có | 3,4 |

Toàn bộ bước huấn luyện và đánh giá (kể cả bootstrap) chạy trong khoảng 1 phút.

## 5.4. Tái lập

Toàn bộ quy trình được đóng gói thành pipeline:

```bash
python run_pipeline.py                        # interim → splits → preprocess → train → report
python run_pipeline.py --only train report    # chỉ huấn luyện và tạo bảng, hình (~1 phút)
```

| Bước | Mã nguồn | Đầu ra |
|---|---|---|
| Chuyển đổi dữ liệu thô | `src/data/make_interim.py` | `data/interim/` |
| Tầng A, chia dữ liệu, B.0 | `src/data/datasets.py`, `src/data/make_splits.py` | `data/splits/` |
| Tầng B.1 | `src/features/preprocess.py`, `src/features/make_processed.py` | `data/processed/classic/` |
| Huấn luyện, đánh giá | `src/models/train_classic.py`, `src/models/metrics.py` | `reports/models/lr_default/` |
| Bảng, hình | `src/models/report_classic.py` | `reports/models/lr_default/tables.md`, `figures/` |

Dự đoán của từng dòng test được lưu trong `reports/models/lr_default/predictions/` để dùng cho các kiểm định thống kê khi so sánh với các mô hình khác.
