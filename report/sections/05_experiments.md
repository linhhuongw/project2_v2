# 5. Thực nghiệm

## 5.1. Thiết lập

| Mục | Giá trị |
|---|---|
| Mô hình | LR, KNN, RF, NB — tham số mặc định của scikit-learn 1.7 (mục 4.5); `random_state = 42` cho LR và RF |
| Tiền xử lý | Tầng A → chia dữ liệu → B.0 → B.1 nhánh classic (chương 4) |
| Số lần fit mỗi mô hình | 5 fold × 6 bộ = 30; Lending Club: 2 (train 2013, và fit lại trên 2013–2014) |
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

Tổng thời gian fit và dự đoán của mỗi mô hình (giây, CPU 10 nhân):

| Bộ dữ liệu | LR | KNN | RF | NB |
|---|---:|---:|---:|---:|
| GMSC | 0,5 | 3,8 | 7,5 | 0,1 |
| Taiwan | 0,3 | 0,7 | 2,2 | < 0,1 |
| Home Credit | 9,3 | 185,2 | 40,6 | 2,4 |
| HMEQ | < 0,1 | 0,1 | 0,6 | < 0,1 |
| South German | < 0,1 | < 0,1 | 0,4 | < 0,1 |
| FICO HELOC | 0,1 | 0,1 | 0,8 | < 0,1 |
| Lending Club | 3,4 | 150,0 | 16,1 | 0,9 |
| **Tổng** | **13,8** | **339,9** | **68,3** | **3,4** |

- LR hội tụ ở mọi lần fit (tối đa 89 vòng lặp).
- KNN gần như không tốn thời gian fit nhưng rất chậm khi dự đoán: mỗi dòng validation/test phải tính khoảng cách tới toàn bộ dòng train. KNN chiếm khoảng 80% tổng thời gian chạy của cả bốn mô hình (78% trên Home Credit, 88% trên Lending Club).
- RF dùng tối đa khoảng 3 GB bộ nhớ trên Home Credit.

## 5.4. Tái lập

Toàn bộ quy trình được đóng gói thành pipeline:

```bash
python run_pipeline.py                        # interim → splits → preprocess → train → report → compare
python run_pipeline.py --only train report    # chỉ huấn luyện và tạo bảng, hình (~1 phút)
```

| Bước | Mã nguồn | Đầu ra |
|---|---|---|
| Chuyển đổi dữ liệu thô | `src/data/make_interim.py` | `data/interim/` |
| Tầng A, chia dữ liệu, B.0 | `src/data/datasets.py`, `src/data/make_splits.py` | `data/splits/` |
| Tầng B.1 | `src/features/preprocess.py`, `src/features/make_processed.py` | `data/processed/classic/` |
| Huấn luyện, đánh giá | `src/models/train_classic.py`, `src/models/metrics.py` | `reports/models/<mô hình>_default/` |
| Bảng, hình từng mô hình | `src/models/report_classic.py` | `reports/models/<mô hình>_default/tables.md`, `figures/` |
| So sánh mô hình | `src/models/compare_models.py` | `reports/models/comparison/` |

Dự đoán của từng dòng validation và test được lưu trong `reports/models/<mô hình>_default/predictions/`, dùng cho kiểm định bootstrap ghép cặp và các so sánh sau này.
