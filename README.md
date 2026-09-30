# Project 2 — v2

Credit default prediction, version 2. Raw data copied from `project2_v1/data/raw`.

## Layout

| Path | Purpose |
|---|---|
| `data/raw/` | Original datasets (read-only) |
| `data/interim/` | Converted / partially cleaned data |
| `data/processed/` | Model-ready feature tables |
| `data/external/` | Third-party reference data |
| `src/` | Code: `data`, `features`, `models`, `visualization` |
| `notebooks/` | Exploration notebooks |
| `configs/` | Experiment configs |
| `scripts/` | CLI entry points |
| `models/` | Saved model artifacts |
| `reports/` | Generated results (JSON/MD, `figures/`, `tables/`) |
| `report/` | Hand-written final report (`report.md` + `sections/`) |

## Reproduce

Data files are not in the repository (`data/` is git-ignored; raw data is ~3.1 GB).
Copy the raw datasets into `data/raw/` (same layout as `project2_v1/data/raw`), then run:

```bash
python -m src.data.make_interim                  # raw -> data/interim (CSV / Parquet, ~30 s)
python scripts/build_eda_notebooks.py --execute  # build + run notebooks/eda_*.ipynb (~1 min)
python -m src.data.make_splits                   # 5-fold train/val/test splits -> data/splits (~3 min)
python -m src.features.make_processed --family classic  # tier B.1 for LR/classic ML -> data/processed/classic (~30 s)
python -m src.models.train_classic --run lr_default     # also knn_default, rf_default, nb_default -> reports/models/<run>
python -m src.models.report_classic --run lr_default    # tables + figures per run
python -m src.models.compare_models                     # ranks, Friedman / Wilcoxon-Holm -> reports/models/comparison
```

Or run every stage in order: `python run_pipeline.py` (see `python run_pipeline.py --help`).

EDA outputs: figures in `reports/figures/eda/`, per-dataset summaries in `reports/eda/*.json`,
comparison table in `reports/tables/eda_comparison.csv`, and the data report in `report/data_report.md`.

## Splits

Implemented per `configs/preprocessing_v1.md` (tier A, split, tier B.0) in `src/data/make_splits.py` (dataset preparation in
`src/data/datasets.py`). Each `data/splits/<name>/` holds `data.parquet`, `folds.parquet`
(`row_id` → role per fold), `fold<k>/{train,val,test}.parquet`, `fold<k>/screening.json` (B.0) and `meta.json`.

```python
from src.data.splits import load_fold, xy
train, val, test = load_fold("lending_club_B", 1)
X_train, y_train = xy(train, "lending_club_B", 1)  # B.0-screened columns
```
