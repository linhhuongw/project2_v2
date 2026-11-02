# Notes: figures in the report

Working note, not part of the report. All figures are in `report/hinh_ve/` and are rebuilt with:

```bash
python scripts/build_report_figures.py
```

The script draws the new figures and copies the pipeline's existing ones from `reports/` under the names below. Rerun it after rerunning the pipeline, so the report keeps the latest results.

Each figure in `sections_new/` has a number, a caption directly below it, and a sentence in the text that refers to it, as Phụ lục A of the template requires.

| Hình | Section | File | Source |
|---|---|---|---|
| 2.1 | 2.3 | `lr_sigmoid.png` | New (script) |
| 2.2 | 2.4 | `bagging_boosting.png` | New (script) |
| 2.3 | 2.4 | `cach_phat_trien_cay.png` | New (script) |
| 3.1 | 3.1 | `quy_trinh.png` | New (script); replaces the ASCII diagram |
| 3.2 | 3.2.2 | `eda_tong_quan.png` | Redrawn (script) with the 7 report datasets; the EDA version also had Statlog German Credit |
| 3.3 | 3.2.3 | `chat_luong_du_lieu.png` | Redrawn (script), same reason |
| 3.4 | 3.2.5 | `lending_club_theo_nam.png` | `reports/figures/eda/lending_club_vintage.png` |
| 3.5 | 3.2.6 | `hmeq_thieu_ty_le_xau.png` | `reports/figures/eda/hmeq_missing_bad_rate.png` |
| 3.6 | 3.2.6 | `hmeq_thieu_tuong_quan.png` | `reports/figures/eda/hmeq_missing_corr.png` |
| 3.7 | 3.3 | `chia_du_lieu.png` | New (script) |
| 4.1 | 4.2.3 | `thoi_gian_chay.png` | New (script), values from the runtime table in 4.2.3 |
| 4.2 | 4.3.1 | `lr_auc_theo_bo.png` | `reports/models/lr_default/figures/auc_by_dataset.png` |
| 4.3 | 4.3.3 | `gmsc_bien_so_theo_nhan.png` | `reports/figures/eda/give_me_some_credit_numeric_by_target.png` |
| 4.4 | 4.3.4 | `lr_roc_calibration.png` | `reports/models/lr_default/figures/roc_calibration.png` |
| 4.5 | 4.4 | `auc_4_mo_hinh_classic.png` | New (script); the old figure here showed 7 models |
| 4.6 | 4.5 | `prf1_7_mo_hinh.png` | `reports/models/comparison/figures/precision_recall_f1.png` (7 models; caption says so) |
| 4.7 | 4.5 | `pr_curves.png` | `reports/models/comparison/figures/pr_curves.png` (7 models; caption says so) |
| 4.8 | 4.6 | `learning_curves.png` | `reports/models/learning_curves/figures/learning_curves.png` |
| 4.9 | 4.7 | `auc_7_mo_hinh.png` | `reports/models/comparison/figures/auc_by_model.png` |
| 4.10 | 4.7 | `hang_trung_binh_7_mo_hinh.png` | `reports/models/comparison/figures/mean_rank.png` |
| 4.11 | 4.7 | `cd_diagram.png` | `reports/models/comparison/figures/cd_diagram.png` |
| 4.12 | 4.7 | `cat_roc_calibration.png` | `reports/models/cat_default/figures/roc_calibration.png` |
| 4.13 | 4.8 | `hmeq_ngan_ro_ri_auc.png` | `reports/figures/eda/hmeq_leakage_control_auc.png` |
| 4.14 | 4.8 | `cd_diagram_hmeq_blind.png` | `reports/models/comparison_hmeq_blind/figures/cd_diagram.png` |
| 4.15 | 4.8 | `overfit_heatmap.png` | New (script) from `reports/models/validity/overfit.csv`; replaces the train − test table |

## Still open

- Figures copied from `reports/` keep their original titles and axis labels, some in English (e.g. "% resolved" and "bad rate" in Hình 3.4, column names in Hình 3.6 and 4.3). For the final version, consider redrawing them in the script so labels match the report.
- Figure 4.1 uses the numbers in the 4.2.3 table. The `run.json` files hold timings from later re-runs (CatBoost 115.5 s instead of 124.4 s), so the table and the figure should be refreshed together if you re-time the models.
- The Phụ lục B candidates from the earlier note (full per-dataset EDA plots, per-model figures) are not added.
