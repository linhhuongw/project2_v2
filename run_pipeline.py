"""Run the project pipeline end to end, or a range of stages.

    python run_pipeline.py                         # all stages
    python run_pipeline.py --from splits           # splits -> preprocess -> train -> report
    python run_pipeline.py --only train report     # just these stages
    python run_pipeline.py --list

Stages (configs/preprocessing_v1.md):
  interim     raw data -> data/interim (CSV / Parquet, target 1 = bad)
  splits      tier A + StratifiedGroupKFold / Lending Club time split + tier B.0 -> data/splits
  preprocess  tier B.1, fit on train: classic (one-hot) -> data/processed/classic,
              gbdt (integer codes) -> data/processed/gbdt
  train       LR, KNN, RF, NB, XGBoost, LightGBM, CatBoost, default hyperparameters -> reports/models/<model>_default
  report      tables and figures per model run (report.md files are hand-written)
  compare     model comparison: AUC, precision, recall, F1, ranks, Friedman / Wilcoxon-Holm,
              precision-recall curves -> reports/models/comparison
  curves      learning curves (AUC vs training size, validation only) -> reports/models/learning_curves
  validity    leakage and overfitting checks -> reports/models/validity
"""

import argparse
import sys
import time

from src.data import make_interim, make_splits
from src.features import make_processed
from src.models import compare_models, learning_curves, report_classic, train_classic, validity_checks

RUNS = ["lr_default", "knn_default", "rf_default", "nb_default", "xgb_default", "lgbm_default", "cat_default"]

STAGES = {
    "interim": lambda: make_interim.main([]),
    "splits": lambda: make_splits.main([]),
    "preprocess": lambda: [make_processed.main(["--family", f]) for f in ("classic", "gbdt")],
    "train": lambda: [train_classic.main(["--run", r]) for r in RUNS],
    "report": lambda: [report_classic.main(["--run", r]) for r in RUNS],
    "compare": lambda: [compare_models.main(["--hmeq", v]) for v in ("hmeq", "hmeq_blind", "hmeq_complete")],
    "curves": lambda: learning_curves.main([]),
    "validity": lambda: validity_checks.main([]),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--from", dest="start", choices=STAGES, help="run this stage and all later ones")
    g.add_argument("--only", nargs="+", choices=STAGES, help="run only these stages")
    p.add_argument("--list", action="store_true", help="list stages and exit")
    args = p.parse_args(argv)
    names = list(STAGES)
    if args.list:
        print("\n".join(names))
        return
    todo = args.only or names[names.index(args.start):] if (args.only or args.start) else names
    for name in todo:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        STAGES[name]()
        print(f"=== {name} done in {time.perf_counter() - t0:.0f}s ===")


if __name__ == "__main__":
    sys.exit(main())
