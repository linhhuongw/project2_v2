"""Run the project pipeline end to end, or a range of stages.

    python run_pipeline.py                         # all stages
    python run_pipeline.py --from splits           # splits -> preprocess -> train -> report
    python run_pipeline.py --only train report     # just these stages
    python run_pipeline.py --list

Stages (configs/preprocessing_v1.md):
  interim     raw data -> data/interim (CSV / Parquet, target 1 = bad)
  splits      tier A + StratifiedGroupKFold / Lending Club time split + tier B.0 -> data/splits
  preprocess  tier B.1 for classic ML (median, standardise, one-hot; fit on train) -> data/processed/classic
  train       Logistic Regression, default hyperparameters -> reports/models/lr_default
  report      tables, figures and report.md for the LR run
"""

import argparse
import sys
import time

from src.data import make_interim, make_splits
from src.features import make_processed
from src.models import report_classic, train_classic

STAGES = {
    "interim": lambda: make_interim.main([]),
    "splits": lambda: make_splits.main([]),
    "preprocess": lambda: make_processed.main(["--family", "classic"]),
    "train": lambda: train_classic.main(["--run", "lr_default"]),
    "report": lambda: report_classic.main(["--run", "lr_default"]),
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
