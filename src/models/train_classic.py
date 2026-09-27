"""Train classic-ML models on the tier-B.1 matrices (data/processed/classic) and evaluate them.

Currently: Logistic Regression with scikit-learn default hyperparameters (C=1.0, L2, lbfgs), except
max_iter=2000 as in configs/preprocessing_v1.md B.3 (solver budget, not a tuned hyperparameter).

Per dataset and fold: fit on train, choose the F1-maximising threshold on validation, evaluate on
validation and test. Lending Club (single time split) also: refit on train + val and evaluate on
test (rule 3), with a 1,000-iteration bootstrap CI of test AUC; the threshold comes from the
train-only model's validation scores.

Output under reports/models/<run>/:
  fold_metrics.csv     one row per dataset x fold x part
  summary.csv          test metrics per dataset (mean ± std over folds; Lending Club: refit)
  predictions/<dataset>.parquet   row_id, fold, part, target, proba (for later statistical tests)
  run.json             model config, fit times, iterations, convergence, bootstrap results

Run from the project root:  python -m src.models.train_classic [--run lr_default] [name ...]
"""

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.data.datasets import ROOT, SPECS
from src.data.splits import load_meta
from src.models.metrics import bootstrap_auc, f1_threshold, score

PROCESSED = ROOT / "data" / "processed" / "classic"
REPORTS = ROOT / "reports" / "models"
SEED = 42

MODELS = {
    "lr_default": lambda: LogisticRegression(max_iter=2000, random_state=SEED),
}


def load_parts(path, parts):
    frames = {p: pd.read_parquet(path / f"{p}.parquet") for p in parts}
    return {p: (f.iloc[:, 2:].to_numpy(np.float32), f["target"].to_numpy(), f["row_id"].to_numpy())
            for p, f in frames.items()}


def fit(model_name, X, y):
    model = MODELS[model_name]()
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, {"fit_seconds": round(time.perf_counter() - t0, 2),
                   "n_iter": int(np.max(model.n_iter_)), "converged": converged}


def run_dataset(name, model_name):
    meta = load_meta(name)
    base = PROCESSED / name
    rows, preds, fits = [], [], {}
    thresholds = {}
    for k in range(1, meta["n_folds"] + 1):
        d = load_parts(base / f"fold{k}", ("train", "val", "test"))
        model, info = fit(model_name, *d["train"][:2])
        fits[f"fold{k}"] = info
        p = {part: model.predict_proba(d[part][0])[:, 1] for part in ("train", "val", "test")}
        thr = f1_threshold(d["val"][1], p["val"])
        thresholds[k] = thr
        for part in ("train", "val", "test"):
            rows.append({"dataset": name, "fold": k, "part": part, "fit": "train",
                         **score(d[part][1], p[part], thr)})
        for part in ("val", "test"):
            preds.append(pd.DataFrame({"row_id": d[part][2], "fold": k, "part": part, "fit": "train",
                                       "target": d[part][1], "proba": p[part]}))

    boot = None
    if meta["method"] == "time_holdout":  # Lending Club rule 3: refit on train + val
        d = load_parts(base / "refit", ("trainval", "test"))
        model, info = fit(model_name, *d["trainval"][:2])
        fits["refit"] = info
        p_test = model.predict_proba(d["test"][0])[:, 1]
        rows.append({"dataset": name, "fold": 1, "part": "test", "fit": "train+val",
                     **score(d["test"][1], p_test, thresholds[1])})
        preds.append(pd.DataFrame({"row_id": d["test"][2], "fold": 1, "part": "test", "fit": "train+val",
                                   "target": d["test"][1], "proba": p_test}))
        boot = bootstrap_auc(d["test"][1], p_test)
    return rows, pd.concat(preds, ignore_index=True), fits, boot


METRICS = ["auc", "gini", "ks", "brier", "logloss", "ap", "precision", "recall", "f1", "accuracy"]


def summarise(fm: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, g in fm.groupby("dataset", sort=False):
        test = g[g.part == "test"]
        final = test[test.fit == "train+val"] if (test.fit == "train+val").any() else test
        row = {"dataset": name, "n_folds": len(final), "eval": "refit train+val" if len(final) == 1 and
               (test.fit == "train+val").any() else "5-fold CV"}
        for m in METRICS:
            row[m] = final[m].mean()
            row[f"{m}_std"] = final[m].std(ddof=1) if len(final) > 1 else np.nan
        row["bad_rate"] = final["bad_rate"].mean()
        row["mean_pred"] = final["mean_pred"].mean()
        row["train_auc"] = g[(g.part == "train")]["auc"].mean()
        row["val_auc"] = g[(g.part == "val")]["auc"].mean()
        out.append(row)
    return pd.DataFrame(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="lr_default", choices=list(MODELS))
    p.add_argument("names", nargs="*", help=f"datasets (default: all): {', '.join(SPECS)}")
    args = p.parse_args(argv)
    unknown = set(args.names) - set(SPECS)
    if unknown:
        p.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")

    out = REPORTS / args.run
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    all_rows, run = [], {"model": repr(MODELS[args.run]()), "seed": SEED, "datasets": {}}
    for name in args.names or SPECS:
        t0 = time.perf_counter()
        rows, preds, fits, boot = run_dataset(name, args.run)
        all_rows += rows
        preds.to_parquet(out / "predictions" / f"{name}.parquet", index=False)
        run["datasets"][name] = {"fits": fits, "bootstrap_test_auc": boot}
        test = [r for r in rows if r["part"] == "test"]
        final = [r for r in test if r["fit"] == "train+val"] or test
        conv = all(f["converged"] for f in fits.values())
        print(f"[{name}] test AUC {np.mean([r['auc'] for r in final]):.4f}"
              + (f" ± {np.std([r['auc'] for r in final], ddof=1):.4f}" if len(final) > 1 else "")
              + (f" (bootstrap 95% CI {boot['ci95'][0]:.4f}–{boot['ci95'][1]:.4f})" if boot else "")
              + f" | converged {conv} | max iter {max(f['n_iter'] for f in fits.values())}"
              + f" | {time.perf_counter() - t0:.1f}s")

    fm = pd.DataFrame(all_rows)
    fm.to_csv(out / "fold_metrics.csv", index=False)
    summarise(fm).to_csv(out / "summary.csv", index=False)
    (out / "run.json").write_text(json.dumps(run, indent=2))
    print(f"written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
