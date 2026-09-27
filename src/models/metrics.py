"""Evaluation metrics (configs/preprocessing_v1.md, "Tinh chỉnh siêu tham số").

Threshold-free: AUC, Gini, KS, Brier, LogLoss, Average Precision.
Threshold-based: precision / recall / F1 / accuracy at the threshold that maximises F1 on validation.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             log_loss, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def f1_threshold(y_val, p_val) -> float:
    """Threshold on validation scores that maximises F1."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def score(y, p, threshold: float | None = None) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    auc = roc_auc_score(y, p)
    out = {
        "auc": auc, "gini": 2 * auc - 1, "ks": ks_stat(y, p),
        "brier": brier_score_loss(y, p), "logloss": log_loss(y, p),
        "ap": average_precision_score(y, p),
        "bad_rate": float(y.mean()), "mean_pred": float(p.mean()),
    }
    if threshold is not None:
        yhat = (p >= threshold).astype(int)
        out.update({
            "threshold": threshold,
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0),
            "accuracy": accuracy_score(y, yhat),
        })
    return {k: float(v) for k, v in out.items()}


def bootstrap_auc(y, p, n: int = 1000, seed: int = 42) -> dict:
    """Percentile 95% CI of test AUC (Lending Club: single split, uncertainty by bootstrap)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], p[idx]))
    aucs = np.array(aucs)
    return {"n": int(len(aucs)), "mean": float(aucs.mean()), "std": float(aucs.std(ddof=1)),
            "ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]}
