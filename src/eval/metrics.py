import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    confusion_matrix, cohen_kappa_score, classification_report,
)
from scipy.stats import binomtest

from src.config import LABELS, BOOTSTRAP_ITER, RANDOM_SEED

def classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    kappa = cohen_kappa_score(y_true, y_pred, labels=LABELS)
    per_class = {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
                 for i, lab in enumerate(LABELS)}
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True)
    return {
        "accuracy": float(acc), "macro_f1": float(macro_f1),
        "per_class": per_class, "confusion_matrix": cm, "labels": LABELS,
        "cohens_kappa": float(kappa), "sklearn_report": report, "n": len(y_true),
    }

def mcnemar_test(y_true: List[str], y_a: List[str], y_b: List[str]) -> Dict:
    b = sum(1 for t, a, bb in zip(y_true, y_a, y_b) if (a == t) and (bb != t))
    c = sum(1 for t, a, bb in zip(y_true, y_a, y_b) if (a != t) and (bb == t))
    if b + c == 0:
        return {"b": b, "c": c, "p_value": 1.0, "note": "no discordant pairs"}
    res = binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided")
    return {"b": b, "c": c, "p_value": float(res.pvalue), "note": "exact binomial"}

def bootstrap_f1_ci(y_true: List[str], y_pred: List[str],
                    n_iter: int = BOOTSTRAP_ITER, ci: float = 0.95, seed: int = RANDOM_SEED) -> Dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s = []
    yt = np.array(y_true); yp = np.array(y_pred)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        f1s.append(f1_score(yt[idx], yp[idx], labels=LABELS, average="macro", zero_division=0))
    f1s = np.array(f1s)
    lo = float(np.quantile(f1s, (1 - ci) / 2))
    hi = float(np.quantile(f1s, 1 - (1 - ci) / 2))
    return {"mean_f1": float(f1s.mean()), "ci": ci, "lo": lo, "hi": hi, "n_iter": n_iter}

def load_predictions(path: Path) -> List[Dict]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out
