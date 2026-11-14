"""Shared helpers for the FordA / FordB / Adiac experiments.

This module imports neither torch nor aeon, so the FCN script and the DTW
script can run in different environments.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "multi_dataset" / "runs"

# Raw UCR files as supplied in the repository. Column 0 is the label.
DATASETS = {
    "FordA": {"train": "FordA/FordA_TRAIN.tsv", "test": "FordA/FordA_TEST.tsv", "delimiter": "\t"},
    "FordB": {"train": "FordB/FordB_TRAIN.tsv", "test": "FordB/FordB_TEST.tsv", "delimiter": "\t"},
    "Adiac": {"train": "Adiac/Adiac_TRAIN", "test": "Adiac/Adiac_TEST", "delimiter": ","},
}


def load_dataset(name: str) -> dict:
    """Load a dataset as provided (no extra normalisation).

    Labels are mapped to 0..K-1 in sorted order of the training labels, so
    FordA/FordB map -1 -> 0 and +1 -> 1 (as in FCN_training.ipynb) and Adiac
    maps 1..37 -> 0..36.
    """
    spec = DATASETS[name]
    train = np.loadtxt(REPO_ROOT / spec["train"], delimiter=spec["delimiter"])
    test = np.loadtxt(REPO_ROOT / spec["test"], delimiter=spec["delimiter"])

    X_train, y_train_raw = train[:, 1:], train[:, 0]
    X_test, y_test_raw = test[:, 1:], test[:, 0]
    if not (np.isfinite(X_train).all() and np.isfinite(X_test).all()):
        raise ValueError(f"{name}: non-finite values in the series")

    classes = np.unique(y_train_raw)
    unknown = np.setdiff1d(np.unique(y_test_raw), classes)
    if unknown.size:
        raise ValueError(f"{name}: test labels {unknown} do not occur in training data")

    return {
        "name": name,
        "X_train": X_train,
        "y_train": np.searchsorted(classes, y_train_raw).astype(np.int64),
        "X_test": X_test,
        "y_test": np.searchsorted(classes, y_test_raw).astype(np.int64),
        "classes": classes,
        "n_classes": int(classes.size),
        "length": int(X_train.shape[1]),
    }


# Training series the FordA Optuna search and FCN-Opt + ES were fitted on
# (3601 minus the stratified 10% validation split).
FORDA_N_FIT = 3240


def scaled_epochs(base_epochs: int, n_fit: int, ref_n_fit: int = FORDA_N_FIT) -> int:
    """Epoch budget that sees as many training examples as `base_epochs` on FordA.

    The FordA budgets (100 epochs per trial, patience 15, pruner warm-up 50 and
    interval 10) were counted in epochs. An epoch on a smaller training set
    holds fewer gradient steps at the same batch size, so each budget is
    multiplied by ref_n_fit / n_fit and rounded up. FordB (3272) keeps the
    FordA values; Adiac gets about 9.2 times more epochs with 351 training
    series and about 11.1 times more in a 4-fold split (292 or 293).
    """
    return -(-base_epochs * ref_n_fit // n_fit)  # ceil in integer arithmetic


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n (same as statsmodels 'wilson')."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def classification_summary(y_true, y_pred, n_classes: int, probs=None) -> dict:
    """Test metrics in the same form as Table III of the report."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(n_classes))
    n = int(y_true.size)
    k = int((y_true == y_pred).sum())
    lo, hi = wilson_interval(k, n)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    out = {
        "n_test": n,
        "n_correct": k,
        "test_acc": float(accuracy_score(y_true, y_pred)),
        "test_error": float(1 - k / n),
        "acc_ci95_low": float(lo),
        "acc_ci95_high": float(hi),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "auc": None,
    }
    if probs is not None:
        try:
            if n_classes == 2:
                out["auc"] = float(roc_auc_score(y_true, probs[:, 1]))
            else:  # one-vs-rest, macro average
                out["auc"] = float(
                    roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels)
                )
        except ValueError as exc:  # e.g. a class missing from the test set
            print(f"AUC not computed: {exc}")
    return out


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def write_run_summary(run_dir: Path, summary: dict, runs_root: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    collect_summaries(runs_root)


def collect_summaries(runs_root: Path) -> Path:
    """Rebuild runs_root/summary.csv from every runs_root/<dataset>/<run>/summary.json."""
    rows = [json.loads(p.read_text()) for p in sorted(runs_root.glob("*/*/summary.json"))]
    keys: list[str] = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    out = runs_root / "summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in keys})
    return out
