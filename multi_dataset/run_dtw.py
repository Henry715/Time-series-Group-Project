"""1-NN DTW reference on FordA, FordB and Adiac.

Same rule as the FordA notebook (preprocessing/visualization-FordA.ipynb):
aeon's dtw_pairwise_distance with a Sakoe-Chiba window of 0.01 (fraction of
the series length), data as provided, nearest training series by argmin
(first index on ties). No training or tuning is involved.

Needs numpy, scikit-learn and aeon (pip install aeon); torch is not needed.

Examples (from the repository root):
  python multi_dataset/run_dtw.py --dataset FordA FordB Adiac
  python multi_dataset/run_dtw.py --dataset Adiac --window 0.1

Outputs in multi_dataset/runs/<dataset>/dtw1nn-w<window>/: predictions.npz
and summary.json; runs/summary.csv collects every run.
"""
from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np

from dataset_utils import DATASETS, DEFAULT_RUNS_DIR, classification_summary, load_dataset, write_run_summary


def run(dataset: str, args) -> dict:
    import aeon
    from aeon.distances import dtw_pairwise_distance

    data = load_dataset(dataset)
    print(f"\n=== {dataset} | 1-NN DTW window {args.window:g} | "
          f"train {len(data['X_train'])} / test {len(data['X_test'])} ===")

    t0 = time.perf_counter()
    dist = dtw_pairwise_distance(data["X_test"], data["X_train"], window=args.window, n_jobs=args.n_jobs)
    seconds = time.perf_counter() - t0

    nearest = np.argmin(dist, axis=1)
    y_pred = data["y_train"][nearest]
    variant = f"dtw1nn-w{args.window:g}"
    summary = {
        "dataset": dataset,
        "model": "1NN-DTW",
        "variant": variant,
        "run": variant,
        "n_classes": data["n_classes"],
        "length": data["length"],
        "n_train_fit": len(data["X_train"]),
        "window": args.window,
        **classification_summary(data["y_test"], y_pred, data["n_classes"]),
        "seconds": round(seconds, 1),
        "aeon": aeon.__version__,
        "python": platform.python_version(),
    }

    run_dir = Path(args.out_dir) / dataset / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "predictions.npz", y_true=data["y_test"], y_pred=y_pred,
             nearest_train_index=nearest, classes=data["classes"])
    write_run_summary(run_dir, summary, Path(args.out_dir))

    print(f"TEST {dataset} {variant}: acc {summary['test_acc']:.4f} "
          f"[{summary['acc_ci95_low']:.3f}, {summary['acc_ci95_high']:.3f}] | "
          f"error {summary['test_error']:.4f} | macro F1 {summary['f1_macro']:.4f} | {seconds:.0f} s")
    print(f"saved to {os.path.relpath(run_dir)}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", required=True, choices=sorted(DATASETS))
    ap.add_argument("--window", type=float, default=0.01, help="Sakoe-Chiba window as a fraction of length")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", default=str(DEFAULT_RUNS_DIR))
    args = ap.parse_args()
    for dataset in args.dataset:
        run(dataset, args)


if __name__ == "__main__":
    main()
