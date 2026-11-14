"""Exact McNemar tests between two FCN models on the same test series.

For every dataset, runs of --model-a and --model-b with the same training seed
are paired (for example opt_tuned-patience139_seed43 with baseline_seed43).
The exact two-sided test is a binomial test on the discordant cases, the same
test as statsmodels' mcnemar(exact=True) used for FordA in the report.

    python multi_dataset/compare_mcnemar.py                 # opt_tuned vs baseline
    python multi_dataset/compare_mcnemar.py --model-a opt_es --dataset FordB
    python multi_dataset/compare_mcnemar.py --model-a opt_es-patience128 --dataset Adiac

Results are printed and written to runs/mcnemar_<model-a>_vs_<model-b>.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from dataset_utils import DATASETS, DEFAULT_RUNS_DIR

RUN_RE = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)$")


def find_runs(dataset_dir: Path, model: str) -> dict[int, Path]:
    """Map seed -> run folder for a model ("opt_es") or a variant ("opt_es-patience128").

    Folders without predictions are skipped.
    """
    runs = {}
    for run_dir in sorted(dataset_dir.iterdir()):
        m = RUN_RE.match(run_dir.name)
        if not m or not (run_dir / "predictions.npz").exists():
            continue
        variant = m["variant"]
        if variant == model or variant.startswith(model + "-"):
            seed = int(m["seed"])
            if seed in runs:
                raise SystemExit(f"Two {model} runs with seed {seed} in {dataset_dir}: "
                                 f"{runs[seed].name}, {run_dir.name}. Pass the variant name "
                                 f"instead, e.g. {run_dir.name.rsplit('_seed', 1)[0]}.")
            runs[seed] = run_dir
    return runs


def mcnemar_exact(run_a: Path, run_b: Path) -> dict:
    a, b = np.load(run_a / "predictions.npz"), np.load(run_b / "predictions.npz")
    if not np.array_equal(a["y_true"], b["y_true"]):
        raise SystemExit(f"Test labels differ between {run_a} and {run_b}")
    correct_a = a["y_pred"] == a["y_true"]
    correct_b = b["y_pred"] == b["y_true"]
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return {
        "error_a": round(1 - correct_a.mean(), 4),
        "error_b": round(1 - correct_b.mean(), 4),
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "both_correct": int((correct_a & correct_b).sum()),
        "both_wrong": int((~correct_a & ~correct_b).sum()),
        "p_exact": float(p),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-a", default="opt_tuned")
    parser.add_argument("--model-b", default="baseline")
    parser.add_argument("--dataset", nargs="+", default=list(DATASETS))
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    args = parser.parse_args()

    rows = []
    for dataset in args.dataset:
        dataset_dir = args.runs_dir / dataset
        if not dataset_dir.is_dir():
            continue
        runs_a, runs_b = find_runs(dataset_dir, args.model_a), find_runs(dataset_dir, args.model_b)
        for seed in sorted(set(runs_a) & set(runs_b)):
            result = mcnemar_exact(runs_a[seed], runs_b[seed])
            rows.append({"dataset": dataset, "seed": seed, "run_a": runs_a[seed].name,
                         "run_b": runs_b[seed].name, **result})

    if not rows:
        raise SystemExit(f"No {args.model_a} / {args.model_b} runs with matching seeds were found.")
    print(f"{'dataset':8s} {'seed':>4s} {'error A':>8s} {'error B':>8s} {'only A':>7s} {'only B':>7s} {'p (exact)':>10s}")
    for r in rows:
        print(f"{r['dataset']:8s} {r['seed']:4d} {r['error_a']:8.4f} {r['error_b']:8.4f} "
              f"{r['only_a_correct']:7d} {r['only_b_correct']:7d} {r['p_exact']:10.2e}")
    out = args.runs_dir / f"mcnemar_{args.model_a}_vs_{args.model_b}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"A = {args.model_a}, B = {args.model_b}. Written to {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
