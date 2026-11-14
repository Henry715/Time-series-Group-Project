"""Optuna search for FCN hyperparameters on one dataset, following the FordA
search in Hyperparam_tuning_FCN.ipynb.

Same as the FordA search:
  search space (filters, kernels, learning rate, batch size, weight decay,
  Adam/AdamW), 50 trials, TPE sampler with seed 42, MedianPruner with 5
  start-up trials, the default FCN configuration enqueued as the first trial,
  set_seed(42) at the start of each trial, shuffled training batches,
  per-trial checkpoint = highest validation macro F1 (validation loss breaks
  ties), patience on that macro F1, objective = that macro F1.

Scaled by training-set size (dataset_utils.scaled_epochs), so that every
dataset sees as many training examples as FordA did:
  maximum epochs per trial (FordA 100), patience (15), pruner warm-up (50)
  and pruner interval (10). FordB keeps the FordA values.

Validation:
  --cv-folds 0  stratified 90/10 split with random_state 42, as on FordA.
  --cv-folds k  stratified k-fold cross-validation over all training series
                (shuffled, random_state 42); objective = mean over folds of
                each fold's best macro F1. The pruner sees the first fold.
                Used for Adiac, whose 39-series validation split covers 37
                classes; with k = 4 every class appears in every fold.

The test series are dropped right after loading and never used here.

Examples (from the repository root):
  python multi_dataset/tune_fcn.py --dataset FordB
  python multi_dataset/tune_fcn.py --dataset Adiac --cv-folds 4

Outputs in multi_dataset/runs/<dataset>/optuna/: study.db (the study can be
resumed by running the same command again), trials.csv (updated after every
trial) and best_params.json (read by train_fcn.py --model opt_tuned).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from dataset_utils import DATASETS, DEFAULT_RUNS_DIR, load_dataset, scaled_epochs
from train_fcn import build_model, build_optimizer, evaluate, pick_device, set_seed, train_one_epoch
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
FORDA_BUDGET = {"max_epochs": 100, "patience": 15, "warmup": 50, "interval": 10}

# First trial of the FordA study (study.enqueue_trial in the notebook).
DEFAULT_TRIAL = {
    "filters1": 128, "filters2": 256, "filters3": 128,
    "kernel1": 8, "kernel2": 5, "kernel3": 3,
    "learning_rate": 0.001, "batch_size": 128, "weight_decay": 0.0, "optimizer": "Adam",
}


def suggest(trial: optuna.Trial) -> dict:
    """Search space of the FordA notebook, suggested in the same order."""
    filters = (
        trial.suggest_categorical("filters1", [64, 128, 256]),
        trial.suggest_categorical("filters2", [128, 256, 512]),
        trial.suggest_categorical("filters3", [64, 128, 256]),
    )
    kernels = (
        trial.suggest_categorical("kernel1", [8, 16, 6]),
        trial.suggest_categorical("kernel2", [5, 10, 4]),
        trial.suggest_categorical("kernel3", [3, 6, 2]),
    )
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    return params_to_config(dict(zip(
        ["filters", "kernels", "lr", "batch_size", "weight_decay", "optimizer"],
        [filters, kernels, lr, batch_size, weight_decay, optimizer],
    )))


def params_to_config(p: dict) -> dict:
    return {
        "arch": "FCN_tune",
        "filters": tuple(p["filters"]),
        "kernels": tuple(p["kernels"]),
        "optimizer": p["optimizer"],
        "lr": p["lr"],
        "weight_decay": p["weight_decay"],
        "batch_size": p["batch_size"],
    }


def config_from_trial_params(tp: dict) -> dict:
    return params_to_config({
        "filters": (tp["filters1"], tp["filters2"], tp["filters3"]),
        "kernels": (tp["kernel1"], tp["kernel2"], tp["kernel3"]),
        "lr": tp["learning_rate"], "batch_size": tp["batch_size"],
        "weight_decay": tp["weight_decay"], "optimizer": tp["optimizer"],
    })


def make_folds(y: np.ndarray, cv_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(y))
    if cv_folds == 0:
        tr, va = train_test_split(idx, test_size=0.1, stratify=y, random_state=SEED)
        return [(tr, va)]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    return list(skf.split(idx, y))


def fit_fold(cfg, X_tr, y_tr, X_va, y_va, n_classes, budget, device, trial=None) -> dict:
    """One training run with the notebook's per-trial loop; reports to the pruner if trial is given."""
    set_seed(SEED)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=cfg["batch_size"], shuffle=True)
    model = build_model(cfg, n_classes).to(device)
    optimizer = build_optimizer(cfg, model)

    best = {"f1": -1.0, "loss": math.inf, "accuracy": None, "epoch": 0}
    since_best = 0
    for epoch in range(budget["max_epochs"]):
        train_one_epoch(model, loader, optimizer, device, n_classes)
        m = evaluate(model, X_va, y_va, device, n_classes)
        if m["f1"] > best["f1"] or (m["f1"] == best["f1"] and m["loss"] < best["loss"]):
            best = {"f1": m["f1"], "loss": m["loss"], "accuracy": m["accuracy"], "epoch": epoch + 1}
            since_best = 0
        else:
            since_best += 1
        if trial is not None:
            trial.report(m["f1"], step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if since_best >= budget["patience"]:
            break
    best["epochs_run"] = epoch + 1
    return best


def write_trials(study: optuna.Study, out_dir: Path) -> None:
    rows = []
    for t in study.trials:
        rows.append({
            "trial": t.number,
            "state": t.state.name,
            "value": t.value,
            "seconds": None if t.duration is None else round(t.duration.total_seconds(), 1),
            **{k: t.user_attrs.get(k) for k in ["fold_f1", "fold_accuracy", "fold_best_epoch", "fold_epochs_run"]},
            **t.params,
        })
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with (out_dir / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--cv-folds", type=int, default=0, help="0 = 90/10 split as on FordA; k = k-fold CV")
    ap.add_argument("--n-trials", type=int, default=50, help="total trials in the study (resumes count)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=str(DEFAULT_RUNS_DIR))
    args = ap.parse_args()

    data = load_dataset(args.dataset)
    X = torch.tensor(data["X_train"][:, None, :], dtype=torch.float32)
    y_np = data["y_train"]
    y = torch.tensor(y_np, dtype=torch.long)
    n_classes = data["n_classes"]
    del data  # the test series are not used in tuning

    folds = make_folds(y_np, args.cv_folds)
    n_fit = min(len(tr) for tr, _ in folds)
    budget = {k: scaled_epochs(v, n_fit) for k, v in FORDA_BUDGET.items()}
    device = pick_device(args.device)

    out_dir = Path(args.out_dir) / args.dataset / "optuna"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {args.dataset} | {len(folds)} fold(s), fit {n_fit}+ / val {len(folds[0][1])} | "
          f"budget {budget} | device {device} ===", flush=True)

    def objective(trial: optuna.Trial) -> float:
        set_seed(SEED)
        cfg = suggest(trial)
        results = []
        for i, (tr, va) in enumerate(folds):
            r = fit_fold(cfg, X[tr], y[tr], X[va], y[va], n_classes, budget, device,
                         trial=trial if i == 0 else None)
            results.append(r)
        trial.set_user_attr("fold_f1", [r["f1"] for r in results])
        trial.set_user_attr("fold_accuracy", [r["accuracy"] for r in results])
        trial.set_user_attr("fold_best_epoch", [r["epoch"] for r in results])
        trial.set_user_attr("fold_epochs_run", [r["epochs_run"] for r in results])
        return float(np.mean([r["f1"] for r in results]))

    study = optuna.create_study(
        study_name=f"{args.dataset}_FCN_MacroF1",
        storage=f"sqlite:///{out_dir / 'study.db'}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=budget["warmup"], interval_steps=budget["interval"]
        ),
    )
    for k, v in {"dataset": args.dataset, "cv_folds": args.cv_folds, "n_fit": n_fit,
                 "budget": budget, "forda_budget": FORDA_BUDGET, "seed": SEED}.items():
        study.set_user_attr(k, v)
    study.enqueue_trial(DEFAULT_TRIAL, skip_if_exists=True)

    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    remaining = args.n_trials - sum(t.state in finished for t in study.trials)
    t0 = time.perf_counter()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True,
                       callbacks=[lambda s, t: write_trials(s, out_dir)])
    write_trials(study, out_dir)

    best = study.best_trial
    cfg = config_from_trial_params(best.params)
    result = {
        "dataset": args.dataset,
        "best_trial": best.number,
        "objective": "validation macro F1" + (f", mean of {args.cv_folds} folds" if args.cv_folds else ""),
        "value": best.value,
        "fold_f1": best.user_attrs.get("fold_f1"),
        "fold_best_epoch": best.user_attrs.get("fold_best_epoch"),
        "params": best.params,
        "config": cfg,
        "cv_folds": args.cv_folds,
        "n_fit": n_fit,
        "budget": budget,
        "n_trials": len(study.trials),
        "n_complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "search_seconds_this_run": round(time.perf_counter() - t0, 1),
        "device": str(device),
        "torch": torch.__version__,
        "optuna": optuna.__version__,
        "python": platform.python_version(),
    }
    (out_dir / "best_params.json").write_text(json.dumps(result, indent=2))
    print(f"BEST {args.dataset}: trial {best.number}, macro F1 {best.value:.4f}, config {cfg}", flush=True)


if __name__ == "__main__":
    main()
