"""Train and evaluate the FCN variants on FordA, FordB and Adiac.

Two models, with the protocols of FCN_training.ipynb:

  baseline  FCN 128/256/128, kernels 8/5/3; Adam, lr 0.001, fixed;
            batch = floor(N_train / 10) (FordA 360, FordB 363, Adiac 39),
            shuffled; 2000 epochs on all training series; checkpoint with
            the lowest training loss. (The notebook saved files only below a
            loss of 0.2; the selected checkpoint is the same whenever the
            loss falls below 0.2, which the log reports.)

  opt_es    FCN 256/128/64, kernels 8/10/6 and the Optuna settings selected
            on FordA (AdamW, lr 0.00403872, weight decay 0, batch 64, no
            shuffling); stratified 90/10 split with random_state 42;
            checkpoint with the lowest validation loss. No re-tuning on
            FordB/Adiac. Stopping rule:
              --stop-rule patience   (default) stop after --patience epochs
                                     (15, as in the Optuna search) without a
                                     lower validation loss;
              --stop-rule threshold  stop when validation accuracy reaches
                                     --val-acc-threshold (0.93, the FordA
                                     rule; Adiac would never reach it).
            --patience auto scales 15 epochs by training-set size
            (dataset_utils.scaled_epochs): FordB 15, Adiac 139.

  opt_tuned Same training protocol as opt_es, with the hyperparameters that
            tune_fcn.py selected on this dataset
            (runs/<dataset>/optuna/best_params.json, or --params).

The test set is evaluated once, after training, on the selected checkpoint.

Examples (from the repository root):
  python multi_dataset/train_fcn.py --dataset FordB Adiac --model baseline opt_es
  python multi_dataset/train_fcn.py --dataset Adiac --model baseline --seeds 42 43 44
  python multi_dataset/train_fcn.py --dataset FordA --model opt_es --stop-rule threshold
  python multi_dataset/train_fcn.py --dataset Adiac --model opt_tuned --patience auto --seeds 42 43 44

Outputs per run in multi_dataset/runs/<dataset>/<run>/: checkpoint.pth,
train_log.csv (1-based epochs), predictions.npz (test labels, predictions,
probabilities) and summary.json; runs/summary.csv collects every run.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repository root, for FCN_model

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from FCN_model import FCN, FCN_tune
from dataset_utils import (
    DATASETS,
    DEFAULT_RUNS_DIR,
    classification_summary,
    load_dataset,
    scaled_epochs,
    write_run_summary,
)

MODEL_CONFIGS = {
    "baseline": {
        "arch": "FCN",
        "filters": (128, 256, 128),
        "kernels": (8, 5, 3),
        "optimizer": "Adam",
        "lr": 0.001,
        "weight_decay": 0.0,
        "batch_size": "auto",  # floor(N_train / 10)
        "shuffle": True,
        "val_fraction": 0.0,
        "selection": "min_train_loss",
    },
    "opt_es": {
        "arch": "FCN_tune",
        "filters": (256, 128, 64),
        "kernels": (8, 10, 6),
        "optimizer": "AdamW",
        "lr": 0.004038721483970501,
        "weight_decay": 0.0,
        "batch_size": 64,
        "shuffle": False,
        "val_fraction": 0.1,
        "selection": "min_val_loss",
    },
}
MODEL_NAMES = sorted(MODEL_CONFIGS) + ["opt_tuned"]


def load_tuned_config(dataset: str, args) -> tuple[dict, str]:
    """opt_es protocol with the hyperparameters tune_fcn.py selected on this dataset."""
    path = Path(args.params) if args.params else Path(args.out_dir) / dataset / "optuna" / "best_params.json"
    tuned = json.loads(path.read_text())
    if tuned.get("dataset") not in (None, dataset):
        raise ValueError(f"{path} was tuned on {tuned['dataset']}, not {dataset}")
    cfg = dict(MODEL_CONFIGS["opt_es"])  # shuffle, validation split, selection
    for key in ["arch", "filters", "kernels", "optimizer", "lr", "weight_decay", "batch_size"]:
        cfg[key] = tuple(tuned["config"][key]) if key in ("filters", "kernels") else tuned["config"][key]
    try:  # keep summaries free of machine-specific absolute paths
        shown = path.resolve().relative_to(HERE.parent)
    except ValueError:
        shown = path
    return cfg, str(shown)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: dict, n_classes: int) -> nn.Module:
    if cfg["arch"] == "FCN":
        return FCN(in_channels=1, num_classes=n_classes)
    return FCN_tune(
        in_channels=1, num_classes=n_classes, filters=cfg["filters"], kernels=cfg["kernels"]
    )


def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    opt_cls = torch.optim.Adam if cfg["optimizer"] == "Adam" else torch.optim.AdamW
    return opt_cls(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def macro_prf(y_true, y_pred, n_classes):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0
    )
    return float(p), float(r), float(f1)


def train_one_epoch(model, loader, optimizer, device, n_classes):
    """One pass in train mode; metrics are running (train-mode) values, as in the notebooks."""
    model.train()
    total_loss, total = 0.0, 0
    preds, labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        total += X.size(0)
        preds.append(logits.argmax(1).detach().cpu())
        labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    p, r, f1 = macro_prf(labels, preds, n_classes)
    return {
        "loss": total_loss / total,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": p,
        "recall": r,
        "f1": f1,
    }


@torch.no_grad()
def predict_logits(model, X: torch.Tensor, device, batch_size: int = 512) -> torch.Tensor:
    model.eval()
    out = [model(X[i : i + batch_size].to(device)).cpu() for i in range(0, len(X), batch_size)]
    return torch.cat(out)


def evaluate(model, X, y, device, n_classes):
    logits = predict_logits(model, X, device)
    preds = logits.argmax(1).numpy()
    p, r, f1 = macro_prf(y.numpy(), preds, n_classes)
    return {
        "loss": float(F.cross_entropy(logits, y).item()),
        "accuracy": float(accuracy_score(y.numpy(), preds)),
        "f1": f1,
        "logits": logits,
        "preds": preds,
    }


def run(dataset: str, model_name: str, seed: int, args) -> dict:
    if model_name == "opt_tuned":
        cfg, params_path = load_tuned_config(dataset, args)
    else:
        cfg, params_path = dict(MODEL_CONFIGS[model_name]), None
    data = load_dataset(dataset)
    n_classes = data["n_classes"]
    device = pick_device(args.device)

    X_all = torch.tensor(data["X_train"][:, None, :], dtype=torch.float32)
    y_all = torch.tensor(data["y_train"], dtype=torch.long)
    X_test = torch.tensor(data["X_test"][:, None, :], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)

    # ----- training / validation split -----
    if cfg["val_fraction"] > 0:
        tr_idx, val_idx = train_test_split(
            np.arange(len(y_all)),
            test_size=cfg["val_fraction"],
            stratify=y_all.numpy(),
            random_state=args.split_seed,
        )
        X_fit, y_fit = X_all[tr_idx], y_all[tr_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
    else:
        X_fit, y_fit, X_val, y_val = X_all, y_all, None, None

    batch_size = args.batch_size or cfg["batch_size"]
    if batch_size == "auto":
        batch_size = max(1, len(X_all) // 10)

    stop_rule = "fixed_budget" if model_name == "baseline" else args.stop_rule
    patience = scaled_epochs(15, len(X_fit)) if args.patience == "auto" else int(args.patience)
    if stop_rule == "patience":
        variant = f"{model_name}-patience{patience}"
    elif stop_rule == "threshold":
        variant = f"{model_name}-thr{args.val_acc_threshold:g}"
    else:
        variant = model_name
    run_name = f"{variant}_seed{seed}" + (f"_{args.tag}" if args.tag else "")
    run_dir = Path(args.out_dir) / dataset / run_name

    print(
        f"\n=== {dataset} | {run_name} | device {device} | classes {n_classes} | "
        f"fit {len(X_fit)} / val {0 if X_val is None else len(X_val)} / test {len(X_test)} | "
        f"batch {batch_size} ==="
    )

    set_seed(seed)
    model = build_model(cfg, n_classes).to(device)
    optimizer = build_optimizer(cfg, model)
    loader = DataLoader(TensorDataset(X_fit, y_fit), batch_size=batch_size, shuffle=cfg["shuffle"])

    log_rows = []
    best_score, best_epoch, best_state = float("inf"), 0, None
    epochs_since_best = 0
    stop_reason = "max_epochs"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        row = {"epoch": epoch, **train_one_epoch(model, loader, optimizer, device, n_classes)}

        if X_val is not None:
            val = evaluate(model, X_val, y_val, device, n_classes)
            row.update(val_loss=val["loss"], val_accuracy=val["accuracy"], val_f1=val["f1"])
            score = val["loss"]
            improved = score < best_score  # strict, as in the early-stopping notebook
        else:
            score = row["loss"]
            improved = score <= best_score  # as in FCN_training.ipynb

        if improved:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        log_rows.append(row)

        if epoch == 1 or epoch % args.log_every == 0:
            msg = f"epoch {epoch:4d} | train loss {row['loss']:.4f} acc {row['accuracy']:.4f}"
            if X_val is not None:
                msg += f" | val loss {row['val_loss']:.4f} acc {row['val_accuracy']:.4f}"
            print(msg + f" | best epoch {best_epoch} ({best_score:.4f})")

        if stop_rule == "patience" and epochs_since_best >= patience:
            stop_reason = f"no lower val loss for {patience} epochs"
            break
        if stop_rule == "threshold" and row["val_accuracy"] >= args.val_acc_threshold:
            stop_reason = f"val accuracy >= {args.val_acc_threshold}"
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    epochs_run = len(log_rows)
    print(f"stopped after {epochs_run} epochs ({stop_reason}); selected epoch {best_epoch}; "
          f"{train_seconds:.0f} s")

    # ----- evaluate the selected checkpoint (test set used once, here) -----
    if best_state is None:
        raise RuntimeError("no checkpoint was selected (loss was NaN in every epoch)")
    model.load_state_dict(best_state)
    test = evaluate(model, X_test, y_test, device, n_classes)
    probs = torch.softmax(test["logits"], dim=1).numpy()
    fit_eval = evaluate(model, X_fit, y_fit, device, n_classes)

    summary = {
        "dataset": dataset,
        "model": model_name,
        "variant": variant,
        "run": run_name,
        "seed": seed,
        "n_classes": n_classes,
        "length": data["length"],
        "n_train_fit": len(X_fit),
        "n_val": 0 if X_val is None else len(X_val),
        "batch_size": batch_size,
        "optimizer": cfg["optimizer"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "filters": "/".join(map(str, cfg["filters"])),
        "kernels": "/".join(map(str, cfg["kernels"])),
        "params_file": params_path,
        "max_epochs": args.epochs,
        "stop_rule": stop_rule,
        "patience": patience if stop_rule == "patience" else None,
        "val_acc_threshold": args.val_acc_threshold if stop_rule == "threshold" else None,
        "epochs_run": epochs_run,
        "stop_reason": stop_reason,
        "selection": cfg["selection"],
        "selected_epoch": best_epoch,
        "selected_score": best_score,
        "train_loss_below_0.2": bool(min(r["loss"] for r in log_rows) <= 0.2),
        "train_acc_eval": fit_eval["accuracy"],
        "val_acc_selected": log_rows[best_epoch - 1].get("val_accuracy"),
        **classification_summary(y_test.numpy(), test["preds"], n_classes, probs),
        "test_loss": test["loss"],
        "train_seconds": round(train_seconds, 1),
        "peak_gpu_mb": None if peak_mb is None else round(peak_mb, 1),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "config": cfg,
            "selected_epoch": best_epoch,
            "n_classes": n_classes,
            "classes": data["classes"].tolist(),
            "seed": seed,
        },
        run_dir / "checkpoint.pth",
    )
    with (run_dir / "train_log.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    np.savez(
        run_dir / "predictions.npz",
        y_true=y_test.numpy(),
        y_pred=test["preds"],
        probs=probs,
        classes=data["classes"],
    )
    write_run_summary(run_dir, summary, Path(args.out_dir))

    print(
        f"TEST {dataset} {run_name}: acc {summary['test_acc']:.4f} "
        f"[{summary['acc_ci95_low']:.3f}, {summary['acc_ci95_high']:.3f}] | "
        f"error {summary['test_error']:.4f} | macro F1 {summary['f1_macro']:.4f} | "
        f"loss {summary['test_loss']:.4f} | AUC {summary['auc']}"
    )
    print(f"saved to {os.path.relpath(run_dir)}")
    return summary


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", required=True, choices=sorted(DATASETS))
    ap.add_argument("--model", nargs="+", default=["baseline", "opt_es"], choices=MODEL_NAMES)
    ap.add_argument("--params", default=None,
                    help="opt_tuned: best_params.json to use (default runs/<dataset>/optuna/best_params.json)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--split-seed", type=int, default=42, help="random_state of the 90/10 split")
    ap.add_argument("--epochs", type=int, default=2000, help="maximum number of epochs")
    ap.add_argument("--batch-size", type=int, default=None, help="override the model's batch size")
    ap.add_argument("--stop-rule", choices=["patience", "threshold"], default="patience",
                    help="opt_es only")
    ap.add_argument("--patience", default="15",
                    help="epochs without a lower val loss, or 'auto' (15 scaled by training-set size)")
    ap.add_argument("--val-acc-threshold", type=float, default=0.93)
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out-dir", default=str(DEFAULT_RUNS_DIR))
    ap.add_argument("--tag", default="", help="suffix for the run folder name")
    return ap.parse_args()


def main():
    args = parse_args()
    for dataset in args.dataset:
        for model_name in args.model:
            for seed in args.seeds:
                run(dataset, model_name, seed, args)


if __name__ == "__main__":
    main()
