# FordA / FordB / Adiac experiments

Scripts for the cross-dataset comparison. FordA stays the main dataset; FordB and Adiac are generalization checks.

| File | What it does | Needs |
|---|---|---|
| `train_fcn.py` | Trains and evaluates `baseline`, `opt_es` and `opt_tuned` on any of the three datasets | torch, numpy, scikit-learn (the `UCR-pytorch` env) |
| `tune_fcn.py` | Optuna search on one dataset, following the FordA search; writes `runs/<dataset>/optuna/best_params.json` for `opt_tuned` | as above, plus optuna (5.0.0, as in `UCR-pytorch.yml`) |
| `run_dtw.py` | 1-NN DTW, window 0.01, the same rule as the FordA notebook | numpy, scikit-learn, aeon 1.6.0 (see Environment) |
| `dataset_utils.py` | Data loading, label mapping, metrics, Wilson CI, `runs/summary.csv` | numpy, scikit-learn |
| `compare_mcnemar.py` | Exact McNemar tests between two models with matching seeds, from the saved `predictions.npz`; writes `runs/mcnemar_<A>_vs_<B>.csv` | numpy, scipy |
| `requirements-mac.txt`, `environment-mac.yml` | The environment used for every run in `runs/` (see below) | |

## Environment

| Experiment | Machine | Environment file |
|---|---|---|
| FordA FCN notebooks (`FCN_training.ipynb`, `Hyperparam_tuning_FCN.ipynb`) | Windows, NVIDIA GPU (CUDA) | `UCR-pytorch.yml` |
| FordA preprocessing and the original DTW notebook | | `UCR-tensorflow.yml` |
| Everything in `multi_dataset/runs/`: FordB and Adiac FCN runs, both Optuna searches, 1-NN DTW on all three datasets | macOS, Apple M5 (PyTorch MPS) | `multi_dataset/requirements-mac.txt` or `multi_dataset/environment-mac.yml` |

Both PyTorch environments use torch 2.14.1. The pins in `requirements-mac.txt` were read from the environment that produced the runs. On a CUDA machine, `UCR-pytorch.yml` already has what `train_fcn.py`, `tune_fcn.py` and `compare_mcnemar.py` need. `run_dtw.py` also needs aeon 1.6.0, which requires numpy < 2.5 and scipy < 1.18, while `UCR-pytorch.yml` pins numpy 2.5.2 and scipy 1.18.1; run DTW in `environment-mac.yml` instead (on Windows or Linux, pip installs the CPU build of torch, which DTW does not use). Results can differ slightly between CUDA and MPS even with the same seed.

```bash
conda env create -f multi_dataset/environment-mac.yml   # or: pip install -r multi_dataset/requirements-mac.txt (Python 3.12)
conda activate UCR-mac
```

## Commands (run from the repository root)

```bash
# FCN on FordB, both models (GPU recommended; patience 15 = about 765 steps)
python multi_dataset/train_fcn.py --dataset FordB --model baseline opt_es

# FCN on Adiac, three seeds (patience 128 = about 765 steps at 6 steps per epoch)
python multi_dataset/train_fcn.py --dataset Adiac --model baseline --seeds 42 43 44
python multi_dataset/train_fcn.py --dataset Adiac --model opt_es --patience 128 --seeds 42 43 44

# Per-dataset tuning (test set not used), then the final runs with the tuned settings
python multi_dataset/tune_fcn.py --dataset FordB
python multi_dataset/tune_fcn.py --dataset Adiac --cv-folds 4
python multi_dataset/train_fcn.py --dataset FordB --model opt_tuned --patience auto
python multi_dataset/train_fcn.py --dataset Adiac --model opt_tuned --patience auto --seeds 42 43 44

# Paired tests behind the report table (opt_tuned vs baseline, matched by seed)
python multi_dataset/compare_mcnemar.py

# 1-NN DTW on all three (FordA should reproduce 0.6909)
python multi_dataset/run_dtw.py --dataset FordA FordB Adiac

# Optional: FordA rerun with the original threshold rule, as a sanity check
python multi_dataset/train_fcn.py --dataset FordA --model opt_es --stop-rule threshold
```

Every run writes `runs/<dataset>/<run>/` with `summary.json`, `predictions.npz` (test labels, predictions, probabilities, for McNemar tests later) and, for FCN, `checkpoint.pth` and `train_log.csv` (1-based epochs). `runs/summary.csv` is rebuilt after each run and holds one row per run.

## Protocol

Data are used as provided (all three are already z-normalized). Labels are mapped to 0..K-1 in sorted order, so FordA/FordB map −1 → 0 and +1 → 1 as in `FCN_training.ipynb`. The test set is evaluated once, after training, on the selected checkpoint.

| | `baseline` | `opt_es` |
|---|---|---|
| Architecture | FCN 128/256/128, kernels 8/5/3 | 256/128/64, kernels 8/10/6 (Optuna, FordA) |
| Optimizer | Adam, lr 0.001, fixed | AdamW, lr 0.00403872, weight decay 0, fixed |
| Batch size | floor(N_train / 10): FordA 360, FordB 363, Adiac 39 | 64 |
| Shuffle | yes | no (as in the notebook) |
| Training data | all training series | stratified 90/10 split, random_state 42 (Adiac: 351 / 39) |
| Epochs | 2000 | at most 2000 |
| Stopping | fixed budget | patience on validation loss, matched in gradient steps (about 765 steps): FordA and FordB 15 epochs, Adiac 128 epochs (`--patience`); `--stop-rule threshold` reproduces the FordA rule (val. acc. ≥ 0.93). `opt_tuned` uses the same protocol with its own hyperparameters and `--patience auto` (see notes) |
| Checkpoint | lowest training loss | lowest validation loss |

Notes:

- **Batch size.** The notebook comment says 360 was computed as N_train / 10, so the same rule is applied to every dataset (`--batch-size` overrides it).
- **Stopping rule for `opt_es`.** The 0.93 threshold was set for FordA. Adiac's published FCN accuracy is about 0.86, so the threshold would never trigger there, and the 39-series validation set covers 37 classes. Patience on validation loss is the same idea as the Optuna search (patience 15).
- **Patience is matched in gradient steps, not epochs.** With batch 64, an epoch is 51 steps on FordA (3240 series), 52 on FordB (3272) and 6 on Adiac (351). Patience 15 epochs is about 765 steps on FordA, but only 90 steps on Adiac. The first Adiac run with patience 15 (`runs/Adiac/opt_es-patience15_seed42`, kept for the record) stopped at epoch 18 and selected epoch 3, where the model predicts one class for every validation series (val. acc. 1/39). A validation-only run of 2000 epochs (no test data) showed that in eval mode the validation loss rises to about 25 during the first 40 epochs. It drops below its epoch-3 value only at epoch 70 and reaches its minimum at epoch 212. The same 765 steps on Adiac is 128 epochs (`--patience 128`); FordB stays at 15. The rule was fixed from the step counts and the validation curve, before the new test results were seen.
- **Seeds.** Adiac is small and has 37 classes, so both models are run with seeds 42, 43 and 44 (the train/validation split stays at random_state 42). Report all runs and their mean.
- **`opt_es` vs `opt_tuned`.** `opt_es` reuses the hyperparameters Optuna selected on FordA; the weights are always trained from scratch. `opt_tuned` repeats the Optuna search on each dataset (`tune_fcn.py`) and then trains with the `opt_es` protocol (90/10 split, no shuffling, lowest validation loss, `--patience auto`). FordA's FCN-Opt + ES is the FordA case of `opt_tuned`, so `opt_tuned` is the row that means the same thing on all three datasets.
- **Per-dataset search (`tune_fcn.py`).** Same as `Hyperparam_tuning_FCN.ipynb`: search space, 50 trials, TPE sampler (seed 42), MedianPruner with 5 start-up trials, the default FCN configuration as the first trial, shuffled batches, per-trial checkpoint and objective = best validation macro F1 (loss breaks ties). The epoch budgets (FordA: 100 epochs per trial, patience 15, pruner warm-up 50, interval 10) are multiplied by 3240 / N_fit and rounded up (`dataset_utils.scaled_epochs`), so every dataset sees as many training examples per trial as FordA: FordB keeps 100 / 15 / 50 / 10; Adiac in 4-fold CV (292 series per fold) gets 1110 / 167 / 555 / 111. `--patience auto` in `train_fcn.py` applies the same rule to the final runs: FordB 15, Adiac 139. (The earlier `opt_es` runs on Adiac used 128, matched in gradient steps at batch 64; the two rules differ only by rounding of partial batches.)
- **Adiac validation in the search.** The 90/10 split leaves 39 validation series for 37 classes, so the best of 50 trials on it would mostly reflect noise. The Adiac search therefore uses stratified 4-fold cross-validation over the 390 training series (random_state 42; the smallest class has 4 series, so each class appears in every fold) and maximizes the mean macro F1 over the folds. The pruner watches the first fold. This is the only deliberate difference from the FordA search.
- **Why Adiac needs more epochs.** `FCN_tune` uses BatchNorm with momentum 0.01, so the running statistics used in eval mode adapt over roughly 100 gradient steps. With 6 steps per epoch on Adiac, they lag far behind the weights for the first tens of epochs. This is a likely cause of the early validation collapse described above (not tested separately).
- **Shuffling.** The FordA search shuffled the training batches but the final FCN-Opt + ES run did not; `tune_fcn.py` and `opt_tuned` keep both choices so that the pipeline matches FordA.
- **FordB 1-NN DTW.** The stored output in `preprocessing/visualization-FordB.ipynb` (accuracy 0.690909…) is identical to FordA's and cannot come from FordB's 810 test series (0.690909 × 810 is not an integer), so it is a stale output. Use `run_dtw.py` instead.
- **Runtime.** `train_seconds` is recorded for every FCN run; `peak_gpu_mb` only on CUDA (it is empty on Apple MPS). The two searches ran at the same time on one GPU (FordB about 3.0 h, Adiac about 6.6 h), and the FordB `opt_tuned` final run (130 s) ran while the Adiac search was still using the GPU, so its time is not comparable with the other runs. The Adiac final runs ran alone.
- **Results.** `runs/summary.csv` holds every run. The report table is `report_figures/figures/table_cross_dataset.tex`.
