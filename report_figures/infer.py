"""Re-run inference for the three models whose weights are in the repository.

Produces report_figures/infer.npz with, per model (M1 = baseline, M2 = + z-score, M4 = FCN-Opt + ES):
  *_p      test-set softmax probabilities            (N_test, 2)
  *_cam    class activation maps for both classes     (N_test, 2, T)
  *_gap    GAP features                               (N_test, C)
  *_trp    eval-mode softmax on the model's OWN training data
  M4_valp  eval-mode softmax on the 10 % validation split (sanity check of the split)
  del_*    CAM deletion test (M1, M4): accuracy after zeroing the top-k CAM time steps
           vs. the same mask circularly shifted to a random position

Usage (from anywhere):  python report_figures/infer.py            # full run
                        python report_figures/infer.py --no-del   # skip the deletion test
"""
import os
import sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get("DATA_DIR", os.path.dirname(HERE)) + "/"
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(HERE)
from FCN_model import FCN, FCN_tune  # noqa: E402

torch.set_grad_enabled(False)
DO_DEL = "--no-del" not in sys.argv

tr = np.load(D + "FordA_processed/FordA_TRAIN.npy")
te = np.load(D + "FordA_processed/FordA_TEST.npy")
Xtr = tr[:, 1:][:, None, :].astype(np.float32)
Xte = te[:, 1:][:, None, :].astype(np.float32)
ytr = (tr[:, 0] == 1).astype(int)
yte = (te[:, 0] == 1).astype(int)
mu, sd = Xtr.mean(), Xtr.std()
print("train", Xtr.shape, "test", Xte.shape, "global mean/std %.4f %.4f" % (mu, sd))

# Same 90/10 split as FCN_training.ipynb ("use a validation set to get early stopping")
tr_idx, val_idx = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=42)

cfg = {
    # key: (model, weight file, z-score input?, indices of the model's own training data)
    "M1": (FCN(1, num_classes=2), "fcn_model_highest_accuracy_epoch1905.pth", False, np.arange(len(ytr))),
    "M2": (FCN(1, num_classes=2), "fcn_model_zscored_min_loss_epoch1898_loss 0.11041.pth", True,
           np.arange(len(ytr))),
    "M4": (FCN_tune(1, num_classes=2, filters=[256, 128, 64], kernels=[8, 10, 6]),
           "fcn_tuned_best0.19576737036995612.pth", False, tr_idx),
}


def prep(X, z):
    return torch.tensor((X - mu) / sd if z else X)


def predict(m, X, bs=330):
    out = [torch.softmax(m(X[i:i + bs]), 1) for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


out = {"Xte": Xte[:, 0], "yte": yte, "ytr": ytr, "tr_idx_es": tr_idx, "val_idx_es": val_idx}
models = {}
for k, (m, f, z, own) in cfg.items():
    st = torch.load(D + f, map_location="cpu", weights_only=True)
    m.load_state_dict(st["model_state_dict"])
    m.eval()
    models[k] = (m, z)
    X = prep(Xte, z)
    body = m.layers[:-3]                       # up to the last ReLU (or Dropout)
    A = body(X)                                # N, C, T
    logits = m.layers[-1](A.mean(-1))
    p = torch.softmax(logits, 1).numpy()
    W = m.layers[-1].weight.numpy()            # 2, C
    cam = np.einsum("kc,nct->nkt", W, A.numpy())
    pred = p.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, average="macro")
    print(k, "TEST  acc %.4f prec %.4f rec %.4f f1 %.4f loss %.4f"
          % (accuracy_score(yte, pred), pr, rc, f1, log_loss(yte, p)))
    out[k + "_p"] = p
    out[k + "_cam"] = cam.astype(np.float32)
    out[k + "_gap"] = A.mean(-1).numpy()

    # eval-mode accuracy on the model's own training data
    ptr = predict(m, prep(Xtr[own], z))
    out[k + "_trp"] = ptr
    print(k, "TRAIN (eval mode, n=%d) acc %.4f loss %.4f"
          % (len(own), accuracy_score(ytr[own], ptr.argmax(1)), log_loss(ytr[own], ptr)))
    if k == "M4":
        pv = predict(m, prep(Xtr[val_idx], z))
        out["M4_valp"] = pv
        print(k, "VAL   (n=%d) acc %.4f loss %.4f  [training log: 0.9335 / 0.1958]"
              % (len(val_idx), accuracy_score(ytr[val_idx], pv.argmax(1)), log_loss(ytr[val_idx], pv)))

# ------------------------------------------------------------------ CAM deletion test
# For each test series, zero (= global mean of the z-normalised data) the fraction f of time
# steps with the highest CAM value for the predicted class. Control: the identical mask,
# circularly shifted by a random offset, so that the number and contiguity of the removed
# points are the same and only their location differs.
if DO_DEL:
    FRAC = np.array([0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
    N_RAND = 5
    rng = np.random.default_rng(0)
    out["del_frac"] = FRAC
    T = Xte.shape[-1]
    for k in ["M1", "M4"]:
        m, z = models[k]
        cam = out[k + "_cam"]
        pred = out[k + "_p"].argmax(1)
        c_pred = cam[np.arange(len(yte)), pred]            # N, T
        order = np.argsort(-c_pred, axis=1)                # descending CAM
        acc_top, acc_rnd = [], []
        for f in FRAC:
            n = int(round(f * T))
            mask = np.zeros_like(c_pred, dtype=bool)
            if n:
                np.put_along_axis(mask, order[:, :n], True, axis=1)
            Xm = Xte[:, 0].copy(); Xm[mask] = 0.0
            acc_top.append(accuracy_score(yte, predict(m, prep(Xm[:, None, :], z)).argmax(1)))
            accs = []
            for r in range(N_RAND if n else 1):
                sh = rng.integers(50, T - 50, size=len(yte))
                ms = np.stack([np.roll(mask[i], sh[i]) for i in range(len(yte))])
                Xr = Xte[:, 0].copy(); Xr[ms] = 0.0
                accs.append(accuracy_score(yte, predict(m, prep(Xr[:, None, :], z)).argmax(1)))
            acc_rnd.append(accs + [np.nan] * (N_RAND - len(accs)))
            print(k, "deletion f=%.2f  top-CAM acc %.4f   shifted acc %.4f"
                  % (f, acc_top[-1], np.nanmean(acc_rnd[-1])))
        out["del_" + k + "_top"] = np.array(acc_top)
        out["del_" + k + "_rnd"] = np.array(acc_rnd)

np.savez_compressed("infer.npz", **out)
print("saved", os.path.join(HERE, "infer.npz"))
