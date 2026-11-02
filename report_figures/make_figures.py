"""Generate all IEEE-style figures for the FordA FCN report.

Inputs : training-log CSVs, Optuna study DB, infer.npz (produced by infer.py)
Outputs: figures/*.pdf (+ .png previews), figures/table_results.tex
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
from sklearn.metrics import confusion_matrix, roc_curve, auc
import ieee_style as S

S.apply()
import os
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get("DATA_DIR", os.path.dirname(HERE)) + "/"
os.chdir(HERE)
R = np.load("infer.npz")
y = R["yte"]
KEYS = ["base", "z", "tune", "es"]

# ---------------------------------------------------------------- data
log_base = pd.read_csv(D + "FCN_original_metrics.csv", index_col=0)
log_z = pd.read_csv(D + "FCN_original_metrics_zscored.csv", index_col=0)
log_tune = pd.read_csv(D + "FCN_tuned_metrics.csv", index_col=0)
log_es = pd.read_csv(D + "FCN_tuned_metrics_early_stop.csv")

SEL = {"base": 1905, "z": 1898, "tune": 407, "es": 10}       # selected checkpoints (0-based epoch)
# test metrics (macro averaged); base/z/es re-computed by infer.py, tune from notebook output
TEST = {
    "base": dict(acc=0.9273, prec=0.9276, rec=0.9279, f1=0.9273, loss=0.1992),
    "z":    dict(acc=0.9205, prec=0.9216, rec=0.9215, f1=0.9205, loss=0.2254),
    "tune": dict(acc=0.8909, prec=0.9015, rec=0.8884, f1=0.8896, loss=0.6632),
    "es":   dict(acc=0.9227, prec=0.9241, rec=0.9238, f1=0.9227, loss=0.2045),
}
PROB = {"base": R["base_p"] if "base_p" in R else R["M1_p"], "z": R["M2_p"], "es": R["M4_p"]}
CAM = {"base": R["M1_cam"], "es": R["M4_cam"]}
GAP = {"base": R["M1_gap"], "es": R["M4_gap"]}
X = R["Xte"]
N_TEST = len(y)

# Training accuracy: eval-mode pass of the final weights over each model's OWN training data
# (3601 series; FCN-Opt + ES: the 3240 left after the validation split), recomputed by infer.py.
# FCN-Opt has no weights in the repository, so its value is the training-log (running) accuracy.
ytr, tr_es = R["ytr"], R["tr_idx_es"]
TRAIN_ACC = {
    "base": (R["M1_trp"].argmax(1) == ytr).mean(),
    "z": (R["M2_trp"].argmax(1) == ytr).mean(),
    "tune": log_tune.loc[SEL["tune"], "accuracy"],
    "es": (R["M4_trp"].argmax(1) == ytr[tr_es]).mean(),
}
TRAIN_FROM_LOG = {"tune"}


# ---------------------------------------------------------------- statistics
def wilson(acc, n=N_TEST, z=1.959964):
    """95 % Wilson score interval for an accuracy measured on n test series."""
    k = round(acc * n); p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


CI = {k: wilson(TEST[k]["acc"]) for k in KEYS}


def mcnemar(a, b):
    """Exact McNemar test on the test-set predictions of models a and b."""
    from scipy.stats import binomtest
    ca, cb = PROB[a].argmax(1) == y, PROB[b].argmax(1) == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return n01, n10, binomtest(n01, n01 + n10).pvalue


def two_prop_p(a, b, n=N_TEST):
    """Two-proportion z-test; used only for FCN-Opt, whose predictions are not available."""
    from scipy.stats import norm
    p1, p2 = TEST[a]["acc"], TEST[b]["acc"]
    p = (p1 + p2) / 2
    zs = (p1 - p2) / np.sqrt(2 * p * (1 - p) / n)
    return 2 * norm.sf(abs(zs))


MCN = {k: mcnemar("base", k) for k in ["z", "es"]}


def cam_concentration(k):
    """Share of the positive CAM evidence (predicted class) that falls in the top 10 % of time steps.
    0.10 = evidence spread uniformly over the series; 1.0 = all evidence in 50 of 500 steps."""
    cam = CAM[k]
    pred = PROB[k].argmax(1)
    c = np.maximum(cam[np.arange(len(y)), pred], 0)
    s = -np.sort(-c, axis=1)
    top = int(round(0.1 * c.shape[1]))
    return s[:, :top].sum(1) / s.sum(1)


def print_stats():
    print("Test accuracy, 95% Wilson CI:")
    for k in KEYS:
        print(f"  {S.NAMES[k]:16s} {TEST[k]['acc']:.4f}  [{CI[k][0]:.4f}, {CI[k][1]:.4f}]"
              f"  train {TRAIN_ACC[k]:.4f}  gap {TRAIN_ACC[k] - TEST[k]['acc']:+.4f}")
    for k, (n01, n10, p) in MCN.items():
        print(f"  McNemar baseline vs {S.NAMES[k]}: only baseline right {n01}, only other right {n10}, p = {p:.3f}")
    print(f"  Two-proportion z-test baseline vs FCN-Opt: p = {two_prop_p('base', 'tune'):.2g}")
    from scipy.stats import mannwhitneyu
    conc = {k: cam_concentration(k) for k in ["base", "es"]}
    for c in [0, 1]:
        m = y == c
        print(f"  CAM top-10% share, class {c}: baseline median {np.median(conc['base'][m]):.3f}, "
              f"Opt+ES median {np.median(conc['es'][m]):.3f}, "
              f"Mann-Whitney p = {mannwhitneyu(conc['base'][m], conc['es'][m]).pvalue:.1e}")


def corner(ax, s, x=0.015, y=0.96):
    ax.text(x, y, s, transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85), zorder=10)


# ---------------------------------------------------------------- Fig. 1
def fig_performance():
    """(a) Test accuracy with 95 % Wilson CI and p-value vs. the baseline;
    (b) train (eval mode) -> test accuracy for every model, with the gap."""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(S.COL2, 1.85), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1], wspace=0.08))
    ypos = {k: len(KEYS) - 1 - i for i, k in enumerate(KEYS)}      # baseline on top

    ax = axes[0]
    lo, hi = CI["base"]
    ax.axvspan(lo, hi, color=S.C["base"], alpha=0.12, lw=0, zorder=1)
    ax.axvline(TEST["base"]["acc"], color=S.C["base"], ls="--", lw=0.6, zorder=2)
    for k in KEYS:
        a, (l, h) = TEST[k]["acc"], CI[k]
        ax.errorbar(a, ypos[k], xerr=[[a - l], [h - a]], fmt=S.MARK[k], ms=4.5, mfc=S.C[k],
                    mec="black", mew=0.5, ecolor=S.C[k], elinewidth=1.2, capsize=2.5, capthick=0.8,
                    zorder=4)
        if k == "base":
            txt = "reference"
        elif k in MCN:
            txt = f"$p$ = {MCN[k][2]:.2f}"
        else:
            p = two_prop_p("base", k)
            txt = "$p$ < 0.001$^{*}$" if p < 1e-3 else f"$p$ = {p:.3f}$^{{*}}$"
        ax.text(h + 0.002, ypos[k], txt, va="center", ha="left", fontsize=8)
    ax.set_xlim(0.865, 0.966)
    ax.set_xlabel("Test accuracy (95% confidence interval)")
    ax.set_yticks([ypos[k] for k in KEYS], [S.NAMES[k] for k in KEYS])
    ax.set_ylim(-0.6, len(KEYS) - 0.4)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.grid(axis="y", visible=False)
    S.panel_label(ax, "(a) Test accuracy and significance vs. baseline", y=-0.33)

    ax = axes[1]
    for k in KEYS:
        a, b = TRAIN_ACC[k], TEST[k]["acc"]
        log = k in TRAIN_FROM_LOG
        ax.plot([b, a], [ypos[k]] * 2, color=S.C[k], lw=1.2, ls=":" if log else "-", zorder=2)
        ax.plot(a, ypos[k], marker=S.MARK[k], ms=4.5, mfc="white", mec=S.C[k], mew=1.0, ls="none", zorder=3)
        ax.plot(b, ypos[k], marker=S.MARK[k], ms=4.5, mfc=S.C[k], mec="black", mew=0.5, ls="none", zorder=4)
        ax.text(max(a, b) + 0.004, ypos[k], f"gap {a - b:.3f}" + ("$^{\\dagger}$" if log else ""),
                va="center", ha="left", fontsize=8)
    ax.set_xlim(0.865, 1.045)
    ax.set_xticks(np.arange(0.88, 1.001, 0.02))       # no ticks above 1.0 (space is for labels)
    ax.set_xlabel("Accuracy")
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ms=4.5, mfc="white", mec="black", mew=0.8, ls="none",
                              label="Train (eval mode)"),
                       Line2D([], [], marker="o", ms=4.5, mfc="black", mec="black", ls="none", label="Test")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.3)
    S.panel_label(ax, "(b) Train vs. test accuracy", y=-0.33)
    S.save(fig, "fig1_performance")

# ---------------------------------------------------------------- Fig. 2
def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(S.COL2, 2.5), gridspec_kw=dict(wspace=0.24))
    logs = {"base": log_base, "z": log_z, "tune": log_tune}
    ax = axes[0]
    for k, L in logs.items():
        ax.plot(L["epoch"] + 1, L["loss"], color=S.C[k], lw=0.6, alpha=0.9, label=S.NAMES[k],
                ls="--" if k == "z" else "-", zorder=3 if k == "z" else 2)
        e = SEL[k]
        ax.plot(e + 1, L.loc[e, "loss"], marker="*", ms=8, mfc=S.C[k], mec="black", mew=0.5,
                ls="none", zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (log scale)")
    ax.set_xlim(0, 2000)
    ax.set_ylim(1e-10, 3)
    ax.yaxis.set_major_locator(LogLocator(numticks=6))
    ax.plot([], [], marker="*", ms=7, mfc="white", mec="black", ls="none", label="Selected checkpoint")
    ax.legend(loc="lower left")
    S.panel_label(ax, "(a) Training loss", y=-0.24)

    ax = axes[1]
    for k, L in logs.items():
        ax.plot(L["epoch"] + 1, L["accuracy"], color=S.C[k], lw=0.4, alpha=0.3)
        ax.plot(L["epoch"] + 1, L["accuracy"].rolling(25, min_periods=1).mean(), color=S.C[k],
                lw=0.9, label=S.NAMES[k], ls="--" if k == "z" else "-")
    # test accuracy exists only for the selected checkpoint -> one marker, not a line over all epochs
    for k in logs:
        ax.plot(SEL[k] + 1, TEST[k]["acc"], marker="D", ms=4.5, mfc=S.C[k], mec="black", mew=0.5,
                ls="none", zorder=6)
    ax.plot([], [], marker="D", ms=4, mfc="white", mec="black", ls="none", label="Test acc. (checkpoint)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training accuracy")
    ax.set_xlim(0, 2000)
    ax.set_ylim(0.85, 1.005)
    ax.legend(loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.6)
    S.panel_label(ax, "(b) Training accuracy (25-epoch moving average)", y=-0.24)
    S.save(fig, "fig2_training_curves")

# ---------------------------------------------------------------- Fig. 3
def fig_early_stop():
    L = log_es
    ep = L["epoch"] + 1
    # one model -> its own colour; training vs. validation by line style and marker
    c = S.C["es"]
    fig, axes = plt.subplots(2, 1, figsize=(S.COL1, 3.5), sharex=True, gridspec_kw=dict(hspace=0.55))
    ax = axes[0]
    ax.plot(ep, L["loss"], color=c, marker="o", ms=3, label="Training")
    ax.plot(ep, L["val_loss"], color=c, marker="s", ms=3, mfc="white", ls="--", label="Validation")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter
    ax.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 4]))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axvline(SEL["es"] + 1, color="black", lw=0.6, ls=":")
    ax.set_ylabel("Loss (log scale)")
    ax.legend(loc="upper right")
    ax.tick_params(labelbottom=True)
    ax.set_xlabel("Epoch")
    S.panel_label(ax, "(a) Loss", y=-0.34)

    ax = axes[1]
    ax.plot(ep, L["accuracy"], color=c, marker="o", ms=3, label="Training")
    ax.plot(ep, L["val_accuracy"], color=c, marker="s", ms=3, mfc="white", ls="--", label="Validation")
    ax.axhline(0.93, color="0.4", lw=0.6, ls="-.")
    ax.text(10.6, 0.955, "stop threshold (0.93)", fontsize=8, ha="right", va="bottom", color="0.25")
    ax.axvline(SEL["es"] + 1, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.45, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=11))
    for a in axes:
        a.tick_params(axis="x", which="minor", bottom=False, top=False)
    S.panel_label(ax, "(b) Accuracy", y=-0.34)
    S.save(fig, "fig3_early_stopping")

# ---------------------------------------------------------------- Fig. 4
def fig_optuna():
    import optuna
    from optuna.importance import get_param_importances
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.load_study(study_name="FordA_FCN_MacroF1",
                           storage=f"sqlite:///{D}fcn_optuna.db")
    tr = [t for t in st.trials if t.state.name == "COMPLETE"]
    num = np.array([t.number for t in tr])
    val = np.array([t.value for t in tr])
    best_so_far = np.maximum.accumulate(val)

    fig, axes = plt.subplots(1, 3, figsize=(S.COL2, 2.35),
                             gridspec_kw=dict(width_ratios=[1, 1, 0.95], wspace=0.62))
    ax = axes[0]
    ax.scatter(num, val, s=10, c=S.C["base"], edgecolor="black", linewidth=0.3, zorder=3, label="Trial")
    ax.step(num, best_so_far, where="post", color=S.C["tune"], lw=1.1, label="Best so far")
    b = st.best_trial
    ax.scatter([b.number], [b.value], marker="*", s=80, c=S.C["tune"], edgecolor="black",
               linewidth=0.4, zorder=4)
    ax.annotate(f"#{b.number}: {b.value:.3f}", (b.number, b.value), xytext=(b.number + 3, 0.82),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.5))
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(0.65, 1.0)
    ax.legend(loc="lower right")
    S.panel_label(ax, "(a) Optimization history", y=-0.26)

    ax = axes[1]
    for t in tr:
        iv = t.intermediate_values
        if not iv:
            continue
        e = np.array(sorted(iv)); v = np.array([iv[i] for i in e])
        ax.plot(e + 1, v, color="0.65", lw=0.4, alpha=0.6)
    iv = b.intermediate_values
    e = np.array(sorted(iv)); v = np.array([iv[i] for i in e])
    ax.plot(e + 1, v, color=S.C["tune"], lw=1.1, label=f"Best (#{b.number})")
    ax.plot([], [], color="0.65", lw=0.6, label="Others")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="lower right")
    S.panel_label(ax, "(b) Per-trial learning curves", y=-0.26)

    ax = axes[2]
    imp = get_param_importances(st, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42))
    pretty = {"filters1": "Filters L1", "filters2": "Filters L2", "filters3": "Filters L3",
              "kernel1": "Kernel L1", "kernel2": "Kernel L2", "kernel3": "Kernel L3",
              "learning_rate": "Learning rate", "batch_size": "Batch size",
              "weight_decay": "Weight decay", "optimizer": "Optimizer"}
    ks = list(imp)[::-1]
    vs = [imp[k] for k in ks]
    ax.barh(range(len(ks)), vs, color=S.C["base"], edgecolor="black", lw=0.4, height=0.65, zorder=3)
    for i, v in enumerate(vs):
        if v >= 0.04:
            ax.text(v + 0.006, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(range(len(ks)), [pretty[k] for k in ks])
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.set_xlabel("fANOVA importance")
    ax.set_xlim(0, max(vs) * 1.3)
    ax.grid(axis="y", visible=False)
    S.panel_label(ax, "(c) Hyperparameter importance", y=-0.26)
    S.save(fig, "fig4_optuna")
    return imp

# ---------------------------------------------------------------- Fig. 5
def fig_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(S.COL1, 1.95), gridspec_kw=dict(wspace=0.5))
    labs = ["$-1$", "$+1$"]
    for ax, k, tag in zip(axes, ["base", "es"], ["(a) " + S.NAMES["base"], "(b) " + S.NAMES["es"]]):
        p = PROB[k].argmax(1)
        cm = confusion_matrix(y, p)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1.15)  # neutral: blue is the baseline colour
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j] * 100:.1f}%)", ha="center", va="center",
                        fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], labs)
        ax.set_yticks([0, 1], labs)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.grid(False)
        ax.minorticks_off()
        ax.tick_params(top=False, right=False, length=0)
        S.panel_label(ax, tag, y=-0.38)
    S.save(fig, "fig5_confusion")

# ---------------------------------------------------------------- Fig. 6  CAM
def _norm(c):
    return (c - c.min()) / (c.max() - c.min() + 1e-12)


def pick_samples(n_per_class=2, seed=0):
    """Test samples correctly classified with high confidence by BOTH models."""
    pb, pe = PROB["base"], PROB["es"]
    ok = (pb.argmax(1) == y) & (pe.argmax(1) == y)
    conf = np.minimum(pb[np.arange(len(y)), y], pe[np.arange(len(y)), y])
    idx = []
    rng = np.random.default_rng(seed)
    for c in [0, 1]:
        cand = np.where(ok & (y == c) & (conf > 0.95))[0]
        idx += list(rng.choice(cand, n_per_class, replace=False))
    return idx


def cam_panel(ax, k, i, cmap="hot_r", s=9):
    c = int(y[i])
    m = _norm(CAM[k][i, c])
    t = np.arange(X.shape[1])
    ax.plot(t, X[i], color="0.75", lw=0.4, zorder=1)
    sc = ax.scatter(t, X[i], c=m, cmap=cmap, vmin=0, vmax=1, s=s, edgecolor="black",
                    linewidth=0.2, zorder=2)
    lab = "$-1$" if c == 0 else "$+1$"
    ax.set_title(f"Label {lab}: {PROB[k][i, c]:.3f}", pad=2)
    ax.set_xlim(-5, X.shape[1] + 4)
    ax.grid(False)
    return sc


def fig_cam():
    idx = pick_samples(1, seed=3)
    fig, axes = plt.subplots(2, 2, figsize=(S.COL2, 3.0), sharex=True, sharey="col",
                             gridspec_kw=dict(hspace=0.35, wspace=0.1))
    for r, k in enumerate(["base", "es"]):
        for col, i in enumerate(idx):
            sc = cam_panel(axes[r, col], k, i)
            if col == 0:
                axes[r, col].set_ylabel(S.NAMES[k] + "\nAmplitude", fontsize=8)
            else:
                axes[r, col].tick_params(labelleft=False)
            if r == 1:
                axes[r, col].set_xlabel("Time step")
    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.015, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"])
    cb.outline.set_linewidth(0.5)
    S.save(fig, "fig6_cam")
    return idx


def fig_cam_errors():
    pb = PROB["base"]
    wrong = np.where(pb.argmax(1) != y)[0]
    idx = []
    for c in [0, 1]:
        w = wrong[y[wrong] == c]
        idx.append(w[np.argmax(pb[w, 1 - c])])
    fig, axes = plt.subplots(2, 1, figsize=(S.COL1, 2.9), sharex=True, gridspec_kw=dict(hspace=0.32))
    for ax, i in zip(axes, idx):
        c = int(y[i]); pc = 1 - c
        m = _norm(CAM["base"][i, pc])
        t = np.arange(X.shape[1])
        ax.plot(t, X[i], color="0.75", lw=0.4, zorder=1)
        sc = ax.scatter(t, X[i], c=m, cmap="hot_r", vmin=0, vmax=1, s=6, edgecolor="black",
                        linewidth=0.15, zorder=2)
        tl = "$-1$" if c == 0 else "$+1$"; pl = "$-1$" if pc == 0 else "$+1$"
        ax.set_title(f"True {tl}, predicted {pl} (prob. {pb[i, pc]:.3f})", pad=2)
        ax.set_ylabel("Amplitude")
        ax.set_xlim(-5, X.shape[1] + 4)
        ax.grid(False)
    axes[1].set_xlabel("Time step")
    cb = fig.colorbar(sc, ax=axes, fraction=0.04, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"])
    cb.outline.set_linewidth(0.5)
    S.save(fig, "fig6b_cam_errors")

def fig_cam_quant():
    """(a) CAM concentration over all test series; (b) deletion test."""
    fig, axes = plt.subplots(2, 1, figsize=(S.COL1, 3.6), gridspec_kw=dict(hspace=0.62))
    models = ["base", "es"]

    ax = axes[0]
    conc = {k: cam_concentration(k) for k in models}
    w = 0.32
    for j, k in enumerate(models):
        data = [conc[k][y == c] for c in [0, 1]]
        pos = np.array([0, 1]) + (j - 0.5) * w * 1.15
        bp = ax.boxplot(data, positions=pos, widths=w, patch_artist=True, showfliers=True,
                        medianprops=dict(color="black", lw=0.9),
                        whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                        flierprops=dict(marker=".", ms=1.5, mfc="0.5", mec="none", alpha=0.6))
        for b in bp["boxes"]:
            b.set(facecolor=S.C[k], edgecolor="black", lw=0.5, hatch=S.HATCH[k])
        ax.plot([], [], marker="s", ms=5, mfc=S.C[k], mec="black", mew=0.5, ls="none", label=S.NAMES[k])
    ax.axhline(0.10, color="0.4", lw=0.6, ls=":")
    ax.text(1.55, 0.11, "uniform", fontsize=8, ha="right", va="bottom", color="0.3")
    ax.set_xticks([0, 1], ["Class $-1$", "Class $+1$"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Evidence in top 10% of steps")
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.2,
              handletextpad=0.2, frameon=False)      # also serves panel (b)
    S.panel_label(ax, "(a) Concentration of the class activation map", y=-0.2)

    ax = axes[1]
    fr = R["del_frac"] * 100
    for k, key in [("base", "M1"), ("es", "M4")]:
        top, rnd = R[f"del_{key}_top"], R[f"del_{key}_rnd"]
        ax.plot(fr, top, color=S.C[k], marker=S.MARK[k], ms=3, mec="black", mew=0.4, lw=1.0, zorder=3)
        ax.plot(fr, np.nanmean(rnd, 1), color=S.C[k], lw=0.9, ls="--", zorder=2)
        ax.fill_between(fr, np.nanmin(rnd, 1), np.nanmax(rnd, 1), color=S.C[k], alpha=0.18, lw=0)
    ax.axhline(0.5, color="0.4", lw=0.6, ls=":")
    ax.text(1, 0.505, "chance", fontsize=8, ha="left", va="bottom", color="0.3")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="black", lw=1.0, marker="o", ms=3, label="Highest-CAM steps removed"),
         Line2D([], [], color="black", lw=0.9, ls="--", label="Same mask, random shift")]
    ax.legend(handles=h, loc="upper right", fontsize=8, ncol=1, handlelength=1.8)
    ax.set_xlabel("Time steps set to zero (%)")
    ax.set_ylabel("Test accuracy")
    ax.set_xlim(0, 50)
    ax.set_ylim(0.38, 1.0)
    S.panel_label(ax, "(b) Deletion test", y=-0.33)
    S.save(fig, "fig7_cam_quant")


# ---------------------------------------------------------------- extra candidates
def fig_roc_calib():
    fig, axes = plt.subplots(1, 2, figsize=(S.COL1, 1.75), gridspec_kw=dict(wspace=0.45))
    ax = axes[0]
    for k in ["base", "z", "es"]:
        f, t, _ = roc_curve(y, PROB[k][:, 1])
        ax.plot(f, t, color=S.C[k], lw=0.9, label=f"{ {'base':'Baseline','z':'+z-score','es':'Opt+ES'}[k]} ({auc(f, t):.3f})")
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.5, ls=":")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(fontsize=8, loc="lower right", title="AUC", title_fontsize=8)
    S.panel_label(ax, "(a) ROC", y=-0.36)

    ax = axes[1]
    bins = np.linspace(0, 1, 11)
    for k in ["base", "z", "es"]:
        p = PROB[k][:, 1]
        idx = np.digitize(p, bins[1:-1])
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 5:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        ax.plot(xs, ys, color=S.C[k], marker=S.MARK[k], ms=2.5, lw=0.8)
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.5, ls=":")
    ax.set_xlabel("Predicted $P(+1)$"); ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    S.panel_label(ax, "(b) Reliability", y=-0.36)
    S.save(fig, "figX_roc_calibration")


def fig_tsne():
    from sklearn.manifold import TSNE
    fig, axes = plt.subplots(1, 2, figsize=(S.COL1, 1.95), gridspec_kw=dict(wspace=0.08))
    for ax, k, tag in zip(axes, ["base", "es"], ["(a) " + S.NAMES["base"], "(b) " + S.NAMES["es"]]):
        Z = TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(GAP[k])
        for c, col, mk in [(0, S.CLS[0], S.CLS_MARK[0]), (1, S.CLS[1], S.CLS_MARK[1])]:
            m = y == c
            ax.scatter(Z[m, 0], Z[m, 1], s=2, c=col, marker=mk, linewidths=0, alpha=0.7,
                       label=f"Class {'$-1$' if c == 0 else '$+1$'}")
        wrong = PROB[k].argmax(1) != y
        ax.scatter(Z[wrong, 0], Z[wrong, 1], s=7, facecolor="none", edgecolor="black",
                   linewidth=0.4, label="Misclassified")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.minorticks_off()
        S.panel_label(ax, tag, y=-0.04)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.04, 1.0), ncol=3, markerscale=2.5,
                   frameon=False, columnspacing=0.8, handletextpad=0.2)
    S.save(fig, "figX_tsne")

def fig_dataset():
    tr = np.load(D + "FordA_processed/FordA_TRAIN.npy")
    Xtr = tr[:, 1:]; ytr = (tr[:, 0] == 1).astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(S.COL1, 3.3), gridspec_kw=dict(hspace=0.62))
    ax = axes[0]
    for c, col, ls in [(0, S.CLS[0], S.CLS_LS[0]), (1, S.CLS[1], S.CLS_LS[1])]:
        i = np.where(ytr == c)[0][0]
        ax.plot(Xtr[i], color=col, lw=0.7, ls=ls, label=f"Class {'$-1$' if c == 0 else '$+1$'}")
    ax.set_xlabel("Time step"); ax.set_ylabel("Amplitude")
    ax.set_xlim(0, 500)
    ax.set_ylim(-3, 3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    S.panel_label(ax, "(a) Example series", y=-0.36)
    ax = axes[1]
    F = np.abs(np.fft.rfft(Xtr, axis=1))
    fr = np.fft.rfftfreq(Xtr.shape[1])
    for c, col, ls in [(0, S.CLS[0], S.CLS_LS[0]), (1, S.CLS[1], S.CLS_LS[1])]:
        mu = F[ytr == c].mean(0); sd = F[ytr == c].std(0)
        ax.plot(fr, mu, color=col, lw=0.9, ls=ls)
        ax.fill_between(fr, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
    ax.set_xlabel("Normalized frequency (cycles per sample)"); ax.set_ylabel("Spectral amplitude")
    ax.set_xlim(0, 0.2)
    S.panel_label(ax, "(b) Class-wise amplitude spectrum (mean and one std)", y=-0.36)
    S.save(fig, "fig0_dataset")

def table_tex():
    rows = []
    # NOTE: the FCN-Opt label is still under discussion (why epoch 408 was chosen); change it here.
    sel = {"base": "Train loss", "z": "Train loss", "tune": "Train loss", "es": "Val. loss"}
    ep = {k: SEL[k] + 1 for k in KEYS}
    aucs = {k: (f"{auc(*roc_curve(y, PROB[k][:, 1])[:2]):.4f}" if k in PROB else "--") for k in KEYS}
    for k in KEYS:
        T = TEST[k]
        tra = f"{TRAIN_ACC[k]:.4f}" + ("$^{\\mathrm{d}}$" if k in TRAIN_FROM_LOG else "")
        ci = f"[{CI[k][0]:.3f}, {CI[k][1]:.3f}]"
        rows.append(f"{S.NAMES[k]} & {sel[k]} ({ep[k]}) & {tra} & {T['acc']:.4f} & {ci} & {T['prec']:.4f} & "
                    f"{T['rec']:.4f} & {T['f1']:.4f} & {T['loss']:.4f} & {aucs[k]} \\\\\n\\hline")
    pz, pe = MCN["z"][2], MCN["es"][2]
    pt = two_prop_p("base", "tune")
    pt_txt = "$p<0.001$" if pt < 1e-3 else f"$p={pt:.3f}$"
    tex = r"""\begin{table*}[htbp]
\caption{Performance of the FCN Variants on the FordA Test Set}
\begin{center}
\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Checkpoint (epoch)} & \textbf{Train Acc.}$^{\mathrm{c}}$ & \textbf{Test Acc.} & \textbf{95\% CI}$^{\mathrm{e}}$ & \textbf{Precision}$^{\mathrm{a}}$ & \textbf{Recall}$^{\mathrm{a}}$ & \textbf{F1}$^{\mathrm{a}}$ & \textbf{Test Loss} & \textbf{AUC}$^{\mathrm{b}}$ \\
\hline
""" + "\n".join(rows) + r"""
\multicolumn{10}{l}{$^{\mathrm{a}}$Macro-averaged over the two classes. Test set: 1320 series.} \\
\multicolumn{10}{l}{$^{\mathrm{b}}$Not available for FCN-Opt because its checkpoint is not in the repository.} \\
\multicolumn{10}{l}{$^{\mathrm{c}}$Final weights evaluated in inference mode on the model's own training data (3601 series; FCN-Opt + ES: 3240 after the validation split).} \\
\multicolumn{10}{l}{$^{\mathrm{d}}$Running accuracy from the training log (weights unavailable for re-evaluation).} \\
\multicolumn{10}{l}{$^{\mathrm{e}}$Wilson interval for test accuracy. Versus the baseline: FCN + z-score $p=""" + f"{pz:.2f}" + r"""$, FCN-Opt + ES $p=""" + f"{pe:.2f}" + r"""$ (exact McNemar test);} \\
\multicolumn{10}{l}{\phantom{$^{\mathrm{e}}$}FCN-Opt """ + pt_txt + r""" (two-proportion $z$-test, as its predictions are unavailable).}
\end{tabular}
\label{tab:results}
\end{center}
\end{table*}
"""
    open("figures/table_results.tex", "w").write(tex)

if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["all"]
    fns = dict(perf=fig_performance, train=fig_training, es=fig_early_stop, optuna=fig_optuna,
               cm=fig_confusion, cam=fig_cam, camerr=fig_cam_errors, camq=fig_cam_quant,
               roc=fig_roc_calib, tsne=fig_tsne, data=fig_dataset, table=table_tex, stats=print_stats)
    if "all" in which:
        fns.pop("roc")      # candidate figure, not part of the report set; run explicitly if needed
    for n, f in fns.items():
        if "all" in which or n in which:
            r = f()
            print(n, "done", r if r is not None else "")
