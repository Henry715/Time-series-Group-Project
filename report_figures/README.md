# Report figures (FordA, FCN)

All figures follow the IEEEtran conference layout. Each PDF is cropped to exactly the template's `\columnwidth` (252 pt) or `\textwidth` (516 pt), so LaTeX inserts it at scale 1.000 and every label stays at exactly 8 pt. This was checked by compiling with the supplied `IEEEtran.cls` (see `layout_check/check.pdf`). All text in the figures is 8 pt Times-style serif, as the template requires for figure labels, colours are colour-blind safe, and the PDFs are vector with embedded fonts. Insert them at exactly these widths (any other width rescales the 8 pt text):

```latex
% single column
\begin{figure}[t]\centering
  \includegraphics[width=\columnwidth]{figures/fig3_early_stopping}
  \caption{...}\label{fig:es}
\end{figure}
% double column
\begin{figure*}[t]\centering
  \includegraphics[width=\textwidth]{figures/fig1_performance}
  \caption{...}\label{fig:perf}
\end{figure*}
```

The table is in `figures/table_results.tex`. It uses the same `\hline` style as the template's sample table and needs no extra package; include it with `\input{figures/table_results}`.

All figure text uses American spelling, and the axis labels use words with units in parentheses, following `conference_101719.tex`.

## Figure list

| File | Width | Content | Suggested section |
|---|---|---|---|
| `fig0_dataset` | single column, panels stacked | (a) one sample from each class; (b) mean ± std amplitude spectrum per class | Dataset |
| `table_results.tex` | double column | Full metrics for the four models: eval-mode train accuracy, test accuracy with 95% Wilson CI, macro P/R/F1, loss, AUC; footnotes give the significance tests | Results |
| `fig1_performance` | double column | (a) test accuracy with 95% CI, shaded band = baseline CI, p-value vs. baseline (McNemar; \* = two-proportion z-test for FCN-Opt); (b) train (open, eval mode) → test (filled) accuracy with the gap († = training-log value) | Results |
| `fig2_training_curves` | double column | Training loss (log scale) and accuracy over 2000 epochs for the first three models; ★ = selected checkpoint, ◆ = test accuracy of that checkpoint | Training analysis |
| `fig3_early_stopping` | single column, panels stacked, shared epoch axis | Train/val loss and accuracy of FCN-Opt + ES during its 11 epochs | Training analysis |
| `fig4_optuna` | double column | (a) optimisation history; (b) validation F1 curves of every trial; (c) fANOVA hyper-parameter importance | Hyper-parameter tuning |
| `fig5_confusion` | single column | Confusion matrices: baseline vs. FCN-Opt + ES | Results |
| `fig6_cam` | double column | CAM in the style of the original paper's Fig. 3 (row 2); rows = models, columns = one sample per class | Interpretability |
| `fig7_cam_quant` | single column, panels stacked | (a) CAM concentration (share of positive evidence in the top 10% of time steps) over all 1320 test series, per class and model; (b) deletion test: accuracy when the highest-CAM steps are zeroed vs. the same mask shifted to a random position | Interpretability |
| `fig6b_cam_errors` | single column, panels stacked | Baseline CAM on its most confidently misclassified sample of each class (map of the wrongly predicted class) | Interpretability / error analysis (optional) |
| `figX_tsne` | single column | t-SNE of GAP features; black circles = misclassified | Feature space (optional) |

## Key observations

- **Data are already z-normalised**: the global mean/std of the FordA training set are about 0 and 1, so the extra z-score in "FCN + z-score" has almost no effect (fig2 curves overlap; test accuracy 0.9273 vs 0.9205).
- **The three models with weights are statistically indistinguishable.** 95% Wilson CIs: baseline [0.912, 0.940], + z-score [0.905, 0.934], Opt + ES [0.907, 0.936]. Exact McNemar vs. baseline: + z-score p = 0.19 (23 vs. 14 discordant series), Opt + ES p = 0.51 (32 vs. 26). Do not call the baseline "best"; say they are comparable.
- **FCN-Opt is significantly worse** (0.891, CI [0.873, 0.907]; two-proportion z-test vs. baseline p = 0.001; its predictions are not available, so McNemar cannot be used). Its training loss falls to 1e-9 and training accuracy reaches 100%, with periodic loss spikes (around epochs 820, 1310 and 1810), probably because the learning rate 0.004 is relatively high and the scheduler never runs (`scheduler.step()` is not called).
- **Train–test gap (eval-mode train accuracy):** baseline 0.034, + z-score 0.034, Opt + ES 0.008; FCN-Opt 0.109 (training-log value). The earlier values (0.036 / 0.042 / 0.002) came from the running accuracy in the training log, which is measured in train mode while the weights change. Opt + ES's eval-mode training accuracy (0.931) is computed on its 3240 training series; its validation accuracy (0.9335) reproduces the notebook exactly, which confirms the split.
- **Error pattern**: both models mostly misclassify class −1 as +1 (63 and 75 cases). FCN-Opt + ES has higher recall on +1 (95.8%) and lower recall on −1.
- **Optuna**: batch size, layer-1 kernel size and layer-2 filter count matter most (fANOVA). The best validation F1 is 0.958, which does not transfer to the test set (0.923); the validation set is small (361 series).
- **CAM is faithful**: zeroing the 10% of time steps with the highest CAM drops accuracy to 0.575 (baseline) / 0.705 (Opt + ES), vs. 0.730 / 0.863 when the same mask is shifted to a random position (fig7 b).
- **CAM: class +1 is more concentrated than class −1** for both models (median top-10% share 0.51 vs. 0.39 baseline, 0.39 vs. 0.31 Opt + ES). This agrees with the qualitative reading of fig6.
- **Correction: FCN-Opt + ES is *not* more focused than the baseline.** On positive evidence the baseline is more concentrated in both classes (Mann–Whitney p < 1e-10); a min–max-normalised version of the metric gives a small difference the other way. The earlier claim, based on one sample per class, should be removed from the report.
- **Robustness side note**: the baseline is also more sensitive to zeroing time steps at random (5% shifted mask: 0.82 vs. 0.90 accuracy).

## Reproduction

```bash
# from the repository root, needs torch, scikit-learn, optuna, matplotlib
python report_figures/infer.py         # reloads models 1, 2, 4: test + eval-mode train inference, CAM deletion test (~6 min on CPU)
python report_figures/infer.py --no-del   # same without the deletion test (~1 min); fig7 then cannot be drawn
python report_figures/make_figures.py  # generates every figure and table into report_figures/figures/
python report_figures/make_figures.py cam   # regenerate a single figure only
python report_figures/make_figures.py stats # print CIs, McNemar, CAM statistics used in the text
```

Notes:
- The weights of FCN-Opt (no ES) (`trained_model/fcn_tuned_model_min_loss_epoch407_loss 0.00409.pth`) are not in the repository, so its test metrics are taken from the output of `FCN_training.ipynb`. The other three models were re-run with `infer.py` and match the notebook results exactly.
- Epochs in the table are 1-based. In the checkpoint file names they are 0-based (e.g. table 1906 = file `epoch1905`).
- The CAM samples are correctly classified by both models with confidence > 0.95 and are drawn at random with a fixed seed. Change `seed` in `pick_samples()` to choose other samples.
