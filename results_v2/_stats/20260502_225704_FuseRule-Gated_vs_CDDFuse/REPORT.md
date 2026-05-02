# Statistical comparison: `CDDFuse-FuseRule-Gated` vs `CDDFuse` baseline

**Run ID**: 20260502_225704
**Alpha**: 0.05, correction: Holm-Bonferroni (per modal)
**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)
**Output**: `results_v2\_stats\20260502_225704_FuseRule-Gated_vs_CDDFuse\significance_FuseRule-Gated_vs_CDDFuse.csv`

## Bottom line

- Verdict: **CONFIRM_IMPROVEMENT**
- Pooled (ALL): 5/25 metrics SIG, 1 marginal, 19 NS

## Top 5 metrics with largest improvement (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QSF | +0.0690 | +0.602 | large | 0.0000 | 0.0000 | **SIG** |
| NABF | -0.0109 | +0.596 | large | 0.0000 | 0.0000 | **SIG** |
| PSNR | +0.2243 | +0.243 | small | 0.0000 | 0.0000 | **SIG** |
| RMSE | -0.0096 | +0.243 | small | 0.0000 | 0.0000 | **SIG** |
| EN | +0.0514 | +0.035 | trivial | 0.0001 | 0.0022 | **SIG** |

## Top 5 metrics with largest regression (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QM | -0.0720 | -0.602 | large | 1.0000 | 1.0000 | NS |
| VAR | -9.2572 | -0.437 | medium | 1.0000 | 1.0000 | NS |
| MI | -9.0999 | -0.303 | small | 1.0000 | 1.0000 | NS |
| QY | -0.0142 | -0.275 | small | 1.0000 | 1.0000 | NS |
| QMI | -0.0411 | -0.196 | small | 1.0000 | 1.0000 | NS |

## Per-modality summary

| modal | n_SIG | n_MARG | n_NS | total |
|---|---|---|---|---|
| CT | 4 | 3 | 18 | 25 |
| PET | 5 | 0 | 20 | 25 |
| SPECT | 2 | 1 | 22 | 25 |
| ALL | 5 | 1 | 19 | 25 |

## Caveats
- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.
- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.
- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.
