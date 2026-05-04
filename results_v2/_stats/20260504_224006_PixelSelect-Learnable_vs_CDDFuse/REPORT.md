# Statistical comparison: `CDDFuse-PixelSelect-Learnable` vs `CDDFuse` baseline

**Run ID**: 20260504_224006
**Alpha**: 0.05, correction: Holm-Bonferroni (per modal)
**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)
**Output**: `results_v2\_stats\20260504_224006_PixelSelect-Learnable_vs_CDDFuse\significance_PixelSelect-Learnable_vs_CDDFuse.csv`

## Bottom line

- Verdict: **CONFIRM_IMPROVEMENT**
- Pooled (ALL): 5/25 metrics SIG, 1 marginal, 19 NS

## Top 5 metrics with largest improvement (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| NABF | -0.0119 | +0.671 | large | 0.0000 | 0.0000 | **SIG** |
| QSF | +0.0850 | +0.543 | large | 0.0000 | 0.0000 | **SIG** |
| PSNR | +0.1618 | +0.228 | small | 0.0000 | 0.0001 | **SIG** |
| RMSE | -0.0079 | +0.228 | small | 0.0000 | 0.0000 | **SIG** |
| QMI | +0.0068 | +0.074 | trivial | 0.0001 | 0.0016 | **SIG** |

## Top 5 metrics with largest regression (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QM | -0.0753 | -0.731 | large | 1.0000 | 1.0000 | NS |
| VAR | -12.7829 | -0.529 | large | 1.0000 | 1.0000 | NS |
| MI | -12.6790 | -0.443 | medium | 1.0000 | 1.0000 | NS |
| QCV | +25.9163 | -0.291 | small | 1.0000 | 1.0000 | NS |
| QY | -0.0145 | -0.267 | small | 1.0000 | 1.0000 | NS |

## Per-modality summary

| modal | n_SIG | n_MARG | n_NS | total |
|---|---|---|---|---|
| CT | 9 | 0 | 16 | 25 |
| PET | 8 | 0 | 17 | 25 |
| SPECT | 2 | 0 | 23 | 25 |
| ALL | 5 | 1 | 19 | 25 |

## Caveats
- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.
- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.
- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.
