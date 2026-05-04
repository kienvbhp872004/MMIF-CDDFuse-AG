# Statistical comparison: `CDDFuse-PixelSelect-Saliency` vs `CDDFuse` baseline

**Run ID**: 20260504_182458
**Alpha**: 0.05, correction: Holm-Bonferroni (per modal)
**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)
**Output**: `results_v2\_stats\20260504_182458_PixelSelect-Saliency_vs_CDDFuse\significance_PixelSelect-Saliency_vs_CDDFuse.csv`

## Bottom line

- Verdict: **CONFIRM_IMPROVEMENT**
- Pooled (ALL): 6/25 metrics SIG, 0 marginal, 19 NS

## Top 5 metrics with largest improvement (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| NABF | -0.0114 | +0.671 | large | 0.0000 | 0.0000 | **SIG** |
| QSF | +0.0836 | +0.568 | large | 0.0000 | 0.0000 | **SIG** |
| PSNR | +0.1746 | +0.236 | small | 0.0000 | 0.0001 | **SIG** |
| RMSE | -0.0085 | +0.236 | small | 0.0000 | 0.0000 | **SIG** |
| QMI | +0.0045 | +0.047 | trivial | 0.0006 | 0.0134 | **SIG** |

## Top 5 metrics with largest regression (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QM | -0.0753 | -0.732 | large | 1.0000 | 1.0000 | NS |
| VAR | -12.5335 | -0.524 | large | 1.0000 | 1.0000 | NS |
| MI | -12.3830 | -0.429 | medium | 1.0000 | 1.0000 | NS |
| QY | -0.0159 | -0.302 | small | 1.0000 | 1.0000 | NS |
| QCV | +24.2539 | -0.291 | small | 1.0000 | 1.0000 | NS |

## Per-modality summary

| modal | n_SIG | n_MARG | n_NS | total |
|---|---|---|---|---|
| CT | 6 | 3 | 16 | 25 |
| PET | 7 | 1 | 17 | 25 |
| SPECT | 2 | 0 | 23 | 25 |
| ALL | 6 | 0 | 19 | 25 |

## Caveats
- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.
- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.
- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.
