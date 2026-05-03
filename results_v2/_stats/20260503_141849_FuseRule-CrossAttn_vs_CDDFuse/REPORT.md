# Statistical comparison: `CDDFuse-FuseRule-CrossAttn` vs `CDDFuse` baseline

**Run ID**: 20260503_141849
**Alpha**: 0.05, correction: Holm-Bonferroni (per modal)
**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)
**Output**: `results_v2\_stats\20260503_141849_FuseRule-CrossAttn_vs_CDDFuse\significance_FuseRule-CrossAttn_vs_CDDFuse.csv`

## Bottom line

- Verdict: **CONFIRM_IMPROVEMENT**
- Pooled (ALL): 5/25 metrics SIG, 0 marginal, 20 NS

## Top 5 metrics with largest improvement (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QSF | +0.0679 | +0.542 | large | 0.0000 | 0.0000 | **SIG** |
| NABF | -0.0093 | +0.529 | large | 0.0000 | 0.0001 | **SIG** |
| PSNR | +0.2412 | +0.238 | small | 0.0000 | 0.0000 | **SIG** |
| RMSE | -0.0096 | +0.238 | small | 0.0000 | 0.0000 | **SIG** |
| EN | +0.0733 | +0.048 | trivial | 0.0000 | 0.0000 | **SIG** |

## Top 5 metrics with largest regression (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QM | -0.0734 | -0.637 | large | 1.0000 | 1.0000 | NS |
| VAR | -8.3160 | -0.421 | medium | 1.0000 | 1.0000 | NS |
| QY | -0.0173 | -0.304 | small | 1.0000 | 1.0000 | NS |
| MI | -7.4208 | -0.258 | small | 1.0000 | 1.0000 | NS |
| QCV | +39.5627 | -0.223 | small | 1.0000 | 1.0000 | NS |

## Per-modality summary

| modal | n_SIG | n_MARG | n_NS | total |
|---|---|---|---|---|
| CT | 4 | 3 | 18 | 25 |
| PET | 5 | 1 | 19 | 25 |
| SPECT | 3 | 2 | 20 | 25 |
| ALL | 5 | 0 | 20 | 25 |

## Caveats
- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.
- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.
- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.
