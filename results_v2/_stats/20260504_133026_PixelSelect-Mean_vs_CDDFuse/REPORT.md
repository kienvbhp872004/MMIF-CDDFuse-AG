# Statistical comparison: `CDDFuse-PixelSelect-Mean` vs `CDDFuse` baseline

**Run ID**: 20260504_133026
**Alpha**: 0.05, correction: Holm-Bonferroni (per modal)
**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)
**Output**: `results_v2\_stats\20260504_133026_PixelSelect-Mean_vs_CDDFuse\significance_PixelSelect-Mean_vs_CDDFuse.csv`

## Bottom line

- Verdict: **MIXED**
- Pooled (ALL): 4/25 metrics SIG, 1 marginal, 20 NS

## Top 5 metrics with largest improvement (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| NABF | -0.0112 | +0.669 | large | 0.0000 | 0.0000 | **SIG** |
| QSF | +0.0840 | +0.591 | large | 0.0000 | 0.0000 | **SIG** |
| PSNR | +0.1909 | +0.251 | small | 0.0000 | 0.0000 | **SIG** |
| RMSE | -0.0091 | +0.251 | small | 0.0000 | 0.0000 | **SIG** |
| QCB | +0.0044 | +0.030 | trivial | 0.0058 | 0.1214 | **MARGINAL** |

## Top 5 metrics with largest regression (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QM | -0.0754 | -0.735 | large | 1.0000 | 1.0000 | NS |
| VAR | -12.4425 | -0.522 | large | 1.0000 | 1.0000 | NS |
| MI | -12.3342 | -0.426 | medium | 1.0000 | 1.0000 | NS |
| QY | -0.0166 | -0.319 | small | 1.0000 | 1.0000 | NS |
| QCV | +23.1380 | -0.287 | small | 1.0000 | 1.0000 | NS |

## Per-modality summary

| modal | n_SIG | n_MARG | n_NS | total |
|---|---|---|---|---|
| CT | 6 | 0 | 19 | 25 |
| PET | 6 | 0 | 19 | 25 |
| SPECT | 2 | 0 | 23 | 25 |
| ALL | 4 | 1 | 20 | 25 |

## Caveats
- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.
- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.
- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.
