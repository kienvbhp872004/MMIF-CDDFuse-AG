# Statistical comparison: `CDDFuse-Combined-CKA-Paper-MIF` vs `CDDFuse` baseline

**Run ID**: 20260515_171823
**Alpha**: 0.05, correction: Holm-Bonferroni (per modal)
**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)
**Output**: `results_v2\_stats\20260515_171823_Combined-CKA-Paper-MIF_vs_CDDFuse\significance_Combined-CKA-Paper-MIF_vs_CDDFuse.csv`

## Bottom line

- Verdict: **MIXED**
- Pooled (ALL): 1/25 metrics SIG, 2 marginal, 22 NS

## Top 5 metrics with largest improvement (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| NABF | -0.0034 | +0.243 | small | 0.0139 | 0.3339 | **MARGINAL** |
| QSF | +0.0039 | +0.123 | trivial | 0.1644 | 1.0000 | **NS** |
| FMI | +0.0098 | +0.109 | trivial | 0.0000 | 0.0000 | **SIG** |
| CE | -0.0566 | +0.064 | trivial | 0.0333 | 0.7648 | **MARGINAL** |
| QY | +0.0022 | +0.062 | trivial | 0.1700 | 1.0000 | **NS** |

## Top 5 metrics with largest regression (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| VAR | -11.1776 | -0.481 | large | 1.0000 | 1.0000 | NS |
| MI | -11.9314 | -0.422 | medium | 1.0000 | 1.0000 | NS |
| SSIM | -0.0514 | -0.293 | small | 1.0000 | 1.0000 | NS |
| QCV | +73.7245 | -0.246 | small | 1.0000 | 1.0000 | NS |
| QS | -0.0394 | -0.188 | small | 1.0000 | 1.0000 | NS |

## Per-modality summary

| modal | n_SIG | n_MARG | n_NS | total |
|---|---|---|---|---|
| CT | 9 | 2 | 14 | 25 |
| PET | 9 | 0 | 16 | 25 |
| SPECT | 2 | 1 | 22 | 25 |
| ALL | 1 | 2 | 22 | 25 |

## Caveats
- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.
- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.
- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.
