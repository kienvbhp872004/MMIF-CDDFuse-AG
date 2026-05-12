# Statistical comparison: `CDDFuse-Paper-MIF` vs `CDDFuse` baseline

**Run ID**: 20260512_142953
**Alpha**: 0.05, correction: Holm-Bonferroni (per modal)
**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)
**Output**: `results_v2\_stats\20260512_142953_Paper-MIF_vs_CDDFuse\significance_Paper-MIF_vs_CDDFuse.csv`

## Bottom line

- Verdict: **MIXED**
- Pooled (ALL): 3/25 metrics SIG, 2 marginal, 20 NS

## Top 5 metrics with largest improvement (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QSF | +0.0115 | +0.350 | medium | 0.0000 | 0.0004 | **SIG** |
| NABF | -0.0012 | +0.122 | trivial | 0.1672 | 1.0000 | **NS** |
| QG | +0.0085 | +0.080 | trivial | 0.0013 | 0.0297 | **SIG** |
| SF | +0.8563 | +0.070 | trivial | 0.1054 | 1.0000 | **NS** |
| QY | +0.0033 | +0.069 | trivial | 0.1937 | 1.0000 | **NS** |

## Top 5 metrics with largest regression (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| VAR | -9.5836 | -0.437 | medium | 1.0000 | 1.0000 | NS |
| MI | -10.4272 | -0.360 | medium | 1.0000 | 1.0000 | NS |
| SSIM | -0.0486 | -0.280 | small | 1.0000 | 1.0000 | NS |
| QCV | +71.7987 | -0.240 | small | 1.0000 | 1.0000 | NS |
| QM | -0.0147 | -0.137 | trivial | 0.9406 | 1.0000 | NS |

## Per-modality summary

| modal | n_SIG | n_MARG | n_NS | total |
|---|---|---|---|---|
| CT | 7 | 2 | 16 | 25 |
| PET | 10 | 2 | 13 | 25 |
| SPECT | 2 | 1 | 22 | 25 |
| ALL | 3 | 2 | 20 | 25 |

## Caveats
- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.
- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.
- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.
