# Statistical comparison: `CDDFuse-FuseRule-ChannelMoE` vs `CDDFuse` baseline

**Run ID**: 20260504_010829
**Alpha**: 0.05, correction: Holm-Bonferroni (per modal)
**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)
**Output**: `results_v2\_stats\20260504_010829_FuseRule-ChannelMoE_vs_CDDFuse\significance_FuseRule-ChannelMoE_vs_CDDFuse.csv`

## Bottom line

- Verdict: **MIXED**
- Pooled (ALL): 4/25 metrics SIG, 1 marginal, 20 NS

## Top 5 metrics with largest improvement (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| NABF | -0.0096 | +0.504 | large | 0.0000 | 0.0004 | **SIG** |
| QSF | +0.0900 | +0.488 | large | 0.0000 | 0.0000 | **SIG** |
| PSNR | +0.2349 | +0.256 | small | 0.0000 | 0.0000 | **SIG** |
| RMSE | -0.0101 | +0.256 | small | 0.0000 | 0.0000 | **SIG** |
| EN | +0.0270 | +0.016 | trivial | 0.0082 | 0.1713 | **MARGINAL** |

## Top 5 metrics with largest regression (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QM | -0.0758 | -0.745 | large | 1.0000 | 1.0000 | NS |
| VAR | -10.2133 | -0.471 | medium | 1.0000 | 1.0000 | NS |
| MI | -9.5962 | -0.332 | medium | 1.0000 | 1.0000 | NS |
| QY | -0.0164 | -0.332 | medium | 1.0000 | 1.0000 | NS |
| QCV | +40.9600 | -0.294 | small | 1.0000 | 1.0000 | NS |

## Per-modality summary

| modal | n_SIG | n_MARG | n_NS | total |
|---|---|---|---|---|
| CT | 7 | 0 | 18 | 25 |
| PET | 5 | 1 | 19 | 25 |
| SPECT | 2 | 0 | 23 | 25 |
| ALL | 4 | 1 | 20 | 25 |

## Caveats
- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.
- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.
- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.
