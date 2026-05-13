# Statistical comparison: `CDDFuse-Combined-Learnable-Paper-MIF` vs `CDDFuse` baseline

**Run ID**: 20260513_081444
**Alpha**: 0.05, correction: Holm-Bonferroni (per modal)
**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)
**Output**: `results_v2\_stats\20260513_081444_Combined-Learnable-Paper-MIF_vs_CDDFuse\significance_Combined-Learnable-Paper-MIF_vs_CDDFuse.csv`

## Bottom line

- Verdict: **MIXED**
- Pooled (ALL): 1/25 metrics SIG, 3 marginal, 21 NS

## Top 5 metrics with largest improvement (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| NABF | -0.0032 | +0.231 | small | 0.0248 | 0.5695 | **MARGINAL** |
| QSF | +0.0064 | +0.175 | small | 0.0381 | 0.8380 | **MARGINAL** |
| FMI | +0.0059 | +0.067 | trivial | 0.0000 | 0.0002 | **SIG** |
| QG | +0.0051 | +0.057 | trivial | 0.0165 | 0.3956 | **MARGINAL** |
| QMI | -0.0237 | +0.039 | trivial | 0.8883 | 1.0000 | **NS** |

## Top 5 metrics with largest regression (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| VAR | -11.2397 | -0.486 | large | 1.0000 | 1.0000 | NS |
| MI | -12.0510 | -0.424 | medium | 1.0000 | 1.0000 | NS |
| SSIM | -0.0491 | -0.279 | small | 1.0000 | 1.0000 | NS |
| QCV | +62.2668 | -0.239 | small | 1.0000 | 1.0000 | NS |
| QM | -0.0189 | -0.161 | small | 0.9595 | 1.0000 | NS |

## Per-modality summary

| modal | n_SIG | n_MARG | n_NS | total |
|---|---|---|---|---|
| CT | 7 | 2 | 16 | 25 |
| PET | 10 | 0 | 15 | 25 |
| SPECT | 1 | 2 | 22 | 25 |
| ALL | 1 | 3 | 21 | 25 |

## Caveats
- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.
- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.
- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.
