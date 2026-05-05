# Statistical comparison: `CDDFuse-Combined-Gated-Saliency` vs `CDDFuse` baseline

**Run ID**: 20260505_151513
**Alpha**: 0.05, correction: Holm-Bonferroni (per modal)
**Test**: Wilcoxon signed-rank, alternative='greater' (variant > baseline)
**Output**: `results_v2\_stats\20260505_151513_Combined-Gated-Saliency_vs_CDDFuse\significance_Combined-Gated-Saliency_vs_CDDFuse.csv`

## Bottom line

- Verdict: **MIXED**
- Pooled (ALL): 4/25 metrics SIG, 1 marginal, 20 NS

## Top 5 metrics with largest improvement (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QSF | +0.0679 | +0.679 | large | 0.0000 | 0.0000 | **SIG** |
| NABF | -0.0126 | +0.659 | large | 0.0000 | 0.0000 | **SIG** |
| PSNR | +0.1629 | +0.225 | small | 0.0002 | 0.0038 | **SIG** |
| RMSE | -0.0080 | +0.225 | small | 0.0000 | 0.0000 | **SIG** |
| EN | +0.0387 | +0.025 | trivial | 0.0088 | 0.1849 | **MARGINAL** |

## Top 5 metrics with largest regression (pooled)

| metric | mean Δ | Cliff's δ | effect | p_value | p_adj | verdict |
|---|---|---|---|---|---|---|
| QM | -0.0715 | -0.587 | large | 1.0000 | 1.0000 | NS |
| VAR | -11.5678 | -0.513 | large | 1.0000 | 1.0000 | NS |
| MI | -11.2653 | -0.384 | medium | 1.0000 | 1.0000 | NS |
| QY | -0.0182 | -0.326 | small | 1.0000 | 1.0000 | NS |
| QMI | -0.0596 | -0.271 | small | 1.0000 | 1.0000 | NS |

## Per-modality summary

| modal | n_SIG | n_MARG | n_NS | total |
|---|---|---|---|---|
| CT | 2 | 1 | 22 | 25 |
| PET | 5 | 0 | 20 | 25 |
| SPECT | 2 | 1 | 22 | 25 |
| ALL | 4 | 1 | 20 | 25 |

## Caveats
- Sample size per modality = 24 (small). Effect sizes ≥ medium needed for thesis claim.
- Holm correction applied **per modal** (K=22 each). For pooled (ALL), correction also K=22.
- Test one-sided (variant > baseline). Two-sided alternative would loosen verdict.
