---
name: fusion-stats
description: Phân tích thống kê có ý nghĩa cho so sánh CDDFuse variants vs baseline / vs SOTA — Wilcoxon paired test, Friedman + Nemenyi post-hoc, hiệu ứng (effect size), Bonferroni/Holm correction, CD diagram. Dùng per-image CSV (KHÔNG dùng aggregate). Trigger khi user nói "phân tích thống kê", "có ý nghĩa không", "so sánh variant với baseline", "p-value", "ranking".
---
# fusion-stats

Mục đích: KHẲNG ĐỊNH (không chỉ hiển thị) rằng cải tiến của variant **thật sự khác baseline**, không phải nhiễu. Đây là điều kiện cần để 1 con số trong bảng kết quả luận văn được chấp nhận như một claim khoa học.

## Khi nào kích hoạt

Trigger:

- "phân tích thống kê" / "có ý nghĩa không" / "p-value" / "Wilcoxon" / "significance" / "ranking" / "CD diagram".
- Sau mỗi lần chạy `fusion-bench-variant`, user hỏi "kết quả có thật sự tốt hơn không".
- Trước khi user đưa số vào ablation table luận văn.

KHÔNG kích hoạt khi user chỉ hỏi raw metric value (đó là `cat csv`).

## Yêu cầu đầu vào (BẮT BUỘC)

Skill này CHỈ chạy trên **per-image CSV** (mỗi dòng = 1 ảnh × 1 metric). Bao gồm:

- `results_v2/CDDFuse/perimage/CDDFuse_<modal>_perimage.csv` (baseline)
- `results_v2/CDDFuse-<Tag>/perimage/CDDFuse-<Tag>_<modal>_perimage.csv` (variant)
- (Optional) per-image của các SOTA model khác cho comparison.

**Nếu file per-image baseline KHÔNG tồn tại** (results hiện tại chỉ có aggregated summary):

- BLOCK ngay. Không substitute bằng aggregated — Wilcoxon trên 3 dòng (CT/PET/SPECT) là vô nghĩa.
- Yêu cầu user re-run baseline với `evaluate_cddfuse.py --save_perimage` để sinh file. Đây là one-time cost, sau đó dùng vĩnh viễn.

## Quy ước thống kê

### Mức ý nghĩa

- α = 0.05 (default).
- Nếu test ≥ 5 metric đồng thời → BẮT BUỘC dùng correction (Bonferroni hoặc Holm-Bonferroni). Holm khuyến nghị (less conservative, vẫn FWER-controlled).

### Test pháp pháp lệnh

| Tình huống                              | Test                        | Lý do                                    |
| ----------------------------------------- | --------------------------- | ----------------------------------------- |
| 2 model,**paired** (same image set) | Wilcoxon signed-rank        | Không assume normality, paired structure |
| 2 model,**unpaired**                | Mann-Whitney U              | Tránh paired test khi không paired      |
| ≥ 3 model, paired                        | Friedman + Nemenyi post-hoc | Standard cho ranking comparison           |
| 1 metric, n nhỏ (< 15)                   | Bootstrap CI thay thế      | Paired test mất power                    |

CDDFuse use case: paired test (cùng 24 ảnh / modal × 3 modal = 72 paired pairs).

### Effect size (BẮT BUỘC, không chỉ p-value)

p-value cho biết có khác không, KHÔNG cho biết khác bao nhiêu. LUÔN report effect size kèm p-value:

- **Cliff's delta** (cho non-parametric): |δ| < 0.147 trivial, < 0.33 small, < 0.474 medium, ≥ 0.474 large.
- **Rank-biserial correlation** (Wilcoxon): tương tự.

Verdict template: "p = 0.003, Cliff's δ = 0.42 (medium effect)" — KHÔNG chỉ "p = 0.003".

### Multiple comparison correction

Khi test K metric paired (ví dụ K=22):

- p_adj_holm[i] = max(p_sorted[i] × (K - i + 1), p_adj_holm[i-1])
- Verdict significant chỉ khi p_adj < α.

→ Trong báo cáo, luôn ghi cả `p` và `p_adj`. Reviewer sẽ check.

## Quy trình chạy

### Bước 1 — Locate per-image data

Find:

```python
baseline_files = glob("results_v2/CDDFuse/perimage/CDDFuse_*_perimage.csv")
variant_files  = glob(f"results_v2/CDDFuse-{tag}/perimage/CDDFuse-{tag}_*_perimage.csv")
```

Verify cùng số ảnh per modal, cùng image names. Mismatched pairs → flag và remove.

### Bước 2 — Pairing

Inner-join trên `image` column per modality:

```python
df_pair = pd.merge(df_baseline, df_variant, on='image', suffixes=('_base', '_var'))
```

Yêu cầu:

- N_paired ≥ 20 per modality (preferable 24 = full Harvard split).
- Nếu n < 15 → warning, kết quả thống kê unreliable.

### Bước 3 — Run tests per metric

Cho mỗi metric trong 22 metrics, **per modality** rồi **pooled across modalities**:

```python
from scipy.stats import wilcoxon
for metric in METRICS_22:
    diff = df_pair[f'{metric}_var'] - df_pair[f'{metric}_base']
    if metric in LOWER_BETTER:  # NABF, CE, RMSE, QCV
        diff = -diff
    stat, pval = wilcoxon(diff, alternative='greater')  # variant > baseline?
    cliffs = cliffs_delta(df_pair[f'{metric}_var'], df_pair[f'{metric}_base'])
    ...
```

### Bước 4 — Apply Holm correction

```python
from statsmodels.stats.multitest import multipletests
reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='holm')
```

### Bước 5 — Build significance table

Output `results_v2/_stats/<run_id>/significance_<variant>_vs_baseline.csv`:

```
metric, modal, n, mean_base, mean_var, delta, cliffs_delta, p_value, p_adj_holm, verdict
SSIM, CT, 24, 1.4799, 1.5021, +0.0222, 0.41, 0.0034, 0.0421, SIG
SSIM, PET, 24, ...
SSIM, SPECT, 24, ...
SSIM, ALL, 72, 1.4387, 1.4612, +0.0225, 0.39, 0.0008, 0.0102, SIG
QG, CT, ...
...
```

`verdict ∈ {SIG, NS, MARGINAL}` — MARGINAL khi p < 0.05 nhưng p_adj ≥ 0.05 (không survive correction → KHÔNG claim được).

### Bước 6 — Multi-model comparison (Friedman + Nemenyi)

Khi có ≥ 3 variants (ví dụ: baseline, A1, A2, A3, full):

```python
from scipy.stats import friedmanchisquare
# X[i,j] = z-score chuẩn hoá metric j của model i, average over images
stat, p = friedmanchisquare(*[X[i] for i in range(n_models)])
# nếu p < α: post-hoc Nemenyi
```

Output **CD diagram** (`results_v2/_stats/<run_id>/cd_diagram.png`):

- Trục: average rank của mỗi model (lower = better).
- Đường nối: models KHÔNG khác biệt significant theo Nemenyi.
  → Hình này là 1 trong những hình "winning" để đặt vào luận văn.

### Bước 7 — Plots

Sinh ra `results_v2/_stats/<run_id>/`:

1. `boxplot_<metric>.png` — paired distribution baseline vs variant, per modality.
2. `delta_distribution_<metric>.png` — histogram of (var - base) với 0-line.
3. `cd_diagram.png` (nếu ≥3 model).
4. `summary_table.md` — bảng markdown ready-to-paste vào luận văn.

Dùng `matplotlib` + `seaborn`. KHÔNG dùng plotly (không reproducible PDF).

### Bước 8 — Composite z-update

Nếu variant SIG trên ≥ 50% metrics quan trọng (SSIM, QG, QY, QCB, QM) → đề xuất user update `results_v2/zscore_ranking.csv`:

```python
# Compute new composite z including variant
# Re-rank all models
# Save to zscore_ranking.csv (BACKUP cũ trước)
```

KHÔNG tự update — yêu cầu user xác nhận bằng `--update-ranking`.

## Format report cuối

`results_v2/_stats/<run_id>/REPORT.md`:

```markdown
# Statistical comparison: CDDFuse-<Tag> vs CDDFuse baseline

**Run ID**: <yyyymmdd_hhmmss>
**N pairs**: 72 (24 CT + 24 PET + 24 SPECT)
**Test**: Wilcoxon signed-rank, alternative='greater', α=0.05, correction=Holm-Bonferroni (K=22)

## Bottom line
- **Significant improvement** on N/22 metrics after correction (X% pass-rate).
- **Largest effects**: <metric_1> (δ=0.X), <metric_2> (δ=0.X)
- **Regression** on N metrics (variant worse): <list>
- **Verdict**: <CONFIRM_IMPROVEMENT | INSUFFICIENT_EVIDENCE | MIXED>

## Per-metric table
[markdown table from significance CSV, top 10]

## Per-modality breakdown
- CT: ...
- PET: ...
- SPECT: ...

## Caveats
- Sample size per modality = 24 (relatively small; effect sizes cần ≥ medium để claim)
- Multiple correction applied (Holm-Bonferroni, K=22)
- Test is paired one-sided (variant > baseline). Two-sided available with --twosided.

## Figures
- [CD diagram](cd_diagram.png) (only if ≥3 models compared)
- [Box plots](boxplot_*.png)
```

## Hard rules

- KHÔNG bao giờ chạy stat trên aggregated summary (3-row CSV). Nếu thiếu per-image → BLOCK.
- KHÔNG report p-value mà không kèm effect size.
- KHÔNG bỏ qua multiple-comparison correction khi test ≥ 5 metrics.
- KHÔNG nói "significantly better" nếu chỉ p < 0.05 mà p_adj ≥ 0.05. Phải nói "marginal, không survive correction".
- LUÔN respect chiều metric (lower-better cần đảo dấu trước test).
- LUÔN log alternative hypothesis ("greater" vs "two-sided") — quan trọng khi reviewer hỏi.

## Output template cho bảng luận văn

```markdown
| Variant | SSIM | QG | QY | QCB | QM | Composite z | p_adj |
|---|---|---|---|---|---|---|---|
| CDDFuse (baseline) | 1.439 | 0.702 | 0.956 | 0.858 | 0.595 | +0.93 | — |
| CDDFuse-<Tag> | 1.46X† | 0.7XX† | 0.9XX | ... | ... | +1.0X | 0.012 |

† Significant improvement (Wilcoxon p_adj < 0.05, Holm correction across 22 metrics)
```

→ Dấu † là chuẩn — reviewer hiểu ngay.

## Khi user hỏi "tại sao không chỉ dùng z-score đã có"

Trả lời:

> Composite z-score xếp hạng tương đối, không cho biết khác biệt có vượt độ nhiễu của ảnh-tới-ảnh không. Cùng 1 model, chạy 2 lần với data khác nhau cũng có thể chênh ±0.1 z. Wilcoxon trả lời câu "chênh này có nằm trong nhiễu không" — đó là claim duy nhất reviewer thừa nhận.
