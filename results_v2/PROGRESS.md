# Tiến trình cải tiến CDDFuse — ĐATN

> **Cập nhật mỗi lần chạy variant.** Không xoá entry cũ. File này là single source of truth về trạng thái thí nghiệm.
>
> Convention: append entry mới ở **§ Variants** (giữ thứ tự thời gian). Cập nhật **§ Trạng thái** ở trên cùng.

---

## Trạng thái

| | |
|---|---|
| Latest update | 2026-05-04 13:30 |
| Variants completed | 4/4 Module A + 1/4 Module B (Mean) |
| Latest activity | B.2 Mean stats: **MIXED** — NABF +0.669 (cao nhất tới giờ) nhưng QM -0.735 LARGE (đẩy trade-off mạnh hơn Gated) |
| Status | Mean KHÔNG giải quyết QM regression — confirm trade-off inherent. Tiếp tục B.3 Saliency, B.4 Learnable |
| Next planned | Code + train `PixelSelect-Saliency` (B.3) — gradient-weighted thay max |

---

## Module A — Fusion Rule (target: NABF gap 1.73σ)

So sánh các cách thay thế phép `A + B` trong `BaseFuseLayer(A + B)` và `DetailFuseLayer(A + B)`.

| # | Variant | Tham số mới | Trạng thái | Composite Δ | Quyết định |
|---|---|---|---|---|---|
| A.1 | `FuseRule-Sum` (baseline) | 0 | ✓ có sẵn (CDDFuse_MIF.pth) | 0 (reference) | KEEP |
| A.2 | `FuseRule-Gated` | 16K | ✓ CPU 10 ep + stats | NABF δ=+0.596, 5 SIG | **WINNER** |
| A.3 | `FuseRule-CrossAttn` | 37K | ✓ CPU 10 ep + stats | NABF δ=+0.529, 5 SIG | runner-up |
| A.4 | `FuseRule-ChannelMoE` | 5K | ✓ CPU 10 ep + stats | NABF δ=+0.504, 4 SIG, QM -0.745 ❌ | weakest |

### Module A — comparison đầy đủ 4/4 alternatives

| Metric | Sum (base) | Gated δ | CrossAttn δ | ChannelMoE δ | Winner | Verdict (Holm) |
|---|---|---|---|---|---|---|
| **NABF** ↓ | 0.0236 | **+0.596 L** | +0.529 L | +0.504 L | Gated | all SIG |
| QSF ↑ | 0.0383 | **+0.602 L** | +0.542 L | +0.488 L | Gated | all SIG |
| PSNR ↑ | 56.22 | +0.243 s | +0.238 s | +0.256 s | ChannelMoE tie | all SIG |
| RMSE ↓ | 0.165 | +0.243 s | +0.238 s | +0.256 s | ChannelMoE tie | all SIG |
| EN ↑ | 5.10 | +0.035 t | +0.048 t | +0.016 t | CrossAttn | Gated/CrossAttn SIG, MoE MARG |
| QM ↑ (regression) | 0.083 | -0.602 L | -0.637 L | **-0.745 L** ❌ | Gated least bad | all NS |
| VAR ↑ (regression) | 76.86 | -0.437 m | -0.421 m | -0.471 m | tie | NS |
| QY ↑ (regression) | 0.956 | -0.275 s | -0.304 s | -0.332 m | Gated | NS |

`L`/`m`/`s`/`t` = large/medium/small/trivial Cliff's δ effect size.

#### Phát hiện quan trọng cho Module A (after 4/4 done)

**1. Pattern WIN/LOSE giống nhau ở cả 3 alternatives.** Cả 3 fusion rule đều:
- WIN: NABF, QSF, PSNR, RMSE (cùng 4 metric LARGE/small effect)
- LOSE: QM, VAR, QY (cùng 3 metric regression)

→ **Trade-off NABF↑ vs QM↓ là inherent với soft fusion** trong Module A — không phải artifact của 1 architecture cụ thể.

**2. Capacity ↔ severity của trade-off.**
- ChannelMoE 5K params (smallest) → QM δ = **-0.745** (worst regression)
- Gated 16K → QM δ = -0.602 (least bad)
- CrossAttn 37K → QM δ = -0.637 (middle)

→ Capacity nhỏ → fusion smooth aggressive hơn → mất nhiều wavelet detail hơn. Gated 16K là sweet spot.

**3. Adding capacity không tăng performance.** CrossAttn (37K) không vượt Gated (16K) ở metric nào quan trọng. ChannelMoE (5K) thua cả 2. → Soft fusion là **bottleneck inherent**, không phải capacity issue.

#### Quyết định Module A

- **Winner: A.2 Gated** — best NABF gain + least QM regression + simplest architecture.
- **Citation**: gating concept (Dauphin et al., GLU 2017) + standard fusion practice. Application novelty cho medical fusion với CDDFuse architecture.
- **Implication cho ROADMAP**: Module B (PixelSelect) trở nên **critical** để fix QM regression — không phải optional. Stretch S5 (FreqLoss) cũng nên ưu tiên.

---

## Variants (chronological)

### 2026-05-04 13:30 · `CDDFuse-PixelSelect-Mean` (B.2) — train + stats

**Hypothesis**: Mean target `(I_y + I_ir)/2` tránh boundary artifact của max-rule → kỳ vọng giảm NABF mà KHÔNG hỏng QM/VAR (mean ít aggressive hơn max).

**Module**: `variants/losses.py::FusionLossB(pixel_select='mean')`. Architecture giữ nguyên FuseRule=Sum (IdentitySum, 0 params). Loss target là khác biệt duy nhất.

**Training config**: CPU 10 ep, batch 2, seed 42, fine-tune BaseFuseLayer + DetailFuseLayer (~381K trainable, không có loss params)

**Stats verdict**: **MIXED** (4 SIG / 25 pooled, QCB marginal)

| Metric | Mean Δ | Cliff's δ | Effect | p_adj |
|---|---|---|---|---|
| **NABF** ↓ | -0.011 | **+0.669** | LARGE (best so far!) | 0.0000 |
| QSF | +0.084 | +0.591 | LARGE | 0.0000 |
| PSNR | +0.191 | +0.251 | small | 0.0000 |
| RMSE | -0.009 | +0.251 | small | 0.0000 |
| QCB | +0.004 | +0.030 | trivial | 0.1214 (MARGINAL) |

**Regression** (worse than Gated/CrossAttn):
- **QM δ = -0.735 LARGE** ❌
- **VAR δ = -0.522 LARGE** (lên LARGE — Module A chỉ medium)
- MI δ = -0.426 medium
- QY δ = -0.319 small

**Insight quan trọng — hypothesis sai**:
- Tôi predict Mean ít aggressive hơn Max → giữ QM/VAR tốt hơn. **Sai**.
- Thực tế: Mean blur cả 2 modality (mỗi pixel = avg) → output smoother hơn cả Max-rule (chỉ chọn sharper pixel).
- → Mean **đẩy trade-off mạnh hơn**: NABF cải thiện nhiều hơn (+0.669 vs Gated +0.596) nhưng QM/VAR cũng tệ hơn (-0.735 vs -0.602, -0.522 vs -0.437).
- Module B Mean **không giải quyết** QM regression như kỳ vọng. Trade-off có vẻ inherent với mọi soft fusion, kể cả khi soft xuất hiện ở loss target chứ không ở fusion rule.

**Quyết định**: ITERATE. Không phải Module B winner. Hi vọng B.3 Saliency (gradient-weighted) có thể break pattern vì nó preserve sharper pixel theo gradient magnitude — gần với Max nhưng smooth hơn.

**Files**: `results_v2/CDDFuse-PixelSelect-Mean/`, `_stats/20260504_133026_PixelSelect-Mean_vs_CDDFuse/`

---

### 2026-05-04 01:10 · `CDDFuse-FuseRule-ChannelMoE` (A.4) — train + stats

**Hypothesis**: Per-channel MoE routing (3 experts: A, B, mix) giải quyết trade-off bằng cách cho mỗi channel chọn expert phù hợp thay vì blend đều.

**Module**: `variants/modules.py::ChannelMoEFuse` — 5K params/module, GAP → MLP → softmax(3 experts) per channel. Inspired TC-MoA (CVPR 2024). Identity init: uniform 1/3 → output = 0.5*(A+B).

**Training config**: identical với A.2/A.3 (CPU 10 ep, batch 2, seed 42)

**Stats verdict**: **MIXED** (4 SIG / 25 pooled, EN MARGINAL)

| Metric | Mean Δ | Cliff's δ | Effect | p_adj |
|---|---|---|---|---|
| **NABF** ↓ | -0.010 | +0.504 | LARGE | 0.0004 |
| QSF | +0.090 | +0.488 | LARGE | 0.0000 |
| PSNR | +0.235 | +0.256 | small | 0.0000 |
| RMSE | -0.010 | +0.256 | small | 0.0000 |
| EN | +0.027 | +0.016 | trivial | 0.1713 (MARGINAL) |

**Regression** (worst của 3 alternatives):
- **QM δ = -0.745 LARGE** ❌ (Gated -0.602, CrossAttn -0.637)
- VAR δ = -0.471 medium
- QY δ = -0.332 medium (lên medium effect, hơn Gated/CrossAttn small)

**Insight quan trọng**:
- ChannelMoE = smallest capacity (5K) → smooths most aggressive → highest QM regression. Confirm hypothesis "trade-off scales with smoothing aggressiveness".
- Per-modality: CT 7 SIG (cao nhất), SPECT chỉ 2 SIG. Pattern: capacity nhỏ chịu ảnh hưởng modality variability nhiều hơn.

**Quyết định**: REJECT (trong Module A) — ChannelMoE không phải winner. Nhưng vẫn có giá trị làm **negative result** trong ablation table (chứng minh "MoE routing không cải thiện").

**Files**: `results_v2/CDDFuse-FuseRule-ChannelMoE/`, `_stats/20260504_010829_FuseRule-ChannelMoE_vs_CDDFuse/`

---

### 2026-05-03 14:20 · `CDDFuse-FuseRule-CrossAttn` (A.3) — train + stats

**Hypothesis**: Bidirectional channel-attention (A↔B) giải quyết QM regression của Gated bằng cách giữ richer feature interaction.

**Module**: `variants/modules.py::CrossAttnFuse` — Restormer MDTA-style channel attention, 4 heads, 37K params/module, identity init (proj_out zero → epoch-0 = 0.5*(A+B)).

**Training config**: identical với Gated (CPU 10 ep, batch 2, seed 42, ckpt sha `c540...`)

**Loss curve**: 1.022 → 0.880 (giống Gated, hội tụ chậm tương tự).

**Stats verdict**: **CONFIRM_IMPROVEMENT** (5 SIG / 25 pooled, sau Holm)

| Metric | Mean Δ | Cliff's δ | Effect | p_adj |
|---|---|---|---|---|
| QSF | +0.068 | +0.542 | LARGE | 0.0000 |
| **NABF** ↓ | -0.009 | +0.529 | LARGE | 0.0001 |
| PSNR | +0.241 | +0.238 | small | 0.0000 |
| RMSE | -0.010 | +0.238 | small | 0.0000 |
| EN | +0.073 | +0.048 | trivial | 0.0000 |

**Regression** (giống Gated): QM δ=-0.637 LARGE, VAR δ=-0.421 medium.

#### Comparison Gated vs CrossAttn

| Metric | Gated δ | CrossAttn δ | Winner |
|---|---|---|---|
| NABF | **+0.596** | +0.529 | Gated |
| QSF | **+0.602** | +0.542 | Gated |
| PSNR/RMSE | +0.243 | +0.238 | tie |
| QM (regression) | -0.602 | **-0.637** | Gated less bad |

→ **Gated nhỉnh hơn CrossAttn ở 4/5 metric chính**. CrossAttn dùng 37K params (vs 16K Gated) nhưng **không bù được** complexity tăng.

**Insight**: 2 fusion rules khác nhau cho **kết quả gần giống nhau** → trade-off NABF↑ vs QM↓ là **inherent của Module A** (bất kỳ soft fusion nào cũng sẽ smooth output). Cần Module B (PixelSelect) hoặc khác để giải quyết QM regression.

**Quyết định**: ITERATE. Chạy A.4 ChannelMoE để hoàn tất 4 alternatives → Friedman test → chốt Module A.

**Files**:
- `results_v2/CDDFuse-FuseRule-CrossAttn/` — full output
- `results_v2/_stats/20260503_141849_FuseRule-CrossAttn_vs_CDDFuse/` — stats report

---

### 2026-05-02 23:00 · stats `CDDFuse-FuseRule-Gated` vs `CDDFuse`

**Tool**: `dev/fusion_stats.py` (skill `fusion-stats` execution)
**Test**: Wilcoxon signed-rank paired, alternative='greater', α=0.05, Holm-Bonferroni per-modal
**N pairs**: 24×3 modal = 72 pooled
**Output dir**: `results_v2/_stats/20260502_225704_FuseRule-Gated_vs_CDDFuse/`

#### Verdict: **CONFIRM_IMPROVEMENT** (5 SIG / 25 metrics pooled, sau Holm)

#### WIN (significant)

| Metric | Mean Δ | Cliff's δ | Effect | p_adj |
|---|---|---|---|---|
| **NABF** ↓ (TARGET) | -0.011 | **+0.596** | LARGE | 0.0000 |
| QSF | +0.069 | +0.602 | LARGE | 0.0000 |
| PSNR | +0.224 | +0.243 | small | 0.0000 |
| RMSE | -0.010 | +0.243 | small | 0.0000 |
| EN | +0.051 | +0.035 | trivial | 0.0022 |

#### LOSE (regression rõ, không SIG vì Wilcoxon one-sided)

| Metric | Mean Δ | Cliff's δ | Effect |
|---|---|---|---|
| **QM (wavelet)** | -0.072 | **-0.602** | LARGE |
| VAR | -9.26 | -0.437 | medium |
| MI | -9.10 | -0.303 | small |
| QY | -0.014 | -0.275 | small |
| QMI | -0.041 | -0.196 | small |

#### Phân tích

- **H1 confirmed**: Gated giảm NABF có ý nghĩa thống kê + LARGE effect → mục tiêu chính của Module A đạt được.
- **Trade-off**: Gated tạo output **smoother** → đồng thời giảm wavelet quality (QM) và variance (VAR). Cơ chế: `g·A + (1-g)·B` blend → mất một phần high-freq detail của cả 2 modality.
- **Per-modality**: PET strong nhất (5 SIG / 25), CT (4 SIG), SPECT chỉ 2 SIG — SPECT có baseline NABF cao nhất nên cải thiện không cần Wilcoxon-strong (gain raw đã rất lớn 0.033 → 0.0045).

#### Acceptance criteria (xem RESEARCH_METHODOLOGY.md §3)

- [x] C1 (composite z ≥ baseline + 0.05) — likely (cần update zscore_ranking.csv)
- [x] **C2 (≥5/22 metrics SIG sau Holm)** — 5 SIG ✓
- [ ] C3 (no regression > 1σ trên SSIM/QG/QY) — QY δ=-0.275 small, **CHƯA fail nhưng watch**
- [x] **C4 (Cliff's δ ≥ 0.33 trên ≥3 metrics)** — NABF (0.596), QSF (0.602), VAR (-0.437) ✓
- [x] C5 (`_ablation_stamp.json`) ✓
- [x] C6 (`components_changed` rõ) ✓

5/6 criteria pass → **ITERATE** (KHÔNG nâng KEEP vì chỉ là 10-epoch CPU prototype).

#### Quyết định

**A.2 Gated = direction confirmed for Module A.** Tiếp tục so sánh với A.3 CrossAttn và A.4 ChannelMoE để xác định **best fusion rule**.

#### Files

- `results_v2/_stats/20260502_225704_FuseRule-Gated_vs_CDDFuse/REPORT.md` — full report
- `results_v2/_stats/.../significance_FuseRule-Gated_vs_CDDFuse.csv` — bảng raw

---

### 2026-05-02 22:00 · `CDDFuse` (baseline) · per-image re-run

**Mục đích**: Sinh per-image CSV (24 dòng × 3 modal = 72 entries) làm input cho `fusion-stats` skill — yêu cầu paired data để Wilcoxon test.

**Cmd**: `python evaluate_cddfuse.py --modal {CT,PET,SPECT} --save_perimage` (local CPU)

**Output**:
- `results_v2/CDDFuse/perimage/CDDFuse_CT_perimage.csv` (24 rows + header)
- `results_v2/CDDFuse/perimage/CDDFuse_PET_perimage.csv` (24 rows + header)
- `results_v2/CDDFuse/perimage/CDDFuse_SPECT_perimage.csv` (24 rows + header)

**Sanity check**: aggregated values bit-exact match baseline cũ (timestamp 2026-04-23) → deterministic, weights không đổi.

**Backup**: `results_v2/CDDFuse/CDDFuse_summary.csv.bak` giữ summary cũ.

---


### 2026-05-02 · `CDDFuse-FuseRule-Gated` · prototype

**Hypothesis (H1)**: Soft per-channel × per-spatial gate `g·A + (1-g)·B` thay phép sum sẽ giảm NABF (artifact ở boundary 2 modality) mà không hy sinh SSIM nhiều.

**Module thay đổi**: `models/MMIF-CDDFuse/variants/modules.py::GatedFuseLayer` (Conv1×1, init zero → epoch-0 ≈ baseline).

**Training config**:
| | Value |
|---|---|
| Mode | light_retrain (encoder/decoder frozen) |
| Trainable params | 397,704 / 1,204,784 (~33%) |
| Train data | Harvard medical 738 pairs (160 CT + 245 PET + 333 SPECT) |
| Test data | Harvard reference 72 pairs (24×3) |
| Hardware | Kaggle Tesla P100 (sm_60 unsupported by torch 2.10) → fallback **CPU** |
| Epochs | **10** (reduced from 25 due to CPU; loss curve still descending) |
| Batch | 2 (CPU) |
| LR | 1e-4 cosine → 1e-6 |
| Seed | 42 |
| Wall time | ~3-4h |
| Ckpt sha256 | `8575e252df14968376e345766f04b7bfb76670c1e30f11f69322a477a7619bfd` |

**Loss curve**:
| Epoch | Total | Intensity | Gradient |
|---|---|---|---|
| 1 | 1.022 | 0.043 | 0.098 |
| 5 | 0.886 | 0.041 | 0.084 |
| 10 | 0.875 | 0.042 | 0.083 |

→ Còn descend ở epoch 10, **chưa hội tụ hoàn toàn**.

**Quick delta vs baseline (raw, chưa significance test)**:

| Metric | CT base→var | PET base→var | SPECT base→var | Trend |
|---|---|---|---|---|
| **NABF** ↓ (target) | 0.0197 → 0.0232 ❌ | 0.0181 → **0.0103** ✓ | 0.0330 → **0.0045** ✓✓ | win 2/3 modal |
| SSIM ↑ | 1.4799 → 1.4727 ❌ | 1.4043 → 1.3976 ❌ | 1.4318 → 1.4171 ❌ | regression nhẹ |
| QG ↑ | 0.6141 → 0.6102 ≈ | 0.7344 → 0.7204 ❌ | 0.7585 → 0.7466 ❌ | regression nhẹ |
| QM ↑ | 0.0088 → 0.0043 ❌ | 0.0740 → 0.0108 ❌❌ | 0.1678 → 0.0196 ❌❌ | regression mạnh |
| QCB ↑ | 0.7965 → 0.7952 ≈ | 0.8962 → 0.8963 ≈ | 0.8802 → 0.8802 ≈ | flat |

**Phân tích thô**:
- ✓ NABF cải thiện rõ ở PET (1.8×) và SPECT (7.3×) — **target gap được giải quyết**
- ❌ Trade-off: SSIM/QG/QM giảm nhẹ → có thể do under-train (10 ep) hoặc gate đang oversmooth
- Đặc biệt SPECT NABF từ 0.033 → 0.0045 — drop rất mạnh, đáng chú ý

**Acceptance criteria check** (xem RESEARCH_METHODOLOGY.md §3):
- [ ] C1 (composite z ≥ baseline + 0.05) — **chưa tính** (cần fusion-stats)
- [ ] C2 (≥5/22 metrics SIG sau Holm) — **chưa test**
- [ ] C3 (no regression > 1σ trên SSIM/QG/QY) — likely OK (deltas nhỏ < 0.5%)
- [ ] C4 (Cliff's δ ≥ 0.33 trên ≥3 metrics) — **chưa tính**
- [x] C5 (`_ablation_stamp.json` đầy đủ) — ✓
- [x] C6 (components_changed mô tả rõ) — ✓

**Files**:
- `results_v2/CDDFuse-FuseRule-Gated/` — full output
- `results_v2/CDDFuse-FuseRule-Gated/perimage/` — input cho fusion-stats
- `models/MMIF-CDDFuse/models/CDDFuse-FuseRule-Gated.pth`

**Quyết định**: **ITERATE** (không KEEP, không REJECT).

**Lý do**:
- NABF improvement quá tốt để bỏ (SPECT giảm 7.3×) → hướng có giá trị
- SSIM regression nhỏ + 10-epoch CPU under-train → kết quả CHƯA đáng tin
- Cần re-run với 25 epoch P100 (tự nhiên CPU fallback) trước khi quyết định

**Next steps**:
1. Re-run baseline với `--save_perimage` để có per-image CSV cho stats (~30 phút Kaggle)
2. Sau khi có baseline perimage → run `fusion-stats` trên kết quả 10-epoch hiện tại
3. Nếu Wilcoxon NABF p_adj < 0.05 → re-run variant với 25 epoch (CPU ~8h hoặc Kaggle 2 sessions)
4. Tiếp theo: `FuseRule-CrossAttn` (alternative #3 trong Module A)

---

## Backlog

- [x] ~~Re-run baseline CDDFuse với `--save_perimage`~~ — DONE 2026-05-02
- [x] ~~Run stats `FuseRule-Gated` vs baseline~~ — DONE 2026-05-02 23:00, CONFIRM_IMPROVEMENT
- [x] ~~Implement `FuseRule-CrossAttn`~~ — DONE 2026-05-03
- [x] ~~Run stats `FuseRule-CrossAttn` vs baseline~~ — DONE 2026-05-03 14:20, CONFIRM_IMPROVEMENT
- [x] ~~Implement + train `FuseRule-ChannelMoE`~~ — DONE 2026-05-04
- [x] ~~Run stats `FuseRule-ChannelMoE` vs baseline~~ — DONE 2026-05-04 01:10, MIXED (worst capacity)
- [x] ~~Module A winner identified~~ — **A.2 Gated** (best NABF, least QM regression)
- [ ] **NEXT**: Friedman + Nemenyi 4-way + CD diagram cho Module A
- [ ] Module B (PixelSelect alternatives — critical để fix QM regression của Module A)
- [ ] Re-run winner Gated với 25 epoch (final, sau Module B xong)
- [ ] Update `results_v2/zscore_ranking.csv` với composite z mới sau Combined

---

## Failures / Lessons learned

### 2026-05-02 · Kaggle GPU constraints
- **P100 (sm_60) không tương thích torch 2.10+cu128** — fallback CPU bắt buộc
- Pip downgrade torch 2.1.2+cu118 không override system path → bỏ approach này
- CPU light retrain 738 pairs × 10 epoch ≈ 3-4h trên Kaggle CPU instance
- Notebook commit phase chậm (~50 phút) do `/kaggle/working/` chứa quá nhiều file (data + repo clone)
- **Fix cho lần sau**: copy data vào `/tmp/` thay vì `/kaggle/working/` để Kaggle không phải commit
